"""
alas/agent_service/memory/manager.py

Multi-layer memory hierarchy for ALAS.

Layer 1 — Short-Term Memory (STM)
  - In-process dict keyed by session_id
  - Holds: recent N turns, current phase, per-turn scores
  - Lifetime: session duration only

Layer 2 — Episodic Memory (EM)
  - Persisted to ChromaDB collection "episodes"
  - Holds: post-session summaries, behavioral observations, skill snapshots
  - Retrieval: by user_id metadata filter (no embedding needed, but kept in
    vector store for unified API and future semantic upgrade)

Layer 3 — Semantic Memory (SM)
  - ChromaDB collections: "scenario_knowledge" + "user_profiles"
  - Holds: chunked scenario documents, embedded user behavioral profiles
  - Retrieval: ANN over embeddings (true semantic search)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# ---------------------------------------------------------------------------
# Config (falls back gracefully if env not loaded)
# ---------------------------------------------------------------------------
import os
from dotenv import load_dotenv
load_dotenv()  # load .env before reading any env vars

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
SESSION_WINDOW = int(os.getenv("SESSION_WINDOW", "8"))
MAX_EPISODES = int(os.getenv("MAX_EPISODE_SUMMARIES", "5"))


# ---------------------------------------------------------------------------
# Layer 1: Short-Term Memory
# ---------------------------------------------------------------------------

@dataclass
class ShortTermMemory:
    """
    Ephemeral, in-process session state.
    Holds the sliding window of recent turns and live session metadata.
    """
    session_id: str
    user_id: str
    scenario_id: str
    scenario_phase: str = "setup"
    turn_index: int = 0
    turns_in_phase: int = 0
    messages: list[BaseMessage] = field(default_factory=list)
    turn_scores: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_turn(self, human_msg: BaseMessage, ai_msg: BaseMessage) -> None:
        self.messages.append(human_msg)
        self.messages.append(ai_msg)
        self.turn_index += 1

    def get_recent_turns(self, n: int = SESSION_WINDOW) -> list[BaseMessage]:
        """Return the last n*2 messages (n human + n ai)."""
        return self.messages[-(n * 2):]

    def add_score(self, score: dict) -> None:
        self.turn_scores.append(score)

    def to_summary_text(self) -> str:
        """Produce a plain-text summary of the session for episodic storage."""
        lines = [f"Session {self.session_id} | User {self.user_id} | Scenario {self.scenario_id}"]
        lines.append(f"Turns completed: {self.turn_index} | Final phase: {self.scenario_phase}")
        if self.turn_scores:
            avg = lambda k: sum(s.get(k, 0) for s in self.turn_scores) / len(self.turn_scores)
            lines.append(
                f"Avg scores — clarity:{avg('clarity'):.2f} empathy:{avg('empathy'):.2f} "
                f"structure:{avg('structure'):.2f} relevance:{avg('relevance'):.2f} "
                f"confidence:{avg('confidence'):.2f} composite:{avg('composite'):.2f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ChromaDB client factory (shared singleton)
# ---------------------------------------------------------------------------

_chroma_client: chromadb.ClientAPI | None = None

def _get_chroma() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def _openai_ef():
    """OpenAI embedding function for ChromaDB."""
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name=EMBED_MODEL,
    )


# ---------------------------------------------------------------------------
# Layer 2: Episodic Memory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Stores and retrieves per-user session summaries and behavioral notes.
    Each document represents one completed session.
    Filtered retrieval: always by user_id, no cross-user leakage.
    """

    COLLECTION = "episodes"

    def __init__(self):
        self._col = _get_chroma().get_or_create_collection(
            name=self.COLLECTION,
            embedding_function=_openai_ef(),
            metadata={"hnsw:space": "cosine"},
        )

    def store_session(
        self,
        session_id: str,
        user_id: str,
        scenario_id: str,
        summary_text: str,
        behavioral_notes: str = "",
        scores: dict | None = None,
    ) -> None:
        """Persist a completed session summary into episodic memory."""
        doc_id = f"ep_{session_id}"
        metadata = {
            "user_id": user_id,
            "scenario_id": scenario_id,
            "session_id": session_id,
            "timestamp": time.time(),
            "behavioral_notes": behavioral_notes,
            "scores_json": json.dumps(scores or {}),
        }
        self._col.upsert(
            ids=[doc_id],
            documents=[summary_text],
            metadatas=[metadata],
        )

    def retrieve_for_user(
        self,
        user_id: str,
        query: str = "",
        n_results: int = MAX_EPISODES,
    ) -> list[dict]:
        """
        Fetch prior session summaries for a user.
        If query is provided, semantic similarity is used to rank results.
        Falls back to fetching by metadata filter when no query is given.
        """
        where = {"user_id": user_id}
        try:
            if query:
                results = self._col.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where,
                )
            else:
                results = self._col.get(
                    where=where,
                    limit=n_results,
                )
        except Exception:
            return []

        docs = results.get("documents") or []
        metas = results.get("metadatas") or []

        # Flatten nested lists returned by .query()
        if docs and isinstance(docs[0], list):
            docs = docs[0]
            metas = metas[0] if metas else []

        return [
            {"summary": d, "meta": m}
            for d, m in zip(docs, metas)
        ]

    def get_behavioral_notes(self, user_id: str) -> list[str]:
        """Extract behavioral_notes strings from all stored episodes for a user."""
        episodes = self.retrieve_for_user(user_id, n_results=10)
        notes = []
        for ep in episodes:
            note = ep.get("meta", {}).get("behavioral_notes", "").strip()
            if note:
                notes.append(note)
        return notes


# ---------------------------------------------------------------------------
# Layer 3: Semantic Memory
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    Vector-search over scenario knowledge chunks and user profile embeddings.

    Two sub-collections:
      scenario_knowledge — chunked documents about scenarios (rubrics, domain context)
      user_profiles      — embedded summaries of user skill trajectories
    """

    SCENARIO_COL = "scenario_knowledge"
    USER_COL = "user_profiles"

    def __init__(self):
        ef = _openai_ef()
        self._scenario_col = _get_chroma().get_or_create_collection(
            name=self.SCENARIO_COL,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._user_col = _get_chroma().get_or_create_collection(
            name=self.USER_COL,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    # --- Scenario knowledge -------------------------------------------------

    def ingest_scenario(self, scenario_id: str, chunks: list[str]) -> None:
        """Embed and store scenario knowledge chunks."""
        ids = [f"{scenario_id}_chunk_{i}" for i in range(len(chunks))]
        metas = [{"scenario_id": scenario_id, "chunk_index": i} for i in range(len(chunks))]
        self._scenario_col.upsert(ids=ids, documents=chunks, metadatas=metas)

    def retrieve_scenario_context(
        self,
        query: str,
        scenario_id: str,
        n_results: int = 3,
    ) -> list[str]:
        """RAG: top-k scenario chunks most relevant to the student's utterance."""
        try:
            results = self._scenario_col.query(
                query_texts=[query],
                n_results=n_results,
                where={"scenario_id": scenario_id},
            )
            docs = results.get("documents", [[]])[0]
            return docs
        except Exception:
            return []

    # --- User profiles -------------------------------------------------------

    def upsert_user_profile(self, user_id: str, profile_text: str) -> None:
        """Store or update an embedded user skill profile."""
        self._user_col.upsert(
            ids=[f"profile_{user_id}"],
            documents=[profile_text],
            metadatas=[{"user_id": user_id, "updated_at": time.time()}],
        )

    def retrieve_similar_users(self, query: str, n_results: int = 2) -> list[str]:
        """Find users with similar profiles — useful for adaptive benchmarking."""
        try:
            results = self._user_col.query(query_texts=[query], n_results=n_results)
            return results.get("documents", [[]])[0]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Unified Memory Manager (facade used by graph nodes)
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Single access point for all three memory layers.
    Graph nodes only interact with this class — they don't know which
    layer is serving each request.
    """

    def __init__(self):
        self._stm: dict[str, ShortTermMemory] = {}   # session_id -> STM
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

    # ---- STM management ----------------------------------------------------

    def create_session(
        self,
        session_id: str,
        user_id: str,
        scenario_id: str,
        scenario_phase: str = "setup",
    ) -> ShortTermMemory:
        stm = ShortTermMemory(
            session_id=session_id,
            user_id=user_id,
            scenario_id=scenario_id,
            scenario_phase=scenario_phase,
        )
        self._stm[session_id] = stm
        return stm

    def get_session(self, session_id: str) -> ShortTermMemory | None:
        return self._stm.get(session_id)

    def end_session(self, session_id: str, behavioral_notes: str = "") -> None:
        """Flush STM to episodic storage, then evict from process memory."""
        stm = self._stm.get(session_id)
        if not stm:
            return

        summary = stm.to_summary_text()
        avg_scores = {}
        if stm.turn_scores:
            keys = ["clarity", "empathy", "structure", "relevance", "confidence", "composite"]
            avg_scores = {
                k: round(sum(s.get(k, 0) for s in stm.turn_scores) / len(stm.turn_scores), 3)
                for k in keys
            }

        self.episodic.store_session(
            session_id=session_id,
            user_id=stm.user_id,
            scenario_id=stm.scenario_id,
            summary_text=summary,
            behavioral_notes=behavioral_notes,
            scores=avg_scores,
        )

        # Update semantic user profile
        profile_text = (
            f"User {stm.user_id} — Scenario: {stm.scenario_id}\n"
            f"{summary}\nNotes: {behavioral_notes}"
        )
        self.semantic.upsert_user_profile(stm.user_id, profile_text)

        del self._stm[session_id]

    # ---- Retrieval façade --------------------------------------------------

    def assemble_context(
        self,
        session_id: str,
        student_utterance: str,
    ) -> dict:
        """
        Called by ContextAssembly node.
        Returns a dict with all three memory layers merged.
        """
        stm = self._stm.get(session_id)
        if not stm:
            return {"scenario_chunks": [], "user_summaries": [], "behavioral_notes": []}

        # Semantic: scenario knowledge relevant to this utterance
        scenario_chunks = self.semantic.retrieve_scenario_context(
            query=student_utterance,
            scenario_id=stm.scenario_id,
        )

        # Episodic: prior sessions for this user
        prior_episodes = self.episodic.retrieve_for_user(
            user_id=stm.user_id,
            query=student_utterance,
        )
        user_summaries = [ep["summary"] for ep in prior_episodes]

        # Episodic: behavioral notes only
        behavioral_notes = self.episodic.get_behavioral_notes(stm.user_id)

        return {
            "scenario_chunks": scenario_chunks,
            "user_summaries": user_summaries,
            "behavioral_notes": behavioral_notes,
        }


# Module-level singleton — imported by graph nodes
memory_manager = MemoryManager()