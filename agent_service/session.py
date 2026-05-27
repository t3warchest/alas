"""
alas/agent_service/session.py

SessionOrchestrator: the single public interface between the API layer
and the LangGraph agent graph.

Responsibilities:
  1. Create / retrieve sessions (STM init + scenario loading)
  2. Route each student message through the graph
  3. Expose a streaming generator for real-time token delivery
  4. Surface evaluation scores and session summaries
  5. Gracefully end sessions

This is the only file the FastAPI routers import from agent_service internals.
Everything else is an implementation detail.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage

from agent_service.graph.agent_graph import agent_graph
from agent_service.graph.state import AgentState
from agent_service.memory.manager import ShortTermMemory, memory_manager
from agent_service.scenarios.registry import get_scenario, ingest_all_scenarios
from agent_service.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Startup hook (call once when the FastAPI app starts)
# ---------------------------------------------------------------------------

def startup() -> None:
    """
    Warm up the agent service:
      - Embed scenario knowledge into ChromaDB (idempotent upsert)
    Called from FastAPI lifespan.
    """
    log.info("alas_startup", detail="Ingesting scenario knowledge into vector store...")
    ingest_all_scenarios(memory_manager)
    log.info("alas_startup", detail="Agent service ready.")


# ---------------------------------------------------------------------------
# DTOs (data flowing in/out of the orchestrator)
# ---------------------------------------------------------------------------

class SessionInfo:
    """Returned when a session is created."""
    __slots__ = ("session_id", "scenario_id", "scenario_title", "persona_name", "opening_line")

    def __init__(
        self,
        session_id: str,
        scenario_id: str,
        scenario_title: str,
        persona_name: str,
        opening_line: str,
    ):
        self.session_id = session_id
        self.scenario_id = scenario_id
        self.scenario_title = scenario_title
        self.persona_name = persona_name
        self.opening_line = opening_line

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "scenario_id": self.scenario_id,
            "scenario_title": self.scenario_title,
            "persona_name": self.persona_name,
            "opening_line": self.opening_line,
        }


class TurnResult:
    """Returned after each student message is processed."""
    __slots__ = (
        "session_id", "turn_index", "avatar_response",
        "emotion", "scenario_phase", "turn_score",
        "session_ended", "session_summary",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


# ---------------------------------------------------------------------------
# Session Orchestrator
# ---------------------------------------------------------------------------

class SessionOrchestrator:
    """
    Stateless facade over the LangGraph agent and memory system.
    All mutable state lives in MemoryManager (STM) or the graph state dict.
    """

    async def create_session(
        self,
        user_id: str,
        scenario_id: str,
        session_id: str | None = None,
    ) -> SessionInfo:
        """
        Initialise a new session:
          - Load scenario config
          - Create STM entry
          - Run one graph turn with a synthetic [SESSION_START] message
            so the avatar delivers an opening line
        """
        session_id = session_id or str(uuid.uuid4())
        scenario = get_scenario(scenario_id)

        # Create the STM slot
        memory_manager.create_session(
            session_id=session_id,
            user_id=user_id,
            scenario_id=scenario_id,
            scenario_phase="setup",
        )

        log.info(
            "session_created",
            session_id=session_id,
            user_id=user_id,
            scenario_id=scenario_id,
        )

        # Trigger an opening turn
        opening_state = await self._run_graph(
            session_id=session_id,
            user_id=user_id,
            scenario_id=scenario_id,
            scenario=scenario,
            student_message="[SESSION_START] Please open the session with a warm, natural greeting appropriate for the scenario. Introduce yourself briefly.",
        )

        opening_line = opening_state.get("avatar_response", "Hello! Let's get started.")

        return SessionInfo(
            session_id=session_id,
            scenario_id=scenario_id,
            scenario_title=scenario["title"],
            persona_name=scenario["persona"]["name"],
            opening_line=opening_line,
        )

    async def send_message(
        self,
        session_id: str,
        user_id: str,
        student_message: str,
    ) -> TurnResult:
        """
        Process one student message through the full agent graph.
        Returns a TurnResult with the avatar response and evaluation score.
        """
        stm = memory_manager.get_session(session_id)
        if stm is None:
            raise ValueError(f"Session {session_id!r} not found. Create it first.")

        scenario_id = stm.scenario_id
        scenario = get_scenario(scenario_id)

        log.info(
            "turn_received",
            session_id=session_id,
            turn_index=stm.turn_index,
            msg_len=len(student_message),
        )

        output_state = await self._run_graph(
            session_id=session_id,
            user_id=user_id,
            scenario_id=scenario_id,
            scenario=scenario,
            student_message=student_message,
            existing_stm=stm,
        )

        directive = output_state.get("avatar_directive") or {}
        turn_scores = output_state.get("turn_scores", [])
        latest_score = turn_scores[-1] if turn_scores else None
        session_ended = output_state.get("should_end_session", False)
        session_summary = output_state.get("session_eval_summary") if session_ended else None

        log.info(
            "turn_completed",
            session_id=session_id,
            turn_index=output_state.get("turn_index", 0),
            emotion=directive.get("emotion"),
            phase=directive.get("scenario_phase"),
            session_ended=session_ended,
            composite=latest_score.get("composite") if latest_score else None,
        )

        return TurnResult(
            session_id=session_id,
            turn_index=output_state.get("turn_index", 0),
            avatar_response=output_state.get("avatar_response", ""),
            emotion=directive.get("emotion", "neutral"),
            scenario_phase=output_state.get("scenario_phase", stm.scenario_phase),
            turn_score=latest_score,
            session_ended=session_ended,
            session_summary=session_summary,
        )

    async def stream_message(
        self,
        session_id: str,
        user_id: str,
        student_message: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Streaming variant of send_message.

        Yields dicts of two types:
          {"type": "token",   "content": "<token>"}          — live tokens
          {"type": "result",  "data": TurnResult.to_dict()}  — final result
          {"type": "error",   "message": "..."}              — on failure

        In this implementation we run the graph normally then simulate
        token streaming by yielding words. In a full production system,
        the LLMCall node would use a streaming callback that pushes tokens
        into an asyncio.Queue that this generator drains concurrently.
        """
        try:
            result = await self.send_message(session_id, user_id, student_message)
            avatar_text = result.avatar_response

            # Simulate word-level streaming (replace with real token streaming in prod)
            words = avatar_text.split()
            for i, word in enumerate(words):
                chunk = word if i == 0 else " " + word
                yield {"type": "token", "content": chunk}
                await asyncio.sleep(0.03)  # ~33 words/sec — natural reading pace

            yield {"type": "result", "data": result.to_dict()}

        except Exception as e:
            log.error("stream_error", session_id=session_id, error=str(e))
            yield {"type": "error", "message": str(e)}

    async def end_session(self, session_id: str) -> dict:
        """
        Force-end a session and retrieve the evaluation summary.
        Safe to call even if the session already ended via phase_router.
        """
        stm = memory_manager.get_session(session_id)
        if stm is None:
            return {"session_id": session_id, "message": "Session already ended or not found."}

        from agent_service.evaluation.engine import (
            aggregate_session_scores,
            generate_behavioral_notes,
        )

        summary = aggregate_session_scores(stm.turn_scores)
        notes = generate_behavioral_notes(summary)
        memory_manager.end_session(session_id, behavioral_notes=notes)

        log.info("session_ended", session_id=session_id)
        return {"session_id": session_id, "summary": summary}

    def get_session_scores(self, session_id: str) -> list[dict]:
        """Return all per-turn scores accumulated so far in this session."""
        stm = memory_manager.get_session(session_id)
        if not stm:
            return []
        return list(stm.turn_scores)

    # -----------------------------------------------------------------------
    # Internal: build initial state and invoke graph
    # -----------------------------------------------------------------------

    async def _run_graph(
        self,
        session_id: str,
        user_id: str,
        scenario_id: str,
        scenario: dict,
        student_message: str,
        existing_stm: ShortTermMemory | None = None,
    ) -> dict[str, Any]:
        """
        Build the AgentState for this turn and run the compiled graph.
        Returns the final output state dict.
        """
        stm = existing_stm or memory_manager.get_session(session_id)
        current_phase = stm.scenario_phase if stm else "setup"
        current_turn = stm.turn_index if stm else 0

        # Reconstruct message history from STM
        history = stm.messages[:] if stm else []

        # Append the new student message
        history.append(HumanMessage(content=student_message))

        initial_state: AgentState = {
            "session_id": session_id,
            "user_id": user_id,
            "scenario_id": scenario_id,
            "messages": history,
            "turn_index": current_turn,
            "scenario_phase": current_phase,
            "turns_in_phase": getattr(stm, "turns_in_phase", 0) if stm else 0,
            "scenario_config": scenario,
            "retrieved_context": {
                "scenario_chunks": [],
                "user_summaries": [],
                "behavioral_notes": [],
            },
            "system_prompt": "",
            "raw_llm_response": "",
            "avatar_response": "",
            "avatar_directive": None,
            "turn_scores": [],
            "session_eval_summary": None,
            "should_end_session": False,
            "error": None,
        }

        output = await agent_graph.ainvoke(initial_state)
        return output


# Module-level singleton
orchestrator = SessionOrchestrator()
