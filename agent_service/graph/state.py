"""
alas/agent_service/graph/state.py

Defines the AgentState TypedDict that flows through the LangGraph graph.
Every node reads from and writes to this object — it is the single source
of truth for a conversation turn.
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


# ---------------------------------------------------------------------------
# Sub-schemas (nested, typed for clarity)
# ---------------------------------------------------------------------------

class TurnScore(TypedDict):
    """Rubric scores emitted by the EvaluationStep for one student turn."""
    turn_index: int
    clarity: float          # 0.0 – 1.0
    empathy: float
    structure: float
    relevance: float
    confidence: float
    composite: float
    rationale: str          # short LLM-generated explanation


class AvatarDirective(TypedDict):
    """Metadata the LLM encodes alongside the spoken response."""
    emotion: str            # e.g. "neutral", "concerned", "encouraging"
    scenario_phase: str     # current phase name
    branch_signal: str | None   # non-null triggers a phase transition
    hidden_notes: str       # internal reasoning — never shown to student


class RetrievedContext(TypedDict):
    """Results from semantic memory retrieval."""
    scenario_chunks: list[str]   # RAG hits from scenario knowledge base
    user_summaries: list[str]    # prior session summaries for this user
    behavioral_notes: list[str]  # observed patterns across sessions


# ---------------------------------------------------------------------------
# Primary graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # ---- Identity ----------------------------------------------------------
    session_id: str
    user_id: str
    scenario_id: str

    # ---- Short-term memory (current session) --------------------------------
    messages: Annotated[list[BaseMessage], operator.add]   # full turn history
    turn_index: int
    scenario_phase: str     # e.g. "setup" | "core" | "escalation" | "resolution"
    turns_in_phase: int      # resets to 0 on each phase transition
    scenario_config: dict[str, Any]   # loaded scenario JSON

    # ---- Retrieved context (populated by ContextAssembly) ------------------
    retrieved_context: RetrievedContext

    # ---- Prompt and LLM outputs --------------------------------------------
    system_prompt: str
    raw_llm_response: str
    avatar_response: str        # the spoken text only
    avatar_directive: AvatarDirective | None

    # ---- Evaluation --------------------------------------------------------
    turn_scores: Annotated[list[TurnScore], operator.add]
    session_eval_summary: dict[str, Any] | None   # computed at session end

    # ---- Control flow ------------------------------------------------------
    should_end_session: bool
    error: str | None
