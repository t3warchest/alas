"""
alas/agent_service/graph/agent_graph.py

LangGraph orchestration graph for the ALAS agent service.

Graph topology (DAG with conditional edges):

  ContextAssembly
       │
       ▼
  PromptBuilder
       │
       ▼
   LLMCall  ──(streaming tokens)──► client
       │
       ▼
  ResponseParser
       │
       ▼
  MemoryUpdate ──────────────────── (async write, non-blocking)
       │
       ▼
  EvaluationStep ────────────────── (async, writes scores to state)
       │
       ▼
  PhaseRouter ──► [continue | end_session]

Each node is a pure function:  AgentState -> dict[str, Any]
The dict is merged into the state by LangGraph (reducer semantics).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agent_service.evaluation.engine import (
    aggregate_session_scores,
    evaluate_turn,
    generate_behavioral_notes,
)
from agent_service.graph.prompt_builder import build_system_prompt
from agent_service.graph.response_parser import parse_llm_response
from agent_service.graph.state import AgentState
from agent_service.memory.manager import memory_manager
from agent_service.scenarios.branching import PhaseContext, PhaseEngine
from agent_service.scenarios.registry import get_scenario, get_scenario_typed

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def _get_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENAI_MODEL,
        streaming=streaming,
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Node 1: ContextAssembly
# ---------------------------------------------------------------------------

async def context_assembly(state: AgentState) -> dict[str, Any]:
    """
    Pull context from all three memory layers.
    
    Short-term : already in state.messages (LangGraph carries it)
    Episodic   : prior session summaries + behavioral notes
    Semantic   : scenario knowledge chunks relevant to this utterance
    """
    session_id = state["session_id"]
    
    # The student's most recent message is the last HumanMessage
    recent_messages = state.get("messages", [])
    student_utterance = ""
    for msg in reversed(recent_messages):
        if isinstance(msg, HumanMessage):
            student_utterance = str(msg.content)
            break

    retrieved = memory_manager.assemble_context(
        session_id=session_id,
        student_utterance=student_utterance,
    )

    return {"retrieved_context": retrieved}


# ---------------------------------------------------------------------------
# Node 2: PromptBuilder
# ---------------------------------------------------------------------------

async def prompt_builder(state: AgentState) -> dict[str, Any]:
    """
    Construct the system prompt dynamically from scenario config + retrieved context.
    Rebuilt every turn so memory and phase changes take effect immediately.
    """
    scenario_config = state.get("scenario_config") or get_scenario(state["scenario_id"])
    current_phase = state.get("scenario_phase", "setup")
    retrieved_context = state.get("retrieved_context", {})

    system_prompt = build_system_prompt(
        scenario_config=scenario_config,
        current_phase=current_phase,
        retrieved_context=retrieved_context,
    )

    return {"system_prompt": system_prompt}


# ---------------------------------------------------------------------------
# Node 3: LLMCall
# ---------------------------------------------------------------------------

async def llm_call(state: AgentState) -> dict[str, Any]:
    """
    Send assembled messages to the LLM.
    Supports streaming — tokens flow back to caller via the state update.
    
    In a real system this node would yield tokens via a callback/queue.
    Here we collect the full response but structure it for easy streaming upgrade.
    """
    llm = _get_llm(streaming=False)

    system_prompt = state["system_prompt"]
    conversation_history = state.get("messages", [])

    # Construct the message list: system + recent turns
    stm = memory_manager.get_session(state["session_id"])
    recent_turns = stm.get_recent_turns() if stm else conversation_history[-16:]

    messages = [SystemMessage(content=system_prompt)] + recent_turns

    try:
        response = await llm.ainvoke(messages)
        raw_content = response.content
    except Exception as e:
        # Graceful degradation: avatar acknowledges delay
        raw_content = (
            "I'm just taking a moment to think through that carefully. "
            "Could you give me a second?\n<<<METADATA>>>\n"
            '{"emotion": "neutral", "scenario_phase": "'
            + state.get("scenario_phase", "core")
            + '", "branch_signal": null, "hidden_notes": "LLM error fallback."}'
        )

    return {"raw_llm_response": raw_content}


# ---------------------------------------------------------------------------
# Node 4: ResponseParser
# ---------------------------------------------------------------------------

async def response_parser(state: AgentState) -> dict[str, Any]:
    """
    Split the raw LLM output into:
      - avatar_response: the spoken conversational text
      - avatar_directive: the hidden metadata JSON
    """
    raw = state.get("raw_llm_response", "")
    spoken_text, directive = parse_llm_response(raw)

    return {
        "avatar_response": spoken_text,
        "avatar_directive": directive,
    }


# ---------------------------------------------------------------------------
# Node 5: MemoryUpdate
# ---------------------------------------------------------------------------

async def memory_update(state: AgentState) -> dict[str, Any]:
    """
    Write the completed turn into STM and run the branching engine.

    Two transition sources are consulted in priority order:
      1. LLM branch_signal — the avatar explicitly signals a transition
         (used for score_threshold and branch_table transitions where the
          LLM has been prompted to watch for conditions and signal them)
      2. PhaseEngine — deterministic evaluation of always_after_max and
         fallback conditions; always fires even if the LLM missed it

    The first source that fires wins. STM is updated to the new phase.
    """
    session_id = state["session_id"]
    stm = memory_manager.get_session(session_id)

    if stm is None:
        return {}

    # --- Persist this turn's messages to STM ---
    messages = state.get("messages", [])
    human_msg = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    ai_response = state.get("avatar_response", "")
    ai_msg = AIMessage(content=ai_response)

    if human_msg:
        stm.add_turn(human_msg, ai_msg)

    # --- Source 1: LLM-signalled branch ---
    directive = state.get("avatar_directive") or {}
    llm_branch = directive.get("branch_signal")

    if llm_branch and llm_branch != stm.scenario_phase:
        stm.scenario_phase = llm_branch
        stm.turns_in_phase = 0
        return {"scenario_phase": llm_branch}

    # --- Source 2: PhaseEngine (deterministic) ---
    try:
        scenario = get_scenario_typed(state["scenario_id"])
        engine = PhaseEngine(scenario)

        # Build rolling score averages from accumulated scores
        all_scores = list(stm.turn_scores)
        rolling: dict[str, float] = {}
        if all_scores:
            dims = ["clarity", "empathy", "structure", "relevance", "confidence", "composite"]
            rolling = {
                d: sum(s.get(d, 0) for s in all_scores) / len(all_scores)
                for d in dims
            }

        ctx = PhaseContext(
            current_phase_name=stm.scenario_phase,
            turns_in_phase=getattr(stm, "turns_in_phase", stm.turn_index),
            total_turns=stm.turn_index,
            rolling_scores=rolling,
            last_student_utterance=str(human_msg.content) if human_msg else "",
        )

        decision = engine.evaluate(ctx)

        if decision.should_transition and decision.target_phase:
            stm.scenario_phase = decision.target_phase
            stm.turns_in_phase = 0
            return {"scenario_phase": decision.target_phase}

    except Exception as e:
        # Branching failure must never crash the conversation
        from agent_service.utils.logging import get_logger
        get_logger(__name__).warning("branching_engine_error", error=str(e))

    # Increment phase-local turn counter
    stm.turns_in_phase = getattr(stm, "turns_in_phase", 0) + 1

    return {}


# ---------------------------------------------------------------------------
# Node 6: EvaluationStep
# ---------------------------------------------------------------------------

async def evaluation_step(state: AgentState) -> dict[str, Any]:
    """
    Score the student's most recent utterance asynchronously.
    Never blocks the conversation — if it's slow, the score is just late.
    
    Scores accumulate in state.turn_scores (Annotated with operator.add).
    """
    session_id = state["session_id"]
    stm = memory_manager.get_session(session_id)
    if not stm:
        return {}

    # Find the student's last utterance and the avatar's preceding message
    messages = state.get("messages", [])
    student_text = ""
    avatar_question = ""

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage) and not student_text:
            student_text = str(msg.content)
        elif isinstance(msg, AIMessage) and not avatar_question and student_text:
            avatar_question = str(msg.content)
            break

    if not student_text:
        return {}

    scenario_config = state.get("scenario_config") or get_scenario(state["scenario_id"])
    skill_dimensions = scenario_config.get("skill_dimensions", {})

    score = await evaluate_turn(
        turn_index=state.get("turn_index", 0),
        scenario_title=scenario_config.get("title", ""),
        avatar_message=avatar_question,
        student_response=student_text,
        skill_dimensions=skill_dimensions,
    )

    # Sync score back to STM for session-end aggregation
    stm.add_score(score)

    return {
        "turn_scores": [score],
        "turn_index": state.get("turn_index", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node 7: PhaseRouter (conditional edge resolver)
# ---------------------------------------------------------------------------

def phase_router(state: AgentState) -> str:
    """
    Terminal gate: decides whether the whole session should end.
    Phase transitions (setup→core→escalation etc.) are handled upstream
    by memory_update + PhaseEngine. This node only gates session termination.

    End conditions (in priority order):
      1. should_end_session flag already set
      2. Total turn count >= max_total_turns
      3. We are in a completion phase AND that phase is exhausted
      4. Early-exit: composite score >= early_exit_score (if configured)
    """
    if state.get("should_end_session"):
        return "end_session"

    current_phase = state.get("scenario_phase", "setup")
    turn_index = state.get("turn_index", 0)

    try:
        scenario = get_scenario_typed(state["scenario_id"])
        exit_conds = scenario.exit_conditions

        # Condition 2: hard turn cap
        if turn_index >= exit_conds.max_total_turns:
            return "end_session"

        # Condition 3: completion phase exhausted
        stm = memory_manager.get_session(state["session_id"])
        if current_phase in exit_conds.completion_phases and stm:
            phase_obj = scenario.get_phase(current_phase)
            turns_in_phase = getattr(stm, "turns_in_phase", 0)
            if phase_obj and turns_in_phase >= phase_obj.max_turns:
                return "end_session"

        # Condition 4: early exit on exceptional performance
        if exit_conds.early_exit_score is not None:
            all_scores = stm.turn_scores if stm else []
            if all_scores:
                composite_avg = sum(s.get("composite", 0) for s in all_scores) / len(all_scores)
                if composite_avg >= exit_conds.early_exit_score:
                    return "end_session"

    except Exception:
        # Fallback to simple dict-based check if typed scenario unavailable
        scenario_config = state.get("scenario_config") or {}
        exit_conditions = scenario_config.get("exit_conditions", {})
        if turn_index >= exit_conditions.get("max_total_turns", 20):
            return "end_session"
        completion_phases = exit_conditions.get("completion_phases", ["resolution"])
        if current_phase in completion_phases:
            directive = state.get("avatar_directive") or {}
            if directive.get("branch_signal") == current_phase:
                return "end_session"

    return END


async def end_session_node(state: AgentState) -> dict[str, Any]:
    """
    Flush session to episodic memory, compute final report.
    """
    session_id = state["session_id"]
    turn_scores = state.get("turn_scores", [])

    summary = aggregate_session_scores(turn_scores)
    notes = generate_behavioral_notes(summary)
    memory_manager.end_session(session_id, behavioral_notes=notes)

    return {
        "session_eval_summary": summary,
        "should_end_session": True,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    """
    Construct and compile the LangGraph state graph.
    
    Flow:
      context_assembly → prompt_builder → llm_call → response_parser
        → memory_update → evaluation_step → [phase_router]
                                              ├─ END (await next message)
                                              └─ end_session
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("context_assembly", context_assembly)
    graph.add_node("prompt_builder", prompt_builder)
    graph.add_node("llm_call", llm_call)
    graph.add_node("response_parser", response_parser)
    graph.add_node("memory_update", memory_update)
    graph.add_node("evaluation_step", evaluation_step)
    graph.add_node("end_session", end_session_node)

    # Entry point
    graph.set_entry_point("context_assembly")

    # Linear edges
    graph.add_edge("context_assembly", "prompt_builder")
    graph.add_edge("prompt_builder", "llm_call")
    graph.add_edge("llm_call", "response_parser")
    graph.add_edge("response_parser", "memory_update")
    graph.add_edge("memory_update", "evaluation_step")

    # Conditional routing after evaluation
    graph.add_conditional_edges(
        "evaluation_step",
        phase_router,
        {
            END: END,
            "end_session": "end_session",
        },
    )
    graph.add_edge("end_session", END)

    return graph.compile()


# Module-level compiled graph — import and use directly
agent_graph = build_agent_graph()
