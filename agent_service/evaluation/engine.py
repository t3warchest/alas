"""
alas/agent_service/evaluation/engine.py

Lightweight evaluation engine that scores each student turn against the
scenario rubric using a fast LLM (gpt-4o-mini).

Design principles:
  - Runs asynchronously alongside the main conversation — never blocks it
  - Scores are accumulated in STM and surfaced at session end
  - Produces structured JSON output: one score object per turn
  - The evaluator is a separate LLM call with a focused, minimal prompt
"""

from __future__ import annotations
import json
import os
import re
from typing import Any

from openai import AsyncOpenAI

EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


# ---------------------------------------------------------------------------
# Evaluator prompt
# ---------------------------------------------------------------------------

_EVALUATOR_SYSTEM = """You are an objective educational evaluator assessing a student's communication skills.

You will receive:
1. The scenario context (brief)
2. The avatar's question/statement
3. The student's response

Score the student's response on these dimensions, each from 0.0 to 1.0:
- clarity: How clearly and precisely did they communicate?
- empathy: Did they show awareness of the other person's perspective?
- structure: Was the response well-organized (e.g., STAR method if applicable)?
- relevance: Did they directly address what was asked?
- confidence: Did they sound assured without being arrogant?

Also produce:
- composite: weighted average (you will be given weights)
- rationale: ONE sentence explaining the most significant strength or weakness

Respond ONLY with a valid JSON object. No preamble, no markdown fences.
Example:
{"clarity": 0.8, "empathy": 0.6, "structure": 0.7, "relevance": 0.9, "confidence": 0.75, "composite": 0.76, "rationale": "Strong STAR structure but lacked specific quantified results."}"""


def _build_eval_prompt(
    scenario_title: str,
    avatar_message: str,
    student_response: str,
    skill_dimensions: dict[str, float],
) -> str:
    weights = ", ".join(f"{k}: {v:.0%}" for k, v in skill_dimensions.items())
    return (
        f"Scenario: {scenario_title}\n"
        f"Avatar said: {avatar_message}\n"
        f"Student responded: {student_response}\n"
        f"Dimension weights for composite: {weights}"
    )


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

async def evaluate_turn(
    turn_index: int,
    scenario_title: str,
    avatar_message: str,
    student_response: str,
    skill_dimensions: dict[str, float],
) -> dict[str, Any]:
    """
    Score one student turn. Returns a TurnScore-compatible dict.
    On any failure, returns a neutral score with an error rationale.
    """
    prompt = _build_eval_prompt(
        scenario_title, avatar_message, student_response, skill_dimensions
    )

    try:
        response = await _get_client().chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": _EVALUATOR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        raw = response.choices[0].message.content or "{}"

        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)

        # Clamp all numeric values to [0, 1]
        keys = ["clarity", "empathy", "structure", "relevance", "confidence", "composite"]
        for k in keys:
            data[k] = max(0.0, min(1.0, float(data.get(k, 0.5))))

        data["turn_index"] = turn_index
        data["rationale"] = str(data.get("rationale", ""))
        return data

    except Exception as e:
        # Evaluation failure must never crash the session
        return {
            "turn_index": turn_index,
            "clarity": 0.5,
            "empathy": 0.5,
            "structure": 0.5,
            "relevance": 0.5,
            "confidence": 0.5,
            "composite": 0.5,
            "rationale": f"[Evaluation error: {str(e)[:60]}]",
        }


# ---------------------------------------------------------------------------
# Session-level aggregation
# ---------------------------------------------------------------------------

def aggregate_session_scores(turn_scores: list[dict]) -> dict[str, Any]:
    """
    Aggregate per-turn scores into a session-level report.
    Includes: averages, trend (improving/declining), notable turns.
    """
    if not turn_scores:
        return {"error": "No turn scores available."}

    dims = ["clarity", "empathy", "structure", "relevance", "confidence", "composite"]
    n = len(turn_scores)

    averages = {d: round(sum(s.get(d, 0) for s in turn_scores) / n, 3) for d in dims}

    # Trend: compare first half vs second half
    mid = max(1, n // 2)
    first_half = turn_scores[:mid]
    second_half = turn_scores[mid:] or turn_scores  # fallback for single-turn sessions
    first_avg = sum(s.get("composite", 0) for s in first_half) / len(first_half)
    second_avg = sum(s.get("composite", 0) for s in second_half) / len(second_half)
    trend_delta = round(second_avg - first_avg, 3)

    if trend_delta > 0.05:
        trend = "improving"
    elif trend_delta < -0.05:
        trend = "declining"
    else:
        trend = "stable"

    # Notable turns (lowest and highest composite)
    sorted_turns = sorted(turn_scores, key=lambda s: s.get("composite", 0))
    weakest = sorted_turns[0] if sorted_turns else None
    strongest = sorted_turns[-1] if sorted_turns else None

    # Strongest and weakest skill dimensions
    best_dim = max(averages, key=lambda k: averages[k] if k != "composite" else -1)
    worst_dim = min(averages, key=lambda k: averages[k] if k != "composite" else 2)

    return {
        "turns_evaluated": n,
        "averages": averages,
        "trend": trend,
        "trend_delta": trend_delta,
        "strongest_dimension": best_dim,
        "weakest_dimension": worst_dim,
        "weakest_turn": weakest,
        "strongest_turn": strongest,
        "rationales": [s.get("rationale", "") for s in turn_scores],
    }


def generate_behavioral_notes(session_summary: dict) -> str:
    """
    Produce a short behavioral note string to store in episodic memory.
    This becomes the 'behavioral_notes' field retrieved in future sessions.
    """
    avg = session_summary.get("averages", {})
    weak = session_summary.get("weakest_dimension", "unknown")
    strong = session_summary.get("strongest_dimension", "unknown")
    trend = session_summary.get("trend", "stable")
    composite = avg.get("composite", 0.5)

    return (
        f"Composite: {composite:.2f} ({trend}). "
        f"Strength: {strong} ({avg.get(strong, 0):.2f}). "
        f"Growth area: {weak} ({avg.get(weak, 0):.2f}). "
        f"Evaluated over {session_summary.get('turns_evaluated', 0)} turns."
    )
