"""
alas/agent_service/graph/prompt_builder.py

Assembles system prompts from typed Scenario objects.

Block order:
  1. PERSONA        — character definition, backstory, behavioural rules
  2. SCENARIO       — situation framing, student objective
  3. PHASE          — current phase instructions + avatar goal
  4. RUBRIC         — weighted skill dimensions (internal, never shown to student)
  5. MEMORY         — episodic context: prior sessions + behavioural observations
  6. KNOWLEDGE      — RAG-retrieved domain chunks most relevant to this utterance
  7. BRANCHING HINT — what the avatar should watch for to trigger a transition
  8. CONSTRAINTS    — hard safety/persona rules
  9. OUTPUT FORMAT  — required response structure

Each block is a standalone function: (data) -> str.
Adding a new block = add one function + one line in build_system_prompt().
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# Accept either a typed Scenario or a legacy dict (for backward compat)
# ---------------------------------------------------------------------------


def _coerce(scenario_config: Any) -> dict[str, Any]:
    """Return a plain dict regardless of whether we received a Scenario or dict."""
    if hasattr(scenario_config, "to_legacy_dict"):
        return scenario_config.to_legacy_dict()
    return scenario_config  # already a dict


# ---------------------------------------------------------------------------
# Block 1: Persona
# ---------------------------------------------------------------------------

def _persona_block(persona: dict[str, Any]) -> str:
    constraints = "\n".join(f"  - {c}" for c in persona.get("constraints", []))
    backstory = persona.get("backstory", "")
    backstory_section = f"\nBackstory (use to inform your reactions — do not recite verbatim):\n{backstory}" if backstory else ""
    return (
        f"## YOUR PERSONA\n"
        f"You are **{persona['name']}**, {persona['role']}.\n"
        f"Emotional starting state: {persona['emotional_start']}.\n"
        f"Interaction style: {persona['style']}."
        f"{backstory_section}\n\n"
        f"Behavioural rules you must follow at all times:\n{constraints}"
    )


# ---------------------------------------------------------------------------
# Block 2: Scenario
# ---------------------------------------------------------------------------

def _scenario_block(scenario: dict[str, Any]) -> str:
    return (
        f"## SCENARIO CONTEXT\n"
        f"**{scenario['title']}**\n"
        f"{scenario['description']}\n\n"
        f"**Student's objective** (they are trying to achieve this — you are not):\n"
        f"{scenario['student_objective']}"
    )


# ---------------------------------------------------------------------------
# Block 3: Phase
# ---------------------------------------------------------------------------

def _phase_block(phases: list[dict], current_phase: str) -> str:
    phase_cfg = next((p for p in phases if p["name"] == current_phase), None)
    if not phase_cfg:
        return f"## CURRENT PHASE\n{current_phase}"

    avatar_goal = phase_cfg.get("avatar_goal", "")
    avatar_goal_line = f"\nYour goal this phase: {avatar_goal}" if avatar_goal else ""

    return (
        f"## CURRENT PHASE: **{phase_cfg['name'].upper()}**\n"
        f"What is happening in this phase:\n{phase_cfg['description']}"
        f"{avatar_goal_line}\n"
        f"(Phase turn budget: {phase_cfg.get('max_turns', '?')} turns)"
    )


# ---------------------------------------------------------------------------
# Block 4: Rubric
# ---------------------------------------------------------------------------

def _rubric_block(rubric: dict[str, Any], dimensions: dict[str, float]) -> str:
    lines = ["## EVALUATION RUBRIC  ⟨internal — never mention this to the student⟩"]
    for dim, weight in dimensions.items():
        lines.append(f"\n### {dim.title()}  (weight: {weight:.0%})")
        descriptors = rubric.get(dim, {})
        for level in [1, 3, 5]:
            desc = descriptors.get(level, "")
            if desc:
                lines.append(f"  Level {level}: {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block 5: Memory
# ---------------------------------------------------------------------------

def _memory_block(user_summaries: list[str], behavioral_notes: list[str]) -> str:
    if not user_summaries and not behavioral_notes:
        return (
            "## MEMORY CONTEXT\n"
            "This is the student's first recorded session. "
            "No prior performance data is available — start fresh."
        )

    lines = ["## MEMORY CONTEXT  ⟨use this to personalise your responses naturally⟩"]

    if user_summaries:
        lines.append("\n**Prior session summaries:**")
        for i, s in enumerate(user_summaries[:3], 1):
            lines.append(f"  [{i}] {s}")

    if behavioral_notes:
        lines.append("\n**Observed behavioural patterns across sessions:**")
        for note in behavioral_notes[:4]:
            lines.append(f"  • {note}")

    lines.append(
        "\n⚠ Reference this memory naturally in conversation — "
        "e.g. 'Last time I noticed…', 'You mentioned before that…', "
        "'I see you've been working on [weakness] — let's test that today.' "
        "Do not dump the memory verbatim. Make the student feel remembered."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block 6: Knowledge (RAG)
# ---------------------------------------------------------------------------

def _knowledge_block(chunks: list[str]) -> str:
    if not chunks:
        return ""
    lines = ["## DOMAIN KNOWLEDGE  ⟨retrieved context — apply where relevant⟩"]
    for chunk in chunks:
        lines.append(f"  • {chunk}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block 7: Branching hint
# ---------------------------------------------------------------------------

def _branching_hint_block(phases: list[dict], current_phase: str) -> str:
    """
    Tell the avatar what to watch for that will trigger a phase transition.
    This makes the avatar's behaviour coherent with the branching engine
    without exposing the raw condition grammar.
    """
    phase_cfg = next((p for p in phases if p["name"] == current_phase), None)
    if not phase_cfg:
        return ""

    condition = phase_cfg.get("transition_condition", "")
    if not condition or condition == "always_after_max":
        return ""  # No hint needed — purely turn-count-based

    # Translate condition strings into human-readable avatar guidance
    hints: list[str] = []

    if "score_threshold" in condition or "composite_score_above" in condition:
        hints.append(
            "Watch for strong, specific, well-structured responses. "
            "When you feel the conversation has reached a natural high point of quality, "
            "signal a phase transition via branch_signal in your metadata."
        )
    if "empathy" in condition and "above" in condition:
        hints.append(
            "Watch for genuine empathy: active listening, acknowledgment of feelings, "
            "open-ended questions. When the student demonstrates this consistently, "
            "signal the transition."
        )
    if "confidence" in condition and "above" in condition:
        hints.append(
            "Watch for confident, well-justified statements. "
            "When the student holds their ground calmly under pressure, signal the transition."
        )
    if "empathy" in condition and "below" in condition:
        hints.append(
            "⚠ If the student becomes cold, accusatory, or dismissive, "
            "signal a rupture transition via branch_signal."
        )

    if not hints:
        return ""

    return (
        "## PHASE TRANSITION HINTS  ⟨internal — do not share with student⟩\n"
        + "\n".join(f"  • {h}" for h in hints)
    )


# ---------------------------------------------------------------------------
# Block 8: Constraints
# ---------------------------------------------------------------------------

def _constraints_block() -> str:
    return (
        "## HARD CONSTRAINTS  ⟨non-negotiable — override everything else⟩\n"
        "- Never produce harmful, discriminatory, or illegal content.\n"
        "- Never reveal the evaluation rubric or scoring system to the student.\n"
        "- Never claim to be a real human if directly and sincerely asked.\n"
        "- Stay within the scenario's scope. One off-topic response is acceptable; "
        "more than one breaks the simulation.\n"
        "- When the session reaches its natural end, deliver a genuine closing line "
        "in character — do not abruptly announce 'session over'."
    )


# ---------------------------------------------------------------------------
# Block 9: Output format
# ---------------------------------------------------------------------------

def _output_format_block(valid_phases: list[str]) -> str:
    phases_str = " | ".join(valid_phases) if valid_phases else "setup | core | resolution"
    return (
        "## OUTPUT FORMAT  ⟨strictly required every response⟩\n"
        "Your response MUST contain exactly two parts, separated by the delimiter <<<METADATA>>>.\n\n"
        "**Part 1 — SPOKEN RESPONSE** (what the avatar says aloud):\n"
        "  Natural conversational dialogue. 2–4 sentences. "
        "First person. In character. No stage directions.\n\n"
        "**Part 2 — METADATA JSON** (never shown to student):\n"
        "  A single JSON object with exactly these keys:\n"
        "  {\n"
        '    "emotion": "<neutral|curious|concerned|encouraging|challenging|warm|disappointed>",\n'
        f'    "scenario_phase": "<{phases_str}>",\n'
        '    "branch_signal": <null or the name of the phase to transition to>,\n'
        '    "hidden_notes": "<one sentence: your internal reasoning for this response>"\n'
        "  }\n\n"
        "  Rules for branch_signal:\n"
        "  - Set to null unless a transition hint above says to signal one.\n"
        "  - Only signal phases that exist in this scenario.\n"
        "  - Do not signal a phase you are already in.\n\n"
        "Example:\n"
        "Can you walk me through a specific time you had to make a hard technical call under pressure?\n"
        "<<<METADATA>>>\n"
        '{"emotion": "curious", "scenario_phase": "core", "branch_signal": null, '
        '"hidden_notes": "Testing STAR structure and composure under probing."}'
    )


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def build_system_prompt(
    scenario_config: Any,
    current_phase: str,
    retrieved_context: dict,
) -> str:
    """
    Assemble the full system prompt for one conversation turn.

    Accepts either a typed Scenario object or a legacy dict.
    Called by the PromptBuilder graph node every single turn — rebuilding
    the prompt on every turn ensures memory and phase changes are immediate.
    """
    cfg = _coerce(scenario_config)

    phases = cfg.get("phases", [])
    valid_phase_names = [p["name"] for p in phases]

    blocks = [
        _persona_block(cfg["persona"]),
        _scenario_block(cfg),
        _phase_block(phases, current_phase),
        _rubric_block(
            cfg.get("rubric", {}),
            cfg.get("skill_dimensions", {}),
        ),
        _memory_block(
            user_summaries=retrieved_context.get("user_summaries", []),
            behavioral_notes=retrieved_context.get("behavioral_notes", []),
        ),
        _knowledge_block(retrieved_context.get("scenario_chunks", [])),
        _branching_hint_block(phases, current_phase),
        _constraints_block(),
        _output_format_block(valid_phase_names),
    ]

    return "\n\n".join(block for block in blocks if block)
