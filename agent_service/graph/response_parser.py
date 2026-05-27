"""
alas/agent_service/graph/response_parser.py

Parses the raw LLM output into two distinct parts:
  1. avatar_response  — the spoken conversational text
  2. avatar_directive — the hidden JSON metadata block

The LLM is instructed to emit these separated by <<<METADATA>>>.
This parser handles malformed output gracefully so a bad response
never crashes the session.
"""

from __future__ import annotations
import json
import re

DELIMITER = "<<<METADATA>>>"

VALID_EMOTIONS = {
    "neutral", "curious", "concerned", "encouraging",
    "challenging", "warm", "disappointed",
}
VALID_PHASES = {
    # job_interview_v1
    "setup", "core", "escalation", "resolution",
    # difficult_coworker_v1
    "opening", "disclosure", "understanding", "rupture", "action_planning", "close",
    # salary_negotiation_v1
    "anchor", "exploration",
}


def parse_llm_response(raw: str) -> tuple[str, dict]:
    """
    Returns (spoken_text, directive_dict).

    spoken_text  — safe to display/TTS to student
    directive    — validated metadata dict (always returns a dict even on failure)
    """
    if DELIMITER in raw:
        parts = raw.split(DELIMITER, 1)
        spoken_text = parts[0].strip()
        metadata_raw = parts[1].strip()
    else:
        # Fallback: try to extract trailing JSON block
        spoken_text, metadata_raw = _extract_trailing_json(raw)

    directive = _parse_metadata(metadata_raw)
    return spoken_text, directive


def _extract_trailing_json(text: str) -> tuple[str, str]:
    """
    Heuristic: if no delimiter, look for a JSON block at the end of the text.
    Returns (spoken_part, json_part).
    """
    match = re.search(r"\{[^{}]*\}$", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        spoken_part = text[: match.start()].strip()
        return spoken_part, json_str
    return text.strip(), "{}"


def _parse_metadata(raw: str) -> dict:
    """
    Safely parse and validate the metadata JSON.
    Returns sensible defaults on any parse error.
    """
    defaults = {
        "emotion": "neutral",
        "scenario_phase": "core",
        "branch_signal": None,
        "hidden_notes": "",
    }

    if not raw or raw.strip() in ("{}", ""):
        return defaults

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to salvage by extracting the first {...}
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return defaults
        else:
            return defaults

    # Validate and sanitize fields
    emotion = data.get("emotion", "neutral")
    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"

    phase = data.get("scenario_phase", "core")
    if phase not in VALID_PHASES:
        phase = defaults["scenario_phase"]

    branch = data.get("branch_signal")
    if branch not in VALID_PHASES and branch is not None:
        branch = None

    return {
        "emotion": emotion,
        "scenario_phase": phase,
        "branch_signal": branch,
        "hidden_notes": str(data.get("hidden_notes", "")),
    }
