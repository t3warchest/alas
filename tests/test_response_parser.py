"""
tests/test_response_parser.py

Unit tests for the response parser.
No LLM calls — pure string parsing logic.
"""

import pytest
from agent_service.graph.response_parser import parse_llm_response, DELIMITER


class TestParseWellFormed:
    def test_clean_delimiter(self):
        raw = (
            "That's a great question. Tell me more about your experience.\n"
            f"{DELIMITER}\n"
            '{"emotion": "curious", "scenario_phase": "core", "branch_signal": null, "hidden_notes": "Testing depth."}'
        )
        spoken, directive = parse_llm_response(raw)
        assert "Tell me more" in spoken
        assert directive["emotion"] == "curious"
        assert directive["scenario_phase"] == "core"
        assert directive["branch_signal"] is None

    def test_trailing_whitespace_stripped(self):
        raw = f"Hello!  \n{DELIMITER}\n" + '{"emotion": "warm", "scenario_phase": "setup", "branch_signal": null, "hidden_notes": ""}'
        spoken, directive = parse_llm_response(raw)
        assert spoken == "Hello!"

    def test_branch_signal_propagated(self):
        raw = (
            f"Let's wrap up.\n{DELIMITER}\n"
            '{"emotion": "neutral", "scenario_phase": "resolution", "branch_signal": "resolution", "hidden_notes": "Transitioning."}'
        )
        _, directive = parse_llm_response(raw)
        assert directive["branch_signal"] == "resolution"


class TestParsemalformed:
    def test_no_delimiter_returns_full_text_as_spoken(self):
        raw = "I didn't include any metadata this time."
        spoken, directive = parse_llm_response(raw)
        assert "metadata" in spoken
        # Should still return a valid (default) directive
        assert directive["emotion"] in {"neutral", "curious", "concerned", "encouraging", "challenging", "warm", "disappointed"}

    def test_invalid_json_returns_defaults(self):
        raw = f"Response text.\n{DELIMITER}\nnot valid json at all"
        spoken, directive = parse_llm_response(raw)
        assert spoken == "Response text."
        assert directive["emotion"] == "neutral"
        assert directive["branch_signal"] is None

    def test_unknown_emotion_clamped_to_neutral(self):
        raw = (
            f"Text.\n{DELIMITER}\n"
            '{"emotion": "VERY_ANGRY", "scenario_phase": "core", "branch_signal": null, "hidden_notes": ""}'
        )
        _, directive = parse_llm_response(raw)
        assert directive["emotion"] == "neutral"

    def test_invalid_phase_falls_back(self):
        raw = (
            f"Text.\n{DELIMITER}\n"
            '{"emotion": "neutral", "scenario_phase": "nonexistent_phase", "branch_signal": null, "hidden_notes": ""}'
        )
        _, directive = parse_llm_response(raw)
        assert directive["scenario_phase"] in {"setup", "core", "escalation", "resolution"}

    def test_empty_metadata_block(self):
        raw = f"Text.\n{DELIMITER}\n"
        spoken, directive = parse_llm_response(raw)
        assert spoken == "Text."
        assert directive["emotion"] == "neutral"

    def test_completely_empty_input(self):
        spoken, directive = parse_llm_response("")
        assert spoken == ""
        assert isinstance(directive, dict)
