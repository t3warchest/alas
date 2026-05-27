"""
tests/test_prompt_builder.py

Unit tests for the dynamic prompt builder.
Verifies that all blocks appear and that memory context is injected.
"""

import pytest
from agent_service.graph.prompt_builder import build_system_prompt
from agent_service.scenarios.registry import get_scenario


SCENARIO = get_scenario("job_interview_v1")


class TestPromptBuilder:
    def _build(self, phase="core", context=None) -> str:
        ctx = context or {"scenario_chunks": [], "user_summaries": [], "behavioral_notes": []}
        return build_system_prompt(SCENARIO, phase, ctx)

    def test_contains_persona_name(self):
        prompt = self._build()
        assert "Alex" in prompt

    def test_contains_scenario_title(self):
        prompt = self._build()
        assert "Software Engineering Job Interview" in prompt

    def test_contains_current_phase(self):
        prompt = self._build(phase="escalation")
        assert "ESCALATION" in prompt.upper()

    def test_contains_rubric_dimensions(self):
        prompt = self._build()
        for dim in ["clarity", "empathy", "structure", "relevance", "confidence"]:
            assert dim in prompt.lower()

    def test_memory_context_injected_when_present(self):
        ctx = {
            "scenario_chunks": ["STAR method is important."],
            "user_summaries": ["Session 1: composite 0.6, improving."],
            "behavioral_notes": ["Tends to ramble under pressure."],
        }
        prompt = self._build(context=ctx)
        assert "STAR method is important" in prompt
        assert "composite 0.6" in prompt
        assert "Tends to ramble" in prompt

    def test_first_session_message_shown_when_no_memory(self):
        prompt = self._build(context={"scenario_chunks": [], "user_summaries": [], "behavioral_notes": []})
        assert "first session" in prompt.lower()

    def test_output_format_block_present(self):
        prompt = self._build()
        assert "<<<METADATA>>>" in prompt
        assert "emotion" in prompt

    def test_constraints_block_present(self):
        prompt = self._build()
        assert "HARD CONSTRAINTS" in prompt

    def test_setup_phase(self):
        prompt = self._build(phase="setup")
        assert "setup" in prompt.lower()

    def test_resolution_phase(self):
        prompt = self._build(phase="resolution")
        assert "resolution" in prompt.lower()

    def test_knowledge_chunks_appear_in_domain_section(self):
        ctx = {
            "scenario_chunks": ["Chunk A from scenario KB.", "Chunk B about interviews."],
            "user_summaries": [],
            "behavioral_notes": [],
        }
        prompt = self._build(context=ctx)
        assert "Chunk A from scenario KB" in prompt
        assert "Chunk B about interviews" in prompt
