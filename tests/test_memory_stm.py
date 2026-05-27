"""
tests/test_memory_stm.py

Unit tests for the Short-Term Memory layer.
Only tests the in-process STM — no ChromaDB / embedding calls.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_service.memory.manager import ShortTermMemory


class TestShortTermMemory:
    def _make_stm(self) -> ShortTermMemory:
        return ShortTermMemory(
            session_id="sess-1",
            user_id="user-1",
            scenario_id="job_interview_v1",
        )

    def test_initial_state(self):
        stm = self._make_stm()
        assert stm.turn_index == 0
        assert stm.messages == []
        assert stm.turn_scores == []
        assert stm.scenario_phase == "setup"

    def test_add_turn_increments_index(self):
        stm = self._make_stm()
        stm.add_turn(HumanMessage(content="Hi"), AIMessage(content="Hello"))
        assert stm.turn_index == 1
        assert len(stm.messages) == 2

    def test_add_multiple_turns(self):
        stm = self._make_stm()
        for i in range(5):
            stm.add_turn(HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))
        assert stm.turn_index == 5
        assert len(stm.messages) == 10

    def test_get_recent_turns_windowing(self):
        stm = self._make_stm()
        for i in range(10):
            stm.add_turn(HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))
        recent = stm.get_recent_turns(n=3)
        # Should return 3 pairs = 6 messages
        assert len(recent) == 6
        # Last human message should be Q9
        human_msgs = [m for m in recent if isinstance(m, HumanMessage)]
        assert human_msgs[-1].content == "Q9"

    def test_get_recent_turns_smaller_than_window(self):
        stm = self._make_stm()
        stm.add_turn(HumanMessage(content="Q0"), AIMessage(content="A0"))
        recent = stm.get_recent_turns(n=8)
        assert len(recent) == 2  # only 1 turn exists

    def test_add_score(self):
        stm = self._make_stm()
        score = {"turn_index": 0, "composite": 0.75, "clarity": 0.8,
                 "empathy": 0.7, "structure": 0.75, "relevance": 0.8,
                 "confidence": 0.7, "rationale": "Good answer."}
        stm.add_score(score)
        assert len(stm.turn_scores) == 1
        assert stm.turn_scores[0]["composite"] == 0.75

    def test_to_summary_text_includes_key_fields(self):
        stm = self._make_stm()
        stm.add_turn(HumanMessage(content="Q"), AIMessage(content="A"))
        stm.turn_scores.append({
            "turn_index": 0, "composite": 0.8, "clarity": 0.8,
            "empathy": 0.8, "structure": 0.8, "relevance": 0.8,
            "confidence": 0.8, "rationale": "r",
        })
        summary = stm.to_summary_text()
        assert "sess-1" in summary
        assert "user-1" in summary
        assert "job_interview_v1" in summary
        assert "0.80" in summary

    def test_summary_no_scores(self):
        stm = self._make_stm()
        summary = stm.to_summary_text()
        assert "sess-1" in summary
        assert "composite" not in summary
