"""
tests/test_evaluation.py

Unit tests for evaluation aggregation logic.
No LLM calls — tests the score math and trend detection only.
"""

import pytest
from agent_service.evaluation.engine import aggregate_session_scores, generate_behavioral_notes


def _make_score(turn_index: int, composite: float, **overrides) -> dict:
    base = {
        "turn_index": turn_index,
        "clarity": composite,
        "empathy": composite,
        "structure": composite,
        "relevance": composite,
        "confidence": composite,
        "composite": composite,
        "rationale": f"Turn {turn_index} rationale.",
    }
    base.update(overrides)
    return base


class TestAggregation:
    def test_empty_scores_returns_error(self):
        result = aggregate_session_scores([])
        assert "error" in result

    def test_single_turn(self):
        scores = [_make_score(0, 0.7)]
        result = aggregate_session_scores(scores)
        assert result["turns_evaluated"] == 1
        assert result["averages"]["composite"] == pytest.approx(0.7)

    def test_average_calculation(self):
        scores = [
            _make_score(0, 0.4),
            _make_score(1, 0.6),
            _make_score(2, 0.8),
        ]
        result = aggregate_session_scores(scores)
        assert result["averages"]["composite"] == pytest.approx(0.6, abs=0.01)

    def test_improving_trend_detected(self):
        # First half: 0.3, second half: 0.8 → improving
        scores = [_make_score(i, v) for i, v in enumerate([0.3, 0.3, 0.8, 0.8])]
        result = aggregate_session_scores(scores)
        assert result["trend"] == "improving"
        assert result["trend_delta"] > 0

    def test_declining_trend_detected(self):
        scores = [_make_score(i, v) for i, v in enumerate([0.9, 0.9, 0.3, 0.3])]
        result = aggregate_session_scores(scores)
        assert result["trend"] == "declining"
        assert result["trend_delta"] < 0

    def test_stable_trend(self):
        scores = [_make_score(i, 0.6) for i in range(4)]
        result = aggregate_session_scores(scores)
        assert result["trend"] == "stable"

    def test_strongest_weakest_dimensions(self):
        scores = [
            _make_score(0, 0.5, clarity=0.9, empathy=0.2, structure=0.5,
                        relevance=0.5, confidence=0.5, composite=0.5),
        ]
        result = aggregate_session_scores(scores)
        assert result["strongest_dimension"] == "clarity"
        assert result["weakest_dimension"] == "empathy"

    def test_strongest_weakest_turns(self):
        scores = [_make_score(0, 0.2), _make_score(1, 0.9)]
        result = aggregate_session_scores(scores)
        assert result["weakest_turn"]["turn_index"] == 0
        assert result["strongest_turn"]["turn_index"] == 1


class TestBehavioralNotes:
    def test_notes_contain_key_fields(self):
        summary = {
            "averages": {
                "clarity": 0.7, "empathy": 0.5, "structure": 0.8,
                "relevance": 0.6, "confidence": 0.7, "composite": 0.68,
            },
            "trend": "improving",
            "weakest_dimension": "empathy",
            "strongest_dimension": "structure",
            "turns_evaluated": 5,
        }
        notes = generate_behavioral_notes(summary)
        assert "0.68" in notes
        assert "improving" in notes
        assert "structure" in notes
        assert "empathy" in notes
        assert "5" in notes
