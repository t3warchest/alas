"""
tests/test_agent_client.py

Unit tests for gateway/services/agent_client.py.

Tests the streaming event translation logic (stream_turn) using a mock
send_message so no real HTTP calls are made.
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("JWT_SECRET", "test-secret-for-client-tests")

import pytest

from gateway.services.agent_client import AgentServiceClient, AgentServiceError
from shared.contracts.events import (
    WS_EVT_TOKEN, WS_EVT_TURN_DONE, WS_EVT_SCORE, WS_EVT_SESSION_END,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> AgentServiceClient:
    return AgentServiceClient(base_url="http://test-agent:8000")


def _make_turn_result(**overrides) -> dict:
    base = {
        "session_id":      "sess-1",
        "turn_index":      1,
        "avatar_response": "Tell me about yourself.",
        "emotion":         "curious",
        "scenario_phase":  "core",
        "session_ended":   False,
        "turn_score":      {
            "turn_index": 1, "clarity": 0.8, "empathy": 0.7,
            "structure": 0.75, "relevance": 0.85, "confidence": 0.7,
            "composite": 0.76, "rationale": "Good STAR structure.",
        },
        "session_summary": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# stream_turn event translation
# ---------------------------------------------------------------------------

class TestStreamTurn:

    @pytest.mark.asyncio
    async def test_yields_tokens(self):
        client = _make_client()
        result = _make_turn_result(avatar_response="Hello world how are you")

        with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = result
            events = [e async for e in client.stream_turn("sess-1", "Hi")]

        token_events = [e for e in events if e["type"] == WS_EVT_TOKEN]
        tokens = "".join(e["content"] for e in token_events)
        assert "Hello" in tokens
        assert "world" in tokens

    @pytest.mark.asyncio
    async def test_token_count_matches_word_count(self):
        client = _make_client()
        text = "One two three four five"
        result = _make_turn_result(avatar_response=text)

        with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = result
            events = [e async for e in client.stream_turn("sess-1", "msg")]

        token_events = [e for e in events if e["type"] == WS_EVT_TOKEN]
        assert len(token_events) == 5

    @pytest.mark.asyncio
    async def test_yields_turn_done(self):
        client = _make_client()
        result = _make_turn_result()

        with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = result
            events = [e async for e in client.stream_turn("sess-1", "msg")]

        turn_done = [e for e in events if e["type"] == WS_EVT_TURN_DONE]
        assert len(turn_done) == 1
        td = turn_done[0]
        assert td["emotion"] == "curious"
        assert td["phase"] == "core"
        assert td["session_ended"] is False

    @pytest.mark.asyncio
    async def test_yields_score_when_present(self):
        client = _make_client()
        result = _make_turn_result()

        with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = result
            events = [e async for e in client.stream_turn("sess-1", "msg")]

        score_events = [e for e in events if e["type"] == WS_EVT_SCORE]
        assert len(score_events) == 1
        assert abs(score_events[0]["composite"] - 0.76) < 0.001

    @pytest.mark.asyncio
    async def test_no_score_when_none(self):
        client = _make_client()
        result = _make_turn_result(turn_score=None)

        with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = result
            events = [e async for e in client.stream_turn("sess-1", "msg")]

        assert not any(e["type"] == WS_EVT_SCORE for e in events)

    @pytest.mark.asyncio
    async def test_yields_session_end_when_ended(self):
        client = _make_client()
        summary = {"averages": {"composite": 0.74}, "trend": "improving"}
        result = _make_turn_result(session_ended=True, session_summary=summary)

        with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = result
            events = [e async for e in client.stream_turn("sess-1", "msg")]

        end_events = [e for e in events if e["type"] == WS_EVT_SESSION_END]
        assert len(end_events) == 1
        assert end_events[0]["summary"]["trend"] == "improving"

    @pytest.mark.asyncio
    async def test_no_session_end_when_not_ended(self):
        client = _make_client()
        result = _make_turn_result(session_ended=False)

        with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = result
            events = [e async for e in client.stream_turn("sess-1", "msg")]

        assert not any(e["type"] == WS_EVT_SESSION_END for e in events)

    @pytest.mark.asyncio
    async def test_event_order_tokens_before_turn_done(self):
        client = _make_client()
        result = _make_turn_result(avatar_response="alpha beta gamma")

        with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = result
            events = [e async for e in client.stream_turn("sess-1", "msg")]

        types = [e["type"] for e in events]
        last_token = max(i for i, t in enumerate(types) if t == WS_EVT_TOKEN)
        first_done = next(i for i, t in enumerate(types) if t == WS_EVT_TURN_DONE)
        assert last_token < first_done

    @pytest.mark.asyncio
    async def test_agent_error_yields_error_event(self):
        client = _make_client()

        with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = AgentServiceError("Agent down", 503)
            events = [e async for e in client.stream_turn("sess-1", "msg")]

        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "agent_error" in events[0]["code"]

    @pytest.mark.asyncio
    async def test_turn_index_propagated_in_tokens(self):
        client = _make_client()
        result = _make_turn_result(turn_index=5, avatar_response="hello world")

        with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = result
            events = [e async for e in client.stream_turn("sess-1", "msg")]

        token_events = [e for e in events if e["type"] == WS_EVT_TOKEN]
        assert all(e["turn_index"] == 5 for e in token_events)


# ---------------------------------------------------------------------------
# AgentServiceError
# ---------------------------------------------------------------------------

class TestAgentServiceError:

    def test_stores_status_code(self):
        e = AgentServiceError("not found", 404)
        assert e.status_code == 404
        assert str(e) == "not found"

    def test_default_status_code(self):
        e = AgentServiceError("oops")
        assert e.status_code == 500
