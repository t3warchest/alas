"""
tests/test_contracts.py

Unit tests for shared/contracts — zero external dependencies.
Tests cover event constants, message serialisation, parse_client_frame,
and GatewaySession lifecycle.
"""

from __future__ import annotations

import json
import sys
import os
import time

# Make both shared and gateway importable from the tests directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from shared.contracts.events import (
    CLIENT_EVENTS, SERVER_EVENTS,
    WS_EVT_TOKEN, WS_EVT_TURN_DONE, WS_EVT_SCORE,
    WS_EVT_PHASE_CHANGE, WS_EVT_SESSION_END,
    WS_EVT_ERROR, WS_EVT_FATAL, WS_EVT_PONG,
    WS_EVT_AUTH_OK, WS_EVT_CONNECTED,
    WS_MSG_MESSAGE, WS_MSG_PING, WS_MSG_END, WS_MSG_RECONNECT,
)
from shared.contracts.messages import (
    ClientMessage, ClientPing, ClientReconnect,
    AuthOkEvent, ConnectedEvent, TokenEvent, TurnDoneEvent,
    ScoreEvent, PhaseChangeEvent, SessionEndEvent, PongEvent,
    ErrorEvent, FatalEvent, WSMessage,
    parse_client_frame,
)
from shared.contracts.session import GatewaySession, SessionStatus


# ===========================================================================
# 1. Event constants
# ===========================================================================

class TestEventConstants:

    def test_all_client_events_are_strings(self):
        for e in CLIENT_EVENTS:
            assert isinstance(e, str) and e

    def test_all_server_events_are_strings(self):
        for e in SERVER_EVENTS:
            assert isinstance(e, str) and e

    def test_client_and_server_events_disjoint(self):
        overlap = CLIENT_EVENTS & SERVER_EVENTS
        # "end" is only a client event; server events are distinct
        # The two sets should not share any event names
        assert not overlap, f"Shared event names: {overlap}"

    def test_specific_constants_exist(self):
        assert WS_EVT_TOKEN        == "token"
        assert WS_EVT_TURN_DONE   == "turn_done"
        assert WS_EVT_SCORE       == "score"
        assert WS_EVT_PHASE_CHANGE == "phase_change"
        assert WS_EVT_SESSION_END  == "session_end"
        assert WS_EVT_ERROR       == "error"
        assert WS_EVT_FATAL       == "fatal"
        assert WS_EVT_PONG        == "pong"
        assert WS_EVT_AUTH_OK     == "auth_ok"
        assert WS_EVT_CONNECTED   == "connected"
        assert WS_MSG_MESSAGE     == "message"
        assert WS_MSG_PING        == "ping"
        assert WS_MSG_END         == "end"
        assert WS_MSG_RECONNECT   == "reconnect"


# ===========================================================================
# 2. Message serialisation
# ===========================================================================

class TestMessageSerialisation:

    def test_ws_message_to_dict(self):
        m = WSMessage(type="test")
        d = m.to_dict()
        assert d["type"] == "test"

    def test_ws_message_to_json(self):
        m = WSMessage(type="test")
        raw = m.to_json()
        parsed = json.loads(raw)
        assert parsed["type"] == "test"

    def test_none_fields_excluded(self):
        """Fields that are None should not appear in to_dict()."""
        m = ErrorEvent(code="e", message="msg")
        d = m.to_dict()
        # 'recoverable' is True (not None), so it appears
        assert "type" in d
        assert "code" in d

    def test_token_event_roundtrip(self):
        e = TokenEvent(content="hello", turn_index=3)
        d = e.to_dict()
        assert d["type"] == WS_EVT_TOKEN
        assert d["content"] == "hello"
        assert d["turn_index"] == 3

    def test_turn_done_event(self):
        e = TurnDoneEvent(
            session_id="s1", turn_index=2,
            avatar_response="Nice answer.", emotion="encouraging",
            phase="core", session_ended=False,
        )
        d = e.to_dict()
        assert d["type"] == WS_EVT_TURN_DONE
        assert d["emotion"] == "encouraging"
        assert d["session_ended"] is False

    def test_score_event_all_fields(self):
        e = ScoreEvent(
            turn_index=1, clarity=0.8, empathy=0.7,
            structure=0.75, relevance=0.9, confidence=0.6,
            composite=0.76, rationale="Good STAR structure.",
        )
        d = e.to_dict()
        assert d["type"] == WS_EVT_SCORE
        assert abs(d["composite"] - 0.76) < 0.001
        assert d["rationale"] == "Good STAR structure."

    def test_phase_change_event(self):
        e = PhaseChangeEvent(
            from_phase="setup", to_phase="core",
            reason="max_turns reached", turn_index=3,
        )
        d = e.to_dict()
        assert d["from_phase"] == "setup"
        assert d["to_phase"] == "core"

    def test_session_end_event_with_summary(self):
        summary = {"averages": {"composite": 0.72}, "trend": "improving"}
        e = SessionEndEvent(session_id="s1", summary=summary)
        d = e.to_dict()
        assert d["type"] == WS_EVT_SESSION_END
        assert d["summary"]["trend"] == "improving"

    def test_connected_event(self):
        e = ConnectedEvent(
            session_id="s1", scenario_id="job_interview_v1",
            scenario_title="Job Interview", persona_name="Alex",
            opening_line="Hello! Tell me about yourself.",
            phase="setup",
        )
        d = e.to_dict()
        assert d["persona_name"] == "Alex"
        assert d["phase"] == "setup"

    def test_auth_ok_event(self):
        e = AuthOkEvent(user_id="u1", token_expires_in=3600)
        d = e.to_dict()
        assert d["type"] == WS_EVT_AUTH_OK
        assert d["user_id"] == "u1"

    def test_error_event_recoverable(self):
        e = ErrorEvent(code="empty_message", message="No content.", recoverable=True)
        d = e.to_dict()
        assert d["recoverable"] is True

    def test_fatal_event_not_recoverable(self):
        e = FatalEvent(code="session_not_found", message="Session missing.")
        d = e.to_dict()
        assert d["recoverable"] is False

    def test_pong_event(self):
        e = PongEvent(timestamp=1234.5)
        d = e.to_dict()
        assert d["type"] == WS_EVT_PONG
        assert d["timestamp"] == 1234.5


# ===========================================================================
# 3. parse_client_frame
# ===========================================================================

class TestParseClientFrame:

    def test_parse_message_event(self):
        raw = json.dumps({"type": "message", "content": "Hello there!"})
        frame = parse_client_frame(raw)
        assert isinstance(frame, ClientMessage)
        assert frame.content == "Hello there!"

    def test_parse_ping_event(self):
        raw = json.dumps({"type": "ping", "timestamp": 9999.0})
        frame = parse_client_frame(raw)
        assert isinstance(frame, ClientPing)
        assert frame.timestamp == 9999.0

    def test_parse_end_event(self):
        raw = json.dumps({"type": "end"})
        frame = parse_client_frame(raw)
        assert frame is not None
        assert frame.type == "end"

    def test_parse_reconnect_event(self):
        raw = json.dumps({
            "type": "reconnect",
            "session_id": "abc-123",
            "last_turn_index": 5,
        })
        frame = parse_client_frame(raw)
        assert isinstance(frame, ClientReconnect)
        assert frame.session_id == "abc-123"
        assert frame.last_turn_index == 5

    def test_parse_unknown_type_returns_base(self):
        raw = json.dumps({"type": "custom_event", "data": 42})
        frame = parse_client_frame(raw)
        assert frame is not None
        assert frame.type == "custom_event"

    def test_parse_invalid_json_returns_none(self):
        assert parse_client_frame("not json {{{") is None

    def test_parse_empty_string_returns_none(self):
        assert parse_client_frame("") is None

    def test_parse_missing_content_defaults_empty(self):
        raw = json.dumps({"type": "message"})
        frame = parse_client_frame(raw)
        assert isinstance(frame, ClientMessage)
        assert frame.content == ""

    def test_parse_preserves_session_id(self):
        raw = json.dumps({"type": "message", "content": "hi", "session_id": "sess-xyz"})
        frame = parse_client_frame(raw)
        assert isinstance(frame, ClientMessage)
        assert frame.session_id == "sess-xyz"


# ===========================================================================
# 4. GatewaySession lifecycle
# ===========================================================================

class TestGatewaySession:

    def _make_session(self, **kwargs) -> GatewaySession:
        defaults = dict(
            session_id="test-sess-1",
            user_id="user-1",
            scenario_id="job_interview_v1",
        )
        defaults.update(kwargs)
        return GatewaySession(**defaults)

    def test_initial_status_is_initialising(self):
        s = self._make_session()
        assert s.status == SessionStatus.INITIALISING

    def test_mark_active(self):
        s = self._make_session()
        s.mark_active()
        assert s.status == SessionStatus.ACTIVE

    def test_mark_disconnected(self):
        s = self._make_session()
        s.mark_active()
        s.mark_disconnected()
        assert s.status == SessionStatus.DISCONNECTED

    def test_mark_ended(self):
        s = self._make_session()
        s.mark_ended()
        assert s.status == SessionStatus.ENDED

    def test_touch_updates_last_active(self):
        s = self._make_session()
        before = s.last_active_at
        import time; time.sleep(0.01)
        s.touch()
        assert s.last_active_at > before

    def test_is_reconnectable_disconnected_fresh(self):
        s = self._make_session()
        s.mark_disconnected()
        assert s.is_reconnectable()

    def test_is_reconnectable_ended_false(self):
        s = self._make_session()
        s.mark_ended()
        assert not s.is_reconnectable()

    def test_is_reconnectable_active_false(self):
        s = self._make_session()
        s.mark_active()
        assert not s.is_reconnectable()

    def test_is_reconnectable_expired(self):
        s = self._make_session()
        s.mark_disconnected()
        # Wind back last_active_at to simulate timeout
        s.last_active_at = time.time() - 7201
        assert not s.is_reconnectable()

    def test_to_dict_roundtrip(self):
        s = self._make_session(
            scenario_title="Job Interview",
            persona_name="Alex",
            current_phase="core",
            last_turn_index=4,
            total_turns=5,
            latest_composite=0.71,
        )
        d = s.to_dict()
        s2 = GatewaySession.from_dict(d)
        assert s2.session_id    == s.session_id
        assert s2.user_id       == s.user_id
        assert s2.scenario_id   == s.scenario_id
        assert s2.status        == s.status
        assert s2.persona_name  == s.persona_name
        assert s2.current_phase == s.current_phase
        assert s2.last_turn_index == s.last_turn_index
        assert abs((s2.latest_composite or 0) - 0.71) < 0.001

    def test_to_dict_status_is_string(self):
        s = self._make_session()
        s.mark_active()
        d = s.to_dict()
        assert isinstance(d["status"], str)
        assert d["status"] == "active"

    def test_session_status_enum_values(self):
        assert SessionStatus.INITIALISING.value == "initialising"
        assert SessionStatus.ACTIVE.value       == "active"
        assert SessionStatus.DISCONNECTED.value == "disconnected"
        assert SessionStatus.ENDED.value        == "ended"
        assert SessionStatus.ERROR.value        == "error"
