"""
tests/test_session_store.py

Unit tests for InMemorySessionStore and SQLiteSessionStore.
SQLite tests use a temp file; both test the same SessionStore interface.
No external dependencies.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from gateway.session_store.store import (
    InMemorySessionStore,
    SQLiteSessionStore,
    SessionStore,
)
from shared.contracts.session import GatewaySession, SessionStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session(sid="s1", uid="u1", scenario="job_interview_v1", **kw) -> GatewaySession:
    s = GatewaySession(session_id=sid, user_id=uid, scenario_id=scenario)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# Parametrised: run every test against both backends
# ---------------------------------------------------------------------------

@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path) -> SessionStore:
    if request.param == "memory":
        return InMemorySessionStore()
    else:
        return SQLiteSessionStore(db_path=tmp_path / "test.db")


class TestSessionStoreInterface:
    """All tests run against both InMemorySessionStore and SQLiteSessionStore."""

    def test_create_and_get(self, store):
        s = _session("abc")
        store.create(s)
        retrieved = store.get("abc")
        assert retrieved is not None
        assert retrieved.session_id == "abc"
        assert retrieved.user_id == "u1"

    def test_get_missing_returns_none(self, store):
        assert store.get("does-not-exist") is None

    def test_update_persists(self, store):
        s = _session("upd")
        store.create(s)
        s.status = SessionStatus.ACTIVE
        s.current_phase = "core"
        store.update(s)
        retrieved = store.get("upd")
        assert retrieved.status == SessionStatus.ACTIVE
        assert retrieved.current_phase == "core"

    def test_delete_removes(self, store):
        s = _session("del")
        store.create(s)
        store.delete("del")
        assert store.get("del") is None

    def test_delete_nonexistent_is_safe(self, store):
        store.delete("ghost")  # should not raise

    def test_list_for_user(self, store):
        store.create(_session("a", uid="alice", scenario="s1"))
        store.create(_session("b", uid="alice", scenario="s2"))
        store.create(_session("c", uid="bob",   scenario="s1"))
        alice = store.list_for_user("alice")
        assert len(alice) == 2
        assert all(s.user_id == "alice" for s in alice)

    def test_list_for_user_empty(self, store):
        assert store.list_for_user("nobody") == []

    def test_update_status_helper(self, store):
        store.create(_session("st"))
        store.update_status("st", SessionStatus.ACTIVE)
        assert store.get("st").status == SessionStatus.ACTIVE

    def test_update_status_missing_is_safe(self, store):
        store.update_status("ghost", SessionStatus.ACTIVE)  # must not raise

    def test_update_turn_increments(self, store):
        store.create(_session("turn"))
        store.update_turn("turn", turn_index=3, phase="core", composite=0.72)
        s = store.get("turn")
        assert s.last_turn_index == 3
        assert s.current_phase == "core"
        assert abs(s.latest_composite - 0.72) < 0.001

    def test_update_turn_total_turns(self, store):
        store.create(_session("tot"))
        store.update_turn("tot", turn_index=0)
        store.update_turn("tot", turn_index=1)
        store.update_turn("tot", turn_index=2)
        s = store.get("tot")
        assert s.total_turns == 3

    def test_multiple_sessions_isolated(self, store):
        store.create(_session("x", uid="ux"))
        store.create(_session("y", uid="uy"))
        store.update_status("x", SessionStatus.ENDED)
        assert store.get("y").status == SessionStatus.INITIALISING

    def test_expire_old_sessions(self, store):
        s = _session("old")
        s.mark_disconnected()
        s.last_active_at = time.time() - 10000  # very old
        store.create(s)

        fresh = _session("fresh")
        fresh.mark_disconnected()
        store.create(fresh)

        removed = store.expire_old_sessions(ttl_seconds=3600)
        assert removed >= 1
        assert store.get("old") is None
        assert store.get("fresh") is not None

    def test_roundtrip_all_fields(self, store):
        s = GatewaySession(
            session_id="full",
            user_id="u-full",
            scenario_id="difficult_coworker_v1",
            scenario_title="Difficult Coworker",
            persona_name="Jordan",
            opening_line="Hello.",
            current_phase="disclosure",
            last_turn_index=7,
            total_turns=8,
            latest_composite=0.65,
        )
        s.mark_active()
        store.create(s)
        r = store.get("full")
        assert r.session_id         == "full"
        assert r.scenario_title     == "Difficult Coworker"
        assert r.persona_name       == "Jordan"
        assert r.current_phase      == "disclosure"
        assert r.last_turn_index    == 7
        assert abs(r.latest_composite - 0.65) < 0.001


# ---------------------------------------------------------------------------
# In-memory specific
# ---------------------------------------------------------------------------

class TestInMemorySpecific:

    def test_len(self):
        store = InMemorySessionStore()
        assert len(store) == 0
        store.create(_session("a"))
        store.create(_session("b"))
        assert len(store) == 2

    def test_thread_safety_basic(self):
        """Crude check: concurrent create calls don't corrupt the dict."""
        import threading
        store = InMemorySessionStore()
        errors = []

        def create_many(prefix):
            for i in range(50):
                try:
                    store.create(_session(f"{prefix}-{i}", uid=prefix))
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=create_many, args=(f"t{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store) == 250


# ---------------------------------------------------------------------------
# SQLite specific
# ---------------------------------------------------------------------------

class TestSQLiteSpecific:

    def test_survives_reopen(self, tmp_path):
        """Data written to SQLite persists after closing and reopening the store."""
        db = tmp_path / "persist.db"
        store1 = SQLiteSessionStore(db_path=db)
        store1.create(_session("persist-me"))
        store1.update_status("persist-me", SessionStatus.ACTIVE)

        # Reopen
        store2 = SQLiteSessionStore(db_path=db)
        s = store2.get("persist-me")
        assert s is not None
        assert s.status == SessionStatus.ACTIVE

    def test_index_on_user_id(self, tmp_path):
        """Verify the user_id index exists (doesn't raise on creation)."""
        db = tmp_path / "idx.db"
        store = SQLiteSessionStore(db_path=db)
        for i in range(20):
            store.create(_session(f"s{i}", uid="bulk-user"))
        result = store.list_for_user("bulk-user")
        assert len(result) == 20
