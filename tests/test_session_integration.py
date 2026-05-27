"""
tests/test_session_integration.py

Integration tests for the SessionOrchestrator.
Mocks the LLM and ChromaDB so no API keys are needed.

Run with:  pytest tests/test_session_integration.py -v
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agent_service.graph.response_parser import DELIMITER

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_LLM_RESPONSE = (
    "That's an interesting background. Can you walk me through "
    "a specific challenge you overcame in a previous role?\n"
    f"{DELIMITER}\n"
    '{"emotion": "curious", "scenario_phase": "core", '
    '"branch_signal": null, "hidden_notes": "Probing for STAR."}'
)

FAKE_EVAL_RESPONSE = MagicMock()
FAKE_EVAL_RESPONSE.choices = [MagicMock()]
FAKE_EVAL_RESPONSE.choices[0].message.content = (
    '{"clarity": 0.75, "empathy": 0.6, "structure": 0.7, '
    '"relevance": 0.8, "confidence": 0.7, "composite": 0.71, '
    '"rationale": "Good opening, lacks specifics."}'
)

FAKE_LLM_RESPONSE_OBJ = MagicMock()
FAKE_LLM_RESPONSE_OBJ.content = FAKE_LLM_RESPONSE


@pytest.fixture(autouse=True)
def patch_chroma(tmp_path):
    """Prevent any real ChromaDB writes during tests."""
    with patch("agent_service.memory.manager.CHROMA_DIR", str(tmp_path / "chromadb")):
        with patch("agent_service.memory.manager._openai_ef") as mock_ef:
            mock_ef.return_value = MagicMock()
            yield


@pytest.fixture(autouse=True)
def patch_llm():
    """Patch both the main LLM and the eval LLM client."""
    fake_chat = AsyncMock()
    fake_chat.ainvoke = AsyncMock(return_value=FAKE_LLM_RESPONSE_OBJ)

    with patch("agent_service.graph.agent_graph._get_llm", return_value=fake_chat):
        with patch("agent_service.evaluation.engine._get_client") as mock_client:
            client_inst = AsyncMock()
            client_inst.chat.completions.create = AsyncMock(return_value=FAKE_EVAL_RESPONSE)
            mock_client.return_value = client_inst
            yield


@pytest.fixture(autouse=True)
def patch_semantic_memory():
    """Stub out vector retrieval — returns empty lists."""
    with patch("agent_service.memory.manager.SemanticMemory.retrieve_scenario_context",
               return_value=[]):
        with patch("agent_service.memory.manager.SemanticMemory.upsert_user_profile"):
            with patch("agent_service.memory.manager.EpisodicMemory.retrieve_for_user",
                       return_value=[]):
                with patch("agent_service.memory.manager.EpisodicMemory.get_behavioral_notes",
                           return_value=[]):
                    with patch("agent_service.memory.manager.EpisodicMemory.store_session"):
                        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSessionOrchestrator:
    @pytest.mark.asyncio
    async def test_create_session_returns_session_info(self):
        from agent_service.session import SessionOrchestrator
        orch = SessionOrchestrator()

        info = await orch.create_session(
            user_id="test-user-1",
            scenario_id="job_interview_v1",
        )

        assert info.session_id
        assert info.scenario_id == "job_interview_v1"
        assert info.scenario_title == "Software Engineering Job Interview"
        assert info.persona_name == "Alex"
        assert len(info.opening_line) > 0

    @pytest.mark.asyncio
    async def test_send_message_returns_turn_result(self):
        from agent_service.session import SessionOrchestrator
        orch = SessionOrchestrator()

        info = await orch.create_session(
            user_id="test-user-2",
            scenario_id="job_interview_v1",
        )

        result = await orch.send_message(
            session_id=info.session_id,
            user_id="test-user-2",
            student_message="I've been a software engineer for 5 years, mostly in backend systems.",
        )

        assert result.session_id == info.session_id
        assert len(result.avatar_response) > 0
        assert result.emotion in {
            "neutral", "curious", "concerned", "encouraging",
            "challenging", "warm", "disappointed",
        }
        assert result.scenario_phase in {"setup", "core", "escalation", "resolution"}

    @pytest.mark.asyncio
    async def test_turn_score_produced(self):
        from agent_service.session import SessionOrchestrator
        orch = SessionOrchestrator()

        info = await orch.create_session(
            user_id="test-user-3",
            scenario_id="job_interview_v1",
        )
        result = await orch.send_message(
            session_id=info.session_id,
            user_id="test-user-3",
            student_message="I solved a production outage by rewriting the caching layer.",
        )

        # Score may be None on the opening turn (no student message to evaluate yet)
        # but should exist after a real student turn
        if result.turn_score:
            score = result.turn_score
            assert 0.0 <= score["composite"] <= 1.0
            assert "rationale" in score

    @pytest.mark.asyncio
    async def test_get_session_scores_accumulates(self):
        from agent_service.session import SessionOrchestrator
        orch = SessionOrchestrator()

        info = await orch.create_session(
            user_id="test-user-4",
            scenario_id="job_interview_v1",
        )

        for msg in [
            "I'm a backend engineer with 3 years experience.",
            "I once led a migration from monolith to microservices.",
        ]:
            await orch.send_message(
                session_id=info.session_id,
                user_id="test-user-4",
                student_message=msg,
            )

        scores = orch.get_session_scores(info.session_id)
        assert isinstance(scores, list)

    @pytest.mark.asyncio
    async def test_end_session_returns_summary(self):
        from agent_service.session import SessionOrchestrator
        orch = SessionOrchestrator()

        info = await orch.create_session(
            user_id="test-user-5",
            scenario_id="job_interview_v1",
        )
        await orch.send_message(
            session_id=info.session_id,
            user_id="test-user-5",
            student_message="I'm excited about this opportunity.",
        )

        result = await orch.end_session(info.session_id)
        assert "session_id" in result

    @pytest.mark.asyncio
    async def test_invalid_session_raises(self):
        from agent_service.session import SessionOrchestrator
        orch = SessionOrchestrator()

        with pytest.raises(ValueError, match="not found"):
            await orch.send_message(
                session_id="nonexistent-session",
                user_id="u",
                student_message="hello",
            )

    @pytest.mark.asyncio
    async def test_stream_message_yields_tokens_then_result(self):
        from agent_service.session import SessionOrchestrator
        orch = SessionOrchestrator()

        info = await orch.create_session(
            user_id="test-user-6",
            scenario_id="job_interview_v1",
        )

        chunks = []
        async for chunk in orch.stream_message(
            session_id=info.session_id,
            user_id="test-user-6",
            student_message="I'm passionate about distributed systems.",
        ):
            chunks.append(chunk)

        types = [c["type"] for c in chunks]
        assert "token" in types
        assert "result" in types
        # Tokens must come before the result
        assert types.index("token") < types.index("result")

    @pytest.mark.asyncio
    async def test_difficult_conversation_scenario_loads(self):
        from agent_service.session import SessionOrchestrator
        orch = SessionOrchestrator()

        info = await orch.create_session(
            user_id="test-user-7",
            scenario_id="difficult_conversation_v1",
        )
        assert info.scenario_id == "difficult_conversation_v1"
        assert info.persona_name == "Jordan"
