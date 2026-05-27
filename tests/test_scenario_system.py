"""
tests/test_scenario_system.py

Comprehensive tests for the scenario system:
  - Pydantic model validation (valid and invalid inputs)
  - ScenarioLoader (load from disk, error handling)
  - PhaseEngine (all four transition types, branch_table conditions)
  - Updated prompt_builder integration
  - Cross-scenario structural invariants

No LLM calls. No ChromaDB. No network.
Run with: pytest tests/test_scenario_system.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_service.scenarios.branching import PhaseContext, PhaseEngine, TransitionDecision
from agent_service.scenarios.loader import ScenarioLoader
from agent_service.scenarios.models import (
    BranchCondition,
    Difficulty,
    EmotionalStart,
    Phase,
    PhaseTransition,
    Persona,
    RubricLevel,
    Scenario,
    ScenarioCategory,
    TransitionType,
)


# ===========================================================================
# Fixtures — minimal valid scenario data
# ===========================================================================

MINIMAL_TRANSITION = {
    "type": "always_after_max",
    "target_phase": "core",
}

MINIMAL_PHASE = {
    "name": "setup",
    "description": "Opening phase of the conversation.",
    "max_turns": 3,
    "transition": MINIMAL_TRANSITION,
}

MINIMAL_RUBRIC_LEVEL = {"1": "Poor.", "3": "Acceptable.", "5": "Excellent."}

MINIMAL_SCENARIO: dict = {
    "id": "test_scenario_v1",
    "version": "1.0",
    "title": "Test Scenario Title",
    "description": "A minimal scenario for unit testing the system.",
    "category": "interview",
    "persona": {
        "name": "Sam",
        "role": "Test interviewer",
        "emotional_start": "neutral",
        "style": "professional",
        "constraints": ["Do not break character."],
    },
    "student_objective": "Demonstrate competence during a structured test scenario.",
    "phases": [
        {
            "name": "setup",
            "description": "The setup phase begins the conversation.",
            "max_turns": 3,
            "transition": {"type": "always_after_max", "target_phase": "core"},
        },
        {
            "name": "core",
            "description": "The core phase is the main body of the conversation.",
            "max_turns": 8,
            "transition": {"type": "always_after_max", "target_phase": "resolution"},
        },
        {
            "name": "resolution",
            "description": "The resolution phase wraps up the conversation.",
            "max_turns": 2,
            "transition": {"type": "always_after_max", "target_phase": "resolution"},
        },
    ],
    "skill_dimensions": {"clarity": 0.5, "confidence": 0.5},
    "rubric": {
        "clarity":    {"1": "Unclear.", "3": "Adequate.", "5": "Crystal-clear."},
        "confidence": {"1": "Hesitant.", "3": "Steady.", "5": "Authoritative."},
    },
    "knowledge_chunks": ["This is a knowledge chunk for the test scenario."],
    "exit_conditions": {
        "max_total_turns": 13,
        "completion_phases": ["resolution"],
    },
}


def make_scenario(**overrides) -> dict:
    import copy
    d = copy.deepcopy(MINIMAL_SCENARIO)
    d.update(overrides)
    return d


def load_scenario(**overrides) -> Scenario:
    return Scenario.model_validate(make_scenario(**overrides))


# ===========================================================================
# 1. Pydantic Model Validation
# ===========================================================================

class TestScenarioModelValidation:

    def test_valid_minimal_scenario(self):
        s = load_scenario()
        assert s.id == "test_scenario_v1"
        assert s.category == ScenarioCategory.INTERVIEW
        assert s.difficulty == Difficulty.INTERMEDIATE

    def test_id_rejects_uppercase(self):
        with pytest.raises(ValidationError, match="id"):
            load_scenario(id="Invalid_ID")

    def test_id_rejects_spaces(self):
        with pytest.raises(ValidationError):
            load_scenario(id="has spaces")

    def test_id_accepts_underscores_and_digits(self):
        s = load_scenario(id="my_scenario_2024")
        assert s.id == "my_scenario_2024"

    def test_version_format_enforced(self):
        with pytest.raises(ValidationError):
            load_scenario(version="v1")
        s = load_scenario(version="2.3")
        assert s.version == "2.3"

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValidationError, match="sum to 1.0"):
            load_scenario(skill_dimensions={"clarity": 0.3, "confidence": 0.4})

    def test_weights_tolerance(self):
        # 0.501 + 0.499 = 1.000 within 0.01 tolerance
        s = load_scenario(skill_dimensions={"clarity": 0.501, "confidence": 0.499})
        assert abs(sum(s.skill_dimensions.values()) - 1.0) < 0.01

    def test_rubric_must_cover_all_dimensions(self):
        with pytest.raises(ValidationError, match="missing entries"):
            load_scenario(
                skill_dimensions={"clarity": 0.5, "empathy": 0.5},
                rubric={"clarity": {"1": "x", "3": "y", "5": "z"}},
                # empathy rubric missing
            )

    def test_phases_must_have_unique_names(self):
        phases = [
            {"name": "setup", "description": "A.", "max_turns": 3,
             "transition": {"type": "always_after_max", "target_phase": "setup"}},
            {"name": "setup", "description": "B.", "max_turns": 3,
             "transition": {"type": "always_after_max", "target_phase": "setup"}},
        ]
        with pytest.raises(ValidationError, match="unique"):
            load_scenario(phases=phases)

    def test_transition_target_must_exist(self):
        phases = [
            {"name": "setup", "description": "A.", "max_turns": 3,
             "transition": {"type": "always_after_max", "target_phase": "nonexistent_phase"}},
        ]
        with pytest.raises(ValidationError, match="unknown phase"):
            load_scenario(phases=phases)

    def test_completion_phases_must_exist(self):
        with pytest.raises(ValidationError, match="unknown phase"):
            load_scenario(exit_conditions={
                "max_total_turns": 10,
                "completion_phases": ["ghost_phase"],
            })

    def test_score_threshold_transition_requires_dimension(self):
        with pytest.raises(ValidationError, match="score_dimension"):
            PhaseTransition.model_validate({
                "type": "score_threshold",
                "score_threshold": 0.7,
            })

    def test_score_threshold_requires_threshold_value(self):
        with pytest.raises(ValidationError, match="score_threshold"):
            PhaseTransition.model_validate({
                "type": "score_threshold",
                "score_dimension": "composite",
            })

    def test_branch_table_requires_branches(self):
        with pytest.raises(ValidationError, match="branch"):
            PhaseTransition.model_validate({"type": "branch_table"})

    def test_keyword_transition_requires_keywords(self):
        with pytest.raises(ValidationError, match="keyword"):
            PhaseTransition.model_validate({
                "type": "keyword_detected",
                "target_phase": "core",
            })

    def test_rubric_level_get_descriptor(self):
        rl = RubricLevel.model_validate({"1": "Bad", "3": "Ok", "5": "Great"})
        assert rl.get_descriptor(1) == "Bad"
        assert rl.get_descriptor(3) == "Ok"
        assert rl.get_descriptor(5) == "Great"
        assert rl.get_descriptor(2) == "Bad"   # falls back to level 1
        assert rl.get_descriptor(4) == "Ok"    # falls back to level 3

    def test_scenario_get_phase(self):
        s = load_scenario()
        phase = s.get_phase("core")
        assert phase is not None
        assert phase.name == "core"
        assert s.get_phase("nonexistent") is None

    def test_scenario_first_phase(self):
        s = load_scenario()
        assert s.first_phase().name == "setup"

    def test_scenario_next_phase(self):
        s = load_scenario()
        nxt = s.next_phase("setup")
        assert nxt is not None
        assert nxt.name == "core"
        assert s.next_phase("resolution") is None

    def test_to_legacy_dict_structure(self):
        s = load_scenario()
        d = s.to_legacy_dict()
        assert "persona" in d
        assert "phases" in d
        assert "skill_dimensions" in d
        assert "rubric" in d
        # Legacy rubric uses int keys
        for dim, levels in d["rubric"].items():
            assert 1 in levels
            assert 3 in levels
            assert 5 in levels

    def test_emotional_start_enum(self):
        for es in ["neutral", "defensive", "curious", "warm", "hostile", "anxious"]:
            p = Persona.model_validate({
                "name": "X", "role": "Y",
                "emotional_start": es, "style": "z",
                "constraints": ["a"],
            })
            assert p.emotional_start.value == es


# ===========================================================================
# 2. Scenario Loader
# ===========================================================================

class TestScenarioLoader:

    @pytest.fixture
    def definitions_dir(self):
        """Use the real definitions directory."""
        return Path(__file__).parent.parent / "agent_service" / "scenarios" / "definitions"

    @pytest.fixture
    def loader_from_real_dir(self, definitions_dir):
        loader = ScenarioLoader(definitions_dir)
        loader.load_all()
        return loader

    def test_loads_all_real_scenarios(self, loader_from_real_dir):
        assert len(loader_from_real_dir) >= 3  # job_interview, difficult_coworker, salary_negotiation

    def test_gets_known_scenario(self, loader_from_real_dir):
        s = loader_from_real_dir.get("job_interview_v1")
        assert s.id == "job_interview_v1"
        assert s.category == ScenarioCategory.INTERVIEW

    def test_gets_coworker_scenario(self, loader_from_real_dir):
        s = loader_from_real_dir.get("difficult_coworker_v1")
        assert s.id == "difficult_coworker_v1"
        assert s.category == ScenarioCategory.FEEDBACK
        assert s.difficulty == Difficulty.ADVANCED

    def test_gets_negotiation_scenario(self, loader_from_real_dir):
        s = loader_from_real_dir.get("salary_negotiation_v1")
        assert s.category == ScenarioCategory.NEGOTIATION

    def test_missing_scenario_raises_key_error(self, loader_from_real_dir):
        with pytest.raises(KeyError, match="ghost"):
            loader_from_real_dir.get("ghost_scenario")

    def test_contains_operator(self, loader_from_real_dir):
        assert "job_interview_v1" in loader_from_real_dir
        assert "nonexistent" not in loader_from_real_dir

    def test_list_summaries_structure(self, loader_from_real_dir):
        summaries = loader_from_real_dir.list_summaries()
        assert len(summaries) >= 3
        for s in summaries:
            assert "id" in s
            assert "title" in s
            assert "category" in s
            assert "difficulty" in s
            assert "phase_count" in s

    def test_load_invalid_json_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("not valid json {{{")
            loader = ScenarioLoader(tmp)
            result = loader.load_all()
            assert len(result) == 0

    def test_load_invalid_schema_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad_schema.json"
            bad.write_text(json.dumps({"id": "x", "title": "short"}))
            loader = ScenarioLoader(tmp)
            result = loader.load_all()
            assert len(result) == 0

    def test_load_valid_custom_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_scenario_v1.json"
            path.write_text(json.dumps(MINIMAL_SCENARIO))
            loader = ScenarioLoader(tmp)
            loader.load_all()
            assert "test_scenario_v1" in loader

    def test_validate_dict_valid(self):
        loader = ScenarioLoader()
        s = loader.validate_dict(MINIMAL_SCENARIO)
        assert s.id == "test_scenario_v1"

    def test_validate_dict_invalid_raises(self):
        loader = ScenarioLoader()
        with pytest.raises(ValidationError):
            loader.validate_dict({"id": "bad_id_FORMAT"})

    def test_real_scenarios_weights_sum_to_one(self, loader_from_real_dir):
        for scenario in loader_from_real_dir.list_all():
            total = sum(scenario.skill_dimensions.values())
            assert abs(total - 1.0) < 0.01, (
                f"Scenario '{scenario.id}' weights sum to {total:.3f}"
            )

    def test_real_scenarios_rubric_covers_all_dims(self, loader_from_real_dir):
        for scenario in loader_from_real_dir.list_all():
            dims = set(scenario.skill_dimensions.keys())
            rubric_dims = set(scenario.rubric.keys())
            assert dims <= rubric_dims, (
                f"Scenario '{scenario.id}' rubric missing: {dims - rubric_dims}"
            )

    def test_real_scenarios_phases_in_exit_conditions(self, loader_from_real_dir):
        for scenario in loader_from_real_dir.list_all():
            phase_names = {p.name for p in scenario.phases}
            for cp in scenario.exit_conditions.completion_phases:
                assert cp in phase_names, (
                    f"Scenario '{scenario.id}': completion phase '{cp}' not in phases"
                )


# ===========================================================================
# 3. PhaseEngine — Branching Logic
# ===========================================================================

def _make_engine(phases_override=None) -> PhaseEngine:
    """Build a PhaseEngine from the minimal scenario."""
    data = make_scenario()
    if phases_override:
        data["phases"] = phases_override
        # Adjust exit conditions to reference real phases
        data["exit_conditions"]["completion_phases"] = [phases_override[-1]["name"]]
    return PhaseEngine(Scenario.model_validate(data))


def _ctx(
    current_phase="setup",
    turns_in_phase=0,
    total_turns=0,
    rolling_scores=None,
    utterance="",
) -> PhaseContext:
    return PhaseContext(
        current_phase_name=current_phase,
        turns_in_phase=turns_in_phase,
        total_turns=total_turns,
        rolling_scores=rolling_scores or {},
        last_student_utterance=utterance,
    )


class TestPhaseEngineAlwaysAfterMax:

    def test_stays_before_max(self):
        engine = _make_engine()
        decision = engine.evaluate(_ctx("setup", turns_in_phase=2))
        assert not decision.should_transition

    def test_transitions_at_max(self):
        engine = _make_engine()
        decision = engine.evaluate(_ctx("setup", turns_in_phase=3))
        assert decision.should_transition
        assert decision.target_phase == "core"

    def test_transitions_past_max(self):
        engine = _make_engine()
        decision = engine.evaluate(_ctx("setup", turns_in_phase=99))
        assert decision.should_transition

    def test_reason_string_populated(self):
        engine = _make_engine()
        decision = engine.evaluate(_ctx("setup", turns_in_phase=3))
        assert "max_turns" in decision.reason


class TestPhaseEngineScoreThreshold:

    def _score_threshold_phases(self) -> list[dict]:
        return [
            {
                "name": "setup", "description": "Open.", "max_turns": 3,
                "transition": {"type": "always_after_max", "target_phase": "core"},
            },
            {
                "name": "core", "description": "Main phase.", "max_turns": 10,
                "transition": {
                    "type": "score_threshold",
                    "score_dimension": "empathy",
                    "score_threshold": 0.70,
                    "target_phase": "resolution",
                },
            },
            {
                "name": "resolution", "description": "End.", "max_turns": 3,
                "transition": {"type": "always_after_max", "target_phase": "resolution"},
            },
        ]

    def test_stays_below_threshold(self):
        engine = _make_engine(self._score_threshold_phases())
        ctx = _ctx("core", turns_in_phase=4, rolling_scores={"empathy": 0.65})
        assert not engine.evaluate(ctx).should_transition

    def test_transitions_above_threshold(self):
        engine = _make_engine(self._score_threshold_phases())
        ctx = _ctx("core", turns_in_phase=4, rolling_scores={"empathy": 0.72})
        decision = engine.evaluate(ctx)
        assert decision.should_transition
        assert decision.target_phase == "resolution"
        assert "0.72" in decision.reason

    def test_fallback_to_max_turns(self):
        engine = _make_engine(self._score_threshold_phases())
        ctx = _ctx("core", turns_in_phase=10, rolling_scores={"empathy": 0.50})
        decision = engine.evaluate(ctx)
        assert decision.should_transition
        assert "fallback" in decision.reason


class TestPhaseEngineKeywordDetected:

    def _keyword_phases(self) -> list[dict]:
        return [
            {
                "name": "setup", "description": "A.", "max_turns": 5,
                "transition": {
                    "type": "keyword_detected",
                    "keywords": ["agree", "sounds good", "let's do it"],
                    "target_phase": "core",
                },
            },
            {
                "name": "core", "description": "B.", "max_turns": 8,
                "transition": {"type": "always_after_max", "target_phase": "resolution"},
            },
            {
                "name": "resolution", "description": "C.", "max_turns": 2,
                "transition": {"type": "always_after_max", "target_phase": "resolution"},
            },
        ]

    def test_no_keyword_stays(self):
        engine = _make_engine(self._keyword_phases())
        ctx = _ctx("setup", turns_in_phase=1, utterance="I'm not sure about this.")
        assert not engine.evaluate(ctx).should_transition

    def test_keyword_triggers_transition(self):
        engine = _make_engine(self._keyword_phases())
        ctx = _ctx("setup", turns_in_phase=1, utterance="Yeah, sounds good to me!")
        decision = engine.evaluate(ctx)
        assert decision.should_transition
        assert decision.target_phase == "core"
        assert "sounds good" in decision.reason

    def test_keyword_case_insensitive(self):
        engine = _make_engine(self._keyword_phases())
        ctx = _ctx("setup", turns_in_phase=1, utterance="I AGREE with that.")
        assert engine.evaluate(ctx).should_transition

    def test_fallback_to_max_turns(self):
        engine = _make_engine(self._keyword_phases())
        ctx = _ctx("setup", turns_in_phase=5, utterance="no keywords here at all")
        decision = engine.evaluate(ctx)
        assert decision.should_transition
        assert "fallback" in decision.reason


class TestPhaseEngineBranchTable:

    def _branch_phases(self) -> list[dict]:
        return [
            {
                "name": "setup", "description": "Open.", "max_turns": 3,
                "transition": {"type": "always_after_max", "target_phase": "good_path"},
            },
            {
                "name": "disclosure", "description": "Core challenge.", "max_turns": 8,
                "transition": {
                    "type": "branch_table",
                    "branches": [
                        {
                            "condition": "empathy_above_0.70_and_clarity_above_0.65",
                            "target_phase": "good_path",
                            "priority": 2,
                        },
                        {
                            "condition": "empathy_below_0.40",
                            "target_phase": "rupture",
                            "priority": 3,
                        },
                        {
                            "condition": "max_turns_reached",
                            "target_phase": "good_path",
                            "priority": 1,
                        },
                    ],
                },
            },
            {
                "name": "good_path", "description": "Good branch.", "max_turns": 4,
                "transition": {"type": "always_after_max", "target_phase": "close"},
            },
            {
                "name": "rupture", "description": "Bad branch.", "max_turns": 4,
                "transition": {"type": "always_after_max", "target_phase": "close"},
            },
            {
                "name": "close", "description": "Wrap up.", "max_turns": 2,
                "transition": {"type": "always_after_max", "target_phase": "close"},
            },
        ]

    def test_high_empathy_takes_good_path(self):
        engine = _make_engine(self._branch_phases())
        ctx = _ctx("disclosure", turns_in_phase=3,
                   rolling_scores={"empathy": 0.80, "clarity": 0.75})
        decision = engine.evaluate(ctx)
        assert decision.should_transition
        assert decision.target_phase == "good_path"

    def test_low_empathy_takes_rupture(self):
        engine = _make_engine(self._branch_phases())
        ctx = _ctx("disclosure", turns_in_phase=3,
                   rolling_scores={"empathy": 0.30, "clarity": 0.70})
        decision = engine.evaluate(ctx)
        assert decision.should_transition
        assert decision.target_phase == "rupture"

    def test_middle_empathy_no_trigger_stays(self):
        engine = _make_engine(self._branch_phases())
        ctx = _ctx("disclosure", turns_in_phase=3,
                   rolling_scores={"empathy": 0.55, "clarity": 0.60})
        assert not engine.evaluate(ctx).should_transition

    def test_max_turns_fallback(self):
        engine = _make_engine(self._branch_phases())
        ctx = _ctx("disclosure", turns_in_phase=8,
                   rolling_scores={"empathy": 0.55, "clarity": 0.55})
        decision = engine.evaluate(ctx)
        assert decision.should_transition
        assert decision.target_phase == "good_path"  # max_turns_reached branch

    def test_priority_ordering(self):
        """High-priority rupture (priority=3) should beat good_path (priority=2)
        when BOTH conditions are satisfied simultaneously."""
        engine = _make_engine(self._branch_phases())
        # empathy=0.35 triggers BOTH rupture (below 0.40) AND NOT good_path (not above 0.70)
        # rupture has higher priority so it wins
        ctx = _ctx("disclosure", turns_in_phase=3,
                   rolling_scores={"empathy": 0.35, "clarity": 0.75})
        decision = engine.evaluate(ctx)
        assert decision.target_phase == "rupture"

    def test_always_condition(self):
        """'always' condition should always match."""
        phases = [
            {
                "name": "setup", "description": "Open.", "max_turns": 5,
                "transition": {
                    "type": "branch_table",
                    "branches": [
                        {"condition": "always", "target_phase": "core", "priority": 1},
                    ],
                },
            },
            {
                "name": "core", "description": "Core.", "max_turns": 5,
                "transition": {"type": "always_after_max", "target_phase": "resolution"},
            },
            {
                "name": "resolution", "description": "End.", "max_turns": 2,
                "transition": {"type": "always_after_max", "target_phase": "resolution"},
            },
        ]
        engine = _make_engine(phases)
        ctx = _ctx("setup", turns_in_phase=0)
        decision = engine.evaluate(ctx)
        assert decision.should_transition
        assert decision.target_phase == "core"


class TestPhaseEngineGlobalLimits:

    def test_max_total_turns_overrides_phase(self):
        """Session should end (jump to last phase) before phase logic even runs."""
        engine = _make_engine()
        ctx = _ctx("setup", turns_in_phase=0, total_turns=13)  # == max_total_turns
        decision = engine.evaluate(ctx)
        assert decision.should_transition
        # Should jump to the last defined phase
        assert decision.target_phase == "resolution"

    def test_session_complete_detection(self):
        engine = _make_engine()
        # In resolution (completion phase) and past its max_turns
        ctx = _ctx("resolution", turns_in_phase=2, total_turns=10)
        assert engine.is_session_complete(ctx)

    def test_session_not_complete_mid_phase(self):
        engine = _make_engine()
        ctx = _ctx("resolution", turns_in_phase=1, total_turns=10)
        assert not engine.is_session_complete(ctx)

    def test_session_not_complete_wrong_phase(self):
        engine = _make_engine()
        ctx = _ctx("setup", turns_in_phase=3, total_turns=5)
        assert not engine.is_session_complete(ctx)

    def test_early_exit_score(self):
        engine = _make_engine()  # early_exit_score not set in minimal
        ctx = _ctx(rolling_scores={"composite": 0.99})
        assert not engine.should_end_on_score(ctx)  # None configured

    def test_unknown_phase_returns_stay(self):
        engine = _make_engine()
        ctx = _ctx("nonexistent_phase", turns_in_phase=99)
        decision = engine.evaluate(ctx)
        assert not decision.should_transition


# ===========================================================================
# 4. Prompt Builder Integration
# ===========================================================================

class TestPromptBuilderIntegration:
    """Verify that the prompt builder correctly uses scenario data."""

    def _build(self, phase="core", memory=None) -> str:
        from agent_service.graph.prompt_builder import build_system_prompt
        s = load_scenario()
        ctx = memory or {"scenario_chunks": [], "user_summaries": [], "behavioral_notes": []}
        return build_system_prompt(s, phase, ctx)

    def test_persona_name_present(self):
        assert "Sam" in self._build()

    def test_current_phase_highlighted(self):
        prompt = self._build(phase="resolution")
        assert "RESOLUTION" in prompt

    def test_setup_phase_highlighted(self):
        prompt = self._build(phase="setup")
        assert "SETUP" in prompt

    def test_rubric_dimensions_listed(self):
        prompt = self._build()
        assert "clarity" in prompt.lower()
        assert "confidence" in prompt.lower()

    def test_output_format_contains_delimiter(self):
        assert "<<<METADATA>>>" in self._build()

    def test_constraints_block_present(self):
        assert "HARD CONSTRAINTS" in self._build()

    def test_first_session_message_when_no_memory(self):
        p = self._build(memory={"scenario_chunks": [], "user_summaries": [], "behavioral_notes": []})
        assert "first" in p.lower()

    def test_memory_chunks_injected(self):
        ctx = {
            "scenario_chunks": ["STAR method is the key framework."],
            "user_summaries": ["Session 1: composite 0.62."],
            "behavioral_notes": ["Tends to give vague examples."],
        }
        p = self._build(memory=ctx)
        assert "STAR method is the key framework" in p
        assert "composite 0.62" in p
        assert "Tends to give vague examples" in p

    def test_valid_phases_in_output_format(self):
        prompt = self._build()
        # The output format block should list valid phase names
        assert "setup" in prompt
        assert "core" in prompt
        assert "resolution" in prompt

    def test_accepts_legacy_dict(self):
        """build_system_prompt must still accept raw dicts for backward compat."""
        from agent_service.graph.prompt_builder import build_system_prompt
        s = load_scenario()
        legacy = s.to_legacy_dict()
        p = build_system_prompt(legacy, "core", {})
        assert "Sam" in p

    def test_accepts_typed_scenario(self):
        """build_system_prompt must accept Scenario objects directly."""
        from agent_service.graph.prompt_builder import build_system_prompt
        s = load_scenario()
        p = build_system_prompt(s, "core", {})
        assert "Sam" in p

    def test_backstory_injected_when_present(self):
        from agent_service.graph.prompt_builder import build_system_prompt
        s = Scenario.model_validate(make_scenario(
            persona={
                "name": "Sam", "role": "test role",
                "emotional_start": "neutral", "style": "professional",
                "backstory": "Sam grew up in a small town and values honesty above all.",
                "constraints": ["Stay in character."],
            }
        ))
        p = build_system_prompt(s, "core", {})
        assert "small town" in p
