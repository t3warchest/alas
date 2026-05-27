"""
validate_scenarios.py — ALAS scenario system standalone validator

Tests every piece of the scenario system without requiring pydantic,
langgraph, or chromadb. Mirrors the logic in the real modules exactly.

Run with: python3 validate_scenarios.py
"""

from __future__ import annotations

import json
import sys
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ── colour helpers ─────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"
DIM = "\033[2m"; B = "\033[1m"; X = "\033[0m"

passed = failed = 0

def ok(name):
    global passed; passed += 1; print(f"  {G}✓{X} {name}")

def fail(name, reason=""):
    global failed; failed += 1; print(f"  {R}✗{X} {name}")
    if reason: print(f"    {DIM}{reason}{X}")

def section(title):
    print(f"\n{B}{C}▶ {title}{X}")

def assert_eq(name, got, expected):
    if got == expected: ok(name)
    else: fail(name, f"got {got!r}, expected {expected!r}")

def assert_in(name, needle, haystack):
    if needle in haystack: ok(name)
    else: fail(name, f"{needle!r} not found")

def assert_true(name, condition, hint=""):
    if condition: ok(name)
    else: fail(name, hint or "condition was False")

def assert_raises(name, fn):
    try:
        fn()
        fail(name, "no exception raised")
    except Exception:
        ok(name)

def run(name, fn):
    import traceback
    try: fn()
    except Exception as e:
        fail(name, f"{type(e).__name__}: {e}\n    {traceback.format_exc().splitlines()[-2]}")


# ══════════════════════════════════════════════════════════════════════════
# Inlined model logic (mirrors models.py without pydantic)
# ══════════════════════════════════════════════════════════════════════════

class ValidationError(Exception):
    pass

def _validate_scenario(data: dict) -> dict:
    """Pure-Python scenario validation, mirrors Pydantic model rules."""
    errors = []

    required = ["id","version","title","description","category",
                 "persona","student_objective","phases",
                 "skill_dimensions","rubric","knowledge_chunks","exit_conditions"]
    for f in required:
        if f not in data:
            errors.append(f"missing required field: '{f}'")

    if errors:
        raise ValidationError("; ".join(errors))

    # id format
    if not re.match(r'^[a-z0-9_]+$', data.get("id", "")):
        errors.append("id must match ^[a-z0-9_]+$")

    # version format
    if not re.match(r'^\d+\.\d+$', data.get("version", "")):
        errors.append("version must match ^\\d+\\.\\d+$")

    # category
    valid_categories = {"interview","feedback","negotiation","onboarding","custom"}
    if data.get("category") not in valid_categories:
        errors.append(f"category must be one of {valid_categories}")

    # persona
    persona = data.get("persona", {})
    for pf in ["name","role","emotional_start","style","constraints"]:
        if pf not in persona:
            errors.append(f"persona.{pf} required")
    valid_emotions = {"neutral","defensive","curious","warm","hostile","anxious"}
    if persona.get("emotional_start") not in valid_emotions:
        errors.append(f"persona.emotional_start must be one of {valid_emotions}")
    if not persona.get("constraints"):
        errors.append("persona.constraints must have at least 1 item")

    # phases
    phases = data.get("phases", [])
    if not phases:
        errors.append("phases must have at least 1 item")
    phase_names = [p["name"] for p in phases if "name" in p]
    if len(phase_names) != len(set(phase_names)):
        errors.append("phase names must be unique")

    # skill_dimensions weights
    dims = data.get("skill_dimensions", {})
    total = sum(dims.values()) if dims else 0
    if abs(total - 1.0) > 0.01:
        errors.append(f"skill_dimensions weights must sum to 1.0, got {total:.3f}")

    # rubric covers all dimensions
    rubric = data.get("rubric", {})
    missing_rubric = set(dims.keys()) - set(rubric.keys())
    if missing_rubric:
        errors.append(f"Rubric missing entries for skill dimensions: {missing_rubric}")

    # transition targets exist
    valid_phase_set = set(phase_names)
    for phase in phases:
        t = phase.get("transition", {})
        target = t.get("target_phase")
        if target and target not in valid_phase_set:
            errors.append(f"Phase '{phase.get('name')}' targets unknown phase '{target}'")
        for branch in t.get("branches", []):
            bt = branch.get("target_phase", "")
            if bt not in valid_phase_set:
                errors.append(f"Branch targets unknown phase '{bt}'")

    # completion phases exist
    exit_c = data.get("exit_conditions", {})
    for cp in exit_c.get("completion_phases", []):
        if cp not in valid_phase_set:
            errors.append(f"exit_conditions.completion_phases contains unknown phase '{cp}'")

    # transition-type-specific rules
    for phase in phases:
        t = phase.get("transition", {})
        tt = t.get("type", "")
        if tt == "score_threshold":
            if not t.get("score_dimension"):
                errors.append(f"Phase '{phase.get('name')}': score_threshold needs score_dimension")
            if t.get("score_threshold") is None:
                errors.append(f"Phase '{phase.get('name')}': score_threshold needs score_threshold value")
        if tt == "keyword_detected" and not t.get("keywords"):
            errors.append(f"Phase '{phase.get('name')}': keyword_detected needs keywords")
        if tt == "branch_table" and not t.get("branches"):
            errors.append(f"Phase '{phase.get('name')}': branch_table needs branches")

    if errors:
        raise ValidationError("; ".join(errors))

    return data


# ══════════════════════════════════════════════════════════════════════════
# Inlined branching engine (mirrors branching.py without pydantic)
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PhaseContext:
    current_phase_name: str
    turns_in_phase: int
    total_turns: int
    rolling_scores: dict
    last_student_utterance: str

@dataclass(frozen=True)
class TransitionDecision:
    should_transition: bool
    target_phase: str | None
    reason: str

    @classmethod
    def stay(cls, reason="staying"):
        return cls(False, None, reason)

    @classmethod
    def move(cls, target, reason):
        return cls(True, target, reason)


class PhaseEngine:
    def __init__(self, scenario: dict):
        self._scenario = scenario
        self._phases = {p["name"]: p for p in scenario.get("phases", [])}

    def evaluate(self, ctx: PhaseContext) -> TransitionDecision:
        phase = self._phases.get(ctx.current_phase_name)
        if not phase:
            return TransitionDecision.stay("unknown phase")

        max_total = self._scenario.get("exit_conditions", {}).get("max_total_turns", 9999)
        if ctx.total_turns >= max_total:
            last = self._scenario["phases"][-1]["name"]
            return TransitionDecision.move(last, f"max_total_turns ({max_total}) reached")

        t = phase.get("transition", {})
        tt = t.get("type", "always_after_max")

        if tt == "always_after_max":
            return self._always_after_max(t, ctx, phase)
        elif tt == "score_threshold":
            return self._score_threshold(t, ctx, phase)
        elif tt == "keyword_detected":
            return self._keyword_detected(t, ctx, phase)
        elif tt == "branch_table":
            return self._branch_table(t, ctx, phase)
        return TransitionDecision.stay("unknown transition type")

    def _always_after_max(self, t, ctx, phase):
        if ctx.turns_in_phase >= phase["max_turns"]:
            target = t.get("target_phase") or self._next(phase["name"])
            if target:
                return TransitionDecision.move(target, f"max_turns ({phase['max_turns']}) reached")
        return TransitionDecision.stay(f"{ctx.turns_in_phase}/{phase['max_turns']}")

    def _score_threshold(self, t, ctx, phase):
        dim = t.get("score_dimension", "composite")
        thresh = t.get("score_threshold", 0.7)
        score = ctx.rolling_scores.get(dim, 0.0)
        if score >= thresh:
            target = t.get("target_phase") or self._next(phase["name"])
            if target:
                return TransitionDecision.move(target, f"{dim}={score:.2f} ≥ {thresh:.2f}")
        if ctx.turns_in_phase >= phase["max_turns"]:
            target = t.get("target_phase") or self._next(phase["name"])
            if target:
                return TransitionDecision.move(target, f"max_turns fallback; {dim}={score:.2f}")
        return TransitionDecision.stay(f"{dim}={score:.2f}<{thresh:.2f}")

    def _keyword_detected(self, t, ctx, phase):
        u = ctx.last_student_utterance.lower()
        for kw in t.get("keywords", []):
            if kw.lower() in u:
                target = t.get("target_phase") or self._next(phase["name"])
                if target:
                    return TransitionDecision.move(target, f"keyword '{kw}' detected")
        if ctx.turns_in_phase >= phase["max_turns"]:
            target = t.get("target_phase") or self._next(phase["name"])
            if target:
                return TransitionDecision.move(target, "max_turns fallback; no keyword")
        return TransitionDecision.stay("no keyword match")

    def _branch_table(self, t, ctx, phase):
        branches = sorted(t.get("branches", []), key=lambda b: b.get("priority", 0), reverse=True)
        for branch in branches:
            if self._eval_condition(branch["condition"], ctx, phase):
                return TransitionDecision.move(branch["target_phase"],
                                               f"branch condition: '{branch['condition']}'")
        return TransitionDecision.stay("no branch matched")

    def _eval_condition(self, cond: str, ctx: PhaseContext, phase: dict) -> bool:
        c = cond.strip().lower()
        if c == "always": return True
        if c == "max_turns_reached": return ctx.turns_in_phase >= phase["max_turns"]
        parts = [p.strip() for p in c.split("_and_")]
        return all(self._eval_simple(p, ctx) for p in parts)

    def _eval_simple(self, expr: str, ctx: PhaseContext) -> bool:
        m = re.match(r'^(?P<dim>[a-z_]+)_(?P<op>above|below)_(?P<val>[\d.]+)$', expr)
        if not m: return False
        dim, op, threshold = m.group("dim"), m.group("op"), float(m.group("val"))
        score = ctx.rolling_scores.get(dim, 0.0)
        return score > threshold if op == "above" else score < threshold

    def is_session_complete(self, ctx: PhaseContext) -> bool:
        cps = self._scenario.get("exit_conditions", {}).get("completion_phases", [])
        if ctx.current_phase_name not in cps: return False
        phase = self._phases.get(ctx.current_phase_name, {})
        return ctx.turns_in_phase >= phase.get("max_turns", 9999)

    def _next(self, name: str) -> str | None:
        phases = self._scenario.get("phases", [])
        for i, p in enumerate(phases):
            if p["name"] == name and i + 1 < len(phases):
                return phases[i + 1]["name"]
        return None


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

DEFS_DIR = Path(__file__).parent / "agent_service" / "scenarios" / "definitions"

def load_json(name: str) -> dict:
    return json.loads((DEFS_DIR / name).read_text())

MINIMAL = {
    "id": "test_v1", "version": "1.0", "title": "Test Scenario Title",
    "description": "A minimal scenario for unit testing purposes here.",
    "category": "interview",
    "persona": {
        "name": "Sam", "role": "Tester", "emotional_start": "neutral",
        "style": "professional", "constraints": ["Stay in character."],
    },
    "student_objective": "Demonstrate competence during this structured test scenario.",
    "phases": [
        {"name": "setup", "description": "Opening phase here.",  "max_turns": 3,
         "transition": {"type": "always_after_max", "target_phase": "core"}},
        {"name": "core",  "description": "Main challenge phase.", "max_turns": 8,
         "transition": {"type": "always_after_max", "target_phase": "resolution"}},
        {"name": "resolution", "description": "Wrap up phase here.", "max_turns": 2,
         "transition": {"type": "always_after_max", "target_phase": "resolution"}},
    ],
    "skill_dimensions": {"clarity": 0.5, "confidence": 0.5},
    "rubric": {
        "clarity":    {"1": "Unclear.", "3": "Adequate.", "5": "Crystal-clear."},
        "confidence": {"1": "Hesitant.", "3": "Steady.",  "5": "Authoritative."},
    },
    "knowledge_chunks": ["This is a knowledge chunk for the test scenario here."],
    "exit_conditions": {"max_total_turns": 13, "completion_phases": ["resolution"]},
}

def make(**overrides) -> dict:
    import copy; d = copy.deepcopy(MINIMAL); d.update(overrides); return d

def ctx(phase="setup", tip=0, total=0, scores=None, utterance="") -> PhaseContext:
    return PhaseContext(phase, tip, total, scores or {}, utterance)


# ══════════════════════════════════════════════════════════════════════════
# 1. MODEL VALIDATION
# ══════════════════════════════════════════════════════════════════════════

section("Schema Validation")

def test_valid():
    _validate_scenario(MINIMAL)
    ok("valid minimal scenario passes")

def test_bad_id():
    assert_raises("uppercase id rejected",
                  lambda: _validate_scenario(make(id="Bad_ID")))

def test_bad_version():
    assert_raises("bad version rejected",
                  lambda: _validate_scenario(make(version="v1")))

def test_bad_category():
    assert_raises("invalid category rejected",
                  lambda: _validate_scenario(make(category="fantasy")))

def test_bad_emotion():
    d = make(); d["persona"]["emotional_start"] = "FURIOUS"
    assert_raises("invalid emotion rejected", lambda: _validate_scenario(d))

def test_weights_not_one():
    assert_raises("weights ≠ 1.0 rejected",
                  lambda: _validate_scenario(make(skill_dimensions={"clarity": 0.3, "confidence": 0.4})))

def test_rubric_missing_dim():
    assert_raises("rubric missing dimension rejected",
                  lambda: _validate_scenario(make(
                      skill_dimensions={"clarity": 0.5, "empathy": 0.5},
                      rubric={"clarity": {"1": "x", "3": "y", "5": "z"}},
                  )))

def test_duplicate_phases():
    phases = [
        {"name": "setup", "description": "A.", "max_turns": 3,
         "transition": {"type": "always_after_max", "target_phase": "setup"}},
        {"name": "setup", "description": "B.", "max_turns": 3,
         "transition": {"type": "always_after_max", "target_phase": "setup"}},
    ]
    assert_raises("duplicate phase names rejected",
                  lambda: _validate_scenario(make(phases=phases)))

def test_unknown_transition_target():
    phases = [
        {"name": "setup", "description": "A.", "max_turns": 3,
         "transition": {"type": "always_after_max", "target_phase": "ghost"}},
    ]
    assert_raises("unknown transition target rejected",
                  lambda: _validate_scenario(make(phases=phases)))

def test_completion_phase_not_exist():
    assert_raises("unknown completion phase rejected",
                  lambda: _validate_scenario(make(exit_conditions={
                      "max_total_turns": 10, "completion_phases": ["ghost"]})))

def test_score_threshold_missing_dim():
    phases = list(MINIMAL["phases"])
    phases[0] = {**phases[0], "transition": {"type": "score_threshold", "score_threshold": 0.7}}
    assert_raises("score_threshold missing dimension rejected",
                  lambda: _validate_scenario(make(phases=phases)))

def test_branch_table_no_branches():
    phases = list(MINIMAL["phases"])
    phases[0] = {**phases[0], "transition": {"type": "branch_table"}}
    assert_raises("branch_table with no branches rejected",
                  lambda: _validate_scenario(make(phases=phases)))

run("valid minimal scenario", test_valid)
run("bad id format", test_bad_id)
run("bad version format", test_bad_version)
run("invalid category", test_bad_category)
run("invalid emotional_start", test_bad_emotion)
run("weights not summing to 1", test_weights_not_one)
run("rubric missing dimension", test_rubric_missing_dim)
run("duplicate phase names", test_duplicate_phases)
run("unknown transition target", test_unknown_transition_target)
run("completion phase not in phases", test_completion_phase_not_exist)
run("score_threshold missing score_dimension", test_score_threshold_missing_dim)
run("branch_table with no branches", test_branch_table_no_branches)


# ══════════════════════════════════════════════════════════════════════════
# 2. JSON SCENARIO FILES
# ══════════════════════════════════════════════════════════════════════════

section("JSON Scenario Files")

SCENARIO_FILES = list(DEFS_DIR.glob("*.json")) if DEFS_DIR.exists() else []

def test_files_exist():
    assert_true(f"{len(SCENARIO_FILES)} JSON files found", len(SCENARIO_FILES) >= 3,
                "Expected at least 3 scenario files")

def test_all_parse():
    for f in SCENARIO_FILES:
        try:
            d = json.loads(f.read_text())
            _validate_scenario(d)
            ok(f"{f.name} parses and validates")
        except Exception as e:
            fail(f"{f.name} validation", str(e))

def test_weights_sum_to_one():
    for f in SCENARIO_FILES:
        d = json.loads(f.read_text())
        dims = d.get("skill_dimensions", {})
        total = sum(dims.values())
        assert_true(f"{f.stem}: weights sum to 1.0",
                    abs(total - 1.0) < 0.01, f"sum={total:.4f}")

def test_rubric_covers_all_dims():
    for f in SCENARIO_FILES:
        d = json.loads(f.read_text())
        dims = set(d.get("skill_dimensions", {}).keys())
        rubric = set(d.get("rubric", {}).keys())
        missing = dims - rubric
        assert_true(f"{f.stem}: rubric covers all dimensions",
                    not missing, f"missing: {missing}")

def test_completion_phases_exist_in_phases():
    for f in SCENARIO_FILES:
        d = json.loads(f.read_text())
        phase_names = {p["name"] for p in d.get("phases", [])}
        for cp in d.get("exit_conditions", {}).get("completion_phases", []):
            assert_true(f"{f.stem}: completion phase '{cp}' exists",
                        cp in phase_names)

def test_knowledge_chunks_non_empty():
    for f in SCENARIO_FILES:
        d = json.loads(f.read_text())
        chunks = d.get("knowledge_chunks", [])
        assert_true(f"{f.stem}: has knowledge chunks", len(chunks) >= 1)

def test_branch_table_scenarios_present():
    has_branch = [f for f in SCENARIO_FILES
                  if any(p.get("transition", {}).get("type") == "branch_table"
                         for p in json.loads(f.read_text()).get("phases", []))]
    assert_true("At least one scenario uses branch_table transitions",
                len(has_branch) >= 1, "No branch_table scenario found")

def test_coworker_has_rupture_branch():
    cf = DEFS_DIR / "difficult_coworker_v1.json"
    if cf.exists():
        d = json.loads(cf.read_text())
        all_targets = [
            b["target_phase"]
            for p in d.get("phases", [])
            for b in p.get("transition", {}).get("branches", [])
        ]
        assert_in("difficult_coworker has rupture branch", "rupture", all_targets)
    else:
        fail("difficult_coworker_v1.json missing")

def test_negotiation_has_anchor_phase():
    nf = DEFS_DIR / "salary_negotiation_v1.json"
    if nf.exists():
        d = json.loads(nf.read_text())
        phase_names = [p["name"] for p in d.get("phases", [])]
        assert_in("salary_negotiation has anchor phase", "anchor", phase_names)
    else:
        fail("salary_negotiation_v1.json missing")

run("definition files exist", test_files_exist)
run("all files parse and validate", test_all_parse)
run("all weights sum to 1.0", test_weights_sum_to_one)
run("rubric covers all dimensions", test_rubric_covers_all_dims)
run("completion phases exist in phases", test_completion_phases_exist_in_phases)
run("knowledge chunks non-empty", test_knowledge_chunks_non_empty)
run("branch_table scenarios present", test_branch_table_scenarios_present)
run("coworker scenario has rupture branch", test_coworker_has_rupture_branch)
run("negotiation scenario has anchor phase", test_negotiation_has_anchor_phase)


# ══════════════════════════════════════════════════════════════════════════
# 3. BRANCHING ENGINE — always_after_max
# ══════════════════════════════════════════════════════════════════════════

section("PhaseEngine — always_after_max")

engine = PhaseEngine(MINIMAL)

def test_stays_before_max():
    d = engine.evaluate(ctx("setup", tip=2))
    assert_true("stays before max", not d.should_transition)

def test_transitions_at_max():
    d = engine.evaluate(ctx("setup", tip=3))
    assert_true("transitions at max", d.should_transition)
    assert_eq("target phase", d.target_phase, "core")

def test_transitions_past_max():
    d = engine.evaluate(ctx("setup", tip=99))
    assert_true("transitions past max", d.should_transition)

def test_reason_populated():
    d = engine.evaluate(ctx("setup", tip=3))
    assert_in("reason contains max_turns", "max_turns", d.reason)

run("stays before max_turns", test_stays_before_max)
run("transitions at max_turns", test_transitions_at_max)
run("transitions past max_turns", test_transitions_past_max)
run("reason string populated", test_reason_populated)


# ══════════════════════════════════════════════════════════════════════════
# 4. BRANCHING ENGINE — score_threshold
# ══════════════════════════════════════════════════════════════════════════

section("PhaseEngine — score_threshold")

SCORE_PHASES = [
    {"name": "setup", "description": "Open.", "max_turns": 3,
     "transition": {"type": "always_after_max", "target_phase": "core"}},
    {"name": "core", "description": "Main.", "max_turns": 10,
     "transition": {"type": "score_threshold", "score_dimension": "empathy",
                    "score_threshold": 0.70, "target_phase": "resolution"}},
    {"name": "resolution", "description": "End.", "max_turns": 3,
     "transition": {"type": "always_after_max", "target_phase": "resolution"}},
]
score_engine = PhaseEngine(make(phases=SCORE_PHASES, exit_conditions={
    "max_total_turns": 16, "completion_phases": ["resolution"]}))

def test_score_stays_below():
    d = score_engine.evaluate(ctx("core", tip=4, scores={"empathy": 0.65}))
    assert_true("stays below threshold", not d.should_transition)

def test_score_transitions_above():
    d = score_engine.evaluate(ctx("core", tip=4, scores={"empathy": 0.72}))
    assert_true("transitions above threshold", d.should_transition)
    assert_eq("target is resolution", d.target_phase, "resolution")
    assert_in("score in reason", "0.72", d.reason)

def test_score_fallback_to_max():
    d = score_engine.evaluate(ctx("core", tip=10, scores={"empathy": 0.50}))
    assert_true("fallback to max_turns", d.should_transition)
    assert_in("fallback in reason", "fallback", d.reason)

run("stays below threshold", test_score_stays_below)
run("transitions above threshold", test_score_transitions_above)
run("fallback to max_turns", test_score_fallback_to_max)


# ══════════════════════════════════════════════════════════════════════════
# 5. BRANCHING ENGINE — keyword_detected
# ══════════════════════════════════════════════════════════════════════════

section("PhaseEngine — keyword_detected")

KW_PHASES = [
    {"name": "setup", "description": "Open.", "max_turns": 5,
     "transition": {"type": "keyword_detected",
                    "keywords": ["agree", "sounds good", "let's do it"],
                    "target_phase": "core"}},
    {"name": "core", "description": "Main.", "max_turns": 8,
     "transition": {"type": "always_after_max", "target_phase": "resolution"}},
    {"name": "resolution", "description": "End.", "max_turns": 2,
     "transition": {"type": "always_after_max", "target_phase": "resolution"}},
]
kw_engine = PhaseEngine(make(phases=KW_PHASES, exit_conditions={
    "max_total_turns": 15, "completion_phases": ["resolution"]}))

def test_kw_no_match():
    d = kw_engine.evaluate(ctx("setup", tip=1, utterance="I'm not sure."))
    assert_true("no keyword stays", not d.should_transition)

def test_kw_match():
    d = kw_engine.evaluate(ctx("setup", tip=1, utterance="Yeah, sounds good!"))
    assert_true("keyword triggers transition", d.should_transition)
    assert_eq("target is core", d.target_phase, "core")

def test_kw_case_insensitive():
    d = kw_engine.evaluate(ctx("setup", tip=1, utterance="I AGREE with that."))
    assert_true("keyword match is case-insensitive", d.should_transition)

def test_kw_fallback():
    d = kw_engine.evaluate(ctx("setup", tip=5, utterance="nothing here"))
    assert_true("fallback to max_turns", d.should_transition)

run("no keyword — stays", test_kw_no_match)
run("keyword present — transitions", test_kw_match)
run("keyword case-insensitive", test_kw_case_insensitive)
run("keyword fallback to max_turns", test_kw_fallback)


# ══════════════════════════════════════════════════════════════════════════
# 6. BRANCHING ENGINE — branch_table
# ══════════════════════════════════════════════════════════════════════════

section("PhaseEngine — branch_table")

BRANCH_PHASES = [
    {"name": "setup", "description": "Open.", "max_turns": 3,
     "transition": {"type": "always_after_max", "target_phase": "disclosure"}},
    {"name": "disclosure", "description": "Core.", "max_turns": 8,
     "transition": {
         "type": "branch_table",
         "branches": [
             {"condition": "empathy_above_0.70_and_clarity_above_0.65",
              "target_phase": "good_path", "priority": 2},
             {"condition": "empathy_below_0.40",
              "target_phase": "rupture", "priority": 3},
             {"condition": "max_turns_reached",
              "target_phase": "good_path", "priority": 1},
         ],
     }},
    {"name": "good_path",  "description": "Good branch.", "max_turns": 4,
     "transition": {"type": "always_after_max", "target_phase": "close"}},
    {"name": "rupture",    "description": "Bad branch.",  "max_turns": 4,
     "transition": {"type": "always_after_max", "target_phase": "close"}},
    {"name": "close",      "description": "Wrap up.",     "max_turns": 2,
     "transition": {"type": "always_after_max", "target_phase": "close"}},
]
bt_engine = PhaseEngine(make(phases=BRANCH_PHASES, exit_conditions={
    "max_total_turns": 21, "completion_phases": ["close"]}))

def test_high_empathy_good_path():
    d = bt_engine.evaluate(ctx("disclosure", tip=3,
                                scores={"empathy": 0.80, "clarity": 0.75}))
    assert_true("high empathy → good_path", d.should_transition)
    assert_eq("target is good_path", d.target_phase, "good_path")

def test_low_empathy_rupture():
    d = bt_engine.evaluate(ctx("disclosure", tip=3,
                                scores={"empathy": 0.30, "clarity": 0.70}))
    assert_true("low empathy → rupture", d.should_transition)
    assert_eq("target is rupture", d.target_phase, "rupture")

def test_middle_stays():
    d = bt_engine.evaluate(ctx("disclosure", tip=3,
                                scores={"empathy": 0.55, "clarity": 0.55}))
    assert_true("middle scores — stays", not d.should_transition)

def test_max_turns_fallback_branch():
    d = bt_engine.evaluate(ctx("disclosure", tip=8,
                                scores={"empathy": 0.55, "clarity": 0.55}))
    assert_true("max_turns fallback fires", d.should_transition)
    assert_eq("target is good_path", d.target_phase, "good_path")

def test_priority_ordering():
    # empathy=0.30 triggers rupture (priority 3) and NOT good_path (needs empathy>0.70)
    d = bt_engine.evaluate(ctx("disclosure", tip=3,
                                scores={"empathy": 0.30, "clarity": 0.80}))
    assert_eq("rupture wins (higher priority)", d.target_phase, "rupture")

def test_always_condition():
    always_phases = [
        {"name": "setup", "description": "Open.", "max_turns": 5,
         "transition": {
             "type": "branch_table",
             "branches": [{"condition": "always", "target_phase": "core", "priority": 1}],
         }},
        {"name": "core", "description": "Core.", "max_turns": 5,
         "transition": {"type": "always_after_max", "target_phase": "resolution"}},
        {"name": "resolution", "description": "End.", "max_turns": 2,
         "transition": {"type": "always_after_max", "target_phase": "resolution"}},
    ]
    ae = PhaseEngine(make(phases=always_phases, exit_conditions={
        "max_total_turns": 12, "completion_phases": ["resolution"]}))
    d = ae.evaluate(ctx("setup", tip=0))
    assert_true("always condition always fires", d.should_transition)
    assert_eq("targets core", d.target_phase, "core")

def test_compound_and_condition():
    # empathy_above_0.70_and_clarity_above_0.65 — test each side
    # Both true → transition
    d = bt_engine.evaluate(ctx("disclosure", tip=2,
                                scores={"empathy": 0.75, "clarity": 0.70}))
    assert_true("compound AND — both true → transition", d.should_transition)

def test_compound_and_partial():
    # empathy high but clarity low — compound should NOT match
    d = bt_engine.evaluate(ctx("disclosure", tip=2,
                                scores={"empathy": 0.75, "clarity": 0.50}))
    # good_path needs clarity > 0.65 — not met; rupture needs empathy < 0.40 — not met
    assert_true("compound AND — partial — stays", not d.should_transition)

run("high empathy → good_path", test_high_empathy_good_path)
run("low empathy → rupture",    test_low_empathy_rupture)
run("middle scores — stays",    test_middle_stays)
run("max_turns fallback fires", test_max_turns_fallback_branch)
run("priority ordering correct", test_priority_ordering)
run("always condition fires",   test_always_condition)
run("compound AND — both true", test_compound_and_condition)
run("compound AND — partial — stays", test_compound_and_partial)


# ══════════════════════════════════════════════════════════════════════════
# 7. GLOBAL SESSION LIMITS
# ══════════════════════════════════════════════════════════════════════════

section("PhaseEngine — Global Session Limits")

def test_max_total_turns_override():
    engine = PhaseEngine(MINIMAL)
    d = engine.evaluate(ctx("setup", tip=0, total=13))
    assert_true("max_total_turns overrides phase", d.should_transition)
    assert_eq("jumps to last phase", d.target_phase, "resolution")

def test_session_complete_in_completion_phase():
    engine = PhaseEngine(MINIMAL)
    assert_true("session complete in resolution @ max",
                engine.is_session_complete(ctx("resolution", tip=2)))

def test_session_not_complete_before_max():
    engine = PhaseEngine(MINIMAL)
    assert_true("session not complete before max",
                not engine.is_session_complete(ctx("resolution", tip=1)))

def test_session_not_complete_wrong_phase():
    engine = PhaseEngine(MINIMAL)
    assert_true("session not complete in non-completion phase",
                not engine.is_session_complete(ctx("setup", tip=99)))

def test_unknown_phase_stays():
    engine = PhaseEngine(MINIMAL)
    d = engine.evaluate(ctx("ghost_phase", tip=99))
    assert_true("unknown phase returns stay", not d.should_transition)

run("max_total_turns overrides phase", test_max_total_turns_override)
run("session complete in completion phase @ max", test_session_complete_in_completion_phase)
run("session not complete before max", test_session_not_complete_before_max)
run("session not complete in wrong phase", test_session_not_complete_wrong_phase)
run("unknown phase returns stay", test_unknown_phase_stays)


# ══════════════════════════════════════════════════════════════════════════
# 8. PROMPT BUILDER INTEGRATION
# ══════════════════════════════════════════════════════════════════════════

section("Prompt Builder Integration")

# Inline the build_system_prompt logic for standalone testing

def _build(phase="core", scenario_override=None, memory=None) -> str:
    cfg = scenario_override or MINIMAL
    ctx_mem = memory or {"scenario_chunks": [], "user_summaries": [], "behavioral_notes": []}
    phases = cfg.get("phases", [])
    valid_phase_names = [p["name"] for p in phases]

    persona = cfg["persona"]
    backstory = persona.get("backstory", "")
    constraints = "\n".join(f"  - {c}" for c in persona.get("constraints", []))
    blocks = []

    # Persona block
    blocks.append(
        f"## YOUR PERSONA\nYou are **{persona['name']}**, {persona['role']}.\n"
        f"Emotional starting state: {persona['emotional_start']}.\n"
        f"Interaction style: {persona['style']}.\n"
        + (f"\nBackstory:\n{backstory}\n" if backstory else "")
        + f"\nBehavioural rules:\n{constraints}"
    )

    # Scenario block
    blocks.append(
        f"## SCENARIO CONTEXT\n**{cfg['title']}**\n{cfg['description']}\n\n"
        f"**Student's objective**:\n{cfg['student_objective']}"
    )

    # Phase block
    phase_cfg = next((p for p in phases if p["name"] == phase), None)
    if phase_cfg:
        ag = phase_cfg.get("avatar_goal", "")
        blocks.append(
            f"## CURRENT PHASE: **{phase.upper()}**\n{phase_cfg['description']}"
            + (f"\nYour goal: {ag}" if ag else "")
        )

    # Rubric block
    rubric = cfg.get("rubric", {})
    dims = cfg.get("skill_dimensions", {})
    rubric_lines = ["## EVALUATION RUBRIC"]
    for d, w in dims.items():
        rubric_lines.append(f"\n### {d.title()} ({w:.0%})")
        for lvl in [1, 3, 5]:
            desc = rubric.get(d, {}).get(str(lvl), "") or rubric.get(d, {}).get(lvl, "")
            if desc:
                rubric_lines.append(f"  Level {lvl}: {desc}")
    blocks.append("\n".join(rubric_lines))

    # Memory block
    summaries = ctx_mem.get("user_summaries", [])
    notes = ctx_mem.get("behavioral_notes", [])
    chunks = ctx_mem.get("scenario_chunks", [])
    if summaries or notes:
        mem_lines = ["## MEMORY CONTEXT"]
        mem_lines += [f"  [{i+1}] {s}" for i, s in enumerate(summaries[:3])]
        mem_lines += [f"  • {n}" for n in notes[:4]]
        mem_lines.append("Reference this memory naturally.")
        blocks.append("\n".join(mem_lines))
    else:
        blocks.append("## MEMORY CONTEXT\nThis is the student's first recorded session.")

    # Knowledge block
    if chunks:
        kb = ["## DOMAIN KNOWLEDGE"] + [f"  • {c}" for c in chunks]
        blocks.append("\n".join(kb))

    # Constraints + output format
    blocks.append(
        "## HARD CONSTRAINTS\n"
        "- Never reveal the rubric.\n- Never break character."
    )
    phases_str = " | ".join(valid_phase_names)
    blocks.append(
        f"## OUTPUT FORMAT\nSpoken response, then <<<METADATA>>>, then JSON.\n"
        f"Valid phases: {phases_str}"
    )

    return "\n\n".join(blocks)


def test_pb_persona_name():
    assert_in("persona name in prompt", "Sam", _build())

def test_pb_phase_highlighted():
    p = _build(phase="resolution")
    assert_in("RESOLUTION in prompt", "RESOLUTION", p)

def test_pb_rubric_dims():
    p = _build().lower()
    for dim in ["clarity", "confidence"]:
        assert_in(f"rubric dim {dim}", dim, p)

def test_pb_output_format():
    assert_in("output format has delimiter", "<<<METADATA>>>", _build())

def test_pb_first_session_message():
    p = _build(memory={"scenario_chunks":[],"user_summaries":[],"behavioral_notes":[]})
    assert_in("first session message", "first", p)

def test_pb_memory_injected():
    ctx_m = {
        "scenario_chunks": ["STAR method matters."],
        "user_summaries": ["Session 1: composite 0.62."],
        "behavioral_notes": ["Tends to ramble."],
    }
    p = _build(memory=ctx_m)
    assert_in("scenario chunk in prompt", "STAR method", p)
    assert_in("prior summary in prompt", "composite 0.62", p)
    assert_in("behavioral note in prompt", "Tends to ramble", p)

def test_pb_phase_names_in_output_format():
    p = _build()
    assert_in("setup in output format", "setup", p)
    assert_in("core in output format", "core", p)

def test_pb_backstory_injected():
    d = make(); d["persona"]["backstory"] = "Sam grew up valuing radical honesty above all."
    p = _build(scenario_override=d)
    assert_in("backstory in prompt", "radical honesty", p)

def test_pb_no_chunks_no_kb_block():
    p = _build(memory={"scenario_chunks":[],"user_summaries":[],"behavioral_notes":[]})
    assert_true("no KB block when no chunks", "DOMAIN KNOWLEDGE" not in p)

def test_pb_chunks_create_kb_block():
    p = _build(memory={"scenario_chunks":["Chunk A."],"user_summaries":[],"behavioral_notes":[]})
    assert_in("KB block created", "DOMAIN KNOWLEDGE", p)
    assert_in("chunk content present", "Chunk A", p)

run("persona name in prompt", test_pb_persona_name)
run("current phase highlighted (RESOLUTION)", test_pb_phase_highlighted)
run("rubric dimensions listed", test_pb_rubric_dims)
run("output format has delimiter", test_pb_output_format)
run("first session fallback message", test_pb_first_session_message)
run("memory context fully injected", test_pb_memory_injected)
run("valid phase names in output format", test_pb_phase_names_in_output_format)
run("backstory injected when present", test_pb_backstory_injected)
run("no KB block when no chunks", test_pb_no_chunks_no_kb_block)
run("chunks create KB block", test_pb_chunks_create_kb_block)


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════

total = passed + failed
print(f"\n{'━'*58}")
print(f"{B}Results:{X}  {G}{passed} passed{X}  "
      f"{(R if failed else DIM)}{failed} failed{X}  of {total} total")
print(f"{'━'*58}\n")
sys.exit(1 if failed else 0)
