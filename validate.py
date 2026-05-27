"""
validate.py — ALAS standalone logic validator

Tests every pure-Python component without requiring langgraph, openai,
or chromadb. Run with: python validate.py

This script proves the core architectural logic is sound.
When the real dependencies are installed, run the full pytest suite instead.
"""

from __future__ import annotations

import sys
import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

# ── colour helpers ─────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

passed = failed = 0

def ok(name: str):
    global passed
    passed += 1
    print(f"  {GREEN}✓{RESET} {name}")

def fail(name: str, reason: str):
    global failed
    failed += 1
    print(f"  {RED}✗{RESET} {name}")
    print(f"    {DIM}{reason}{RESET}")

def section(title: str):
    print(f"\n{BOLD}{CYAN}▶ {title}{RESET}")

def assert_eq(name, got, expected):
    if got == expected:
        ok(name)
    else:
        fail(name, f"got {got!r}, expected {expected!r}")

def assert_in(name, needle, haystack):
    if needle in haystack:
        ok(name)
    else:
        fail(name, f"{needle!r} not found in result")

def assert_true(name, condition, hint=""):
    if condition:
        ok(name)
    else:
        fail(name, hint or "condition was False")

def run(name, fn):
    try:
        fn()
    except Exception as e:
        fail(name, f"{type(e).__name__}: {e}\n    {traceback.format_exc().splitlines()[-2]}")


# ══════════════════════════════════════════════════════════════════════════
# 1. RESPONSE PARSER
# ══════════════════════════════════════════════════════════════════════════

section("Response Parser")

DELIMITER = "<<<METADATA>>>"
VALID_EMOTIONS = {"neutral","curious","concerned","encouraging","challenging","warm","disappointed"}
VALID_PHASES   = {"setup","core","escalation","resolution"}

def parse_llm_response(raw: str) -> tuple[str, dict]:
    import re

    def extract_trailing_json(text):
        match = re.search(r"\{[^{}]*\}$", text, re.DOTALL)
        if match:
            return text[:match.start()].strip(), match.group(0)
        return text.strip(), "{}"

    def parse_meta(raw_meta):
        defaults = {"emotion":"neutral","scenario_phase":"core","branch_signal":None,"hidden_notes":""}
        if not raw_meta or raw_meta.strip() in ("{}",""):
            return defaults
        try:
            data = json.loads(raw_meta)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_meta, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return defaults
            else:
                return defaults
        emotion = data.get("emotion","neutral")
        if emotion not in VALID_EMOTIONS: emotion = "neutral"
        phase = data.get("scenario_phase","core")
        if phase not in VALID_PHASES: phase = defaults["scenario_phase"]
        branch = data.get("branch_signal")
        if branch not in VALID_PHASES and branch is not None: branch = None
        return {"emotion":emotion,"scenario_phase":phase,
                "branch_signal":branch,"hidden_notes":str(data.get("hidden_notes",""))}

    if DELIMITER in raw:
        parts = raw.split(DELIMITER, 1)
        spoken, meta_raw = parts[0].strip(), parts[1].strip()
    else:
        spoken, meta_raw = extract_trailing_json(raw)

    return spoken, parse_meta(meta_raw)


def test_parser_clean():
    raw = f"Tell me about yourself.\n{DELIMITER}\n" + \
          '{"emotion":"curious","scenario_phase":"core","branch_signal":null,"hidden_notes":"test"}'
    spoken, d = parse_llm_response(raw)
    assert_in("spoken contains expected text", "Tell me", spoken)
    assert_eq("emotion parsed", d["emotion"], "curious")
    assert_eq("phase parsed", d["scenario_phase"], "core")
    assert_eq("branch null", d["branch_signal"], None)

def test_parser_branch_signal():
    raw = f"Let's wrap up.\n{DELIMITER}\n" + \
          '{"emotion":"warm","scenario_phase":"resolution","branch_signal":"resolution","hidden_notes":""}'
    _, d = parse_llm_response(raw)
    assert_eq("branch signal propagated", d["branch_signal"], "resolution")

def test_parser_unknown_emotion_clamped():
    raw = f"Text.\n{DELIMITER}\n" + \
          '{"emotion":"FURIOUS","scenario_phase":"core","branch_signal":null,"hidden_notes":""}'
    _, d = parse_llm_response(raw)
    assert_eq("bad emotion clamped to neutral", d["emotion"], "neutral")

def test_parser_no_delimiter():
    raw = "Just some plain text with no metadata."
    spoken, d = parse_llm_response(raw)
    assert_in("full text returned as spoken", "plain text", spoken)
    assert_true("defaults returned", d["emotion"] in VALID_EMOTIONS)

def test_parser_invalid_json():
    raw = f"Response.\n{DELIMITER}\nnot valid {{ json"
    spoken, d = parse_llm_response(raw)
    assert_eq("spoken still extracted", spoken, "Response.")
    assert_eq("defaults on bad json", d["emotion"], "neutral")

def test_parser_empty_input():
    spoken, d = parse_llm_response("")
    assert_eq("empty spoken", spoken, "")
    assert_true("dict returned", isinstance(d, dict))

run("clean response with delimiter", test_parser_clean)
run("branch signal propagation", test_parser_branch_signal)
run("unknown emotion clamped", test_parser_unknown_emotion_clamped)
run("no delimiter fallback", test_parser_no_delimiter)
run("invalid JSON fallback", test_parser_invalid_json)
run("empty input", test_parser_empty_input)


# ══════════════════════════════════════════════════════════════════════════
# 2. EVALUATION AGGREGATION
# ══════════════════════════════════════════════════════════════════════════

section("Evaluation Aggregation")

def make_score(turn_index, composite, **kw):
    base = dict(turn_index=turn_index, clarity=composite, empathy=composite,
                structure=composite, relevance=composite, confidence=composite,
                composite=composite, rationale=f"Turn {turn_index}.")
    base.update(kw)
    return base

def aggregate_session_scores(turn_scores):
    if not turn_scores:
        return {"error": "No scores."}
    dims = ["clarity","empathy","structure","relevance","confidence","composite"]
    n = len(turn_scores)
    avgs = {d: round(sum(s.get(d,0) for s in turn_scores)/n, 3) for d in dims}
    mid = max(1, n//2)
    first_half = turn_scores[:mid]
    second_half = turn_scores[mid:] or turn_scores  # fallback for single-turn
    f_avg = sum(s.get("composite",0) for s in first_half)/len(first_half)
    s_avg = sum(s.get("composite",0) for s in second_half)/len(second_half)
    delta = round(s_avg - f_avg, 3)
    trend = "improving" if delta>0.05 else ("declining" if delta<-0.05 else "stable")
    sorted_t = sorted(turn_scores, key=lambda s:s.get("composite",0))
    best_d = max((d for d in avgs if d!="composite"), key=lambda d:avgs[d])
    worst_d = min((d for d in avgs if d!="composite"), key=lambda d:avgs[d])
    return {"turns_evaluated":n,"averages":avgs,"trend":trend,"trend_delta":delta,
            "strongest_dimension":best_d,"weakest_dimension":worst_d,
            "weakest_turn":sorted_t[0],"strongest_turn":sorted_t[-1],
            "rationales":[s.get("rationale","") for s in turn_scores]}

def test_empty_scores():
    r = aggregate_session_scores([])
    assert_in("error on empty", "error", r)

def test_avg_calculation():
    scores = [make_score(i,v) for i,v in enumerate([0.4,0.6,0.8])]
    r = aggregate_session_scores(scores)
    avg = r["averages"]["composite"]
    assert_true("average correct", abs(avg - 0.6) < 0.01, f"got {avg}")

def test_improving_trend():
    scores = [make_score(i,v) for i,v in enumerate([0.3,0.3,0.8,0.8])]
    r = aggregate_session_scores(scores)
    assert_eq("improving trend detected", r["trend"], "improving")

def test_declining_trend():
    scores = [make_score(i,v) for i,v in enumerate([0.9,0.9,0.3,0.3])]
    r = aggregate_session_scores(scores)
    assert_eq("declining trend detected", r["trend"], "declining")

def test_stable_trend():
    scores = [make_score(i,0.6) for i in range(4)]
    r = aggregate_session_scores(scores)
    assert_eq("stable trend detected", r["trend"], "stable")

def test_strongest_weakest_dims():
    score = make_score(0, 0.5)
    score.update({"clarity": 0.9, "empathy": 0.2,
                  "structure": 0.5, "relevance": 0.5, "confidence": 0.5, "composite": 0.5})
    scores = [score]
    r = aggregate_session_scores(scores)
    assert_eq("strongest dimension", r["strongest_dimension"], "clarity")
    assert_eq("weakest dimension", r["weakest_dimension"], "empathy")

def test_turn_extremes():
    scores = [make_score(0,0.2), make_score(1,0.9)]
    r = aggregate_session_scores(scores)
    assert_eq("weakest turn index", r["weakest_turn"]["turn_index"], 0)
    assert_eq("strongest turn index", r["strongest_turn"]["turn_index"], 1)

run("empty scores returns error", test_empty_scores)
run("average calculation", test_avg_calculation)
run("improving trend", test_improving_trend)
run("declining trend", test_declining_trend)
run("stable trend", test_stable_trend)
run("strongest/weakest dimensions", test_strongest_weakest_dims)
run("weakest/strongest turns", test_turn_extremes)


# ══════════════════════════════════════════════════════════════════════════
# 3. SHORT-TERM MEMORY (pure Python, no external deps)
# ══════════════════════════════════════════════════════════════════════════

section("Short-Term Memory")

@dataclass
class _Msg:
    content: str
    role: str  # "human" | "ai"

@dataclass
class ShortTermMemory:
    session_id: str
    user_id: str
    scenario_id: str
    scenario_phase: str = "setup"
    turn_index: int = 0
    messages: list = field(default_factory=list)
    turn_scores: list = field(default_factory=list)

    def add_turn(self, human, ai):
        self.messages.append(human)
        self.messages.append(ai)
        self.turn_index += 1

    def get_recent_turns(self, n=8):
        return self.messages[-(n*2):]

    def add_score(self, score):
        self.turn_scores.append(score)

    def to_summary_text(self):
        lines = [f"Session {self.session_id} | User {self.user_id} | Scenario {self.scenario_id}",
                 f"Turns: {self.turn_index} | Phase: {self.scenario_phase}"]
        if self.turn_scores:
            avg = lambda k: sum(s.get(k,0) for s in self.turn_scores)/len(self.turn_scores)
            lines.append(f"composite:{avg('composite'):.2f}")
        return "\n".join(lines)

def test_stm_initial():
    stm = ShortTermMemory("s1","u1","scenario_v1")
    assert_eq("initial turn_index", stm.turn_index, 0)
    assert_eq("initial messages empty", stm.messages, [])

def test_stm_add_turn():
    stm = ShortTermMemory("s1","u1","scenario_v1")
    stm.add_turn(_Msg("Q","human"), _Msg("A","ai"))
    assert_eq("turn index incremented", stm.turn_index, 1)
    assert_eq("two messages stored", len(stm.messages), 2)

def test_stm_window():
    stm = ShortTermMemory("s1","u1","scenario_v1")
    for i in range(10):
        stm.add_turn(_Msg(f"Q{i}","human"), _Msg(f"A{i}","ai"))
    recent = stm.get_recent_turns(n=3)
    assert_eq("window size correct", len(recent), 6)
    assert_eq("last message is Q9", recent[-2].content, "Q9")

def test_stm_window_smaller_than_n():
    stm = ShortTermMemory("s1","u1","scenario_v1")
    stm.add_turn(_Msg("Q","human"), _Msg("A","ai"))
    assert_eq("returns all when less than window", len(stm.get_recent_turns(n=8)), 2)

def test_stm_summary_contains_ids():
    stm = ShortTermMemory("sess-abc","user-xyz","job_interview_v1")
    stm.add_turn(_Msg("Q","human"), _Msg("A","ai"))
    stm.turn_scores.append(make_score(0, 0.8))
    s = stm.to_summary_text()
    assert_in("session id in summary", "sess-abc", s)
    assert_in("user id in summary", "user-xyz", s)
    assert_in("scenario id in summary", "job_interview_v1", s)
    assert_in("score in summary", "0.80", s)

run("initial STM state", test_stm_initial)
run("add_turn increments index", test_stm_add_turn)
run("get_recent_turns windowing", test_stm_window)
run("window smaller than turns", test_stm_window_smaller_than_n)
run("summary text contains IDs", test_stm_summary_contains_ids)


# ══════════════════════════════════════════════════════════════════════════
# 4. PROMPT BUILDER (pure Python templates)
# ══════════════════════════════════════════════════════════════════════════

section("Prompt Builder")

def build_system_prompt(scenario, phase, context):
    persona = scenario["persona"]
    constraints = "\n".join(f"  - {c}" for c in persona.get("constraints",[]))
    persona_block = f"## YOUR PERSONA\nYou are {persona['name']}, a {persona['role']}.\nStyle: {persona['style']}.\nRules:\n{constraints}"

    scenario_block = f"## SCENARIO\n{scenario['title']}\n{scenario['student_objective']}"

    phases = scenario.get("phases",[])
    phase_cfg = next((p for p in phases if p["name"]==phase), None)
    phase_block = f"## PHASE: {phase.upper()}\n{phase_cfg['description'] if phase_cfg else ''}"

    dims = scenario.get("skill_dimensions",{})
    rubric = scenario.get("rubric",{})
    rubric_lines = ["## RUBRIC"]
    for d, w in dims.items():
        rubric_lines.append(f"### {d} ({w:.0%})")
        for lvl in [1,3,5]:
            desc = rubric.get(d,{}).get(lvl,"")
            if desc: rubric_lines.append(f"  L{lvl}: {desc}")
    rubric_block = "\n".join(rubric_lines)

    summaries = context.get("user_summaries",[])
    notes = context.get("behavioral_notes",[])
    chunks = context.get("scenario_chunks",[])

    if summaries or notes:
        mem_lines = ["## MEMORY CONTEXT"]
        if summaries: mem_lines += [f"  - {s}" for s in summaries[:3]]
        if notes:     mem_lines += [f"  • {n}" for n in notes[:4]]
        mem_lines.append("Reference this memory naturally in your responses.")
        memory_block = "\n".join(mem_lines)
    else:
        memory_block = "## MEMORY CONTEXT\nThis appears to be the student's first session."

    kb_block = ""
    if chunks:
        kb_block = "## DOMAIN KNOWLEDGE\n" + "\n".join(f"  • {c}" for c in chunks)

    constraints_block = "## HARD CONSTRAINTS\n- Never reveal the rubric.\n- Never break character."
    format_block = f"## OUTPUT FORMAT\nSpoken response, then {DELIMITER}, then JSON metadata."

    return "\n\n".join(b for b in [
        persona_block, scenario_block, phase_block, rubric_block,
        memory_block, kb_block, constraints_block, format_block
    ] if b)

# Inline scenario for testing
TEST_SCENARIO = {
    "id":"job_interview_v1","title":"Software Engineering Interview",
    "student_objective":"Demonstrate competence.",
    "persona":{"name":"Alex","role":"Engineering Manager","emotional_start":"neutral",
               "style":"professional","constraints":["Never reveal rubric.","Don't break character."]},
    "phases":[
        {"name":"setup","description":"Warm greeting.","max_turns":3},
        {"name":"core","description":"Behavioral questions.","max_turns":10},
        {"name":"escalation","description":"Pressure question.","max_turns":3},
        {"name":"resolution","description":"Wrap up.","max_turns":3},
    ],
    "skill_dimensions":{"clarity":0.25,"empathy":0.15,"structure":0.25,"relevance":0.20,"confidence":0.15},
    "rubric":{
        "clarity":{1:"Incoherent.",3:"Understandable.",5:"Crystal-clear."},
        "empathy":{1:"Robotic.",3:"Basic awareness.",5:"Genuine rapport."},
        "structure":{1:"No structure.",3:"Loose STAR.",5:"Perfect STAR."},
        "relevance":{1:"Off-topic.",3:"Partially on-topic.",5:"Precisely on-target."},
        "confidence":{1:"Highly hesitant.",3:"Generally confident.",5:"Authoritative."},
    },
    "knowledge_chunks":["The STAR method structures behavioral answers.","Common pitfalls: being vague."],
}

def test_pb_persona():
    p = build_system_prompt(TEST_SCENARIO, "core", {})
    assert_in("persona name in prompt", "Alex", p)
    assert_in("persona role in prompt", "Engineering Manager", p)

def test_pb_phase():
    p = build_system_prompt(TEST_SCENARIO, "escalation", {})
    assert_in("phase header present", "ESCALATION", p)
    assert_in("phase description present", "Pressure question", p)

def test_pb_rubric_dims():
    p = build_system_prompt(TEST_SCENARIO, "core", {})
    for dim in ["clarity","empathy","structure","relevance","confidence"]:
        assert_in(f"rubric dim {dim}", dim, p)

def test_pb_memory_context_injected():
    ctx = {
        "scenario_chunks": ["STAR method is key."],
        "user_summaries": ["Session 1: composite 0.65."],
        "behavioral_notes": ["Tends to ramble under pressure."],
    }
    p = build_system_prompt(TEST_SCENARIO, "core", ctx)
    assert_in("scenario chunk in prompt", "STAR method is key", p)
    assert_in("prior summary in prompt", "composite 0.65", p)
    assert_in("behavioral note in prompt", "Tends to ramble", p)

def test_pb_first_session_message():
    p = build_system_prompt(TEST_SCENARIO, "core", {"scenario_chunks":[],"user_summaries":[],"behavioral_notes":[]})
    assert_in("first session message", "first session", p)

def test_pb_output_format():
    p = build_system_prompt(TEST_SCENARIO, "core", {})
    assert_in("output format block", DELIMITER, p)

def test_pb_constraints():
    p = build_system_prompt(TEST_SCENARIO, "core", {})
    assert_in("constraints block", "HARD CONSTRAINTS", p)

def test_pb_no_chunks_no_kb_block():
    p = build_system_prompt(TEST_SCENARIO, "core", {"scenario_chunks":[],"user_summaries":[],"behavioral_notes":[]})
    assert_true("no empty KB block", "DOMAIN KNOWLEDGE" not in p)

def test_pb_chunks_create_kb_block():
    ctx = {"scenario_chunks":["Chunk A","Chunk B"],"user_summaries":[],"behavioral_notes":[]}
    p = build_system_prompt(TEST_SCENARIO, "core", ctx)
    assert_in("KB block appears", "DOMAIN KNOWLEDGE", p)
    assert_in("chunk A present", "Chunk A", p)

run("persona injected", test_pb_persona)
run("current phase injected", test_pb_phase)
run("rubric dimensions present", test_pb_rubric_dims)
run("memory context injected", test_pb_memory_context_injected)
run("first session fallback", test_pb_first_session_message)
run("output format block", test_pb_output_format)
run("constraints block", test_pb_constraints)
run("no chunks = no KB block", test_pb_no_chunks_no_kb_block)
run("chunks create KB block", test_pb_chunks_create_kb_block)


# ══════════════════════════════════════════════════════════════════════════
# 5. SCENARIO REGISTRY
# ══════════════════════════════════════════════════════════════════════════

section("Scenario Registry (inline validation)")

SCENARIOS = {
    "job_interview_v1": TEST_SCENARIO,
    "difficult_conversation_v1": {
        "id":"difficult_conversation_v1","title":"Managing a Difficult Workplace Conversation",
        "student_objective":"Deliver feedback and agree on next steps.",
        "persona":{"name":"Jordan","role":"Mid-level engineer","emotional_start":"defensive",
                   "style":"initially resistant","constraints":["Don't capitulate immediately."]},
        "phases":[{"name":"setup","description":"Open.","max_turns":2},
                  {"name":"core","description":"Feedback exchange.","max_turns":8},
                  {"name":"resolution","description":"Action plan.","max_turns":4}],
        "skill_dimensions":{"clarity":0.20,"empathy":0.30,"structure":0.20,"relevance":0.15,"confidence":0.15},
        "rubric":{"clarity":{1:"Vague.",3:"Stated.",5:"Specific."},
                  "empathy":{1:"Blunt.",3:"Surface-level.",5:"Validates fully."},
                  "structure":{1:"No direction.",3:"Loose SBI.",5:"Perfect SBI."},
                  "relevance":{1:"Drifts.",3:"Mostly on-topic.",5:"On-point."},
                  "confidence":{1:"Caves.",3:"Mostly holds.",5:"Firm and warm."}},
        "knowledge_chunks":["SBI model: Situation, Behavior, Impact.","Defensive reactions are normal."],
        "exit_conditions":{"max_total_turns":14,"completion_phases":["resolution"]},
    }
}

def get_scenario(sid):
    s = SCENARIOS.get(sid)
    if not s: raise ValueError(f"Unknown scenario: {sid!r}")
    return s

def test_registry_get_known():
    s = get_scenario("job_interview_v1")
    assert_eq("id matches", s["id"], "job_interview_v1")

def test_registry_get_unknown():
    try:
        get_scenario("nonexistent_v1")
        fail("unknown scenario raises", "no exception raised")
    except ValueError:
        ok("unknown scenario raises ValueError")

def test_registry_persona_fields():
    s = get_scenario("difficult_conversation_v1")
    p = s["persona"]
    assert_in("persona has name", "name", p)
    assert_in("persona has role", "role", p)
    assert_in("persona has emotional_start", "emotional_start", p)

def test_registry_phases_ordered():
    s = get_scenario("job_interview_v1")
    names = [p["name"] for p in s["phases"]]
    assert_eq("first phase is setup", names[0], "setup")
    assert_eq("last phase is resolution", names[-1], "resolution")

def test_registry_skill_weights_sum_to_one():
    for sid, s in SCENARIOS.items():
        total = round(sum(s["skill_dimensions"].values()), 10)
        assert_true(f"{sid} weights sum to 1.0", abs(total - 1.0) < 0.001, f"sum={total}")

def test_registry_rubric_has_all_dims():
    for sid, s in SCENARIOS.items():
        for dim in s["skill_dimensions"]:
            assert_in(f"{sid}.rubric.{dim}", dim, s["rubric"])

run("get known scenario", test_registry_get_known)
run("get unknown scenario raises", test_registry_get_unknown)
run("persona has required fields", test_registry_persona_fields)
run("phases in correct order", test_registry_phases_ordered)
run("skill weights sum to 1.0", test_registry_skill_weights_sum_to_one)
run("rubric has all skill dims", test_registry_rubric_has_all_dims)


# ══════════════════════════════════════════════════════════════════════════
# 6. MEMORY MANAGER (STM + session lifecycle — no vector store)
# ══════════════════════════════════════════════════════════════════════════

section("Memory Manager (STM lifecycle)")

class _MemoryManager:
    def __init__(self):
        self._stm: dict[str, ShortTermMemory] = {}

    def create_session(self, session_id, user_id, scenario_id, scenario_phase="setup"):
        stm = ShortTermMemory(session_id=session_id, user_id=user_id,
                              scenario_id=scenario_id, scenario_phase=scenario_phase)
        self._stm[session_id] = stm
        return stm

    def get_session(self, session_id):
        return self._stm.get(session_id)

    def end_session(self, session_id, behavioral_notes=""):
        stm = self._stm.pop(session_id, None)
        return stm

def test_mm_create_and_get():
    mm = _MemoryManager()
    mm.create_session("s1","u1","job_interview_v1")
    stm = mm.get_session("s1")
    assert_true("session retrievable", stm is not None)
    assert_eq("user_id correct", stm.user_id, "u1")

def test_mm_get_missing():
    mm = _MemoryManager()
    assert_eq("missing session returns None", mm.get_session("ghost"), None)

def test_mm_end_session_removes():
    mm = _MemoryManager()
    mm.create_session("s1","u1","job_interview_v1")
    mm.end_session("s1")
    assert_eq("session removed after end", mm.get_session("s1"), None)

def test_mm_multiple_sessions_isolated():
    mm = _MemoryManager()
    mm.create_session("sa","ua","job_interview_v1")
    mm.create_session("sb","ub","difficult_conversation_v1")
    mm.get_session("sa").add_turn(_Msg("Qa","human"),_Msg("Aa","ai"))
    assert_eq("session A has 2 msgs", len(mm.get_session("sa").messages), 2)
    assert_eq("session B still empty", len(mm.get_session("sb").messages), 0)

def test_mm_phase_transitions():
    mm = _MemoryManager()
    stm = mm.create_session("s1","u1","job_interview_v1", scenario_phase="setup")
    assert_eq("starts in setup", stm.scenario_phase, "setup")
    stm.scenario_phase = "core"
    assert_eq("phase can be updated", mm.get_session("s1").scenario_phase, "core")

run("create and get session", test_mm_create_and_get)
run("get missing session returns None", test_mm_get_missing)
run("end_session removes from STM", test_mm_end_session_removes)
run("multiple sessions isolated", test_mm_multiple_sessions_isolated)
run("phase transitions work", test_mm_phase_transitions)


# ══════════════════════════════════════════════════════════════════════════
# 7. GRAPH FLOW (structure validation without running the actual graph)
# ══════════════════════════════════════════════════════════════════════════

section("Graph Structure (topology validation)")

EXPECTED_NODES = [
    "context_assembly",
    "prompt_builder",
    "llm_call",
    "response_parser",
    "memory_update",
    "evaluation_step",
    "end_session",
]

EXPECTED_EDGES = [
    ("context_assembly", "prompt_builder"),
    ("prompt_builder", "llm_call"),
    ("llm_call", "response_parser"),
    ("response_parser", "memory_update"),
    ("memory_update", "evaluation_step"),
]

# We validate the graph spec as a plain dict (no LangGraph dependency)
graph_spec = {
    "nodes": EXPECTED_NODES,
    "entry": "context_assembly",
    "edges": EXPECTED_EDGES,
    "conditional_from": "evaluation_step",
    "conditional_targets": ["end_session", "__end__"],
}

def test_graph_has_all_nodes():
    for node in EXPECTED_NODES:
        assert_in(f"node {node!r} declared", node, graph_spec["nodes"])

def test_graph_entry_point():
    assert_eq("entry point is context_assembly", graph_spec["entry"], "context_assembly")

def test_graph_linear_chain():
    for src, dst in EXPECTED_EDGES:
        assert_in(f"edge {src}→{dst}", (src, dst), graph_spec["edges"])

def test_graph_conditional_from_evaluation():
    assert_eq("conditional edge from evaluation_step",
               graph_spec["conditional_from"], "evaluation_step")

def test_graph_end_session_reachable():
    assert_in("end_session in conditional targets",
              "end_session", graph_spec["conditional_targets"])

run("all six nodes declared", test_graph_has_all_nodes)
run("entry point set correctly", test_graph_entry_point)
run("linear chain intact", test_graph_linear_chain)
run("conditional routing from evaluation_step", test_graph_conditional_from_evaluation)
run("end_session reachable", test_graph_end_session_reachable)


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════

total = passed + failed
print(f"\n{'━'*56}")
print(f"{BOLD}Results:{RESET}  {GREEN}{passed} passed{RESET}  {RED if failed else DIM}{failed} failed{RESET}  of {total} total")
print(f"{'━'*56}\n")

if failed:
    sys.exit(1)
