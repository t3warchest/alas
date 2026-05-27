"""
alas/agent_service/scenarios/branching.py

PhaseEngine: evaluates transition conditions and decides when and where
to move between scenario phases.

This is the orchestration brain of the scenario system. It translates the
declarative JSON transition specs into live decisions based on session state.

Transition types (from models.py):

  always_after_max   — deterministic: move after turn count exhausted
  score_threshold    — data-driven: move when rolling avg of a score dim crosses threshold
  keyword_detected   — signal-based: move when student utterance contains a keyword
  branch_table       — conditional: evaluate ordered conditions, first match wins

Branch condition grammar (evaluated by _evaluate_condition):
  "always"                                  — unconditional
  "max_turns_reached"                       — turn count >= phase max_turns
  "<dim>_above_<threshold>"                 — rolling avg of dim > threshold
  "<dim>_below_<threshold>"                 — rolling avg of dim < threshold
  "<dim1>_above_<t1>_and_<dim2>_above_<t2>" — compound AND
  "<dim1>_above_<t1>_and_<dim2>_below_<t2>" — compound AND with mixed direction

Architecture note:
  PhaseEngine is stateless — it takes SessionPhaseContext and returns a decision.
  The decision is then applied by MemoryUpdate in the graph.
  This keeps branching logic fully unit-testable without mocking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent_service.scenarios.models import (
    BranchCondition,
    Phase,
    PhaseTransition,
    Scenario,
    TransitionType,
)
from agent_service.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Context object passed into the engine each turn
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseContext:
    """
    Everything the branching engine needs to make a decision.
    Populated from STM + graph state before PhaseEngine.evaluate() is called.
    """
    current_phase_name: str
    turns_in_phase: int          # how many turns have elapsed in the current phase
    total_turns: int             # total turns across the whole session
    rolling_scores: dict[str, float]   # dimension → rolling average so far this session
    last_student_utterance: str  # the most recent student message (for keyword detection)


@dataclass(frozen=True)
class TransitionDecision:
    """
    The branching engine's output for one turn.
    """
    should_transition: bool
    target_phase: str | None    # None if should_transition is False
    reason: str                 # human-readable explanation (for logs and hidden_notes)

    @classmethod
    def stay(cls, reason: str = "staying in phase") -> "TransitionDecision":
        return cls(should_transition=False, target_phase=None, reason=reason)

    @classmethod
    def move(cls, target: str, reason: str) -> "TransitionDecision":
        return cls(should_transition=True, target_phase=target, reason=reason)


# ---------------------------------------------------------------------------
# PhaseEngine
# ---------------------------------------------------------------------------

class PhaseEngine:
    """
    Evaluates phase transition conditions for a given scenario and context.

    Usage:
        engine = PhaseEngine(scenario)
        decision = engine.evaluate(phase_context)
        if decision.should_transition:
            new_phase = decision.target_phase
    """

    def __init__(self, scenario: Scenario):
        self._scenario = scenario
        self._phase_map: dict[str, Phase] = {p.name: p for p in scenario.phases}

    def evaluate(self, ctx: PhaseContext) -> TransitionDecision:
        """
        Check whether the current phase should transition, and if so, to where.
        Called every turn after EvaluationStep updates scores.
        """
        phase = self._phase_map.get(ctx.current_phase_name)
        if phase is None:
            log.warning(
                "phase_not_found",
                phase=ctx.current_phase_name,
                scenario_id=self._scenario.id,
            )
            return TransitionDecision.stay("unknown phase — cannot evaluate transition")

        # Check global session exit first (max total turns)
        exit_conds = self._scenario.exit_conditions
        if ctx.total_turns >= exit_conds.max_total_turns:
            last_phase = self._scenario.phases[-1]
            return TransitionDecision.move(
                target=last_phase.name,
                reason=f"max_total_turns ({exit_conds.max_total_turns}) reached — jumping to final phase",
            )

        decision = self._evaluate_transition(phase.transition, ctx, phase)

        if decision.should_transition:
            log.info(
                "phase_transition",
                scenario_id=self._scenario.id,
                from_phase=ctx.current_phase_name,
                to_phase=decision.target_phase,
                reason=decision.reason,
                turns_in_phase=ctx.turns_in_phase,
                total_turns=ctx.total_turns,
            )

        return decision

    def is_session_complete(self, ctx: PhaseContext) -> bool:
        """True when we are in a completion phase and at or past max_turns."""
        completion_phases = self._scenario.exit_conditions.completion_phases
        if ctx.current_phase_name not in completion_phases:
            return False
        phase = self._phase_map.get(ctx.current_phase_name)
        return phase is not None and ctx.turns_in_phase >= phase.max_turns

    def should_end_on_score(self, ctx: PhaseContext) -> bool:
        """True if early-exit score threshold is configured and met."""
        early = self._scenario.exit_conditions.early_exit_score
        if early is None:
            return False
        composite = ctx.rolling_scores.get("composite", 0.0)
        return composite >= early

    # -----------------------------------------------------------------------
    # Transition type dispatchers
    # -----------------------------------------------------------------------

    def _evaluate_transition(
        self,
        transition: PhaseTransition,
        ctx: PhaseContext,
        phase: Phase,
    ) -> TransitionDecision:
        dispatch = {
            TransitionType.ALWAYS_AFTER_MAX: self._always_after_max,
            TransitionType.SCORE_THRESHOLD:  self._score_threshold,
            TransitionType.KEYWORD_DETECTED: self._keyword_detected,
            TransitionType.BRANCH_TABLE:     self._branch_table,
        }
        handler = dispatch.get(transition.type)
        if handler is None:
            return TransitionDecision.stay(f"unknown transition type: {transition.type}")
        return handler(transition, ctx, phase)

    def _always_after_max(
        self,
        transition: PhaseTransition,
        ctx: PhaseContext,
        phase: Phase,
    ) -> TransitionDecision:
        """Move to target_phase as soon as turns_in_phase >= phase.max_turns."""
        if ctx.turns_in_phase >= phase.max_turns:
            target = transition.target_phase or self._next_phase_name(phase.name)
            if target:
                return TransitionDecision.move(
                    target=target,
                    reason=f"max_turns ({phase.max_turns}) reached in phase '{phase.name}'",
                )
        return TransitionDecision.stay(
            f"turns_in_phase={ctx.turns_in_phase}/{phase.max_turns}"
        )

    def _score_threshold(
        self,
        transition: PhaseTransition,
        ctx: PhaseContext,
        phase: Phase,
    ) -> TransitionDecision:
        """
        Move when the rolling average of score_dimension exceeds score_threshold.
        Falls back to always_after_max if turns run out.
        """
        dim = transition.score_dimension or "composite"
        threshold = transition.score_threshold or 0.7
        current_score = ctx.rolling_scores.get(dim, 0.0)

        if current_score >= threshold:
            target = transition.target_phase or self._next_phase_name(phase.name)
            if target:
                return TransitionDecision.move(
                    target=target,
                    reason=f"{dim}={current_score:.2f} ≥ threshold {threshold:.2f}",
                )

        # Fallback: move after max_turns even if threshold not met
        if ctx.turns_in_phase >= phase.max_turns:
            target = transition.target_phase or self._next_phase_name(phase.name)
            if target:
                return TransitionDecision.move(
                    target=target,
                    reason=(
                        f"max_turns fallback ({phase.max_turns}); "
                        f"{dim}={current_score:.2f} never reached {threshold:.2f}"
                    ),
                )

        return TransitionDecision.stay(
            f"{dim}={current_score:.2f} < {threshold:.2f}, "
            f"turns={ctx.turns_in_phase}/{phase.max_turns}"
        )

    def _keyword_detected(
        self,
        transition: PhaseTransition,
        ctx: PhaseContext,
        phase: Phase,
    ) -> TransitionDecision:
        """Move when any keyword appears in the student's last utterance."""
        utterance_lower = ctx.last_student_utterance.lower()
        for kw in transition.keywords:
            if kw.lower() in utterance_lower:
                target = transition.target_phase or self._next_phase_name(phase.name)
                if target:
                    return TransitionDecision.move(
                        target=target,
                        reason=f"keyword '{kw}' detected in student utterance",
                    )

        # Fallback: max_turns
        if ctx.turns_in_phase >= phase.max_turns:
            target = transition.target_phase or self._next_phase_name(phase.name)
            if target:
                return TransitionDecision.move(
                    target=target,
                    reason=f"max_turns fallback ({phase.max_turns}); no keyword detected",
                )

        return TransitionDecision.stay(
            f"no keyword match in utterance; turns={ctx.turns_in_phase}/{phase.max_turns}"
        )

    def _branch_table(
        self,
        transition: PhaseTransition,
        ctx: PhaseContext,
        phase: Phase,
    ) -> TransitionDecision:
        """
        Evaluate an ordered list of conditions, return on first match.
        Branches are sorted by priority (descending) then evaluated in order.
        """
        sorted_branches = sorted(
            transition.branches,
            key=lambda b: b.priority,
            reverse=True,
        )

        for branch in sorted_branches:
            if self._evaluate_condition(branch.condition, ctx, phase):
                return TransitionDecision.move(
                    target=branch.target_phase,
                    reason=f"branch condition matched: '{branch.condition}'",
                )

        return TransitionDecision.stay(
            f"no branch condition matched; turns={ctx.turns_in_phase}/{phase.max_turns}"
        )

    # -----------------------------------------------------------------------
    # Condition evaluator (branch_table grammar)
    # -----------------------------------------------------------------------

    def _evaluate_condition(
        self,
        condition: str,
        ctx: PhaseContext,
        phase: Phase,
    ) -> bool:
        """
        Evaluate a condition string against the current PhaseContext.

        Supported grammar:
          "always"
          "max_turns_reached"
          "<dim>_above_<float>"
          "<dim>_below_<float>"
          "<dim1>_above_<t1>_and_<dim2>_above_<t2>"
          "<dim1>_above_<t1>_and_<dim2>_below_<t2>"
          "<dim1>_below_<t1>_and_<dim2>_above_<t2>"
        """
        cond = condition.strip().lower()

        if cond == "always":
            return True

        if cond == "max_turns_reached":
            return ctx.turns_in_phase >= phase.max_turns

        # Split compound conditions on "_and_"
        parts = [p.strip() for p in cond.split("_and_")]
        return all(self._evaluate_simple(part, ctx) for part in parts)

    def _evaluate_simple(self, expr: str, ctx: PhaseContext) -> bool:
        """
        Evaluate a single simple condition:
          "<dim>_above_<threshold>"  or  "<dim>_below_<threshold>"
        """
        # Pattern: word(s) then _above_ or _below_ then float
        match = re.match(
            r"^(?P<dim>[a-z_]+)_(?P<op>above|below)_(?P<val>[\d.]+)$",
            expr,
        )
        if not match:
            log.warning("unrecognised_condition_fragment", expr=expr)
            return False

        dim = match.group("dim")
        op  = match.group("op")
        threshold = float(match.group("val"))
        score = ctx.rolling_scores.get(dim, 0.0)

        if op == "above":
            return score > threshold
        else:  # below
            return score < threshold

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _next_phase_name(self, current_name: str) -> str | None:
        """Return the name of the phase that follows current_name in the scenario."""
        phases = self._scenario.phases
        for i, p in enumerate(phases):
            if p.name == current_name and i + 1 < len(phases):
                return phases[i + 1].name
        return None
