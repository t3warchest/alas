"""
alas/agent_service/scenarios/models.py

Pydantic v2 models that mirror the JSON schema exactly.
Every scenario loaded from disk is validated against these models before
it can be used — invalid scenarios are rejected at startup, not mid-session.

Design rules:
  - Models are the single source of type truth for scenario data.
  - The JSON schema (schema.json) and these models must stay in sync.
  - All downstream code (loader, prompt builder, branching engine) imports
    from here — never raw dicts.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ScenarioCategory(str, Enum):
    INTERVIEW    = "interview"
    FEEDBACK     = "feedback"
    NEGOTIATION  = "negotiation"
    ONBOARDING   = "onboarding"
    CUSTOM       = "custom"


class Difficulty(str, Enum):
    BEGINNER     = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"


class EmotionalStart(str, Enum):
    NEUTRAL   = "neutral"
    DEFENSIVE = "defensive"
    CURIOUS   = "curious"
    WARM      = "warm"
    HOSTILE   = "hostile"
    ANXIOUS   = "anxious"


class TransitionType(str, Enum):
    ALWAYS_AFTER_MAX  = "always_after_max"
    SCORE_THRESHOLD   = "score_threshold"
    KEYWORD_DETECTED  = "keyword_detected"
    BRANCH_TABLE      = "branch_table"


# ---------------------------------------------------------------------------
# Transition models
# ---------------------------------------------------------------------------

class BranchCondition(BaseModel):
    """One row in a branch_table transition."""
    model_config = ConfigDict(frozen=True)

    condition:    str = Field(..., description="Evaluable condition string")
    target_phase: str = Field(..., description="Phase to transition to if condition is met")
    priority:     int = Field(default=0, description="Higher priority evaluated first")


class PhaseTransition(BaseModel):
    """
    Describes when and how to leave a phase.

    Transition types:
      always_after_max  — move to target_phase after max_turns is exhausted
      score_threshold   — move when score_dimension's rolling avg exceeds score_threshold
      keyword_detected  — move when any keyword in keywords appears in student utterance
      branch_table      — evaluate ordered conditions; first match wins
    """
    model_config = ConfigDict(frozen=True)

    type:             TransitionType
    target_phase:     str | None = None
    score_dimension:  str | None = None
    score_threshold:  float | None = Field(default=None, ge=0.0, le=1.0)
    keywords:         list[str] = Field(default_factory=list)
    branches:         list[BranchCondition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_transition_fields(self) -> "PhaseTransition":
        t = self.type
        if t == TransitionType.SCORE_THRESHOLD:
            if not self.score_dimension:
                raise ValueError("score_threshold transition requires score_dimension")
            if self.score_threshold is None:
                raise ValueError("score_threshold transition requires score_threshold value")
        if t == TransitionType.KEYWORD_DETECTED:
            if not self.keywords:
                raise ValueError("keyword_detected transition requires at least one keyword")
        if t == TransitionType.BRANCH_TABLE:
            if not self.branches:
                raise ValueError("branch_table transition requires at least one branch")
        return self


# ---------------------------------------------------------------------------
# Phase model
# ---------------------------------------------------------------------------

class Phase(BaseModel):
    model_config = ConfigDict(frozen=True)

    name:        str = Field(..., min_length=1)
    description: str = Field(..., min_length=10)
    max_turns:   int = Field(..., ge=1, le=30)
    avatar_goal: str | None = None
    transition:  PhaseTransition


# ---------------------------------------------------------------------------
# Persona model
# ---------------------------------------------------------------------------

class Persona(BaseModel):
    model_config = ConfigDict(frozen=True)

    name:            str
    role:            str
    emotional_start: EmotionalStart
    style:           str
    backstory:       str | None = None
    constraints:     list[str] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Rubric models
# ---------------------------------------------------------------------------

class RubricLevel(BaseModel):
    """Level descriptors for one skill dimension. Keys are '1'-'5' (str in JSON)."""
    model_config = ConfigDict(frozen=True, extra="allow")

    # Required levels
    l1: str = Field(..., alias="1")
    l3: str = Field(..., alias="3")
    l5: str = Field(..., alias="5")
    # Optional intermediate levels
    l2: str | None = Field(default=None, alias="2")
    l4: str | None = Field(default=None, alias="4")

    def get_descriptor(self, level: int) -> str:
        """Return the descriptor for the closest defined level."""
        mapping = {
            1: self.l1,
            2: self.l2 or self.l1,
            3: self.l3,
            4: self.l4 or self.l3,
            5: self.l5,
        }
        return mapping.get(level, self.l3)

    def as_prompt_text(self, dimension_name: str) -> str:
        lines = [f"**{dimension_name.title()}**"]
        for lvl in [1, 3, 5]:
            lines.append(f"  Level {lvl}: {self.get_descriptor(lvl)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exit conditions
# ---------------------------------------------------------------------------

class ExitConditions(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_total_turns:   int = Field(..., ge=2, le=60)
    completion_phases: list[str] = Field(..., min_length=1)
    early_exit_score:  float | None = Field(default=None, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Root scenario model
# ---------------------------------------------------------------------------

class Scenario(BaseModel):
    """
    The complete, validated scenario definition.
    Loaded from JSON → validated → used everywhere in the system.
    Raw dicts never leave the loader.
    """
    model_config = ConfigDict(frozen=True)

    id:                str = Field(..., pattern=r"^[a-z0-9_]+$")
    version:           str = Field(..., pattern=r"^\d+\.\d+$")
    title:             str = Field(..., min_length=5)
    description:       str = Field(..., min_length=10)
    category:          ScenarioCategory
    difficulty:        Difficulty = Difficulty.INTERMEDIATE
    tags:              list[str] = Field(default_factory=list)

    persona:           Persona
    student_objective: str = Field(..., min_length=20)

    phases:            list[Phase] = Field(..., min_length=1)

    skill_dimensions:  dict[str, float] = Field(..., min_length=1)
    rubric:            dict[str, RubricLevel]

    knowledge_chunks:  list[str] = Field(..., min_length=1)
    exit_conditions:   ExitConditions

    # ----- Cross-field validators -------------------------------------------

    @field_validator("skill_dimensions")
    @classmethod
    def weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"skill_dimensions weights must sum to 1.0, got {total:.3f}"
            )
        for dim, w in v.items():
            if not (0 < w <= 1):
                raise ValueError(f"Weight for '{dim}' must be in (0, 1], got {w}")
        return v

    @model_validator(mode="after")
    def rubric_covers_all_dimensions(self) -> "Scenario":
        dims = set(self.skill_dimensions.keys())
        rubric_dims = set(self.rubric.keys())
        missing = dims - rubric_dims
        if missing:
            raise ValueError(
                f"Rubric missing entries for skill dimensions: {missing}"
            )
        return self

    @model_validator(mode="after")
    def phase_names_unique(self) -> "Scenario":
        names = [p.name for p in self.phases]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Phase names must be unique; duplicates: {set(dupes)}")
        return self

    @model_validator(mode="after")
    def transition_targets_exist(self) -> "Scenario":
        """All transition targets must refer to a real phase name."""
        valid_phases = {p.name for p in self.phases}
        for phase in self.phases:
            t = phase.transition
            if t.target_phase and t.target_phase not in valid_phases:
                raise ValueError(
                    f"Phase '{phase.name}' transition targets unknown phase "
                    f"'{t.target_phase}'. Valid phases: {valid_phases}"
                )
            for branch in t.branches:
                if branch.target_phase not in valid_phases:
                    raise ValueError(
                        f"Phase '{phase.name}' branch targets unknown phase "
                        f"'{branch.target_phase}'. Valid phases: {valid_phases}"
                    )
        return self

    @model_validator(mode="after")
    def completion_phases_exist(self) -> "Scenario":
        valid = {p.name for p in self.phases}
        for cp in self.exit_conditions.completion_phases:
            if cp not in valid:
                raise ValueError(
                    f"exit_conditions.completion_phases contains unknown phase "
                    f"'{cp}'. Valid: {valid}"
                )
        return self

    # ----- Convenience helpers ----------------------------------------------

    def get_phase(self, name: str) -> Phase | None:
        return next((p for p in self.phases if p.name == name), None)

    def first_phase(self) -> Phase:
        return self.phases[0]

    def phase_index(self, name: str) -> int:
        for i, p in enumerate(self.phases):
            if p.name == name:
                return i
        raise ValueError(f"Phase '{name}' not found in scenario '{self.id}'")

    def next_phase(self, current_name: str) -> Phase | None:
        idx = self.phase_index(current_name)
        if idx + 1 < len(self.phases):
            return self.phases[idx + 1]
        return None

    def to_legacy_dict(self) -> dict[str, Any]:
        """
        Produce the flat dict format the existing prompt_builder expects.
        Bridges the new typed model with the existing graph nodes
        until prompt_builder is upgraded to accept Scenario directly.
        """
        rubric_dict: dict[str, dict[int, str]] = {}
        for dim, rl in self.rubric.items():
            rubric_dict[dim] = {1: rl.l1, 3: rl.l3, 5: rl.l5}

        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "persona": self.persona.model_dump(exclude_none=True),
            "student_objective": self.student_objective,
            "phases": [
                {
                    "name": p.name,
                    "description": p.description,
                    "max_turns": p.max_turns,
                    "transition_condition": p.transition.type.value,
                }
                for p in self.phases
            ],
            "skill_dimensions": dict(self.skill_dimensions),
            "rubric": rubric_dict,
            "knowledge_chunks": list(self.knowledge_chunks),
            "exit_conditions": {
                "max_total_turns": self.exit_conditions.max_total_turns,
                "completion_phases": list(self.exit_conditions.completion_phases),
            },
        }
