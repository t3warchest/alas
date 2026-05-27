"""
alas/agent_service/scenarios/loader.py

ScenarioLoader: reads scenario JSON files from disk, validates them through
the Pydantic model layer, and populates the in-memory registry.

Design:
  - JSON files are the source of truth. The Python registry.py is replaced
    by this loader + the definitions/ directory.
  - Validation errors are surfaced at load time with clear field-level messages.
  - The loader is callable at startup and can also be called to hot-reload
    scenarios during development (e.g. a PUT /scenarios/{id}/reload endpoint).
  - Falls back gracefully: a bad scenario file logs an error and is skipped;
    the rest of the registry remains available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from agent_service.scenarios.models import Scenario
from agent_service.utils.logging import get_logger

log = get_logger(__name__)

# Default directory: alongside this file
_DEFAULT_DEFINITIONS_DIR = Path(__file__).parent / "definitions"


class ScenarioLoader:
    """
    Loads, validates, and caches Scenario objects from a definitions directory.

    Usage:
        loader = ScenarioLoader()
        loader.load_all()                     # populate registry at startup
        scenario = loader.get("job_interview_v1")
        all_scenarios = loader.list_all()
    """

    def __init__(self, definitions_dir: Path | str | None = None):
        self._dir = Path(definitions_dir or _DEFAULT_DEFINITIONS_DIR)
        self._registry: dict[str, Scenario] = {}

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def load_all(self) -> dict[str, Scenario]:
        """
        Scan the definitions directory, load and validate every .json file.
        Returns the populated registry dict.
        Skips (and logs) any files that fail validation.
        """
        if not self._dir.exists():
            log.warning("scenario_dir_missing", path=str(self._dir))
            return {}

        loaded = 0
        failed = 0

        for path in sorted(self._dir.glob("*.json")):
            scenario = self._load_file(path)
            if scenario is not None:
                self._registry[scenario.id] = scenario
                loaded += 1
            else:
                failed += 1

        log.info(
            "scenarios_loaded",
            directory=str(self._dir),
            loaded=loaded,
            failed=failed,
            ids=list(self._registry.keys()),
        )
        return dict(self._registry)

    def load_file(self, path: Path | str) -> Scenario:
        """
        Load and validate a single JSON file. Raises on failure (unlike load_all).
        Use this for explicit single-file loading or testing.
        """
        path = Path(path)
        scenario = self._load_file(path, raise_on_error=True)
        assert scenario is not None  # guaranteed when raise_on_error=True
        self._registry[scenario.id] = scenario
        return scenario

    def get(self, scenario_id: str) -> Scenario:
        """Return a validated Scenario by ID. Raises KeyError if not found."""
        if scenario_id not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Scenario '{scenario_id}' not found. "
                f"Available: {available}"
            )
        return self._registry[scenario_id]

    def list_all(self) -> list[Scenario]:
        return list(self._registry.values())

    def list_summaries(self) -> list[dict]:
        """Lightweight list for the GET /scenarios endpoint."""
        return [
            {
                "id":          s.id,
                "title":       s.title,
                "description": s.description,
                "category":    s.category.value,
                "difficulty":  s.difficulty.value,
                "tags":        list(s.tags),
                "phase_count": len(s.phases),
            }
            for s in self._registry.values()
        ]

    def reload(self, scenario_id: str) -> Scenario:
        """
        Re-read a single scenario file from disk (hot-reload).
        Raises if the file can't be found or fails validation.
        """
        path = self._dir / f"{scenario_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No definition file for '{scenario_id}': {path}")
        return self.load_file(path)

    def validate_dict(self, data: dict) -> Scenario:
        """
        Validate a raw dict as a Scenario (useful for the scenario authoring API).
        Raises pydantic.ValidationError on failure with full field-level detail.
        """
        return Scenario.model_validate(data)

    def __contains__(self, scenario_id: str) -> bool:
        return scenario_id in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _load_file(
        self,
        path: Path,
        raise_on_error: bool = False,
    ) -> Scenario | None:
        """
        Load, parse, and validate one JSON file.
        Returns None on failure (unless raise_on_error=True).
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as e:
            msg = f"Cannot read scenario file {path}: {e}"
            log.error("scenario_file_read_error", path=str(path), error=str(e))
            if raise_on_error:
                raise RuntimeError(msg) from e
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in {path}: {e}"
            log.error("scenario_json_parse_error", path=str(path), error=str(e))
            if raise_on_error:
                raise ValueError(msg) from e
            return None

        try:
            scenario = Scenario.model_validate(data)
        except ValidationError as e:
            # Surface every field-level error clearly
            errors = e.errors()
            log.error(
                "scenario_validation_error",
                path=str(path),
                error_count=len(errors),
                errors=[
                    {
                        "field": " → ".join(str(loc) for loc in err["loc"]),
                        "message": err["msg"],
                        "type": err["type"],
                    }
                    for err in errors
                ],
            )
            if raise_on_error:
                raise
            return None

        log.debug("scenario_loaded", id=scenario.id, title=scenario.title, path=str(path))
        return scenario

    def _iter_json_paths(self) -> Iterator[Path]:
        return self._dir.glob("*.json")


# ---------------------------------------------------------------------------
# Module-level singleton used across the service
# ---------------------------------------------------------------------------

scenario_loader = ScenarioLoader()
