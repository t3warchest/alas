"""
alas/agent_service/scenarios/registry.py

Thin facade over ScenarioLoader.
All external code that previously imported from this module continues to work.
Internally, scenarios now come from JSON files via the typed loader.
"""

from __future__ import annotations
from typing import Any

from agent_service.scenarios.loader import scenario_loader
from agent_service.scenarios.models import Scenario


def get_scenario(scenario_id: str) -> dict[str, Any]:
    """
    Return a scenario as a legacy dict (for backward-compat with graph nodes
    that still expect raw dicts). Prefer get_scenario_typed() for new code.
    """
    return scenario_loader.get(scenario_id).to_legacy_dict()


def get_scenario_typed(scenario_id: str) -> Scenario:
    """Return the fully-typed, validated Scenario object."""
    return scenario_loader.get(scenario_id)


def list_scenarios() -> list[dict[str, str]]:
    """Return lightweight summaries for API listing."""
    return scenario_loader.list_summaries()


def ingest_all_scenarios(memory_manager) -> None:
    """
    Called at startup: load all JSON scenarios from disk, validate them,
    then embed knowledge chunks into ChromaDB.
    Idempotent — upsert is safe to call multiple times.
    """
    scenario_loader.load_all()
    for scenario in scenario_loader.list_all():
        chunks = list(scenario.knowledge_chunks)
        if chunks:
            memory_manager.semantic.ingest_scenario(scenario.id, chunks)
