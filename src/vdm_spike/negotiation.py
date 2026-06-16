from __future__ import annotations

from enum import Enum

from vdm_spike.adapters.base import Capabilities


class DriftPlan(str, Enum):
    FULL = "FULL"
    QUERY = "QUERY"
    MINIMAL = "MINIMAL"


# Which detector families each plan is allowed to run.
PLAN_FAMILIES: dict[DriftPlan, set[str]] = {
    DriftPlan.FULL: {"distribution", "retrieval", "ops"},
    DriftPlan.QUERY: {"retrieval", "ops"},
    DriftPlan.MINIMAL: {"ops"},
}


def select_drift_plan(c: Capabilities) -> DriftPlan:
    """Pick the richest viable plan. Distribution drift requires an UNBIASED sample,
    not merely the ability to return vectors (PRD v0.2 correction)."""
    if c.returns_vectors and c.unbiased_sample:
        return DriftPlan.FULL
    if c.live_query:
        return DriftPlan.QUERY
    return DriftPlan.MINIMAL
