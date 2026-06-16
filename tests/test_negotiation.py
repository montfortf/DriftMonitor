from vdm_spike.adapters.base import Capabilities
from vdm_spike.negotiation import PLAN_FAMILIES, DriftPlan, select_drift_plan


def _caps(returns, unbiased, live):
    return Capabilities(returns_vectors=returns, unbiased_sample=unbiased, live_query=live)


def test_full_plan_when_unbiased_vectors_available():
    assert select_drift_plan(_caps(True, True, True)) is DriftPlan.FULL


def test_query_plan_when_vectors_returnable_but_not_unbiased():
    # returns_vectors True but unbiased_sample False -> still QUERY (the v0.2 correction)
    assert select_drift_plan(_caps(True, False, True)) is DriftPlan.QUERY


def test_query_plan_when_no_vectors_but_live_query():
    assert select_drift_plan(_caps(False, False, True)) is DriftPlan.QUERY


def test_minimal_plan_when_no_query():
    assert select_drift_plan(_caps(False, False, False)) is DriftPlan.MINIMAL


def test_plan_families_mapping():
    assert PLAN_FAMILIES[DriftPlan.FULL] == {"distribution", "retrieval", "ops"}
    assert PLAN_FAMILIES[DriftPlan.QUERY] == {"retrieval", "ops"}
    assert PLAN_FAMILIES[DriftPlan.MINIMAL] == {"ops"}
