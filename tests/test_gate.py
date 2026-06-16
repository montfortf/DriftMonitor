import pytest

from vdm_spike.adapters.fakes import FakeMinimalAdapter, FakeQueryOnlyAdapter
from vdm_spike.adapters.pgvector import PgVectorAdapter
from vdm_spike.adapters.qdrant import QdrantAdapter
from vdm_spike.negotiation import DriftPlan, select_drift_plan
from vdm_spike.run import evaluate_scenario, gate_ok
from vdm_spike.scenarios import SCENARIO_NAMES, build_scenario

# Analyze the FULL loaded population so the gate is DETERMINISTIC. Both adapters'
# sample() use random subsampling (pgvector ORDER BY random(); qdrant unseeded RNG);
# a partial subsample makes borderline scenarios flaky. Bounded-sampling power is a
# separate concern, covered by power.py (MDE) — not what this gate tests. _FULL is
# larger than any namespace built here (topic-shift current = 2*n).
_FULL = 100_000


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_full_adapter_pgvector_meets_expectation(conn, name):
    adapter = PgVectorAdapter(conn, dim=384)
    sc = build_scenario(name, n=600, seed=0)
    _, results, _ = evaluate_scenario(sc, adapter, sample_k=_FULL)
    assert gate_ok(sc, results), f"pgvector/{name} failed its expectation"


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_full_adapter_qdrant_meets_expectation(name):
    adapter = QdrantAdapter(dim=384)
    sc = build_scenario(name, n=600, seed=0)
    _, results, _ = evaluate_scenario(sc, adapter, sample_k=_FULL)
    assert gate_ok(sc, results), f"qdrant/{name} failed its expectation"


def test_negotiator_covers_all_three_plans(conn):
    assert select_drift_plan(PgVectorAdapter(conn, dim=384).capabilities()) is DriftPlan.FULL
    assert select_drift_plan(QdrantAdapter(dim=384).capabilities()) is DriftPlan.FULL
    assert select_drift_plan(FakeQueryOnlyAdapter(dim=384).capabilities()) is DriftPlan.QUERY
    assert select_drift_plan(FakeMinimalAdapter(dim=384).capabilities()) is DriftPlan.MINIMAL
