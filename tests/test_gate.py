import pytest

from vdm_spike.run import evaluate_scenario
from vdm_spike.scenarios import SCENARIO_NAMES, build_scenario
from vdm_spike.store import PgVectorStore


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_scenario_meets_expectation(conn, name):
    store = PgVectorStore(conn, dim=384)
    sc = build_scenario(name, n=600, seed=0)
    _, gate_ok = evaluate_scenario(sc, store, sample_k=600)
    assert gate_ok, f"scenario {name} did not meet its detector expectations"
