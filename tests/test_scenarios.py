import numpy as np

from vdm_spike.scenarios import SCENARIO_NAMES, build_scenario


def test_all_scenarios_build_with_expected_shapes():
    for name in SCENARIO_NAMES:
        sc = build_scenario(name, n=120, seed=0)
        assert sc.baseline.dim == 384 and sc.current.dim == 384
        assert sc.query_vectors.shape[1] == 384
        assert len(sc.expectation.fires) >= 1


def test_null_control_is_identical_corpus():
    sc = build_scenario("null-control", n=80, seed=0)
    assert sc.baseline.ids == sc.current.ids
    assert np.allclose(sc.baseline.vectors, sc.current.vectors)


def test_benign_growth_retains_baseline_docs():
    sc = build_scenario("benign-growth", n=80, seed=0)
    assert set(sc.baseline.ids).issubset(set(sc.current.ids))
    assert sc.current.n > sc.baseline.n


def test_broken_writes_injects_zero_vectors():
    sc = build_scenario("broken-writes", n=80, seed=0)
    assert (np.linalg.norm(sc.current.vectors, axis=1) == 0).sum() > 0
