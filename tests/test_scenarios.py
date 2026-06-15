from vdm_spike.scenarios import SCENARIO_NAMES, build_scenario


def test_all_scenarios_build_with_expected_shapes():
    for name in SCENARIO_NAMES:
        sc = build_scenario(name, n=120, seed=0)
        assert sc.name == name
        assert sc.baseline.dim == 384
        assert sc.current.dim == 384
        assert sc.query_vectors.shape[1] == 384
        assert isinstance(sc.expectation.fires, dict)
        assert len(sc.expectation.fires) >= 1


def test_model_swap_keeps_same_ids():
    sc = build_scenario("model-swap", n=80, seed=0)
    assert set(sc.baseline.ids) == set(sc.current.ids)


def test_broken_writes_injects_zero_vectors():
    import numpy as np
    sc = build_scenario("broken-writes", n=80, seed=0)
    zero = (np.linalg.norm(sc.current.vectors, axis=1) == 0).sum()
    assert zero > 0
