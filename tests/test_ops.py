import numpy as np

from vdm_spike.ops import invalid_vectors


def test_invalid_quiet_on_clean_vectors():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 16)).astype(np.float32)
    res = invalid_vectors(v)
    assert res.fired is False
    assert res.statistic == 0


def test_invalid_detects_nan_inf_and_zero():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 16)).astype(np.float32)
    v[0] = np.nan
    v[1] = np.inf
    v[2] = 0.0
    res = invalid_vectors(v)
    assert res.fired is True
    assert res.statistic == 3
    assert res.detail["nan"] == 1
    assert res.detail["inf"] == 1
    assert res.detail["zero_norm"] == 1
