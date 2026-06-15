import numpy as np

from vdm_spike.detectors.distribution import centroid_distance, mmd_rbf


def test_centroid_quiet_on_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 32)).astype(np.float32)
    b = rng.normal(size=(400, 32)).astype(np.float32)
    res = centroid_distance(a, b)
    assert res.fired is False


def test_centroid_fires_on_shifted_mean():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 32)).astype(np.float32)
    b = rng.normal(loc=0.5, size=(400, 32)).astype(np.float32)
    res = centroid_distance(a, b)
    assert res.fired is True
    assert res.statistic > 0


def test_mmd_quiet_on_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 16)).astype(np.float32)
    b = rng.normal(size=(300, 16)).astype(np.float32)
    res = mmd_rbf(a, b, n_perm=200, seed=0)
    assert res.p_value is not None and res.p_value > 0.05
    assert res.fired is False


def test_mmd_fires_on_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 16)).astype(np.float32)
    b = rng.normal(loc=0.6, size=(300, 16)).astype(np.float32)
    res = mmd_rbf(a, b, n_perm=200, seed=0)
    assert res.p_value is not None and res.p_value < 0.05
    assert res.fired is True
