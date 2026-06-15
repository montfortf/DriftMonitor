import numpy as np

from vdm_spike.detectors.distribution import centroid_distance


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
