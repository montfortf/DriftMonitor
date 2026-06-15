import numpy as np

from vdm_spike.features import fit_pca, median_bandwidth, rbf_kernel


def test_fit_pca_reduces_dimensionality_and_transforms():
    rng = np.random.default_rng(0)
    baseline = rng.normal(size=(500, 50)).astype(np.float32)
    pca = fit_pca(baseline, var=0.95)
    assert 0 < pca.n_components_ <= 50
    projected = pca.transform(baseline)
    assert projected.shape == (500, pca.n_components_)


def test_median_bandwidth_is_positive_and_frozen_value():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 10)).astype(np.float32)
    gamma = median_bandwidth(x)
    assert gamma > 0
    # deterministic for same input
    assert gamma == median_bandwidth(x)


def test_rbf_kernel_diagonal_is_one():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 8)).astype(np.float32)
    k = rbf_kernel(x, x, gamma=0.5)
    assert k.shape == (30, 30)
    assert np.allclose(np.diag(k), 1.0)
