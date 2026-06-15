import numpy as np

from vdm_spike.detectors.distribution import centroid_distance, classifier_drift, mmd_rbf, norm_ks, perdim_psi
from vdm_spike.features import fit_pca


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


def test_classifier_quiet_on_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 30)).astype(np.float32)
    b = rng.normal(size=(400, 30)).astype(np.float32)
    pca = fit_pca(a, var=0.95)
    res = classifier_drift(a, b, pca, n_perm=30, seed=0)
    assert res.fired is False


def test_classifier_fires_on_separable_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 30)).astype(np.float32)
    b = rng.normal(loc=0.7, size=(400, 30)).astype(np.float32)
    pca = fit_pca(a, var=0.95)
    res = classifier_drift(a, b, pca, n_perm=30, seed=0)
    assert res.fired is True
    assert res.statistic > 0.55  # AUC well above chance


def test_norm_ks_fires_when_norms_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 20)).astype(np.float32)
    b = (rng.normal(size=(400, 20)) * 3.0).astype(np.float32)  # magnitude change
    res = norm_ks(a, b)
    assert res.fired is True


def test_norm_ks_quiet_on_same_norms():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 20)).astype(np.float32)
    b = rng.normal(size=(400, 20)).astype(np.float32)
    res = norm_ks(a, b)
    assert res.fired is False


def test_perdim_psi_fires_on_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 20)).astype(np.float32)
    pca = fit_pca(a, var=0.95)
    b = rng.normal(loc=1.0, size=(400, 20)).astype(np.float32)
    res = perdim_psi(a, b, pca)
    assert res.fired is True


def test_perdim_psi_quiet_on_same():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 20)).astype(np.float32)
    pca = fit_pca(a, var=0.95)
    b = rng.normal(size=(400, 20)).astype(np.float32)
    res = perdim_psi(a, b, pca)
    assert res.fired is False
