from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from vdm_spike.core import DetectorResult
from vdm_spike.features import median_bandwidth, rbf_kernel


def centroid_distance(baseline: np.ndarray, current: np.ndarray,
                      threshold: float = 0.1) -> DetectorResult:
    """Euclidean distance between mean vectors, normalised by sqrt(dim).

    Normalisation makes the statistic comparable across embedding dimensions
    and avoids the degenerate cosine-distance behaviour for near-zero means.
    Coarse first signal (informational).
    """
    mb = baseline.mean(axis=0)
    mc = current.mean(axis=0)
    dist = float(np.linalg.norm(mb - mc) / np.sqrt(baseline.shape[1]))
    return DetectorResult(
        name="centroid",
        statistic=dist,
        p_value=None,
        fired=dist > threshold,
        detail={"threshold": threshold},
    )


def _mmd2_from_kernel(k: np.ndarray, n: int) -> float:
    kxx = k[:n, :n]
    kyy = k[n:, n:]
    kxy = k[:n, n:]
    m = k.shape[0] - n
    # unbiased estimator
    sxx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    syy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    sxy = kxy.sum() / (n * m)
    return float(sxx + syy - 2 * sxy)


def mmd_rbf(baseline: np.ndarray, current: np.ndarray,
            n_perm: int = 200, alpha: float = 0.05, seed: int = 0) -> DetectorResult:
    """MMD^2 with RBF kernel; bandwidth frozen via median heuristic on baseline.

    Significance via a label-permutation test on the precomputed kernel matrix.
    """
    gamma = median_bandwidth(baseline)  # frozen on baseline
    n = baseline.shape[0]
    z = np.vstack([baseline, current])
    k = rbf_kernel(z, z, gamma=gamma)
    observed = _mmd2_from_kernel(k, n)

    rng = np.random.default_rng(seed)
    total = z.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(total)
        kp = k[np.ix_(perm, perm)]
        if _mmd2_from_kernel(kp, n) >= observed:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return DetectorResult(
        name="mmd",
        statistic=observed,
        p_value=p_value,
        fired=p_value < alpha,
        detail={"gamma": gamma, "n_perm": n_perm},
    )


def _cv_auc(features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    clf = LogisticRegression(max_iter=1000)
    proba = cross_val_predict(
        clf, features, labels, cv=5, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(labels, proba))


def classifier_drift(baseline: np.ndarray, current: np.ndarray, pca: PCA,
                     n_perm: int = 30, alpha: float = 0.05,
                     seed: int = 0) -> DetectorResult:
    """Domain-classifier AUC on PCA-reduced features with a permutation null."""
    fb = pca.transform(baseline)
    fc = pca.transform(current)
    features = np.vstack([fb, fc])
    labels = np.concatenate([np.zeros(len(fb)), np.ones(len(fc))]).astype(int)

    observed = _cv_auc(features, labels, seed)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        permuted = rng.permutation(labels)
        if _cv_auc(features, permuted, seed) >= observed:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return DetectorResult(
        name="classifier",
        statistic=observed,
        p_value=p_value,
        fired=(p_value < alpha) and (observed > 0.55),
        detail={"n_perm": n_perm, "n_components": int(pca.n_components_)},
    )


def norm_ks(baseline: np.ndarray, current: np.ndarray,
            alpha: float = 0.05) -> DetectorResult:
    """KS test on L2 norm distributions — strong signal for a silent model swap."""
    nb = np.linalg.norm(baseline, axis=1)
    nc = np.linalg.norm(current, axis=1)
    stat, p = ks_2samp(nb, nc)
    return DetectorResult(
        name="norm_ks", statistic=float(stat), p_value=float(p),
        fired=bool(p < alpha), detail={},
    )


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def perdim_psi(baseline: np.ndarray, current: np.ndarray, pca: PCA,
               psi_threshold: float = 0.2,
               share_threshold: float = 0.1) -> DetectorResult:
    """Per-dimension PSI on the frozen PCA basis; fire if share-drifted exceeds threshold."""
    fb = pca.transform(baseline)
    fc = pca.transform(current)
    psis = np.array([_psi(fb[:, j], fc[:, j]) for j in range(fb.shape[1])])
    share_drifted = float(np.mean(psis > psi_threshold))
    return DetectorResult(
        name="psi", statistic=share_drifted, p_value=None,
        fired=share_drifted > share_threshold,
        detail={"max_psi": float(psis.max()), "psi_threshold": psi_threshold},
    )
