from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA


def fit_pca(baseline: np.ndarray, var: float = 0.95) -> PCA:
    """Fit a PCA basis on the baseline retaining `var` fraction of variance.

    The basis is fit ONCE on the baseline and reused to transform current data,
    so projections are axis-comparable (frozen basis).
    """
    max_comp = min(baseline.shape)
    pca = PCA(n_components=var, svd_solver="full", random_state=0)
    pca.fit(baseline)
    # n_components float can occasionally select all dims; clamp is implicit via fit.
    if pca.n_components_ > max_comp:  # defensive, should not happen
        pca = PCA(n_components=max_comp, svd_solver="full", random_state=0).fit(baseline)
    return pca


def median_bandwidth(x: np.ndarray) -> float:
    """RBF gamma via the median heuristic, computed on `x` and meant to be frozen."""
    dists = pdist(x, metric="euclidean")
    median = float(np.median(dists))
    sigma = median if median > 0 else 1.0
    return 1.0 / (2.0 * sigma * sigma)


def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    sq = cdist(a, b, metric="sqeuclidean")
    return np.exp(-gamma * sq)
