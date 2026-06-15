from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from vdm_spike.core import DetectorResult


def rbo(list1: list[str], list2: list[str], p: float = 0.9) -> float:
    """Rank-Biased Overlap (extrapolated) — weights agreement at the top of the ranking.

    Returns a value in [0, 1] where 1.0 means the lists are identical.
    Uses the RBO_ext formula from Webber et al. (2010) which extrapolates
    agreement beyond the observed depth, ensuring identical lists score 1.0.
    """
    k = max(len(list1), len(list2))
    if k == 0:
        return 1.0
    s1: set[str] = set()
    s2: set[str] = set()
    sum_term = 0.0
    for d in range(k):
        if d < len(list1):
            s1.add(list1[d])
        if d < len(list2):
            s2.add(list2[d])
        overlap = len(s1 & s2)
        sum_term += (overlap / (d + 1)) * (p ** d)
    # RBO_ext: truncated sum + tail extrapolation assuming same agreement at depth k
    overlap_at_k = len(s1 & s2)
    rbo_ext = (1 - p) / p * sum_term + overlap_at_k / k * (p ** k)
    # Clamp to [0, 1] to guard against floating-point overshoot
    return min(1.0, max(0.0, rbo_ext))


def retrieval_overlap(
    baseline_hits: list[list[str]],
    current_hits: list[list[str]],
    p: float = 0.9,
    rbo_threshold: float = 0.8,
) -> DetectorResult:
    """Mean RBO across a query set; fire when top-k results drift apart."""
    scores = [rbo(b, c, p=p) for b, c in zip(baseline_hits, current_hits)]
    mean_rbo = float(np.mean(scores)) if scores else 1.0
    return DetectorResult(
        name="retrieval_rbo",
        statistic=mean_rbo,
        p_value=None,
        fired=bool(mean_rbo < rbo_threshold),
        detail={"p": p, "n_queries": len(scores)},
    )


def score_ks(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    alpha: float = 0.05,
) -> DetectorResult:
    """KS test on similarity-score distributions — works even when vectors are unreadable."""
    stat, p = ks_2samp(np.asarray(baseline_scores), np.asarray(current_scores))
    return DetectorResult(
        name="retrieval_score_ks",
        statistic=float(stat),
        p_value=float(p),
        fired=bool(p < alpha),
        detail={},
    )
