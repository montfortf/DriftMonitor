import numpy as np

from vdm_spike.detectors.retrieval import rbo, retrieval_overlap, score_ks


def test_rbo_identical_lists_is_one():
    assert rbo(["a", "b", "c"], ["a", "b", "c"], p=0.9) == 1.0


def test_rbo_disjoint_lists_is_low():
    assert rbo(["a", "b", "c"], ["x", "y", "z"], p=0.9) < 0.1


def test_retrieval_overlap_quiet_when_results_stable():
    baseline = [["a", "b", "c"], ["d", "e", "f"]]
    current = [["a", "b", "c"], ["d", "e", "f"]]
    res = retrieval_overlap(baseline, current)
    assert res.fired is False
    assert res.statistic > 0.95  # mean RBO


def test_retrieval_overlap_fires_when_results_diverge():
    baseline = [["a", "b", "c"], ["d", "e", "f"]]
    current = [["x", "y", "z"], ["p", "q", "r"]]
    res = retrieval_overlap(baseline, current)
    assert res.fired is True


def test_score_ks_fires_when_scores_drop():
    base_scores = np.full(100, 0.9)
    curr_scores = np.full(100, 0.5)
    res = score_ks(base_scores, curr_scores)
    assert res.fired is True
