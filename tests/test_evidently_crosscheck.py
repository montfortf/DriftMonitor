"""Confirm our MMD verdict agrees with Evidently's embedding-drift detector
on the same data, building trust that our implementation is correct.

Evidently version: 0.7.21
API used: evidently.legacy.report.Report + EmbeddingsDriftMetric
  - The legacy API (evidently.legacy.*) is still present in 0.7.x and exposes
    a clean ``drift_detected`` bool in the result dict via ``report.as_dict()``.
  - The newer ``evidently.future.*`` API uses SingleValue results (drift score
    only) without a direct ``drift_detected`` flag; the legacy path is cleaner.
"""
import numpy as np
import pandas as pd

from evidently.legacy.metrics import EmbeddingsDriftMetric
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

from vdm_spike.detectors.distribution import mmd_rbf

_EMB_COLS = [f"e{i}" for i in range(16)]


def _evidently_drift_detected(baseline: np.ndarray, current: np.ndarray) -> bool:
    """Return Evidently's embedding drift verdict for the given arrays.

    Uses the legacy ``Report`` + ``EmbeddingsDriftMetric`` path which is
    available and stable in Evidently 0.7.x.  The default method is a
    classifier-based approach (``model``) with threshold 0.55.
    """
    ref_df = pd.DataFrame(baseline.astype(np.float64), columns=_EMB_COLS)
    cur_df = pd.DataFrame(current.astype(np.float64), columns=_EMB_COLS)

    cm = ColumnMapping()
    cm.embeddings = {"emb": _EMB_COLS}

    report = Report(metrics=[EmbeddingsDriftMetric("emb")])
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=cm)

    result = report.as_dict()["metrics"][0]["result"]
    return bool(result["drift_detected"])


def test_mmd_agrees_with_evidently_on_shift():
    """Both detectors should fire on clearly shifted embeddings (loc=0.6)."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 16)).astype(np.float32)
    b = rng.normal(loc=0.6, size=(300, 16)).astype(np.float32)

    assert mmd_rbf(a, b, n_perm=200, seed=0).fired is True
    assert _evidently_drift_detected(a, b) is True


def test_mmd_agrees_with_evidently_on_no_shift():
    """Both detectors should stay silent on same-distribution embeddings."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 16)).astype(np.float32)
    b = rng.normal(size=(300, 16)).astype(np.float32)

    assert mmd_rbf(a, b, n_perm=200, seed=0).fired is False
    assert _evidently_drift_detected(a, b) is False
