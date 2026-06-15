from __future__ import annotations

import numpy as np

from vdm_spike.core import DetectorResult


def invalid_vectors(vectors: np.ndarray) -> DetectorResult:
    """Count NaN/Inf/zero-norm rows — broken embedding writes."""
    nan_rows = np.isnan(vectors).any(axis=1)
    inf_rows = np.isinf(vectors).any(axis=1)
    finite = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    zero_rows = (np.linalg.norm(finite, axis=1) == 0) & ~nan_rows & ~inf_rows
    n_nan = int(nan_rows.sum())
    n_inf = int(inf_rows.sum())
    n_zero = int(zero_rows.sum())
    total = n_nan + n_inf + n_zero
    return DetectorResult(
        name="ops_invalid", statistic=total, p_value=None, fired=total > 0,
        detail={"nan": n_nan, "inf": n_inf, "zero_norm": n_zero},
    )
