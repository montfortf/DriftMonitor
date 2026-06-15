from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from vdm_spike.core import DetectorResult  # noqa: E402


def format_verdict_table(
    scenario: str,
    results: dict[str, DetectorResult],
    expectation: dict[str, bool],
) -> tuple[str, bool]:
    """Render one scenario's gated detectors; return (table_text, all_passed)."""
    lines = [f"Scenario: {scenario}"]
    all_ok = True
    for name, expected in expectation.items():
        res = results.get(name)
        actual = bool(res.fired) if res else False
        ok = actual == expected
        all_ok = all_ok and ok
        mark = "PASS" if ok else "FAIL"
        stat = f"{res.statistic:.4f}" if res else "n/a"
        p = f"{res.p_value:.4f}" if (res and res.p_value is not None) else "-"
        lines.append(
            f"  [{mark}] {name:18s} expected={expected!s:5s} "
            f"fired={actual!s:5s} stat={stat} p={p}"
        )
    return "\n".join(lines), all_ok


def save_overlay_plot(
    baseline: np.ndarray,
    current: np.ndarray,
    pca: PCA,
    path: str,
) -> None:
    """2D PCA overlay: fit-on-baseline, transform current through the frozen basis."""
    pb = pca.transform(baseline)[:, :2]
    pc = pca.transform(current)[:, :2]
    plt.figure(figsize=(6, 6))
    plt.scatter(pb[:, 0], pb[:, 1], s=8, alpha=0.4, label="baseline")
    plt.scatter(pc[:, 0], pc[:, 1], s=8, alpha=0.4, label="current")
    plt.legend()
    plt.title("Embedding-space overlay (frozen PCA basis)")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
