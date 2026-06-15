import numpy as np

from vdm_spike.core import DetectorResult
from vdm_spike.features import fit_pca
from vdm_spike.report import format_verdict_table, save_overlay_plot


def test_format_verdict_table_marks_pass_and_fail():
    results = {"mmd": DetectorResult("mmd", 0.1, 0.01, True)}
    expectation = {"mmd": True}
    table, ok = format_verdict_table("topic-shift", results, expectation)
    assert "topic-shift" in table
    assert "mmd" in table
    assert ok is True

    bad_expectation = {"mmd": False}
    table2, ok2 = format_verdict_table("x", results, bad_expectation)
    assert ok2 is False


def test_save_overlay_plot_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 20)).astype(np.float32)
    b = rng.normal(loc=0.5, size=(200, 20)).astype(np.float32)
    pca = fit_pca(a, var=0.95)
    out = tmp_path / "overlay.png"
    save_overlay_plot(a, b, pca, str(out))
    assert out.exists() and out.stat().st_size > 0
