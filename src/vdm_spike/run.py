from __future__ import annotations

import numpy as np

from vdm_spike.core import DetectorResult
from vdm_spike.detectors import distribution as dist
from vdm_spike.detectors import retrieval as ret
from vdm_spike.features import fit_pca
from vdm_spike.ops import invalid_vectors
from vdm_spike.power import is_underpowered, min_detectable_effect
from vdm_spike.report import format_verdict_table, save_overlay_plot
from vdm_spike.scenarios import Scenario
from vdm_spike.store import PgVectorStore


def evaluate_scenario(sc: Scenario, store: PgVectorStore,
                      sample_k: int = 1000, query_k: int = 10,
                      plot_path: str | None = None) -> tuple[dict[str, DetectorResult], bool]:
    """Load both snapshots, read back over SQL, run detectors, return (results, gate_ok)."""
    store.ensure_schema()
    store.conn.execute("DELETE FROM items")
    store.conn.commit()
    store.load(sc.baseline, namespace="baseline")
    store.load(sc.current, namespace="current")

    base_s = store.sample("baseline", k=sample_k)
    curr_s = store.sample("current", k=sample_k)

    # Sanitize for distribution math (zero-norm rows survive; NaN/Inf cannot enter pgvector).
    pca = fit_pca(base_s.vectors, var=0.95)

    results: dict[str, DetectorResult] = {}
    results["centroid"] = dist.centroid_distance(base_s.vectors, curr_s.vectors)
    results["mmd"] = dist.mmd_rbf(base_s.vectors, curr_s.vectors)
    results["classifier"] = dist.classifier_drift(base_s.vectors, curr_s.vectors, pca)
    results["norm_ks"] = dist.norm_ks(base_s.vectors, curr_s.vectors)
    results["psi"] = dist.perdim_psi(base_s.vectors, curr_s.vectors, pca)

    base_hits, curr_hits, base_scores, curr_scores = [], [], [], []
    for qv in sc.query_vectors:
        bh = store.query(qv, namespace="baseline", k=query_k)
        ch = store.query(qv, namespace="current", k=query_k)
        base_hits.append([h.id for h in bh])
        curr_hits.append([h.id for h in ch])
        base_scores.extend([h.score for h in bh])
        curr_scores.extend([h.score for h in ch])
    results["retrieval_rbo"] = ret.retrieval_overlap(base_hits, curr_hits)
    results["retrieval_score_ks"] = ret.score_ks(
        np.array(base_scores), np.array(curr_scores)
    )

    results["ops_invalid"] = invalid_vectors(sc.current.vectors)

    if plot_path:
        save_overlay_plot(base_s.vectors, curr_s.vectors, pca, plot_path)

    _, gate_ok = format_verdict_table(sc.name, results, sc.expectation.fires)
    return results, gate_ok


def power_note(n: int, target_effect: float = 0.2) -> str:
    mde = min_detectable_effect(n)
    flag = "UNDER-POWERED" if is_underpowered(n, target_effect) else "ok"
    return f"n={n} mde={mde:.3f} (target {target_effect}) -> {flag}"


def main() -> int:
    import os

    import psycopg
    from pgvector.psycopg import register_vector

    from vdm_spike.scenarios import SCENARIO_NAMES, build_scenario

    dsn = os.environ.get("VDM_DSN", "postgresql://vdm:vdm@localhost:5432/vdm")
    overall_ok = True
    with psycopg.connect(dsn) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        store = PgVectorStore(conn, dim=384)
        for name in SCENARIO_NAMES:
            sc = build_scenario(name, n=2000, seed=0)
            results, ok = evaluate_scenario(
                sc, store, plot_path=f"overlay_{name}.png"
            )
            table, _ = format_verdict_table(name, results, sc.expectation.fires)
            print(table)
            # Print non-gated detectors as informational (visible, not asserted).
            informational = [k for k in results if k not in sc.expectation.fires]
            for k in informational:
                r = results[k]
                p = f"{r.p_value:.4f}" if r.p_value is not None else "-"
                print(f"  [info] {k:18s} fired={str(r.fired):5s} "
                      f"stat={r.statistic:.4f} p={p}")
            print("  " + power_note(min(sc.baseline.n, 1000)))
            print()
            overall_ok = overall_ok and ok
    print("GATE:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
