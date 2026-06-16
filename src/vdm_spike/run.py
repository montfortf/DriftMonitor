from __future__ import annotations

import numpy as np

from vdm_spike.core import DetectorResult
from vdm_spike.detectors import distribution as dist
from vdm_spike.detectors import retrieval as ret
from vdm_spike.features import fit_pca
from vdm_spike.negotiation import PLAN_FAMILIES, DriftPlan, select_drift_plan
from vdm_spike.ops import invalid_vectors
from vdm_spike.sampling import query_seeded, vector_sample


def evaluate_scenario(sc, adapter, sample_k: int = 1000, query_k: int = 10
                      ) -> tuple[DriftPlan, dict[str, DetectorResult], list[str]]:
    """Load both snapshots into the adapter, negotiate the plan, run allowed families."""
    if hasattr(adapter, "ensure_schema"):
        adapter.ensure_schema()
        adapter.conn.execute("DELETE FROM items")
        adapter.conn.commit()
    adapter.load(sc.baseline, namespace="baseline")
    adapter.load(sc.current, namespace="current")

    plan = select_drift_plan(adapter.capabilities())
    families = PLAN_FAMILIES[plan]
    results: dict[str, DetectorResult] = {}
    notes: list[str] = []

    if "distribution" in families:
        base = vector_sample(adapter, "baseline", sample_k)
        curr = vector_sample(adapter, "current", sample_k)
        pca = fit_pca(base.vectors, var=0.95)
        results["centroid"] = dist.centroid_distance(base.vectors, curr.vectors)
        results["mmd"] = dist.mmd_rbf(base.vectors, curr.vectors)
        results["classifier"] = dist.classifier_drift(base.vectors, curr.vectors, pca)
        results["norm_ks"] = dist.norm_ks(base.vectors, curr.vectors)
        results["psi"] = dist.perdim_psi(base.vectors, curr.vectors, pca)
    else:
        notes.append(f"distribution drift UNAVAILABLE under {plan.value} plan "
                     "(store returns no unbiased vector sample)")

    if "retrieval" in families:
        qb = query_seeded(adapter, "baseline", sc.query_vectors, query_k)
        qc = query_seeded(adapter, "current", sc.query_vectors, query_k)
        results["retrieval_rbo"] = ret.retrieval_overlap(qb.hit_ids, qc.hit_ids)
        results["retrieval_score_ks"] = ret.score_ks(np.array(qb.scores), np.array(qc.scores))
    else:
        notes.append(f"retrieval drift UNAVAILABLE under {plan.value} plan (no live query)")

    results["ops_invalid"] = invalid_vectors(sc.current.vectors)
    return plan, results, notes


def gate_ok(sc, results: dict[str, DetectorResult]) -> bool:
    """A scenario passes if every GATED detector that is AVAILABLE matches its expectation.
    Detectors unavailable under the active plan are not asserted (honest degradation)."""
    for name, expected in sc.expectation.fires.items():
        if name in results and bool(results[name].fired) != expected:
            return False
    return True


def main() -> int:
    import os

    import psycopg
    from pgvector.psycopg import register_vector

    from vdm_spike.adapters.fakes import FakeMinimalAdapter, FakeQueryOnlyAdapter
    from vdm_spike.adapters.pgvector import PgVectorAdapter
    from vdm_spike.adapters.qdrant import QdrantAdapter
    from vdm_spike.scenarios import SCENARIO_NAMES, build_scenario

    dsn = os.environ.get("VDM_DSN", "postgresql://vdm:vdm@localhost:5432/vdm")
    overall_ok = True
    with psycopg.connect(dsn) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        adapters = {
            "pgvector": lambda: PgVectorAdapter(conn, dim=384),
            "qdrant": lambda: QdrantAdapter(dim=384),
            "fake-query-only": lambda: FakeQueryOnlyAdapter(dim=384),
            "fake-minimal": lambda: FakeMinimalAdapter(dim=384),
        }
        for aname, make in adapters.items():
            for sname in SCENARIO_NAMES:
                sc = build_scenario(sname, n=600, seed=0)
                # sample_k above any namespace size → analyze full population,
                # making the headline gate deterministic (see test_gate.py).
                plan, results, notes = evaluate_scenario(sc, make(), sample_k=100_000)
                ok = gate_ok(sc, results)
                overall_ok = overall_ok and ok
                print(f"[{aname:16s}] {sname:14s} plan={plan.value:7s} "
                      f"{'PASS' if ok else 'FAIL'}")
                for note in notes:
                    print(f"    fidelity: {note}")
    print("GATE:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
