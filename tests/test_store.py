import numpy as np

from vdm_spike.adapters.pgvector import PgVectorAdapter
from vdm_spike.core import Snapshot
from vdm_spike.negotiation import DriftPlan, select_drift_plan


def test_pgvector_extension_available(conn):
    row = conn.execute(
        "SELECT extname FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    assert row is not None
    assert row[0] == "vector"


def _snapshot(seed: int, n: int) -> Snapshot:
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, 384)).astype(np.float32)
    ids = [f"x{i}" for i in range(n)]
    return Snapshot(ids=ids, vectors=vecs)


def test_load_count_sample_roundtrip(conn):
    store = PgVectorAdapter(conn, dim=384)
    store.ensure_schema()
    snap = _snapshot(seed=0, n=200)
    store.load(snap, namespace="baseline")
    assert store.count("baseline") == 200
    sampled = store.sample("baseline", k=50)
    assert sampled.n == 50
    assert sampled.dim == 384
    assert set(sampled.ids).issubset(set(snap.ids))


def test_query_returns_ranked_hits(conn):
    store = PgVectorAdapter(conn, dim=384)
    store.ensure_schema()
    snap = _snapshot(seed=1, n=100)
    store.load(snap, namespace="current")
    hits = store.query(snap.vectors[0], namespace="current", k=5)
    assert len(hits) == 5
    ranks = [h.rank for h in hits]
    assert ranks == [0, 1, 2, 3, 4]
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)  # cosine similarity descending
    assert hits[0].id == "x0"  # a vector is its own nearest neighbor



def test_pgvector_capabilities_select_full(conn):
    store = PgVectorAdapter(conn, dim=384)
    caps = store.capabilities()
    assert caps.returns_vectors and caps.unbiased_sample and caps.live_query
    assert select_drift_plan(caps) is DriftPlan.FULL


def test_pgvector_fetch_by_ids_roundtrip(conn):
    store = PgVectorAdapter(conn, dim=384)
    store.ensure_schema()
    rng = np.random.default_rng(0)
    snap = Snapshot(ids=[f"x{i}" for i in range(20)],
                    vectors=rng.normal(size=(20, 384)).astype(np.float32))
    store.load(snap, namespace="baseline")
    got = store.fetch_by_ids(["x0", "x5", "x9"], namespace="baseline")
    assert set(got.ids) == {"x0", "x5", "x9"}
    assert got.vectors.shape == (3, 384)
