import numpy as np

from vdm_spike.adapters.qdrant import QdrantAdapter
from vdm_spike.core import Snapshot
from vdm_spike.negotiation import DriftPlan, select_drift_plan


def _snap(seed, n):
    rng = np.random.default_rng(seed)
    return Snapshot(ids=[f"d{i}" for i in range(n)],
                    vectors=rng.normal(size=(n, 384)).astype(np.float32))


def test_qdrant_is_full_plan():
    a = QdrantAdapter(dim=384)
    assert select_drift_plan(a.capabilities()) is DriftPlan.FULL


def test_qdrant_load_count_sample_roundtrip():
    a = QdrantAdapter(dim=384)
    a.load(_snap(0, 100), namespace="baseline")
    assert a.count("baseline") == 100
    s = a.sample("baseline", k=30)
    assert s.n == 30 and s.dim == 384
    assert set(s.ids).issubset({f"d{i}" for i in range(100)})


def test_qdrant_query_self_is_top_hit():
    a = QdrantAdapter(dim=384)
    snap = _snap(1, 50)
    a.load(snap, namespace="current")
    hits = a.query(snap.vectors[0], namespace="current", k=5)
    assert len(hits) == 5
    assert [h.rank for h in hits] == [0, 1, 2, 3, 4]
    assert hits[0].id == "d0"


def test_qdrant_fetch_by_ids():
    a = QdrantAdapter(dim=384)
    a.load(_snap(2, 40), namespace="baseline")
    got = a.fetch_by_ids(["d1", "d2", "d3"], namespace="baseline")
    assert set(got.ids) == {"d1", "d2", "d3"}
