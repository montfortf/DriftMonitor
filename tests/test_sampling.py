import numpy as np

from vdm_spike.adapters.fakes import FakeQueryOnlyAdapter
from vdm_spike.adapters.qdrant import QdrantAdapter
from vdm_spike.core import Snapshot
from vdm_spike.sampling import query_seeded, vector_sample


def _snap(seed, n):
    rng = np.random.default_rng(seed)
    return Snapshot(ids=[f"d{i}" for i in range(n)],
                    vectors=rng.normal(size=(n, 384)).astype(np.float32))


def test_vector_sample_returns_bounded_snapshot():
    a = QdrantAdapter(dim=384)
    a.load(_snap(0, 80), namespace="baseline")
    s = vector_sample(a, "baseline", k=25)
    assert s.n == 25 and s.dim == 384


def test_query_seeded_collects_hits_and_scores():
    a = FakeQueryOnlyAdapter(dim=384)
    snap = _snap(1, 60)
    a.load(snap, namespace="current")
    seeds = snap.vectors[:5]
    qs = query_seeded(a, "current", seeds, k=10)
    assert len(qs.hit_ids) == 5          # one hit-list per seed query
    assert all(len(h) == 10 for h in qs.hit_ids)
    assert len(qs.scores) == 5 * 10      # flattened score pool
    assert qs.strategy == "query-seeded"
