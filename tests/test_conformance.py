import numpy as np

from vdm_spike.adapters.base import Capabilities, ProbeResult, QueryHit, StoreDescriptor
from vdm_spike.adapters.fakes import FakeMinimalAdapter, FakeQueryOnlyAdapter
from vdm_spike.adapters.qdrant import QdrantAdapter
from vdm_spike.conformance import run_conformance
from vdm_spike.core import Snapshot


def test_qdrant_passes_conformance():
    report = run_conformance(QdrantAdapter(dim=384), dim=384)
    assert report.ok, report.failures


def test_query_only_passes_conformance():
    report = run_conformance(FakeQueryOnlyAdapter(dim=384), dim=384)
    assert report.ok, report.failures


def test_minimal_passes_conformance():
    report = run_conformance(FakeMinimalAdapter(dim=384), dim=384)
    assert report.ok, report.failures


class _DishonestAdapter:
    """Declares returns_vectors=True but sample() returns nothing — must FAIL conformance."""

    def __init__(self, dim):
        self.dim = dim
        self._ns = {}
    def describe(self): return StoreDescriptor(name="liar", dimension=self.dim)
    def capabilities(self):
        return Capabilities(returns_vectors=True, unbiased_sample=True, live_query=True)
    def load(self, snap, namespace): self._ns[namespace] = snap
    def count(self, namespace): return self._ns[namespace].n
    def sample(self, namespace, k):
        return Snapshot(ids=[], vectors=np.zeros((0, self.dim), dtype=np.float32))
    def fetch_by_ids(self, ids, namespace): return self.sample(namespace, len(ids))
    def query(self, vector, namespace, k):
        return [QueryHit(id="x", score=0.5, rank=0)]
    def probe(self): return ProbeResult(ok=True, latency_ms=0.0)


def test_conformance_has_teeth_dishonest_adapter_fails():
    report = run_conformance(_DishonestAdapter(dim=384), dim=384)
    assert report.ok is False
    assert any("returns_vectors" in f or "sample" in f for f in report.failures)
