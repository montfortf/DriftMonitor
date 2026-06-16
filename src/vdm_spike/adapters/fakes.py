from __future__ import annotations

import numpy as np

from vdm_spike.adapters.base import (
    Capabilities,
    CapabilityError,
    ProbeResult,
    QueryHit,
    StoreDescriptor,
)
from vdm_spike.core import Snapshot


class FakeQueryOnlyAdapter:
    """QUERY-plan store: refuses to return raw/unbiased vectors but answers queries."""

    def __init__(self, dim: int):
        self.dim = dim
        self._ns: dict[str, Snapshot] = {}

    def describe(self) -> StoreDescriptor:
        return StoreDescriptor(name="fake-query-only", dimension=self.dim)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            returns_vectors=False, unbiased_sample=False, live_query=True,
            id_listing=False, random_sample="query-seeded", max_batch=100,
        )

    def load(self, snap: Snapshot, namespace: str) -> None:
        self._ns[namespace] = snap

    def count(self, namespace: str) -> int:
        return self._ns[namespace].n

    def sample(self, namespace: str, k: int) -> Snapshot:
        raise CapabilityError("fake-query-only: raw vector sampling not supported")

    def fetch_by_ids(self, ids: list[str], namespace: str) -> Snapshot:
        raise CapabilityError("fake-query-only: fetch_by_ids not supported")

    def query(self, vector: np.ndarray, namespace: str, k: int) -> list[QueryHit]:
        snap = self._ns[namespace]
        v = np.asarray(vector, dtype=np.float32)
        mat = snap.vectors
        denom = np.linalg.norm(mat, axis=1) * (np.linalg.norm(v) or 1.0)
        denom[denom == 0] = 1.0
        sims = (mat @ v) / denom
        order = np.argsort(-sims)[:k]
        return [QueryHit(id=snap.ids[j], score=float(sims[j]), rank=i)
                for i, j in enumerate(order)]

    def probe(self) -> ProbeResult:
        return ProbeResult(ok=True, latency_ms=0.0)


class FakeMinimalAdapter(FakeQueryOnlyAdapter):
    """MINIMAL-plan store: not even live query (e.g. count/health only)."""

    def describe(self) -> StoreDescriptor:
        return StoreDescriptor(name="fake-minimal", dimension=self.dim)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            returns_vectors=False, unbiased_sample=False, live_query=False,
            id_listing=False, random_sample="none", max_batch=0,
        )

    def query(self, vector: np.ndarray, namespace: str, k: int) -> list[QueryHit]:
        raise CapabilityError("fake-minimal: live query not supported")
