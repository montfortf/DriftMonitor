from __future__ import annotations

import time

import numpy as np
from qdrant_client import QdrantClient, models

from vdm_spike.adapters.base import (
    Capabilities,
    ProbeResult,
    QueryHit,
    StoreDescriptor,
)
from vdm_spike.core import Snapshot


class QdrantAdapter:
    """FULL-plan adapter backed by an in-memory Qdrant (no Docker). Second SDK = genericity proof."""

    def __init__(self, dim: int):
        self.dim = dim
        self.client = QdrantClient(location=":memory:")

    def describe(self) -> StoreDescriptor:
        return StoreDescriptor(name="qdrant", dimension=self.dim, metric="cosine")

    def capabilities(self) -> Capabilities:
        return Capabilities(
            returns_vectors=True, unbiased_sample=True, live_query=True,
            id_listing=True, random_sample="reservoir", max_batch=256,
        )

    def _ensure(self, namespace: str) -> None:
        if not self.client.collection_exists(namespace):
            self.client.create_collection(
                namespace,
                vectors_config=models.VectorParams(
                    size=self.dim, distance=models.Distance.COSINE
                ),
            )

    def load(self, snap: Snapshot, namespace: str) -> None:
        self._ensure(namespace)
        points = [
            models.PointStruct(id=i, vector=snap.vectors[i].tolist(),
                               payload={"doc_id": snap.ids[i]})
            for i in range(snap.n)
        ]
        self.client.upsert(namespace, points=points)

    def count(self, namespace: str) -> int:
        return int(self.client.count(namespace).count)

    def _scroll_all(self, namespace: str):
        points, _ = self.client.scroll(
            namespace, limit=self.count(namespace),
            with_vectors=True, with_payload=True,
        )
        return points

    def sample(self, namespace: str, k: int) -> Snapshot:
        points = self._scroll_all(namespace)
        rng = np.random.default_rng()
        if k < len(points):
            idx = rng.choice(len(points), size=k, replace=False)
            points = [points[i] for i in idx]
        ids = [p.payload["doc_id"] for p in points]
        vecs = np.array([p.vector for p in points], dtype=np.float32)
        return Snapshot(ids=ids, vectors=vecs)

    def fetch_by_ids(self, ids: list[str], namespace: str) -> Snapshot:
        wanted = set(ids)
        points = [p for p in self._scroll_all(namespace) if p.payload["doc_id"] in wanted]
        out_ids = [p.payload["doc_id"] for p in points]
        vecs = np.array([p.vector for p in points], dtype=np.float32)
        return Snapshot(ids=out_ids, vectors=vecs)

    def query(self, vector: np.ndarray, namespace: str, k: int) -> list[QueryHit]:
        res = self.client.query_points(
            namespace, query=np.asarray(vector, dtype=np.float32).tolist(),
            limit=k, with_payload=True,
        ).points
        return [QueryHit(id=p.payload["doc_id"], score=float(p.score), rank=i)
                for i, p in enumerate(res)]

    def probe(self) -> ProbeResult:
        t0 = time.perf_counter()
        self.client.get_collections()
        return ProbeResult(ok=True, latency_ms=(time.perf_counter() - t0) * 1000.0)
