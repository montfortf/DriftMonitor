from __future__ import annotations

import json
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from vdm_spike.adapters.base import (
    Capabilities,
    ProbeResult,
    QueryHit,
    StoreDescriptor,
)
from vdm_spike.core import Snapshot


class PgVectorAdapter:
    """FULL-plan adapter. Returns raw vectors and supports unbiased sampling."""

    def __init__(self, conn: psycopg.Connection, dim: int):
        self.conn = conn
        self.dim = dim
        register_vector(self.conn)

    def describe(self) -> StoreDescriptor:
        return StoreDescriptor(name="pgvector", dimension=self.dim, metric="cosine")

    def capabilities(self) -> Capabilities:
        return Capabilities(
            returns_vectors=True, unbiased_sample=True, live_query=True,
            id_listing=True, random_sample="reservoir", max_batch=1000,
        )

    def count(self, namespace: str) -> int:
        row = self.conn.execute(
            "SELECT count(*) FROM items WHERE namespace = %s", (namespace,)
        ).fetchone()
        return int(row[0])

    def sample(self, namespace: str, k: int) -> Snapshot:
        rows = self.conn.execute(
            "SELECT id, embedding FROM items WHERE namespace = %s "
            "ORDER BY random() LIMIT %s",
            (namespace, k),
        ).fetchall()
        ids = [r[0] for r in rows]
        vecs = np.array([r[1] for r in rows], dtype=np.float32)
        return Snapshot(ids=ids, vectors=vecs)

    def fetch_by_ids(self, ids: list[str], namespace: str) -> Snapshot:
        rows = self.conn.execute(
            "SELECT id, embedding FROM items WHERE namespace = %s AND id = ANY(%s)",
            (namespace, list(ids)),
        ).fetchall()
        out_ids = [r[0] for r in rows]
        vecs = np.array([r[1] for r in rows], dtype=np.float32)
        return Snapshot(ids=out_ids, vectors=vecs)

    def query(self, vector: np.ndarray, namespace: str, k: int) -> list[QueryHit]:
        v = np.asarray(vector, dtype=np.float32)
        rows = self.conn.execute(
            "SELECT id, 1 - (embedding <=> %s) AS score FROM items "
            "WHERE namespace = %s ORDER BY embedding <=> %s LIMIT %s",
            (v, namespace, v, k),
        ).fetchall()
        return [QueryHit(id=r[0], score=float(r[1]), rank=i) for i, r in enumerate(rows)]

    def probe(self) -> ProbeResult:
        t0 = time.perf_counter()
        self.conn.execute("SELECT 1").fetchone()
        return ProbeResult(ok=True, latency_ms=(time.perf_counter() - t0) * 1000.0)

    # --- harness affordance (not part of the read-only contract) ---
    def ensure_schema(self) -> None:
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS items (
                id text, namespace text, embedding vector({self.dim}),
                metadata jsonb, PRIMARY KEY (id, namespace)
            )
            """
        )
        self.conn.commit()

    def load(self, snap: Snapshot, namespace: str) -> None:
        meta = snap.metadata or [{} for _ in snap.ids]
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO items (id, namespace, embedding, metadata) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id, namespace) DO UPDATE SET embedding = EXCLUDED.embedding",
                [(i, namespace, snap.vectors[k], json.dumps(meta[k]))
                 for k, i in enumerate(snap.ids)],
            )
        self.conn.commit()
