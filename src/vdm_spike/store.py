from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from vdm_spike.core import Snapshot


@dataclass
class QueryHit:
    id: str
    score: float
    rank: int


class PgVectorStore:
    """Thin pgvector read path. Method shapes mirror the future VectorStoreAdapter."""

    def __init__(self, conn: psycopg.Connection, dim: int):
        self.conn = conn
        self.dim = dim
        register_vector(self.conn)

    def ensure_schema(self) -> None:
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS items (
                id text,
                namespace text,
                embedding vector({self.dim}),
                metadata jsonb,
                PRIMARY KEY (id, namespace)
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
                [
                    (i, namespace, snap.vectors[k], json.dumps(meta[k]))
                    for k, i in enumerate(snap.ids)
                ],
            )
        self.conn.commit()

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

    def query(self, vector: np.ndarray, namespace: str, k: int) -> list[QueryHit]:
        rows = self.conn.execute(
            "SELECT id, 1 - (embedding <=> %s) AS score FROM items "
            "WHERE namespace = %s ORDER BY embedding <=> %s LIMIT %s",
            (
                np.asarray(vector, dtype=np.float32),
                namespace,
                np.asarray(vector, dtype=np.float32),
                k,
            ),
        ).fetchall()
        return [QueryHit(id=r[0], score=float(r[1]), rank=i) for i, r in enumerate(rows)]
