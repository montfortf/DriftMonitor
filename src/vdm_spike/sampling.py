from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vdm_spike.core import Snapshot


@dataclass
class QuerySeeded:
    hit_ids: list[list[str]]   # one ranked id-list per seed query
    scores: list[float]        # flattened similarity scores across all seeds
    strategy: str = "query-seeded"


def vector_sample(adapter, namespace: str, k: int) -> Snapshot:
    """Unbiased vector sample for FULL-plan adapters (delegates to adapter.sample)."""
    return adapter.sample(namespace, k)


def query_seeded(adapter, namespace: str, seed_vectors: np.ndarray, k: int) -> QuerySeeded:
    """Issue a seed-query panel and collect returned neighbors as a biased pseudo-sample.
    Powers retrieval + score-distribution drift only — never distribution stats."""
    hit_ids: list[list[str]] = []
    scores: list[float] = []
    for v in seed_vectors:
        hits = adapter.query(v, namespace, k)
        hit_ids.append([h.id for h in hits])
        scores.extend(h.score for h in hits)
    return QuerySeeded(hit_ids=hit_ids, scores=scores)
