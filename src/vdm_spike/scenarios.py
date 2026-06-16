from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vdm_spike.core import Expectation, Snapshot
from vdm_spike.corpus import BALANCED_MIX, SHIFTED_MIX, make_docs
from vdm_spike.embed import MODEL_A, MODEL_NORM_DIVERGENT, Embedder, inject_invalid

SCENARIO_NAMES = [
    "null-control",
    "benign-growth",
    "topic-shift",
    "model-swap",
    "broken-writes",
]


@dataclass
class Scenario:
    name: str
    baseline: Snapshot
    current: Snapshot
    query_vectors: np.ndarray
    expectation: Expectation


def _snapshot(docs, emb: Embedder) -> Snapshot:
    vecs = emb.encode([d.text for d in docs])
    return Snapshot(ids=[d.id for d in docs], vectors=vecs,
                    metadata=[{"topic": d.topic} for d in docs])


def _concat(a: Snapshot, b: Snapshot) -> Snapshot:
    return Snapshot(
        ids=a.ids + b.ids,
        vectors=np.vstack([a.vectors, b.vectors]).astype(np.float32),
        metadata=(a.metadata or []) + (b.metadata or []),
    )


def build_scenario(name: str, n: int = 2000, seed: int = 0) -> Scenario:
    emb_a = Embedder(MODEL_A)
    base_docs = make_docs(seed=seed, n=n, topic_mix=BALANCED_MIX)
    baseline = _snapshot(base_docs, emb_a)
    query_docs = make_docs(seed=seed + 999, n=50, topic_mix=BALANCED_MIX, id_prefix="q")
    query_vectors = emb_a.encode([d.text for d in query_docs])

    if name == "null-control":
        # True no-op: the SAME corpus, re-presented unchanged. All families quiet.
        current = Snapshot(ids=list(baseline.ids), vectors=baseline.vectors.copy(),
                           metadata=baseline.metadata)
        fires = {"mmd": False, "classifier": False, "retrieval_rbo": False}

    elif name == "benign-growth":
        # Incremental growth: baseline docs RETAINED + ~10% new same-distribution docs.
        extra = make_docs(seed=seed + 2, n=max(1, n // 10), topic_mix=BALANCED_MIX,
                          id_prefix="g")
        current = _concat(baseline, _snapshot(extra, emb_a))
        fires = {"mmd": False, "classifier": False, "retrieval_rbo": False}

    elif name == "topic-shift":
        # Retained docs + a large injection of off-topic docs that displaces top-k.
        injected = make_docs(seed=seed + 3, n=n, topic_mix=SHIFTED_MIX, id_prefix="s")
        current = _concat(baseline, _snapshot(injected, emb_a))
        fires = {"mmd": True, "classifier": True, "retrieval_rbo": True}

    elif name == "model-swap":
        # Same retained docs/ids, re-embedded with a norm-divergent model.
        # NOTE (Finding B contingency): sentence-transformers models all include a
        # Normalize module, producing unit-norm vectors (std=0). norm_ks cannot
        # fire on this class of models. norm_ks remains informational, not gated.
        # The swap is still robustly caught by mmd + classifier + retrieval_rbo.
        emb_div = Embedder(MODEL_NORM_DIVERGENT)
        current = _snapshot(base_docs, emb_div)
        fires = {"mmd": True, "classifier": True, "retrieval_rbo": True}

    elif name == "broken-writes":
        current = _snapshot(base_docs, emb_a)
        current = Snapshot(
            ids=current.ids,
            vectors=inject_invalid(current.vectors, n_zero=max(1, n // 20), seed=seed),
            metadata=current.metadata,
        )
        fires = {"ops_invalid": True}

    else:
        raise ValueError(f"unknown scenario: {name}")

    return Scenario(name=name, baseline=baseline, current=current,
                    query_vectors=query_vectors, expectation=Expectation(fires=fires))
