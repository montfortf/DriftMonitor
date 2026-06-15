from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vdm_spike.core import Expectation, Snapshot
from vdm_spike.corpus import BALANCED_MIX, SHIFTED_MIX, make_docs
from vdm_spike.embed import MODEL_A, MODEL_B, Embedder, inject_invalid

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


def build_scenario(name: str, n: int = 2000, seed: int = 0) -> Scenario:
    emb_a = Embedder(MODEL_A)
    base_docs = make_docs(seed=seed, n=n, topic_mix=BALANCED_MIX)
    baseline = _snapshot(base_docs, emb_a)
    # fixed query set: a held-out balanced sample embedded with the baseline model
    query_docs = make_docs(seed=seed + 999, n=50, topic_mix=BALANCED_MIX,
                           id_prefix="q")
    query_vectors = emb_a.encode([d.text for d in query_docs])

    # Gating note (Phase 0 finding): only detectors that are scientifically valid
    # FOR HOW THIS SCENARIO IS CONSTRUCTED are gated. Others are still computed and
    # printed (informational) but not asserted. See README "Findings". This is a
    # correction of expectations, NOT a tuning of detector thresholds.

    if name == "null-control":
        cur_docs = make_docs(seed=seed + 1, n=n, topic_mix=BALANCED_MIX)
        current = _snapshot(cur_docs, emb_a)
        # retrieval_rbo NOT gated: `current` is an INDEPENDENT resample, so the
        # specific documents a query returns legitimately differ even though the
        # distribution is unchanged. RBO is only a valid no-change signal on an
        # incrementally evolving index (shared doc identity) — see Finding A.
        fires = {"mmd": False, "classifier": False}

    elif name == "benign-growth":
        extra = make_docs(seed=seed + 2, n=n // 2, topic_mix=BALANCED_MIX,
                          id_prefix="g")
        cur_docs = base_docs + extra
        current = _snapshot(cur_docs, emb_a)
        # retrieval_rbo NOT gated: large same-distribution growth genuinely
        # reshuffles top-k. Per PRD §7.3, benign growth should be judged by
        # rate-of-change, not an absolute RBO threshold — see Finding A.
        fires = {"mmd": False, "classifier": False}

    elif name == "topic-shift":
        cur_docs = make_docs(seed=seed + 3, n=n, topic_mix=SHIFTED_MIX)
        current = _snapshot(cur_docs, emb_a)
        fires = {"mmd": True, "classifier": True, "retrieval_rbo": True}

    elif name == "model-swap":
        emb_b = Embedder(MODEL_B)
        current = _snapshot(base_docs, emb_b)  # same docs/ids, different model
        # norm_ks NOT gated: the two same-dim MiniLM models have near-identical
        # L2-norm distributions, so norm-shift is a weak channel FOR THIS PAIR.
        # The swap is robustly caught by mmd + classifier + retrieval — see Finding B.
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
