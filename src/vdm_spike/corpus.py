from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOPIC_VOCAB: dict[str, list[str]] = {
    "finance": ["loan", "interest", "market", "stock", "bank", "credit",
                "invest", "fund", "rate", "bond", "equity", "yield"],
    "medicine": ["patient", "dose", "clinic", "surgery", "drug", "therapy",
                 "nurse", "symptom", "diagnosis", "health", "vaccine", "trial"],
    "sports": ["match", "goal", "team", "score", "coach", "player",
               "league", "season", "tournament", "win", "defense", "striker"],
    "tech": ["server", "code", "data", "network", "cloud", "model",
             "query", "cache", "latency", "deploy", "kernel", "buffer"],
}

BALANCED_MIX: dict[str, float] = {"finance": 0.25, "medicine": 0.25,
                                  "sports": 0.25, "tech": 0.25}
# Heavily reweighted toward two topics → a genuine semantic shift.
SHIFTED_MIX: dict[str, float] = {"finance": 0.05, "medicine": 0.05,
                                 "sports": 0.45, "tech": 0.45}


@dataclass
class Doc:
    id: str
    text: str
    topic: str


def make_docs(seed: int, n: int, topic_mix: dict[str, float],
              words_per_doc: int = 12, id_prefix: str = "d") -> list[Doc]:
    """Generate `n` synthetic documents whose topics follow `topic_mix`."""
    rng = np.random.default_rng(seed)
    topics = list(topic_mix.keys())
    weights = np.array([topic_mix[t] for t in topics], dtype=float)
    weights = weights / weights.sum()
    chosen = rng.choice(topics, size=n, p=weights)
    docs: list[Doc] = []
    for i, topic in enumerate(chosen):
        vocab = TOPIC_VOCAB[topic]
        words = rng.choice(vocab, size=words_per_doc, replace=True)
        docs.append(Doc(id=f"{id_prefix}{i}", text=" ".join(words), topic=str(topic)))
    return docs
