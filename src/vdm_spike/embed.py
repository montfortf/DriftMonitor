from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# Same output dimension (384) so a model swap is NOT trivially detectable by dim.
MODEL_A = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_B = "sentence-transformers/all-MiniLM-L12-v2"
# A different-family 384-dim model used for the model-swap scenario. NOTE (Phase 1.1
# Finding B): sentence-transformers models emit unit-norm vectors, so norm_ks CANNOT
# distinguish a swap between two such models — it is informational, not gated. The swap
# is robustly caught by mmd + classifier + retrieval_rbo instead.
MODEL_NORM_DIVERGENT = "BAAI/bge-small-en-v1.5"

_CACHE: dict[str, SentenceTransformer] = {}


class Embedder:
    def __init__(self, model_name: str):
        if model_name not in _CACHE:
            _CACHE[model_name] = SentenceTransformer(model_name)
        self.model = _CACHE[model_name]

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, normalize_embeddings=False, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


def inject_invalid(vectors: np.ndarray, n_zero: int, seed: int = 0) -> np.ndarray:
    """Return a copy with `n_zero` rows replaced by zero vectors (simulates broken writes).

    NaN/Inf are intentionally NOT injected here: pgvector rejects them on insert,
    so NaN/Inf detection is unit-tested directly in test_ops.py instead.
    """
    out = vectors.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(out.shape[0], size=n_zero, replace=False)
    out[idx] = 0.0
    return out
