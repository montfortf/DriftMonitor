import numpy as np

from vdm_spike.corpus import BALANCED_MIX, make_docs
from vdm_spike.embed import MODEL_A, MODEL_B, Embedder, inject_invalid


def test_embedder_a_and_b_share_dimension():
    docs = make_docs(seed=0, n=8, topic_mix=BALANCED_MIX)
    texts = [d.text for d in docs]
    va = Embedder(MODEL_A).encode(texts)
    vb = Embedder(MODEL_B).encode(texts)
    assert va.shape[0] == vb.shape[0] == 8
    assert va.shape[1] == vb.shape[1] == 384


def test_embedding_is_deterministic():
    docs = make_docs(seed=0, n=4, topic_mix=BALANCED_MIX)
    texts = [d.text for d in docs]
    emb = Embedder(MODEL_A)
    assert np.allclose(emb.encode(texts), emb.encode(texts))


def test_model_a_and_b_differ_on_same_text():
    emb_a = Embedder(MODEL_A).encode(["loan interest market"])
    emb_b = Embedder(MODEL_B).encode(["loan interest market"])
    # same dimension, genuinely different representation
    assert not np.allclose(emb_a, emb_b)


def test_inject_invalid_creates_zero_norm_rows():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(10, 384)).astype(np.float32)
    out = inject_invalid(v, n_zero=3, seed=1)
    zero_rows = np.where(np.linalg.norm(out, axis=1) == 0)[0]
    assert len(zero_rows) == 3
