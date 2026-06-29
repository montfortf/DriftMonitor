# VDM Phase 0 Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-Python harness that proves the drift signal fires on harmful shifts and stays quiet on benign growth, legibly, against real embeddings in a real pgvector store.

**Architecture:** Synthetic topical corpus → embedded with real sentence-transformers models → loaded into a Dockerized pgvector store → read back over SQL → run through a distribution + retrieval + operational detector catalog → verdict table + PCA-overlay plot. Labeled scenarios double as test fixtures; the run-level gate asserts each detector fires/stays-quiet as expected.

**Tech Stack:** Python 3.11+, uv, pytest, ruff, numpy, scipy, scikit-learn, sentence-transformers, psycopg[binary], pgvector, matplotlib, Evidently (one cross-check), Docker (pgvector/pgvector:pg16).

---

## File Structure

```
vdm-spike/
├─ scripts/setup_db.sh         # brew postgres@16 + pgvector setup (idempotent)
├─ pyproject.toml              # uv project + deps + tool config
├─ .gitignore
├─ src/vdm_spike/
│  ├─ __init__.py
│  ├─ core.py                  # Snapshot, Expectation, DetectorResult dataclasses
│  ├─ features.py              # frozen-PCA fit + transform; rbf kernel + median-heuristic bandwidth
│  ├─ corpus.py                # synthetic topical corpus + shift operators
│  ├─ embed.py                 # model A / model B wrappers; invalid-vector injection
│  ├─ store.py                 # pgvector read path: count / sample / query (namespaced)
│  ├─ detectors/
│  │  ├─ __init__.py
│  │  ├─ distribution.py       # centroid, mmd, classifier, norm_ks, psi
│  │  └─ retrieval.py          # rbo/jaccard overlap, score_ks
│  ├─ ops.py                   # invalid-vector (NaN/Inf/zero-norm) checks
│  ├─ power.py                 # minimum detectable effect / under-powered flag
│  ├─ scenarios.py             # labeled scenario builders
│  ├─ report.py                # verdict table + PCA overlay plot
│  └─ run.py                   # orchestrator + gate verdict
└─ tests/
   ├─ conftest.py              # pgvector store fixture
   ├─ test_features.py
   ├─ test_corpus.py
   ├─ test_embed.py
   ├─ test_store.py
   ├─ test_distribution.py
   ├─ test_retrieval.py
   ├─ test_ops.py
   ├─ test_power.py
   ├─ test_scenarios.py
   └─ test_gate.py             # the success-criterion: fires-on-harmful, quiet-on-benign
```

All commands run from `vdm-spike/`. The Postgres service must be running (`brew services start postgresql@16`) with the `vdm` role/database and `vector` extension installed — see Task 2 — for `test_store.py`, `test_retrieval.py`, `test_scenarios.py`, and `test_gate.py`.

---

## Task 1: Project scaffold, tooling, and git

**Files:**
- Create: `vdm-spike/pyproject.toml`
- Create: `vdm-spike/.gitignore`
- Create: `vdm-spike/src/vdm_spike/__init__.py`
- Create: `vdm-spike/src/vdm_spike/detectors/__init__.py`
- Create: `vdm-spike/tests/test_smoke.py`

- [ ] **Step 1: Create the project and structure**

```bash
cd "/Users/montfortfernando/Dropbox/Montfort/Dev2026/Vector/Drift monitor"
mkdir -p vdm-spike/src/vdm_spike/detectors vdm-spike/tests vdm-spike/scripts
cd vdm-spike
git init
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "vdm-spike"
version = "0.0.1"
description = "VDM Phase 0 drift-signal validation spike"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "scikit-learn>=1.4",
    "sentence-transformers>=2.7",
    "psycopg[binary]>=3.1",
    "pgvector>=0.2.5",
    "matplotlib>=3.8",
    "evidently>=0.4.25",
]

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.4"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vdm_spike"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
*.png
.env
```

- [ ] **Step 4: Create package init files**

`src/vdm_spike/__init__.py`:
```python
"""VDM Phase 0 drift-signal validation spike."""
```

`src/vdm_spike/detectors/__init__.py`:
```python
"""Drift detector catalog."""
```

- [ ] **Step 5: Write a smoke test**

`tests/test_smoke.py`:
```python
import vdm_spike


def test_package_imports():
    assert vdm_spike is not None
```

- [ ] **Step 6: Install and verify**

Run: `uv sync && uv run pytest tests/test_smoke.py -v && uv run ruff check`
Expected: 1 passed; ruff reports no errors.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold vdm-spike project, tooling, and smoke test"
```

---

## Task 2: Local Postgres + pgvector (Homebrew) + connection fixture

> **Environment note:** Docker is not available on this machine; pgvector runs as a native Homebrew Postgres service. The DSN (`postgresql://vdm:vdm@localhost:5432/vdm`) and all test code are identical to a Docker setup — only provisioning differs. The `setup_db.sh` script must be idempotent (safe to re-run). The acceptance criterion is the verified end state (role + db + `vector` extension reachable), so adapt the brew specifics if a formula name/version differs on this machine.

**Files:**
- Create: `vdm-spike/scripts/setup_db.sh`
- Create: `vdm-spike/tests/conftest.py`
- Create: `vdm-spike/tests/test_store.py` (connection smoke only in this task)

- [ ] **Step 1: Write `scripts/setup_db.sh`**

```bash
#!/usr/bin/env bash
# Idempotent local pgvector setup via Homebrew. Target: a `vdm` role/db on
# localhost:5432 with the `vector` extension, reachable by the test DSN.
set -euo pipefail

brew install postgresql@16 pgvector
brew services start postgresql@16

# Wait for the server to accept connections.
PG_BIN="$(brew --prefix postgresql@16)/bin"
export PATH="$PG_BIN:$PATH"
for _ in $(seq 1 30); do
  if pg_isready -h localhost -p 5432 >/dev/null 2>&1; then break; fi
  sleep 1
done

# Create the vdm role + database if absent (connect to default 'postgres' db
# as the bootstrap superuser, which is the current OS user under brew).
psql -d postgres -v ON_ERROR_STOP=1 <<'SQL'
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'vdm') THEN
    CREATE ROLE vdm LOGIN PASSWORD 'vdm' SUPERUSER;
  END IF;
END
$$;
SQL

psql -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'vdm'" \
  | grep -q 1 || createdb -O vdm vdm

psql -d vdm -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector"
echo "pgvector ready at postgresql://vdm:vdm@localhost:5432/vdm"
```

- [ ] **Step 2: Run the setup script**

Run: `chmod +x scripts/setup_db.sh && ./scripts/setup_db.sh`
Expected: ends with `pgvector ready at postgresql://vdm:vdm@localhost:5432/vdm`. (First run compiles/installs Postgres + pgvector via brew — allow a few minutes.)

- [ ] **Step 3: Write the pgvector fixture in `conftest.py`**

```python
import os

import psycopg
import pytest
from pgvector.psycopg import register_vector

DSN = os.environ.get("VDM_DSN", "postgresql://vdm:vdm@localhost:5432/vdm")


@pytest.fixture()
def conn():
    with psycopg.connect(DSN) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(c)
        yield c
        c.execute("DROP TABLE IF EXISTS items")
        c.commit()
```

- [ ] **Step 4: Write the connection smoke test**

`tests/test_store.py`:
```python
def test_pgvector_extension_available(conn):
    row = conn.execute(
        "SELECT extname FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    assert row is not None
    assert row[0] == "vector"
```

- [ ] **Step 5: Run the smoke test**

Run: `uv run pytest tests/test_store.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: dockerized pgvector service and connection fixture"
```

---

## Task 3: Core types and feature helpers

**Files:**
- Create: `vdm-spike/src/vdm_spike/core.py`
- Create: `vdm-spike/src/vdm_spike/features.py`
- Create: `vdm-spike/tests/test_features.py`

- [ ] **Step 1: Write `core.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Snapshot:
    """A set of vectors read from (or destined for) a store."""

    ids: list[str]
    vectors: np.ndarray  # shape (n, d), float32
    metadata: list[dict] | None = None

    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]


@dataclass
class Expectation:
    """Which detectors a scenario asserts should fire (True) or stay quiet (False).

    Only the listed detector names are gated; others are reported but not asserted.
    """

    fires: dict[str, bool]


@dataclass
class DetectorResult:
    name: str
    statistic: float
    p_value: float | None
    fired: bool
    detail: dict = field(default_factory=dict)
```

- [ ] **Step 2: Write the failing test for `features.py`**

`tests/test_features.py`:
```python
import numpy as np

from vdm_spike.features import fit_pca, median_bandwidth, rbf_kernel


def test_fit_pca_reduces_dimensionality_and_transforms():
    rng = np.random.default_rng(0)
    baseline = rng.normal(size=(500, 50)).astype(np.float32)
    pca = fit_pca(baseline, var=0.95)
    assert 0 < pca.n_components_ <= 50
    projected = pca.transform(baseline)
    assert projected.shape == (500, pca.n_components_)


def test_median_bandwidth_is_positive_and_frozen_value():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 10)).astype(np.float32)
    gamma = median_bandwidth(x)
    assert gamma > 0
    # deterministic for same input
    assert gamma == median_bandwidth(x)


def test_rbf_kernel_diagonal_is_one():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 8)).astype(np.float32)
    k = rbf_kernel(x, x, gamma=0.5)
    assert k.shape == (30, 30)
    assert np.allclose(np.diag(k), 1.0)
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL (`ModuleNotFoundError: vdm_spike.features`).

- [ ] **Step 4: Write `features.py`**

```python
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA


def fit_pca(baseline: np.ndarray, var: float = 0.95) -> PCA:
    """Fit a PCA basis on the baseline retaining `var` fraction of variance.

    The basis is fit ONCE on the baseline and reused to transform current data,
    so projections are axis-comparable (frozen basis).
    """
    max_comp = min(baseline.shape)
    pca = PCA(n_components=var, svd_solver="full", random_state=0)
    pca.fit(baseline)
    # n_components float can occasionally select all dims; clamp is implicit via fit.
    if pca.n_components_ > max_comp:  # defensive, should not happen
        pca = PCA(n_components=max_comp, svd_solver="full", random_state=0).fit(baseline)
    return pca


def median_bandwidth(x: np.ndarray) -> float:
    """RBF gamma via the median heuristic, computed on `x` and meant to be frozen."""
    dists = pdist(x, metric="euclidean")
    median = float(np.median(dists))
    sigma = median if median > 0 else 1.0
    return 1.0 / (2.0 * sigma * sigma)


def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    sq = cdist(a, b, metric="sqeuclidean")
    return np.exp(-gamma * sq)
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_features.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: core dataclasses and frozen-PCA / rbf feature helpers"
```

---

## Task 4: Synthetic corpus + shift operators

**Files:**
- Create: `vdm-spike/src/vdm_spike/corpus.py`
- Create: `vdm-spike/tests/test_corpus.py`

- [ ] **Step 1: Write the failing test**

`tests/test_corpus.py`:
```python
from vdm_spike.corpus import BALANCED_MIX, SHIFTED_MIX, make_docs


def test_make_docs_is_deterministic_for_same_seed():
    a = make_docs(seed=0, n=50, topic_mix=BALANCED_MIX)
    b = make_docs(seed=0, n=50, topic_mix=BALANCED_MIX)
    assert [d.text for d in a] == [d.text for d in b]


def test_make_docs_count_and_unique_ids():
    docs = make_docs(seed=1, n=120, topic_mix=BALANCED_MIX)
    assert len(docs) == 120
    assert len({d.id for d in docs}) == 120


def test_shifted_mix_changes_topic_distribution():
    base = make_docs(seed=2, n=200, topic_mix=BALANCED_MIX)
    shifted = make_docs(seed=2, n=200, topic_mix=SHIFTED_MIX)
    base_topics = [d.topic for d in base]
    shifted_topics = [d.topic for d in shifted]
    assert base_topics != shifted_topics
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: FAIL (`ModuleNotFoundError: vdm_spike.corpus`).

- [ ] **Step 3: Write `corpus.py`**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: synthetic topical corpus with balanced/shifted topic mixes"
```

---

## Task 5: Embedding models (A baseline / B swap) + invalid injection

**Files:**
- Create: `vdm-spike/src/vdm_spike/embed.py`
- Create: `vdm-spike/tests/test_embed.py`

- [ ] **Step 1: Write the failing test**

`tests/test_embed.py`:
```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_embed.py -v`
Expected: FAIL (`ModuleNotFoundError: vdm_spike.embed`).

- [ ] **Step 3: Write `embed.py`**

```python
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# Same output dimension (384) so a model swap is NOT trivially detectable by dim.
MODEL_A = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_B = "sentence-transformers/all-MiniLM-L12-v2"

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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_embed.py -v`
Expected: 4 passed (first run downloads both models; allow time).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: dual same-dim embedders (A/B) and invalid-vector injection"
```

---

## Task 6: pgvector read path (store.py)

**Files:**
- Create: `vdm-spike/src/vdm_spike/store.py`
- Modify: `vdm-spike/tests/test_store.py`

- [ ] **Step 1: Write the failing test (append to `tests/test_store.py`)**

```python
import numpy as np

from vdm_spike.core import Snapshot
from vdm_spike.store import PgVectorStore


def _snapshot(seed: int, n: int) -> Snapshot:
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, 384)).astype(np.float32)
    ids = [f"x{i}" for i in range(n)]
    return Snapshot(ids=ids, vectors=vecs)


def test_load_count_sample_roundtrip(conn):
    store = PgVectorStore(conn, dim=384)
    store.ensure_schema()
    snap = _snapshot(seed=0, n=200)
    store.load(snap, namespace="baseline")
    assert store.count("baseline") == 200
    sampled = store.sample("baseline", k=50)
    assert sampled.n == 50
    assert sampled.dim == 384
    assert set(sampled.ids).issubset(set(snap.ids))


def test_query_returns_ranked_hits(conn):
    store = PgVectorStore(conn, dim=384)
    store.ensure_schema()
    snap = _snapshot(seed=1, n=100)
    store.load(snap, namespace="current")
    hits = store.query(snap.vectors[0], namespace="current", k=5)
    assert len(hits) == 5
    ranks = [h.rank for h in hits]
    assert ranks == [0, 1, 2, 3, 4]
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)  # cosine similarity descending
    assert hits[0].id == "x0"  # a vector is its own nearest neighbor
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL (`ModuleNotFoundError: vdm_spike.store`).

- [ ] **Step 3: Write `store.py`**

```python
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
            (np.asarray(vector, dtype=np.float32), namespace,
             np.asarray(vector, dtype=np.float32), k),
        ).fetchall()
        return [QueryHit(id=r[0], score=float(r[1]), rank=i) for i, r in enumerate(rows)]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_store.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: pgvector read path (count/sample/query) with roundtrip tests"
```

---

## Task 7: Distribution detector — centroid distance

**Files:**
- Create: `vdm-spike/src/vdm_spike/detectors/distribution.py`
- Create: `vdm-spike/tests/test_distribution.py`

- [ ] **Step 1: Write the failing test**

`tests/test_distribution.py`:
```python
import numpy as np

from vdm_spike.detectors.distribution import centroid_distance


def test_centroid_quiet_on_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 32)).astype(np.float32)
    b = rng.normal(size=(400, 32)).astype(np.float32)
    res = centroid_distance(a, b)
    assert res.fired is False


def test_centroid_fires_on_shifted_mean():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 32)).astype(np.float32)
    b = rng.normal(loc=0.5, size=(400, 32)).astype(np.float32)
    res = centroid_distance(a, b)
    assert res.fired is True
    assert res.statistic > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_distribution.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `distribution.py` (centroid only for now)**

```python
from __future__ import annotations

import numpy as np

from vdm_spike.core import DetectorResult


def centroid_distance(baseline: np.ndarray, current: np.ndarray,
                      threshold: float = 0.02) -> DetectorResult:
    """Cosine distance between mean vectors. Coarse first signal (informational)."""
    mb = baseline.mean(axis=0)
    mc = current.mean(axis=0)
    denom = (np.linalg.norm(mb) * np.linalg.norm(mc)) or 1.0
    cosine_sim = float(np.dot(mb, mc) / denom)
    dist = 1.0 - cosine_sim
    return DetectorResult(
        name="centroid",
        statistic=dist,
        p_value=None,
        fired=dist > threshold,
        detail={"threshold": threshold},
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_distribution.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: centroid-distance detector"
```

---

## Task 8: Distribution detector — MMD (RBF, frozen bandwidth, permutation)

**Files:**
- Modify: `vdm-spike/src/vdm_spike/detectors/distribution.py`
- Modify: `vdm-spike/tests/test_distribution.py`

- [ ] **Step 1: Write the failing test (append)**

```python
from vdm_spike.detectors.distribution import mmd_rbf


def test_mmd_quiet_on_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 16)).astype(np.float32)
    b = rng.normal(size=(300, 16)).astype(np.float32)
    res = mmd_rbf(a, b, n_perm=200, seed=0)
    assert res.p_value is not None and res.p_value > 0.05
    assert res.fired is False


def test_mmd_fires_on_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 16)).astype(np.float32)
    b = rng.normal(loc=0.6, size=(300, 16)).astype(np.float32)
    res = mmd_rbf(a, b, n_perm=200, seed=0)
    assert res.p_value is not None and res.p_value < 0.05
    assert res.fired is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_distribution.py::test_mmd_fires_on_shifted_distribution -v`
Expected: FAIL (`ImportError: cannot import name 'mmd_rbf'`).

- [ ] **Step 3: Add `mmd_rbf` to `distribution.py`**

```python
from vdm_spike.features import median_bandwidth, rbf_kernel


def _mmd2_from_kernel(k: np.ndarray, n: int) -> float:
    kxx = k[:n, :n]
    kyy = k[n:, n:]
    kxy = k[:n, n:]
    m = k.shape[0] - n
    # unbiased estimator
    sxx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    syy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    sxy = kxy.sum() / (n * m)
    return float(sxx + syy - 2 * sxy)


def mmd_rbf(baseline: np.ndarray, current: np.ndarray,
            n_perm: int = 200, alpha: float = 0.05, seed: int = 0) -> DetectorResult:
    """MMD^2 with RBF kernel; bandwidth frozen via median heuristic on baseline.

    Significance via a label-permutation test on the precomputed kernel matrix.
    """
    gamma = median_bandwidth(baseline)  # frozen on baseline
    n = baseline.shape[0]
    z = np.vstack([baseline, current])
    k = rbf_kernel(z, z, gamma=gamma)
    observed = _mmd2_from_kernel(k, n)

    rng = np.random.default_rng(seed)
    total = z.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(total)
        kp = k[np.ix_(perm, perm)]
        if _mmd2_from_kernel(kp, n) >= observed:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return DetectorResult(
        name="mmd",
        statistic=observed,
        p_value=p_value,
        fired=p_value < alpha,
        detail={"gamma": gamma, "n_perm": n_perm},
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_distribution.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: MMD detector with frozen RBF bandwidth and permutation test"
```

---

## Task 9: Distribution detector — domain-classifier AUC (PCA + k-fold + permutation null)

**Files:**
- Modify: `vdm-spike/src/vdm_spike/detectors/distribution.py`
- Modify: `vdm-spike/tests/test_distribution.py`

- [ ] **Step 1: Write the failing test (append)**

```python
from vdm_spike.detectors.distribution import classifier_drift
from vdm_spike.features import fit_pca


def test_classifier_quiet_on_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 30)).astype(np.float32)
    b = rng.normal(size=(400, 30)).astype(np.float32)
    pca = fit_pca(a, var=0.95)
    res = classifier_drift(a, b, pca, n_perm=30, seed=0)
    assert res.fired is False


def test_classifier_fires_on_separable_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 30)).astype(np.float32)
    b = rng.normal(loc=0.7, size=(400, 30)).astype(np.float32)
    pca = fit_pca(a, var=0.95)
    res = classifier_drift(a, b, pca, n_perm=30, seed=0)
    assert res.fired is True
    assert res.statistic > 0.55  # AUC well above chance
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_distribution.py::test_classifier_fires_on_separable_distribution -v`
Expected: FAIL (`ImportError: cannot import name 'classifier_drift'`).

- [ ] **Step 3: Add `classifier_drift` to `distribution.py`**

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score


def _cv_auc(features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    clf = LogisticRegression(max_iter=1000)
    proba = cross_val_predict(
        clf, features, labels, cv=5, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(labels, proba))


def classifier_drift(baseline: np.ndarray, current: np.ndarray, pca: PCA,
                     n_perm: int = 30, alpha: float = 0.05,
                     seed: int = 0) -> DetectorResult:
    """Domain-classifier AUC on PCA-reduced features with a permutation null."""
    fb = pca.transform(baseline)
    fc = pca.transform(current)
    features = np.vstack([fb, fc])
    labels = np.concatenate([np.zeros(len(fb)), np.ones(len(fc))]).astype(int)

    observed = _cv_auc(features, labels, seed)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        permuted = rng.permutation(labels)
        if _cv_auc(features, permuted, seed) >= observed:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return DetectorResult(
        name="classifier",
        statistic=observed,
        p_value=p_value,
        fired=(p_value < alpha) and (observed > 0.55),
        detail={"n_perm": n_perm, "n_components": int(pca.n_components_)},
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_distribution.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: domain-classifier-AUC detector with permutation null"
```

---

## Task 10: Distribution detectors — norm-KS and per-dim PSI

**Files:**
- Modify: `vdm-spike/src/vdm_spike/detectors/distribution.py`
- Modify: `vdm-spike/tests/test_distribution.py`

> Note: PSI is reported with the standard threshold convention (>0.2 = significant shift) and `share_drifted`. Benjamini–Hochberg-corrected per-dimension KS is deferred to Phase 1; the Phase 0 gate does not depend on per-dim localization.

- [ ] **Step 1: Write the failing test (append)**

```python
from vdm_spike.detectors.distribution import norm_ks, perdim_psi


def test_norm_ks_fires_when_norms_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 20)).astype(np.float32)
    b = (rng.normal(size=(400, 20)) * 3.0).astype(np.float32)  # magnitude change
    res = norm_ks(a, b)
    assert res.fired is True


def test_norm_ks_quiet_on_same_norms():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 20)).astype(np.float32)
    b = rng.normal(size=(400, 20)).astype(np.float32)
    res = norm_ks(a, b)
    assert res.fired is False


def test_perdim_psi_fires_on_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 20)).astype(np.float32)
    pca = fit_pca(a, var=0.95)
    b = rng.normal(loc=1.0, size=(400, 20)).astype(np.float32)
    res = perdim_psi(a, b, pca)
    assert res.fired is True


def test_perdim_psi_quiet_on_same(  ):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 20)).astype(np.float32)
    pca = fit_pca(a, var=0.95)
    b = rng.normal(size=(400, 20)).astype(np.float32)
    res = perdim_psi(a, b, pca)
    assert res.fired is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_distribution.py::test_perdim_psi_fires_on_shift -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Add `norm_ks` and `perdim_psi` to `distribution.py`**

```python
from scipy.stats import ks_2samp


def norm_ks(baseline: np.ndarray, current: np.ndarray,
            alpha: float = 0.05) -> DetectorResult:
    """KS test on L2 norm distributions — strong signal for a silent model swap."""
    nb = np.linalg.norm(baseline, axis=1)
    nc = np.linalg.norm(current, axis=1)
    stat, p = ks_2samp(nb, nc)
    return DetectorResult(
        name="norm_ks", statistic=float(stat), p_value=float(p),
        fired=p < alpha, detail={},
    )


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def perdim_psi(baseline: np.ndarray, current: np.ndarray, pca: PCA,
               psi_threshold: float = 0.2,
               share_threshold: float = 0.1) -> DetectorResult:
    """Per-dimension PSI on the frozen PCA basis; fire if share-drifted exceeds threshold."""
    fb = pca.transform(baseline)
    fc = pca.transform(current)
    psis = np.array([_psi(fb[:, j], fc[:, j]) for j in range(fb.shape[1])])
    share_drifted = float(np.mean(psis > psi_threshold))
    return DetectorResult(
        name="psi", statistic=share_drifted, p_value=None,
        fired=share_drifted > share_threshold,
        detail={"max_psi": float(psis.max()), "psi_threshold": psi_threshold},
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_distribution.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: norm-KS and per-dim PSI distribution detectors"
```

---

## Task 11: Retrieval-quality detectors (RBO / Jaccard / score-KS)

**Files:**
- Create: `vdm-spike/src/vdm_spike/detectors/retrieval.py`
- Create: `vdm-spike/tests/test_retrieval.py`

- [ ] **Step 1: Write the failing test**

`tests/test_retrieval.py`:
```python
import numpy as np

from vdm_spike.detectors.retrieval import rbo, retrieval_overlap, score_ks


def test_rbo_identical_lists_is_one():
    assert rbo(["a", "b", "c"], ["a", "b", "c"], p=0.9) == 1.0


def test_rbo_disjoint_lists_is_low():
    assert rbo(["a", "b", "c"], ["x", "y", "z"], p=0.9) < 0.1


def test_retrieval_overlap_quiet_when_results_stable():
    baseline = [["a", "b", "c"], ["d", "e", "f"]]
    current = [["a", "b", "c"], ["d", "e", "f"]]
    res = retrieval_overlap(baseline, current)
    assert res.fired is False
    assert res.statistic > 0.95  # mean RBO


def test_retrieval_overlap_fires_when_results_diverge():
    baseline = [["a", "b", "c"], ["d", "e", "f"]]
    current = [["x", "y", "z"], ["p", "q", "r"]]
    res = retrieval_overlap(baseline, current)
    assert res.fired is True


def test_score_ks_fires_when_scores_drop():
    base_scores = np.full(100, 0.9)
    curr_scores = np.full(100, 0.5)
    res = score_ks(base_scores, curr_scores)
    assert res.fired is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `retrieval.py`**

```python
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from vdm_spike.core import DetectorResult


def rbo(list1: list[str], list2: list[str], p: float = 0.9) -> float:
    """Rank-Biased Overlap — weights agreement at the top of the ranking."""
    k = max(len(list1), len(list2))
    if k == 0:
        return 1.0
    score = 0.0
    s1: set[str] = set()
    s2: set[str] = set()
    for d in range(k):
        if d < len(list1):
            s1.add(list1[d])
        if d < len(list2):
            s2.add(list2[d])
        overlap = len(s1 & s2)
        score += (overlap / (d + 1)) * (p ** d)
    return (1 - p) * score


def retrieval_overlap(baseline_hits: list[list[str]], current_hits: list[list[str]],
                      p: float = 0.9, rbo_threshold: float = 0.8) -> DetectorResult:
    """Mean RBO across a query set; fire when top-k results drift apart."""
    scores = [rbo(b, c, p=p) for b, c in zip(baseline_hits, current_hits)]
    mean_rbo = float(np.mean(scores)) if scores else 1.0
    return DetectorResult(
        name="retrieval_rbo", statistic=mean_rbo, p_value=None,
        fired=mean_rbo < rbo_threshold, detail={"p": p, "n_queries": len(scores)},
    )


def score_ks(baseline_scores: np.ndarray, current_scores: np.ndarray,
             alpha: float = 0.05) -> DetectorResult:
    """KS test on similarity-score distributions — works even when vectors are unreadable."""
    stat, p = ks_2samp(np.asarray(baseline_scores), np.asarray(current_scores))
    return DetectorResult(
        name="retrieval_score_ks", statistic=float(stat), p_value=float(p),
        fired=p < alpha, detail={},
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: retrieval-quality detectors (RBO overlap, score-KS)"
```

---

## Task 12: Operational checks (ops.py)

**Files:**
- Create: `vdm-spike/src/vdm_spike/ops.py`
- Create: `vdm-spike/tests/test_ops.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ops.py`:
```python
import numpy as np

from vdm_spike.ops import invalid_vectors


def test_invalid_quiet_on_clean_vectors():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 16)).astype(np.float32)
    res = invalid_vectors(v)
    assert res.fired is False
    assert res.statistic == 0


def test_invalid_detects_nan_inf_and_zero():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 16)).astype(np.float32)
    v[0] = np.nan
    v[1] = np.inf
    v[2] = 0.0
    res = invalid_vectors(v)
    assert res.fired is True
    assert res.statistic == 3
    assert res.detail["nan"] == 1
    assert res.detail["inf"] == 1
    assert res.detail["zero_norm"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_ops.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `ops.py`**

```python
from __future__ import annotations

import numpy as np

from vdm_spike.core import DetectorResult


def invalid_vectors(vectors: np.ndarray) -> DetectorResult:
    """Count NaN/Inf/zero-norm rows — broken embedding writes."""
    nan_rows = np.isnan(vectors).any(axis=1)
    inf_rows = np.isinf(vectors).any(axis=1)
    finite = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    zero_rows = (np.linalg.norm(finite, axis=1) == 0) & ~nan_rows & ~inf_rows
    n_nan = int(nan_rows.sum())
    n_inf = int(inf_rows.sum())
    n_zero = int(zero_rows.sum())
    total = n_nan + n_inf + n_zero
    return DetectorResult(
        name="ops_invalid", statistic=total, p_value=None, fired=total > 0,
        detail={"nan": n_nan, "inf": n_inf, "zero_norm": n_zero},
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_ops.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: operational invalid-vector detector (NaN/Inf/zero-norm)"
```

---

## Task 13: Statistical power / minimum detectable effect

**Files:**
- Create: `vdm-spike/src/vdm_spike/power.py`
- Create: `vdm-spike/tests/test_power.py`

- [ ] **Step 1: Write the failing test**

`tests/test_power.py`:
```python
from vdm_spike.power import is_underpowered, min_detectable_effect


def test_mde_decreases_with_sample_size():
    small = min_detectable_effect(n=50)
    large = min_detectable_effect(n=5000)
    assert small > large > 0


def test_underpowered_flag():
    assert is_underpowered(n=20, target_effect=0.1) is True
    assert is_underpowered(n=5000, target_effect=0.5) is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_power.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `power.py`**

```python
from __future__ import annotations

from scipy.stats import norm


def min_detectable_effect(n: int, power: float = 0.8, alpha: float = 0.05) -> float:
    """Heuristic two-sample minimum detectable standardized effect (Cohen's d).

    mde = (z_alpha/2 + z_power) * sqrt(2/n). A first-order honesty signal for the
    spike, NOT a full high-dimensional power analysis.
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    return float((z_alpha + z_power) * (2.0 / n) ** 0.5)


def is_underpowered(n: int, target_effect: float, power: float = 0.8,
                    alpha: float = 0.05) -> bool:
    """True when the achieved sample size can't reliably detect `target_effect`."""
    return min_detectable_effect(n, power, alpha) > target_effect
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_power.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: minimum-detectable-effect / under-powered heuristic"
```

---

## Task 14: Labeled scenario builders

**Files:**
- Create: `vdm-spike/src/vdm_spike/scenarios.py`
- Create: `vdm-spike/tests/test_scenarios.py`

- [ ] **Step 1: Write the failing test**

`tests/test_scenarios.py`:
```python
from vdm_spike.scenarios import SCENARIO_NAMES, build_scenario


def test_all_scenarios_build_with_expected_shapes():
    for name in SCENARIO_NAMES:
        sc = build_scenario(name, n=120, seed=0)
        assert sc.name == name
        assert sc.baseline.dim == 384
        assert sc.current.dim == 384
        assert sc.query_vectors.shape[1] == 384
        assert isinstance(sc.expectation.fires, dict)
        assert len(sc.expectation.fires) >= 1


def test_model_swap_keeps_same_ids():
    sc = build_scenario("model-swap", n=80, seed=0)
    assert set(sc.baseline.ids) == set(sc.current.ids)


def test_broken_writes_injects_zero_vectors():
    import numpy as np
    sc = build_scenario("broken-writes", n=80, seed=0)
    zero = (np.linalg.norm(sc.current.vectors, axis=1) == 0).sum()
    assert zero > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_scenarios.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `scenarios.py`**

```python
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

    if name == "null-control":
        cur_docs = make_docs(seed=seed + 1, n=n, topic_mix=BALANCED_MIX)
        current = _snapshot(cur_docs, emb_a)
        fires = {"mmd": False, "classifier": False, "retrieval_rbo": False}

    elif name == "benign-growth":
        extra = make_docs(seed=seed + 2, n=n // 2, topic_mix=BALANCED_MIX,
                          id_prefix="g")
        cur_docs = base_docs + extra
        current = _snapshot(cur_docs, emb_a)
        fires = {"mmd": False, "classifier": False, "retrieval_rbo": False}

    elif name == "topic-shift":
        cur_docs = make_docs(seed=seed + 3, n=n, topic_mix=SHIFTED_MIX)
        current = _snapshot(cur_docs, emb_a)
        fires = {"mmd": True, "classifier": True, "retrieval_rbo": True}

    elif name == "model-swap":
        emb_b = Embedder(MODEL_B)
        current = _snapshot(base_docs, emb_b)  # same docs/ids, different model
        fires = {"mmd": True, "classifier": True, "norm_ks": True,
                 "retrieval_rbo": True}

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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_scenarios.py -v`
Expected: 3 passed (uses small `n=80..120`; models already cached).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: labeled scenario builders (null/benign/topic/model-swap/broken)"
```

---

## Task 15: Report (verdict table + PCA overlay plot)

**Files:**
- Create: `vdm-spike/src/vdm_spike/report.py`
- Create: `vdm-spike/tests/test_report.py`

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:
```python
import numpy as np

from vdm_spike.core import DetectorResult
from vdm_spike.features import fit_pca
from vdm_spike.report import format_verdict_table, save_overlay_plot


def test_format_verdict_table_marks_pass_and_fail():
    results = {"mmd": DetectorResult("mmd", 0.1, 0.01, True)}
    expectation = {"mmd": True}
    table, ok = format_verdict_table("topic-shift", results, expectation)
    assert "topic-shift" in table
    assert "mmd" in table
    assert ok is True

    bad_expectation = {"mmd": False}
    table2, ok2 = format_verdict_table("x", results, bad_expectation)
    assert ok2 is False


def test_save_overlay_plot_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 20)).astype(np.float32)
    b = rng.normal(loc=0.5, size=(200, 20)).astype(np.float32)
    pca = fit_pca(a, var=0.95)
    out = tmp_path / "overlay.png"
    save_overlay_plot(a, b, pca, str(out))
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `report.py`**

```python
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from vdm_spike.core import DetectorResult  # noqa: E402


def format_verdict_table(scenario: str, results: dict[str, DetectorResult],
                         expectation: dict[str, bool]) -> tuple[str, bool]:
    """Render one scenario's gated detectors; return (table_text, all_passed)."""
    lines = [f"Scenario: {scenario}"]
    all_ok = True
    for name, expected in expectation.items():
        res = results.get(name)
        actual = bool(res.fired) if res else False
        ok = actual == expected
        all_ok = all_ok and ok
        mark = "PASS" if ok else "FAIL"
        stat = f"{res.statistic:.4f}" if res else "n/a"
        p = f"{res.p_value:.4f}" if (res and res.p_value is not None) else "-"
        lines.append(
            f"  [{mark}] {name:18s} expected={expected!s:5s} "
            f"fired={actual!s:5s} stat={stat} p={p}"
        )
    return "\n".join(lines), all_ok


def save_overlay_plot(baseline: np.ndarray, current: np.ndarray, pca: PCA,
                      path: str) -> None:
    """2D PCA overlay: fit-on-baseline, transform current through the frozen basis."""
    pb = pca.transform(baseline)[:, :2]
    pc = pca.transform(current)[:, :2]
    plt.figure(figsize=(6, 6))
    plt.scatter(pb[:, 0], pb[:, 1], s=8, alpha=0.4, label="baseline")
    plt.scatter(pc[:, 0], pc[:, 1], s=8, alpha=0.4, label="current")
    plt.legend()
    plt.title("Embedding-space overlay (frozen PCA basis)")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: verdict-table formatting and PCA overlay plot"
```

---

## Task 16: Orchestrator + the success-criterion gate test

**Files:**
- Create: `vdm-spike/src/vdm_spike/run.py`
- Create: `vdm-spike/tests/test_gate.py`

- [ ] **Step 1: Write `run.py`**

```python
from __future__ import annotations

import numpy as np

from vdm_spike.core import DetectorResult, Snapshot
from vdm_spike.detectors import distribution as dist
from vdm_spike.detectors import retrieval as ret
from vdm_spike.features import fit_pca
from vdm_spike.ops import invalid_vectors
from vdm_spike.power import is_underpowered, min_detectable_effect
from vdm_spike.report import format_verdict_table, save_overlay_plot
from vdm_spike.scenarios import Scenario
from vdm_spike.store import PgVectorStore


def evaluate_scenario(sc: Scenario, store: PgVectorStore,
                      sample_k: int = 1000, query_k: int = 10,
                      plot_path: str | None = None) -> tuple[dict[str, DetectorResult], bool]:
    """Load both snapshots, read back over SQL, run detectors, return (results, gate_ok)."""
    store.ensure_schema()
    store.conn.execute("DELETE FROM items")
    store.conn.commit()
    store.load(sc.baseline, namespace="baseline")
    store.load(sc.current, namespace="current")

    base_s = store.sample("baseline", k=sample_k)
    curr_s = store.sample("current", k=sample_k)

    # Sanitize for distribution math (zero-norm rows survive; NaN/Inf cannot enter pgvector).
    pca = fit_pca(base_s.vectors, var=0.95)

    results: dict[str, DetectorResult] = {}
    results["centroid"] = dist.centroid_distance(base_s.vectors, curr_s.vectors)
    results["mmd"] = dist.mmd_rbf(base_s.vectors, curr_s.vectors)
    results["classifier"] = dist.classifier_drift(base_s.vectors, curr_s.vectors, pca)
    results["norm_ks"] = dist.norm_ks(base_s.vectors, curr_s.vectors)
    results["psi"] = dist.perdim_psi(base_s.vectors, curr_s.vectors, pca)

    base_hits, curr_hits, base_scores, curr_scores = [], [], [], []
    for qv in sc.query_vectors:
        bh = store.query(qv, namespace="baseline", k=query_k)
        ch = store.query(qv, namespace="current", k=query_k)
        base_hits.append([h.id for h in bh])
        curr_hits.append([h.id for h in ch])
        base_scores.extend([h.score for h in bh])
        curr_scores.extend([h.score for h in ch])
    results["retrieval_rbo"] = ret.retrieval_overlap(base_hits, curr_hits)
    results["retrieval_score_ks"] = ret.score_ks(
        np.array(base_scores), np.array(curr_scores)
    )

    results["ops_invalid"] = invalid_vectors(sc.current.vectors)

    if plot_path:
        save_overlay_plot(base_s.vectors, curr_s.vectors, pca, plot_path)

    _, gate_ok = format_verdict_table(sc.name, results, sc.expectation.fires)
    return results, gate_ok


def power_note(n: int, target_effect: float = 0.2) -> str:
    mde = min_detectable_effect(n)
    flag = "UNDER-POWERED" if is_underpowered(n, target_effect) else "ok"
    return f"n={n} mde={mde:.3f} (target {target_effect}) -> {flag}"


def main() -> int:
    import psycopg
    from pgvector.psycopg import register_vector

    from vdm_spike.scenarios import SCENARIO_NAMES, build_scenario

    dsn = "postgresql://vdm:vdm@localhost:5432/vdm"
    overall_ok = True
    with psycopg.connect(dsn) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        store = PgVectorStore(conn, dim=384)
        for name in SCENARIO_NAMES:
            sc = build_scenario(name, n=2000, seed=0)
            results, ok = evaluate_scenario(
                sc, store, plot_path=f"overlay_{name}.png"
            )
            table, _ = format_verdict_table(name, results, sc.expectation.fires)
            print(table)
            print("  " + power_note(min(sc.baseline.n, 1000)))
            print()
            overall_ok = overall_ok and ok
    print("GATE:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write the gate test**

`tests/test_gate.py`:
```python
import pytest

from vdm_spike.run import evaluate_scenario
from vdm_spike.scenarios import SCENARIO_NAMES, build_scenario
from vdm_spike.store import PgVectorStore


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_scenario_meets_expectation(conn, name):
    store = PgVectorStore(conn, dim=384)
    sc = build_scenario(name, n=600, seed=0)
    _, gate_ok = evaluate_scenario(sc, store, sample_k=600)
    assert gate_ok, f"scenario {name} did not meet its detector expectations"
```

- [ ] **Step 3: Run the gate**

Run: `uv run pytest tests/test_gate.py -v`
Expected: 5 passed — harmful scenarios fire, benign/null stay quiet.

> If `benign-growth` or `null-control` fails by firing, that is a real finding, not a test bug: tune detector thresholds/sample size and document it. If `topic-shift`/`model-swap` fail to fire at `n=600`, increase `n` and re-check `power_note` — an under-powered miss is itself a Phase 0 result worth recording.

- [ ] **Step 4: Run the full end-to-end harness**

Run: `uv run python -m vdm_spike.run`
Expected: per-scenario verdict tables, `overlay_*.png` files written, final line `GATE: PASS`.

- [ ] **Step 5: Run the full suite + lint**

Run: `uv run pytest -v && uv run ruff check`
Expected: all tests pass; no lint errors.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: orchestrator and end-to-end drift-signal gate (Phase 0 complete)"
```

---

## Task 17: Evidently cross-check (trust the math)

**Files:**
- Create: `vdm-spike/tests/test_evidently_crosscheck.py`

- [ ] **Step 1: Write the cross-check test**

`tests/test_evidently_crosscheck.py`:
```python
"""Confirm our MMD verdict agrees with Evidently's embedding-drift detector
on the same data, building trust that our implementation is correct."""

import numpy as np
import pandas as pd
from evidently.metrics import EmbeddingsDriftMetric
from evidently.report import Report

from vdm_spike.detectors.distribution import mmd_rbf


def _evidently_drift_detected(baseline: np.ndarray, current: np.ndarray) -> bool:
    cols = [f"e{i}" for i in range(baseline.shape[1])]
    ref = pd.DataFrame(baseline, columns=cols)
    cur = pd.DataFrame(current, columns=cols)
    report = Report(metrics=[EmbeddingsDriftMetric("emb")])
    report.run(
        reference_data=ref, current_data=cur,
        column_mapping=_mapping(cols),
    )
    result = report.as_dict()["metrics"][0]["result"]
    return bool(result["drift_detected"])


def _mapping(cols):
    from evidently import ColumnMapping
    cm = ColumnMapping()
    cm.embeddings = {"emb": cols}
    return cm


def test_mmd_agrees_with_evidently_on_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 16)).astype(np.float32)
    b = rng.normal(loc=0.6, size=(300, 16)).astype(np.float32)
    assert mmd_rbf(a, b, n_perm=200, seed=0).fired is True
    assert _evidently_drift_detected(a, b) is True


def test_mmd_agrees_with_evidently_on_no_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 16)).astype(np.float32)
    b = rng.normal(size=(300, 16)).astype(np.float32)
    assert mmd_rbf(a, b, n_perm=200, seed=0).fired is False
    assert _evidently_drift_detected(a, b) is False
```

- [ ] **Step 2: Run the cross-check**

Run: `uv run pytest tests/test_evidently_crosscheck.py -v`
Expected: 2 passed (our detector and Evidently agree on both shift and no-shift).

> Evidently's API surface shifts across versions. If imports fail, check the installed version's embedding-drift API and adjust the import/mapping; the *assertion* (agreement on shift / no-shift) is what matters, not the exact import path.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: cross-check MMD detector against Evidently"
```

---

## Self-Review

**1. Spec coverage** (against [../specs/2026-06-15-vdm-phase0-spike-design.md](../specs/2026-06-15-vdm-phase0-spike-design.md)):
- §3 success criteria → Task 16 gate test + run.py `GATE` line + `power_note` (statistical honesty). ✓
- §4 modules: corpus(T4), embed(T5), store(T6), scenarios(T14), distribution(T7–10), retrieval(T11), power(T13), report(T15), run(T16). ✓ (`ops.py` added as T12 — it was implied by the `broken-writes` scenario and Family C; surfaced explicitly.)
- §5 scenario set: all five built in T14, gated in T16. ✓
- §6 tooling: uv/pytest/ruff (T1), Docker pgvector (T2), Evidently cross-check (T17). ✓
- §7 testing approach: TDD throughout; MMD/classifier permutation, RBO drop, NaN/zero detection, under-powered flag all asserted. ✓
- §8 out-of-scope: no adapter contract/UI/API/scheduler/alerting — confirmed none added. ✓

**2. Placeholder scan:** No TBD/TODO; every code step contains complete, runnable code. ✓

**3. Type consistency:** `Snapshot`(ids, vectors, metadata, .n/.dim), `Expectation`(fires), `DetectorResult`(name, statistic, p_value, fired, detail), `QueryHit`(id, score, rank), `Scenario`(name, baseline, current, query_vectors, expectation) used consistently across T3, T6, T7–12, T14, T15, T16. Detector names (`centroid`, `mmd`, `classifier`, `norm_ks`, `psi`, `retrieval_rbo`, `retrieval_score_ks`, `ops_invalid`) match between `scenarios.py` expectations and `run.py` results keys. ✓

**Deviation noted:** Per-dim PSI uses the standard >0.2 threshold convention rather than BH-corrected per-dim KS (design §4.1 mentioned BH). This is an intentional Phase-0 simplification — the gate does not depend on per-dim localization — documented inline at Task 10. Not a coverage gap.
