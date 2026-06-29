# VDM Phase 0 Spike — Design Spec

**Date:** 2026-06-15
**Status:** Approved (design); pending implementation plan
**Parent spec:** [PRD-v0.2.md](../PRD-v0.2.md) (Draft v0.2), §13.1 Phase 0
**Scope:** Single implementation increment — the de-risking spike that precedes Phase 1.

---

## 1. Purpose

Phase 0 exists to answer **one question with evidence, before any ecosystem investment**:

> Does the drift signal fire on **harmful** shifts and stay **quiet** on **benign** corpus growth — legibly enough to trust?

This is the gate from §13.1 of the parent spec. If the spike cannot cleanly separate harmful drift from benign growth, the whole product premise is in doubt and Phase 1 should not start.

## 2. Decisions locked during brainstorming

| Decision | Choice | Rationale |
|---|---|---|
| Starting increment | Phase 0 spike (not Phase 1 skeleton) | Smallest thing that de-risks the project; matches parent-spec recommendation. |
| Language | Pure Python | Maximum statistical-library maturity for the math being validated. |
| Store | pgvector only | One store; capability negotiation is out of scope here. |
| Corpus | Synthetic, scripted, **labeled** shifts | Fully reproducible; lets us *assert* "fires on harmful, quiet on benign". |
| Embeddings | **Real** model (local `sentence-transformers`) | "Model swap" becomes a literal second model — the headline silent-failure case. |
| Read path | **Real** pgvector via Docker, read back over SQL | Exercises a real store read path; the thin `store.py` shape seeds the Phase 1 pgvector adapter. |

## 3. Success criteria (the gate)

The spike **passes** when, across the scenario set (§5):

1. **Harmful** scenarios (`topic-shift`, `model-swap`, `broken-writes`) fire clearly in the expected detector families.
2. **Benign** scenarios (`benign-growth`) and the `null-control` stay quiet — this is the false-positive guard from parent-spec §7.3.
3. The run output makes the result **legible**: a verdict table (scenario × detector → fired? matches expectation?) plus a PCA overlay plot that shows the corpus moving.
4. Every run reports **statistical honesty**: sample size, and a minimum-detectable-effect / under-powered flag (parent-spec §8.5). A "quiet" result on an under-powered run is never reported as a clean pass.

A failing gate (detectors can't separate harmful from benign, or only do so when over-powered) is itself a valid, valuable outcome — it stops Phase 1.

## 4. Architecture

Control flow: `run.py` builds each labeled scenario → loads baseline + current into pgvector → reads samples/queries back through `store.py` → runs detectors → `report.py` renders the verdict and plot. Each module has one job and is testable in isolation.

```
vdm-spike/
├─ docker-compose.yml            # pgvector (postgres + pgvector extension)
├─ pyproject.toml                # uv-managed
└─ src/vdm_spike/
   ├─ corpus.py      # synthetic topical corpus + shift operators (deterministic seed)
   ├─ embed.py       # model A (baseline) & model B (swap) — local sentence-transformers
   ├─ store.py       # pgvector read path: count / sample(reservoir) / query(cosine)
   ├─ scenarios.py   # labeled scenario builders → (baseline, current, expectation)
   ├─ detectors/
   │   ├─ distribution.py
   │   └─ retrieval.py
   ├─ power.py       # minimum-detectable-effect; under-powered flag
   ├─ report.py      # verdict table + PCA overlay plot
   └─ run.py         # orchestrator
```

### 4.1 Module responsibilities

- **`corpus.py`** — Generates a synthetic corpus of short text docs organized into topical clusters from a fixed seed. Exposes shift operators: `grow_same_distribution(n)`, `shift_topics(mix)`, and a hook for re-embedding (model swap) and write-corruption handled downstream. Deterministic: same seed → same corpus.
- **`embed.py`** — Wraps two local `sentence-transformers` models: model A (baseline embedder) and model B (a *different* model used to simulate a silent model upgrade). Returns `numpy` arrays. No network at run time after first model download.
- **`store.py`** — Thin pgvector read path. `count(scope)`, `sample(scope, k)` (reservoir/`ORDER BY random()` bounded by budget), `query(vector, k)` (cosine top-k). Deliberately *not* the full adapter contract — but its method shapes mirror the Phase 1 `VectorStoreAdapter` so the code carries forward.
- **`scenarios.py`** — Each scenario builder returns `(baseline_snapshot, current_snapshot, Expectation)` where `Expectation` declares which detector families *should* fire and which *should* stay quiet. The scenarios double as test fixtures.
- **`detectors/distribution.py`** — Centroid distance (cosine + Euclidean); MMD (RBF kernel, bandwidth via median heuristic **frozen on the baseline sample**, significance via permutation test, default 200 permutations); domain-classifier AUC (classifier on **PCA-reduced** features, k-fold cross-validated, with a permutation null → calibrated p-value); norm-distribution shift (KS on vector norms); per-dimension PSI on the **frozen PCA basis** with Benjamini–Hochberg correction + share-drifted.
- **`detectors/retrieval.py`** — The headline family. A fixed query set run against baseline vs current index: top-k ID overlap (Jaccard + Rank-Biased Overlap), and score-distribution drift (KS on similarity scores).
- **`power.py`** — For the achieved sample size `n` and dimensionality `d`, computes the minimum detectable effect at the configured power (default 0.8) and flags runs that are under-powered for the target effect.
- **`report.py`** — Verdict table (per scenario × detector: fired? expected? ✓/✗) and a PCA overlay plot fitted on the baseline and `transform`-ing the current sample through the **frozen basis** (parent-spec §11) so the overlay is axis-comparable.
- **`run.py`** — Orchestrates the full sweep and prints the pass/fail gate summary.

## 5. The scenario set (validation matrix)

| Scenario | Construction | Expected detector behavior |
|---|---|---|
| `null-control` | Resample the same distribution with a different seed | **All quiet** — primary false-positive guard |
| `benign-growth` | Add same-distribution docs to the index | Distribution **quiet or mild**; retrieval **stable** |
| `topic-shift` | Change the topic mixture of current vs baseline | **Fires**: MMD, classifier-AUC, retrieval overlap/score |
| `model-swap` | Re-embed the *same* docs with model B | **Fires hard**: norm-KS + MMD (the silent-upgrade headline) |
| `broken-writes` | Inject NaN / zero vectors into current | **Fires**: operational checks (null/NaN), integrity |

## 6. Tech & tooling

- Python 3.11+, `uv` for environment/deps, `pytest`, `ruff`.
- Core: `numpy`, `scipy`, `scikit-learn`, `sentence-transformers`, `psycopg[binary]`, `pgvector`, `matplotlib` (for the overlay plot).
- Optional trust cross-check: validate **one** detector (e.g. MMD or classifier drift) against **Evidently** to confirm our implementation agrees with an established library (parent-spec principle: "wrap proven detectors so the math is trusted from day one").
- pgvector via `docker-compose` (`pgvector/pgvector` image).

## 7. Testing approach

TDD, with the scenarios as fixtures. Tests assert detector verdicts against each scenario's declared `Expectation`:
- `MMD permutation p < 0.05` on `model-swap` and `topic-shift`; `p > 0.05` on `null-control`.
- Classifier-AUC permutation p significant on harmful, not on `null-control`/`benign-growth`.
- Retrieval overlap (RBO) drops materially on `topic-shift`/`model-swap`, stable on `benign-growth`.
- NaN/zero detection catches 100% of injected bad vectors in `broken-writes`.
- `power.py` correctly flags a deliberately tiny-sample run as under-powered.

The `store.py` read path is tested against the real Dockerized pgvector instance (round-trip: load → sample → fetch returns consistent records).

## 8. Out of scope (YAGNI for Phase 0)

No adapter abstraction or plugin registry; no capability negotiation (pgvector only); no API server; no React/Streamlit UI (CLI + saved plot only); no scheduler; no alerting/policy engine; no baseline persistence beyond a run; no cost model; no TypeScript. `store.py` is shaped to seed the Phase 1 adapter but implements no formal contract.

## 9. Deliverable

A runnable `vdm-spike` that, via a single command, executes all scenarios and prints the gate verdict plus saves the PCA overlay plot — producing the evidence to decide go/no-go on Phase 1.
