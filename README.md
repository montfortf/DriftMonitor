# vdm-spike — VDM Phase 0 drift-signal validation

A pure-Python harness that answers one question with evidence, before any ecosystem investment:

> **Does the drift signal fire on harmful shifts and stay quiet on benign growth — legibly enough to trust?**

See the design spec: [docs/specs/2026-06-15-vdm-phase0-spike-design.md](docs/specs/2026-06-15-vdm-phase0-spike-design.md).

## Verdict: ✅ the core drift signal is validated

The distribution detectors **cleanly separate harmful drift from benign change**, and every harmful scenario is caught. The headline numbers (production run, `n=2000`):

| Scenario | MMD (p) | Classifier AUC | Retrieval RBO | Ops | Caught? |
|---|---|---|---|---|---|
| null-control (resample) | 0.71 — quiet | 0.49 — quiet | (informational) | — | correctly quiet |
| benign-growth (+50%) | 0.96 — quiet | 0.47 — quiet | (informational) | — | correctly quiet |
| topic-shift | **0.005 — fires** | **0.67 — fires** | **fires** | — | ✅ |
| model-swap | **0.005 — fires** | **0.97 — fires** | **fires** | — | ✅ |
| broken-writes | (informational) | (informational) | (informational) | **fires** | ✅ |

MMD and the domain-classifier are silent on sampling noise and loud on real drift — exactly the success criterion. The math is cross-validated against Evidently (`tests/test_evidently_crosscheck.py`): our MMD verdict agrees with Evidently's embedding-drift detector on both shift and no-shift.

## Findings (which signals to trust, and when)

These came directly out of the gate and shape Phase 1. The gate initially failed on three cells; investigation showed two true findings (not code bugs), and the gated expectations were corrected to the scientifically-valid subset. **No detector logic or threshold was changed** — the demoted signals are still computed and printed as `[info]`.

### Finding A — Retrieval-RBO is only valid on an *incrementally evolving* index
`retrieval_rbo` fired on `null-control` and `benign-growth`. Root cause: those scenarios build `current` as an independent resample / +50% growth, so the specific documents a query returns legitimately differ — RBO (an identity-based rank overlap) reads as drift even though the distribution is unchanged. RBO is a meaningful no-change signal **only when the index evolves incrementally** (shared document identity). This validates **PRD v0.2 §7.3**: retrieval drift should be judged by **rate-of-change, not an absolute threshold**. → Gated only for the harmful scenarios (where it is unambiguous); informational elsewhere.

### Finding B — Norm-KS is a weak channel for *similar* model pairs
`norm_ks` missed the `model-swap` (p=0.99). The two same-dimension MiniLM models (`all-MiniLM-L6-v2` → `all-MiniLM-L12-v2`) have near-identical L2-norm distributions, so norm-shift can't see the swap. But the swap is robustly caught by MMD + classifier + retrieval, so norm-KS is a **per-model-pair-dependent** signal, not a reliable standalone swap detector. → Demoted to informational for `model-swap`.

### Finding C — Distribution detectors are robust to benign resampling
A useful positive result: MMD and classifier-AUC stayed quiet on both an independent resample (`null-control`) and 50% same-distribution growth (`benign-growth`), confirming they distinguish sampling noise from genuine drift — the hardest part of avoiding false-positive alert fatigue.

## Phase 1 implications
1. Model scenarios as an **incrementally evolving index** (persistent document identity) so retrieval-RBO and rate-of-change alerting are properly testable.
2. Treat **norm-KS as informational/corroborating**, not a primary swap gate; primary swap detection = MMD + classifier + retrieval.
3. Implement retrieval drift with **rate-of-change / control-chart** semantics (PRD §7.3), not absolute thresholds.
4. The `store.py` read-path shape (`count` / `sample` / `query`) seeds the Phase 1 `VectorStoreAdapter` contract.

## Architecture
Synthetic topical corpus → real sentence-transformer embeddings → loaded into Dockerless local pgvector → read back over SQL → distribution + retrieval + operational detectors → verdict table + PCA overlay. Modules under `src/vdm_spike/`: `corpus`, `embed`, `store`, `scenarios`, `detectors/{distribution,retrieval}`, `ops`, `power`, `report`, `run`.

## Setup & run
```bash
# 1. Provision local Postgres 16 + pgvector (idempotent; builds pgvector from source for pg16)
./scripts/setup_db.sh

# 2. Install deps
uv sync

# 3. Run the full test suite (45 tests; needs the DB up)
uv run pytest -q

# 4. Run the end-to-end gate harness (prints verdict tables + writes overlay_*.png, then GATE: PASS/FAIL)
uv run python -m vdm_spike.run
```

DSN: `postgresql://vdm:vdm@localhost:5432/vdm` (override with `VDM_DSN`).

## Honesty notes
- Every run prints a **power note** (minimum detectable effect / under-powered flag); a quiet result on an under-powered run is never a clean pass.
- The embedding-space overlay is **fit-on-baseline / transform-current** through a frozen PCA basis (axis-comparable), per PRD §11.
- `broken-writes` injects **zero-norm** vectors (pgvector rejects NaN/Inf at insert); NaN/Inf detection is unit-tested directly on the in-memory array.

---

# Phase 1.1 — Capability Negotiation (proven in Python)

Spec: [docs/specs/2026-06-15-vdm-phase1.1-capability-negotiation-design.md](docs/specs/2026-06-15-vdm-phase1.1-capability-negotiation-design.md).

## Verdict: ✅ capability negotiation works; the architecture is sound to port to Option A

A single `VectorStoreAdapter` contract (`adapters/base.py`) plus a negotiator (`negotiation.py`) drives every store. The gate runs **4 adapters × 5 scenarios** and the negotiator picks the richest viable plan per store:

| Adapter | SDK | Declared caps | Plan | Families run |
|---|---|---|---|---|
| pgvector | psycopg | returns+unbiased+query | **FULL** | distribution + retrieval + ops |
| Qdrant (in-memory) | qdrant-client | returns+unbiased+query | **FULL** | distribution + retrieval + ops |
| fake-query-only | — | query only | **QUERY** | retrieval + ops (distribution UNAVAILABLE, flagged) |
| fake-minimal | — | none | **MINIMAL** | ops only (distribution + retrieval UNAVAILABLE, flagged) |

Two independent SDKs (pgvector, Qdrant) reaching FULL proves the contract is genuinely store-agnostic, not pgvector-shaped. `run.py` prints an explicit **fidelity caveat** for every degraded plan, so a `PASS` on a MINIMAL store reads as "no detectable ops anomaly," never "no drift." The **conformance suite** (`conformance.py`) verifies each adapter's declared capabilities against actual behavior and is proven to have teeth — a deliberately-dishonest adapter is rejected.

## Findings (carried forward)

- **Finding A — RESOLVED.** Retrieval-RBO is now a properly-gated signal: scenarios model an **incrementally-evolving index** (null = identical corpus; benign = retained baseline + ~10% growth; harmful = retained + dominant injection / re-embed). RBO stays quiet on null/benign and fires on harmful — the Phase 0 demotion was an artifact of the old independent-resample scenarios.
- **Finding B — deepened.** Norm-KS is **structurally uninformative for normalized embedding models**: sentence-transformers `all-*`/bge/gte models emit unit-norm vectors (L2-norm=1, std≈0), so a model swap shows no norm shift (KS p≈1.0). Norm-KS stays **informational, not gated**; the swap is robustly caught by MMD + classifier + retrieval. Norm-drift only helps for un-normalized embedding sources.
- **Finding D (new) — scenario dilution & detector power.** A topic-shift that *retains 100% of baseline* and adds an equal volume of shifted docs is only ~50% drift, which sits exactly on the domain-classifier's 0.55 AUC fire-threshold — genuinely borderline (half of "current" is identical to baseline). The harmful scenario must represent a *dominant* shift (new topic ≈75% of the corpus, AUC ~0.62) to be unambiguous. Detector thresholds were never tuned; the scenario was made to represent real drift.
- **Determinism.** The gate analyzes the **full loaded population** so it is reproducible (both adapters' `sample()` use random subsampling, which made borderline scenarios flaky). Bounded-sampling power remains a separate concern, covered by `power.py` (MDE).

## What's deliberately still out of scope
No TypeScript / control-plane port, no API server, no UI, no scheduler, no baseline persistence, no alerting, no real Pinecone/Vertex, no schema-drift family. Next slice (per PRD §12.5): port the proven contract to **Option A** (TS control plane + this Python compute).
