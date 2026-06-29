# TECHNICAL DESIGN SPECIFICATION
## vector-drift-monitor
**A store-agnostic drift & retrieval-quality monitor for any vector database**
Pluggable Adapters · Capability Negotiation · Drift Engine · UI · Stack Options

Prepared for Montfort Fernando
**Draft v0.2 · Design & brainstorming spec**

> **Changelog v0.1 → v0.2**
> - Corrected the capability framing: Pinecone serverless *can* return raw vectors via `list` + `fetch`; the only true query-only store in the matrix is Vertex AI Vector Search. The narrative wedge has been re-centered on **retrieval-quality drift over a live connection** (the genuinely novel part) rather than embedding-distribution math (the commodity part). (§1, §6, §15)
> - Added **§7.3 — Distinguishing healthy change from harmful drift**, the core product problem.
> - Added **§8.5 — Statistical power & detector parameters** (MMD kernel/bandwidth policy, minimum detectable effect vs. sample budget).
> - Added **§10.4 — Cost model** and read-unit budgeting; cost surfaced at registration.
> - Clarified the **embedding-space projection** must be fit-on-baseline / transform-current (PCA default; parametric UMAP optional). (§11)
> - Labeled **write-tap** explicitly as the one non-agnostic strategy. (§7.1)
> - Committed the API to **REST-only** (dropped GraphQL). (§10.3)
> - Added handling of **connection-secret management**, **golden-label staleness**, **conformance CI cost**, and an expanded **landscape** (Ragas/TruLens/LangSmith). (§7.2, §8.2, §9.2, §2.1)

---

## 1. Executive summary

vector-drift-monitor (VDM) is an open-source service that continuously watches a vector index for the things that silently degrade RAG and semantic-search systems: retrieval-quality decay, embedding distribution drift, index-health regressions, and silent embedding-model swaps. The differentiator is that it is **store-agnostic by design** — you point it at Pinecone, Qdrant, Weaviate, Milvus, Chroma, pgvector, LanceDB, Vespa, Vertex AI Vector Search, Turbopuffer, Redis, MongoDB Atlas, Elasticsearch/OpenSearch, or a store that does not exist yet, through a thin pluggable adapter.

The central engineering insight is that vector stores disagree on the capabilities that matter for monitoring — and the one that varies most is **how (and whether) you can get a representative sample of the index back out**. Most stores (pgvector, Qdrant, Chroma, Milvus, Weaviate, LanceDB, Elastic, Mongo, Redis) let you scan or list-and-fetch raw embeddings directly. A few are query-first: **Vertex AI Vector Search** will not hand back raw vectors at all, and others (Pinecone serverless, Cloudflare Vectorize) *can* return vectors by ID but make an *unbiased* bulk sample awkward. VDM treats this as a first-class **capability-negotiation problem**: each adapter declares what it can do, and the drift engine picks the richest analysis the store can actually support, degrading gracefully from full distribution analysis to query-result-based drift when an unbiased raw-vector sample is out of reach.

### The one-line pitch
> **“Datadog for your vector index”** — a drift and **retrieval-quality** monitor that plugs into any vector store through a conformance-tested adapter, and tells you the moment your index stops behaving like it did yesterday.

### Where the real novelty is
The drift *math* is well understood and commoditized (Evidently, Alibi-Detect). What no existing tool does is **watch retrieval quality — what the user actually feels — over a live connection to an arbitrary production store, on a schedule, with honest capability negotiation.** For the ~10 stores that readily return vectors, distribution drift is table stakes a determined user could assemble from Evidently in an afternoon; VDM's defensible value is the *connector + retrieval lens + conformance ecosystem*, and that value is **highest exactly on the query-first stores** (Vertex-class) that defeat every DataFrame-based tool. The pitch leads with retrieval-quality drift; distribution drift is supporting evidence, not the headline.

This document is a brainstorming-grade design spec: it defines the architecture, the adapter contract, the capability model, the drift-metric catalog, the plugin and conformance system, the UI, the cost model, and four concrete tech-stack options with conviction ratings and a recommendation.

---

## 2. The gap this fills

Embedding drift is now widely acknowledged as an operational metric, not a research curiosity. Recent 2026 vector-database guidance explicitly recommends monitoring embedding distribution shift — norms, cosine-similarity histograms, centroid movement — and retrieval stability over time, calling drift “the hidden” failure mode. The detection methods are well understood (Evidently’s open-source library implements Euclidean/cosine centroid distance, a domain classifier, share-of-drifted-components, and Maximum Mean Discrepancy, with UMAP-based visualization).

What is missing is the connective tissue. Every mature tool assumes the embeddings have already been extracted into a DataFrame or pushed to a platform. None of them plug directly into an arbitrary production vector store, negotiate what that store can return, sample from it on a schedule, and watch **retrieval quality** — not just embedding statistics — over time. That connector-plus-capability-negotiation layer, delivered with a retrieval lens, is the wedge.

### 2.1 Landscape — what exists and where it stops

| Tool / approach | What it does well | Why it is not this |
|---|---|---|
| **Evidently AI (OSS)** | Embedding drift via centroid distance, MMD, domain-classifier AUC, share-of-drifted-components, UMAP viz | DataFrame-in. You bring the embeddings; it has no concept of a live vector store or sampling from one. |
| **Arize / Phoenix** | Euclidean centroid drift, embedding monitors, alerting | Push/SDK-based platform. You log data to it; not a connector to your production index. |
| **NannyML / Alibi-Detect** | Strong statistical drift + performance estimation | Tabular/ML-feature focused; not vector-store-aware, no retrieval-quality lens. |
| **WhyLabs / whylogs** | Data profiling & drift on logged profiles | Profile-logging model; you instrument the pipeline, not the store. |
| **Ragas / TruLens / LangSmith** | RAG-output evaluation (faithfulness, answer relevance, retrieval eval), tracing | Evaluate the *application's responses/traces*, not the *store's index health*; require LLM-as-judge and instrumented app traffic. Store-blind. Reinforces the wedge rather than competing for it. |
| **Vector DB native stats** | Some stores expose counts, basic index stats, per-index dashboards | No cross-store standard, no drift semantics, no retrieval-quality regression, no fleet UI for it. |

### The defensible wedge
Existing drift tooling is **push-based and store-blind**. VDM is **pull-based and store-aware**. The moat is the **adapter contract plus a published conformance suite**: once a clean, capability-negotiating contract exists and vendors/community can certify their own adapters against it, VDM becomes the default monitoring layer for the whole vector-store category rather than one more DataFrame library. The retrieval-quality lens is what makes it indispensable even where the math is commodity.

---

## 3. Design goals & principles

- **Store-agnostic by contract, not by special-casing.** Support for a new store = one adapter implementing a small interface and passing the conformance suite. No core changes.
- **Capability negotiation over assumption.** Adapters declare capabilities; the engine selects the richest viable analysis and is explicit in the UI about what it could and could not measure for a given store.
- **Graceful degradation.** A store that will not return raw vectors still gets meaningful monitoring — retrieval-quality drift, cardinality, latency, and metadata/schema drift — instead of nothing.
- **Retrieval quality is the headline signal.** Distribution drift is often benign (the corpus legitimately grows); a *drop in retrieval quality against a labeled set* is what is actionable. Alerting defaults reflect this priority (§7.3).
- **Reference math, not reinvented math.** Use established detectors (PSI, MMD, domain-classifier AUC, centroid distance, RBO) with **explicitly pinned parameters** (§8.5) so results are trusted and comparable to Evidently/Arize conventions.
- **Sampling is bounded and honest.** Never assume you can scan a billion vectors. Every method has a sampling budget, and any sampling bias (e.g. query-seeded sampling) is surfaced as a confidence caveat — and so is insufficient statistical power.
- **Cost is bounded and honest.** Monitoring runs against the user's *production* store and costs them money on metered stores. Every run has a read-unit/$ budget, surfaced at registration (§10.4).
- **Observable itself.** VDM emits OpenTelemetry spans/metrics so it composes with existing enterprise observability, rather than being one more silo.
- **Privacy- and secret-aware.** Embeddings can be partially inverted to source content; connection credentials grant read access to production data. Baseline snapshots and store secrets are access-controlled and encryptable; raw-vector retention is configurable and can be disabled in favor of statistics-only baselines (§7.2).
- **OSS-first, ecosystem-driven.** Apache-2.0 core, first-party adapters for the top stores, and a contribution path that lets the community and vendors add the long tail.

---

## 4. System architecture

VDM separates the **control plane** (adapters, API, scheduling, UI) from the **compute plane** (sampling and drift math). This keeps the store-integration surface thin and lets the statistically heavy work scale independently.

### 4.1 Component responsibilities

| Component | Responsibility |
|---|---|
| **Adapter layer** | Pluggable connectors implementing the `VectorStoreAdapter` contract. One per store. Declares capabilities; provides sampling, fetch-by-id, query, count, and probe. |
| **Capability negotiator** | Reads each adapter’s declared capabilities and selects a “drift plan” — the set of metrics that the store can actually support. |
| **Collector / sampler** | Scheduled workers that pull samples, run query sets, and probe latency/health according to the drift plan and a sampling + cost budget. |
| **Baseline store** | Versioned reference snapshots (sampled vectors and/or precomputed statistics), query sets, and the time-series of computed metrics. |
| **Drift engine** | Computes the metric catalog (distribution, retrieval, operational, schema) against the active baseline, with pinned detector parameters. The statistically heavy component. |
| **Policy & alerting** | Threshold/control-chart evaluation, suppression windows, severity tiers, healthy-change discrimination, and fan-out to webhooks, Slack, PagerDuty, email, OTel, or a GitHub issue. |
| **API server** | REST surface for registration, runs, metrics, baselines, query sets, and policies; plus an SSE event stream for the UI. |
| **Dashboard (UI)** | Fleet overview, drift timelines, embedding-space projection, per-dimension heatmap, retrieval-drift panel, index-health panel, and policy editor. |
| **Scheduler** | Cron/interval orchestration of collection and baseline recomputation per store, within cost budgets. |

### 4.2 Data flow

```
  +-------------+   declares caps   +----------------------+
  |  Adapter(s) | ----------------> |  Capability          |
  |  pinecone.. |                   |  Negotiator -> plan  |
  +------+------+                   +----------+-----------+
         |  sample / query / probe             |  drift plan
         v                                     v
  +-------------+   samples + hits  +----------------------+
  |  Collector  | ----------------> |    Drift Engine      |
  | (scheduled) |                   |   PSI . MMD . RBO    |
  +-------------+                   +----------+-----------+
         ^  baseline                           |  metrics
         |                                     v
  +-------------+ timeseries/alerts +----------------------+
  |  Baseline   | <---------------- |  Policy & Alerting   |
  |    Store    |                   +----------+-----------+
  +------+------+                              |  events
         |                                     v
         |       +------------+    SSE   +--------------+
         +------ | API server | -------> |  Dashboard   |
                 +------------+          +--------------+
```
*High-level data flow. Adapters are the only store-specific code; everything downstream is generic.*

---

## 5. The vector-store adapter contract

This is the heart of genericity. An adapter is the only store-specific code in the system. It implements a small interface and declares, honestly, what it can do. The contract is shown in TypeScript for clarity but is language-neutral — a Python or Go adapter implements the same shape.

### 5.1 Core interface

```ts
interface VectorStoreAdapter {
  // identity & shape of the index
  describe(): Promise<StoreDescriptor>;
    // { name, dimension, metric, indexType,
    //   namespaces[], approxVectorCount }

  // honest declaration of what this store supports
  capabilities(): Capabilities;

  // counting (optionally by namespace/filter)
  count(scope: Scope): Promise<number>;

  // sampling — the critical, capability-gated method
  sample(scope: Scope, opts: SampleOptions)
    : AsyncIterable<VectorRecord>;

  // pull specific records for tracked-cohort drift
  fetchByIds(ids: string[], scope: Scope)
    : Promise<VectorRecord[]>;

  // run a known query to measure retrieval drift
  query(req: QueryRequest): Promise<QueryHit[]>;

  // latency + reachability probe
  probe(): Promise<ProbeResult>;
}

type VectorRecord = {
  id: string; vector?: number[];        // vector may be absent
  metadata?: Record<string, unknown>; namespace?: string;
};
type QueryHit = { id: string; score: number; rank: number;
                  metadata?: Record<string, unknown> };
```

### 5.2 The capability declaration

Capabilities are **declared, not inferred at runtime**, and are verified by the conformance suite (§9). The engine reads them to choose a drift plan. Note the addition of `unbiasedSample` — the distinction that actually drives plan selection (see §6).

```ts
interface Capabilities {
  returnsVectors: boolean;     // can sample/fetch return raw embeddings?
  unbiasedSample: boolean;     // can we obtain a representative (non-query-biased)
                               //   sample of vectors? (fullScan OR native random
                               //   OR id-listing+fetch). This — not returnsVectors —
                               //   gates distribution drift.
  fullScan: boolean;           // deterministic full iteration?
  randomSample: 'native' | 'reservoir'
               | 'query-seeded' | 'none';
  idListing: boolean;          // can enumerate IDs?
  metadataFilter: boolean;     // server-side filter support?
  countByFilter: boolean;
  liveQuery: boolean;          // can run search queries?
  maxBatch: number;            // max records per page
}
```

### 5.3 What each method requires from a store

| Method | Needs | If unavailable |
|---|---|---|
| `describe()` | Basic index metadata | Adapter supplies best-effort defaults |
| `count()` | Cardinality endpoint | Estimate via listing or skip cardinality drift |
| `sample()` | Scan, ID-listing, native random, or query-seeding | Fall back to query-only drift plan |
| `fetchByIds()` | Point lookup returning vectors | Tracked-cohort drift disabled |
| `query()` | Standard similarity search | Retrieval-quality drift disabled (rare) |
| `probe()` | Any cheap call | Latency drift disabled |

---

## 6. Capability negotiation & the store matrix

Given an adapter’s capabilities, the engine selects one of three escalating drift plans. The richest plan a store can support is chosen automatically, and the UI labels which plan is active so the operator knows the fidelity they are getting.

**The key correction from v0.1:** the gate for distribution drift is **`unbiasedSample`**, not merely `returnsVectors`. A store can return vectors by ID (so `returnsVectors === true`) yet have no way to draw a representative sample (`unbiasedSample === false`) — in which case distribution claims would be biased and we route to the QUERY plan for the distribution family while still using fetched vectors for tracked-cohort drift.

```ts
function selectDriftPlan(c: Capabilities): DriftPlan {
  const canDistribution =
      c.returnsVectors && c.unbiasedSample;   // unbiased sample, not just readback

  if (canDistribution) return 'FULL';   // distribution + retrieval + ops + schema
  if (c.liveQuery)      return 'QUERY';  // retrieval + ops + schema (+ tracked-cohort
                                         //   drift if fetchByIds returns vectors)
  return 'MINIMAL';                      // ops + schema only
}
```
*Plan selection. FULL unlocks unbiased embedding-distribution drift; QUERY and MINIMAL degrade gracefully. Stores that return vectors by ID but cannot be sampled without bias get QUERY-plan distribution metrics plus tracked-cohort drift on a fixed ID set.*

### 6.1 Reference capability matrix

*Illustrative, design-time reference — exact behavior varies by version, tier, and SDK, so each adapter’s declared capabilities are verified by the conformance suite rather than trusted from this table.* The decisive column is **“Unbiased sample”**: only stores that can produce a representative vector sample qualify for the FULL distribution plan. Note that **Pinecone serverless and Cloudflare Vectorize *do* return vectors** (via `list` IDs + `fetch`), so they support **tracked-cohort drift and ID-listing-based sampling** — they are *not* purely query-only. **Vertex AI Vector Search is the one genuinely query-only store here.**

| Store | Returns vectors | Unbiased sample | Full scan | ID listing | Live query | Sampling strategy | Plan |
|---|---|---|---|---|---|---|---|
| pgvector | Yes | Yes | Yes | Yes | Yes | native / reservoir | FULL |
| Qdrant | Yes | Yes | Yes | Yes | Yes | native | FULL |
| Weaviate | Yes | Yes | Yes | Yes | Yes | cursor | FULL |
| Milvus / Zilliz | Yes | Yes | Yes | Yes | Yes | iterator | FULL |
| Chroma | Yes | Yes | Yes | Yes | Yes | scan | FULL |
| LanceDB | Yes | Yes | Yes | Yes | Yes | native | FULL |
| Elastic / OpenSearch | Yes | Yes | Yes | Yes | Yes | scroll | FULL |
| MongoDB Atlas | Yes | Yes | Yes | Yes | Yes | scan | FULL |
| Redis | Yes | Yes | partial | Yes | Yes | scan | FULL |
| Turbopuffer | Yes | Yes | partial | Yes | Yes | id-listing + fetch | FULL |
| **Pinecone (serverless)** | **Yes** | **Approx** | No | paginated (`list`) | Yes | **id-listing + fetch** (preferred) / query-seeded | FULL\* / QUERY |
| **Cloudflare Vectorize** | **Yes** | **Approx** | No | partial | Yes | id-listing + fetch / query-seeded | FULL\* / QUERY |
| **Vertex AI Vector Search** | **No** | **No** | No | No | Yes | query-seeded | QUERY |

\* *Pinecone/Vectorize reach a FULL-equivalent distribution plan when `list` + `fetch` yields a representative ID sample within budget; if ID enumeration is namespace-skewed or budget-bounded below a representative threshold, the engine falls back to QUERY-plan distribution metrics and flags reduced confidence.*

### Why query-seeded sampling still matters
For the genuinely query-only stores (**Vertex AI Vector Search**), and as a fallback when ID-listing can't yield a representative sample within budget, VDM issues a panel of seed query vectors and collects the returned neighbors as a pseudo-sample. This is biased toward dense regions of the space, so it is **never presented as an unbiased distribution** — it powers retrieval-quality and score-distribution drift, and is flagged with a sampling-confidence indicator in the UI. It is the difference between “no monitoring” and “honest, caveated monitoring” for Vertex-class stores.

---

## 7. Sampling & baseline strategies

### 7.1 Sampling strategies

Sampling is the bridge across the store-capability gap. VDM ships five strategies; the negotiator picks the best one a store supports within a configured **sampling + cost budget** (max records, max bytes, max wall-clock, max read-units/$).

| Strategy | How it works | Best for | Agnostic? |
|---|---|---|---|
| **Native random** | Store supports random sampling directly | Best when available | Yes |
| **Full-scan reservoir** | Iterate all, reservoir-sample *k* in one pass | Small/medium indexes, pgvector, LanceDB | Yes |
| **ID-listing + fetch** | Enumerate IDs, random-subset, fetch vectors | **Pinecone / Vectorize / Turbopuffer**: listable IDs, point fetch | Yes |
| **Query-seeded** | Panel of seed queries; collect neighbors | Vertex / no-scan stores (biased, caveated) | Yes |
| **Write-tap (sidecar)** | Mirror a fraction of upserts at write time | Highest fidelity | **No — the one non-agnostic strategy.** Requires pipeline integration; it is an opt-in fidelity upgrade, not the default "just point it at your store" path. |

### 7.2 Baseline strategies

A baseline is what “normal” means. To keep recomputation cheap, a baseline stores both a sampled vector set (optional, privacy-gated) and **precomputed reference statistics** — centroid, covariance summary, per-dimension histograms, a **frozen PCA basis** (also used for the projection, §11), and score distributions for the query set.

| Baseline type | Definition |
|---|---|
| **Fixed snapshot** | A golden reference captured at t0 and frozen. Drift is always measured against this known-good state. |
| **Rolling window** | Compare current sample against a trailing N-day window. Good for slow, expected evolution. |
| **Seasonal / periodic** | Compare against the same weekday/hour to absorb known cyclical patterns. |
| **Champion (promoted)** | Last manually approved “known-good” baseline; promotion is an explicit, audited action. |

#### Statistics-only baselines for privacy
Because embeddings can be partially inverted to source content, VDM supports a **statistics-only baseline mode** that retains the reference distributions (histograms, centroid, frozen PCA basis, score profiles) but discards the raw vectors after computing them. You keep drift detection; you stop storing potentially sensitive embeddings at rest.

#### Connection-secret management
Connecting to a production store requires storing **read credentials** for it — often the bigger enterprise blocker than embeddings-at-rest. VDM:
- Stores connection secrets in a dedicated secret backend (env-injected, or pluggable: Vault / cloud KMS / Supabase Vault), **never** in the metrics database alongside results.
- Encrypts secrets at rest and scopes them to the registering principal.
- Requests **read-only / least-privilege** credentials in every adapter's setup docs, and the connection form states exactly which operations VDM will perform.
- Logs every store access through the same OTel surface so credential use is auditable.

### 7.3 Distinguishing healthy change from harmful drift

**This is the make-or-break product problem.** A healthy RAG corpus shifts every day — new documents, new topics, re-chunking. Centroid moves, MMD rises, top-k changes. A naïve monitor pages on normal operation, gets muted, and dies. VDM's position:

1. **Retrieval quality against labels is the primary alert.** A drop in `recall@k` / `nDCG` against a golden labeled set (§8.2) is *actionable by definition* and is largely invariant to benign corpus growth. Distribution-drift metrics default to **info/context severity**, not paging, unless explicitly promoted.
2. **Rate-of-change, not absolute change.** Alert when the *rate* of distribution drift departs from the store's own established baseline rate (drift-of-drift / control charts on the first difference), so steady, expected evolution does not trip thresholds — only anomalous accelerations do.
3. **Cardinality-correlated change is reclassified.** A centroid move that coincides with a cardinality increase reads as ingestion (expected); the same move with **flat** cardinality reads as a silent model swap or re-embedding (suspect). The engine cross-references families before assigning severity.
4. **Suppression windows for known events.** Re-embedding / bulk-load jobs mute expected change (§10.1), and `embedding_model` / `chunk_version` metadata changes (§8.4) auto-open a suppression-and-rebaseline prompt rather than paging.
5. **Champion baselines make "known-good" explicit.** Promotion is the operator asserting "this is the new normal," resetting the rate baselines deliberately rather than letting drift accumulate silently.

The net rule: **page on quality regressions and anomalous *accelerations*; inform on steady distribution evolution.**

---

## 8. Drift detection catalog

Metrics are grouped into four families. The active drift plan determines which families run. Methods follow established conventions so results are comparable to Evidently/Arize-style monitoring, with parameters pinned in §8.5.

### 8.1 Family A — Embedding-distribution drift (needs an unbiased vector sample)

| Metric | Detects | Notes |
|---|---|---|
| Centroid distance | Mean-vector shift (cosine/Euclidean) | Cheap, coarse; Arize-style first signal |
| Per-dim PSI + share-drifted | How many dimensions individually shifted | Apply on PCA-reduced dims to cut noise |
| Maximum Mean Discrepancy | Whole-distribution shift | Robust, distribution-level; Evidently method (kernel/bandwidth pinned, §8.5) |
| Domain-classifier AUC | Whether ref vs current are separable | AUC ≫ 0.5 = drift; cross-validated, with a permutation null (§8.5) |
| Norm distribution shift | Magnitude-profile change | Strong signal for a silent model swap |
| PCA overlap | Cluster movement, qualitative shift | Powers the embedding-space UI view (frozen-basis projection, §11) |

### 8.2 Family B — Retrieval-quality drift (needs live query + a query set) — **the headline family**

| Metric | Detects | Notes |
|---|---|---|
| Top-k ID overlap (Jaccard/RBO) | Same queries returning different docs | Rank-Biased Overlap weights the top ranks |
| Score-distribution drift (KS) | Similarity scores trending down | Works even when vectors are unreadable |
| recall@k / nDCG vs labels | Hard quality regression | Needs a labeled golden set; **highest value; primary alert (§7.3)** |
| Answer-set stability | Volatility of returned cohorts | Catches index churn the scores hide |

#### Golden-label staleness
`recall@k` / `nDCG` depend on a labeled golden set, and **labels rot as the corpus changes** — a doc that was the right answer may be deleted or superseded. VDM mitigates:
- **Label health monitoring:** flag golden queries whose labeled target IDs no longer exist in the index (deleted) or have fallen far out of top-k across consecutive runs — surfaced as "labels need review," not as drift.
- **Versioned query sets** tied to a corpus/champion-baseline version, so re-labeling is an explicit, audited event (mirrors champion promotion).
- **Graceful partial scoring:** queries with stale labels are excluded from the aggregate and counted separately, so quality scores never silently degrade due to label rot.

### 8.3 Family C — Operational / index health

| Metric | Detects | Notes |
|---|---|---|
| Cardinality drift | Ingestion failures, deletions | Sudden count drops = pipeline incident; *also used to reclassify distribution drift (§7.3)* |
| Latency drift (p50/p95/p99) | Index degradation, load issues | From `probe()`; plots on the health panel |
| Null / zero / NaN vectors | Broken embedding writes | Quarantine-and-alert |
| Duplicate ratio | Re-ingestion bugs | Cheap integrity check |

### 8.4 Family D — Schema & provenance drift (needs metadata)

| Metric | Detects | Notes |
|---|---|---|
| New / missing metadata fields | Pipeline schema changes | Often the first sign of an upstream change |
| Field type changes | Silent contract breaks | Type histograms per field |
| `embedding_model` mismatch | Silent model upgrade | If model/version is tracked in metadata; auto-triggers rebaseline prompt (§7.3) |
| chunk/template version drift | Re-chunking without re-baselining | Pairs with cardinality spikes |

### 8.5 Statistical power & detector parameters

Bounded sampling against high-dimensional, billion-scale indexes makes **statistical power**, not just method choice, the deciding factor. v0.2 pins the parameters that make "trusted math" actually trustworthy:

- **MMD kernel & bandwidth.** RBF kernel; bandwidth via the **median heuristic computed on the *baseline* sample and frozen into the baseline** (recomputing per-run would let the bandwidth chase the very shift we measure). Significance via a permutation test (default 200 permutations) rather than an absolute threshold.
- **Domain-classifier AUC.** Logistic-regression or gradient-boosted classifier on **PCA-reduced** features, **k-fold cross-validated**, with a **permutation null** to convert AUC into a calibrated p-value — guarding against "AUC > 0.5 from sampling noise / curse of dimensionality."
- **Dimensionality reduction first.** Per-dim PSI and the classifier run on a **frozen PCA basis** (fit on baseline) retaining ~95% variance, cutting high-dimensional noise and multiple-testing burden. Per-dim PSI is reported with **Benjamini–Hochberg** correction.
- **Minimum detectable effect (MDE).** Each run computes and surfaces, for the achieved sample size *n* and dimensionality *d*, the **smallest drift effect it could reliably detect** at the configured power (default 0.8). When the budget yields a sample too small for the target MDE, the run is labeled **"under-powered"** in the UI — the same honesty principle as the sampling-confidence flag. A clean result on an under-powered run is never shown as a clean bill of health.
- **Sample-size guidance.** Registration estimates the *n* required to detect a target effect at the store's dimensionality and translates it into a sampling budget (and, on metered stores, a cost — §10.4), so operators choose fidelity vs. cost with eyes open.

---

## 9. Plugin system & conformance suite

“Plug in any vector store, including new ones” only works if adding a store is trivial and trustworthy. VDM solves this with a registry-based plugin model and a published conformance kit.

### 9.1 Plugin model

- **Convention-based discovery:** adapters ship as packages following a naming convention — e.g. `@vdm/adapter-*` on npm or a `vdm.adapters` entry-point group in Python — and self-register via a manifest.
- **Manifest:** each adapter declares `{ id, displayName, configSchema, capabilities, factory }`. The config schema drives an auto-generated connection form in the UI (and documents the least-privilege credentials it needs, §7.2).
- **First-party vs community:** core ships certified adapters for the top stores; the long tail (and brand-new stores) come from community/vendor packages that certify against the conformance suite.

```ts
// a complete adapter is small
export default defineAdapter({
  id: 'qdrant',
  displayName: 'Qdrant',
  configSchema: z.object({ url: z.string(), apiKey: z.string().optional(),
                           collection: z.string() }),
  capabilities: { returnsVectors: true, unbiasedSample: true, fullScan: true,
                  randomSample: 'native', idListing: true,
                  metadataFilter: true, countByFilter: true,
                  liveQuery: true, maxBatch: 256 },
  factory: (cfg) => new QdrantAdapter(cfg),
});
```

### 9.2 Conformance suite — the trust mechanism

A published, runnable test battery that every adapter must pass against a standard fixture dataset. This is what lets you (or a vendor) certify a brand-new store and what keeps capability declarations honest across SDK versions.

- **Capability honesty:** for each declared capability, the suite verifies the adapter actually behaves that way on the fixture (e.g. if `returnsVectors` is true, `sample()` really returns vectors of the declared dimension; if `unbiasedSample` is true, repeated samples are representative, not query-biased).
- **Sampling correctness:** `sample(n)` returns at most *n* records, within `count()`, de-duplicated, with stable IDs.
- **Query contract:** `query()` returns ranked hits with monotonic ranks and scores in the metric’s expected range.
- **Round-trip:** `fetchByIds()` returns the same records a prior `sample()` produced.
- **Resilience:** pagination, empty scopes, filter handling, and timeout behavior.

#### CI cost of the conformance suite
Verifying capability honesty "across SDK versions" requires running the suite against **real store instances**, and that has an ongoing operational cost worth naming:
- **Local/testcontainers** cover the self-hostable stores cheaply: pgvector, Qdrant, Chroma, Milvus, Weaviate, Redis, Elastic, LanceDB, Mongo.
- **No local emulator exists** for Pinecone, Vertex AI Vector Search, Turbopuffer, or Cloudflare Vectorize — these require **free-tier/sandbox cloud accounts and credentials in CI**, run on a **scheduled (e.g. nightly) matrix** rather than per-PR to bound cost, with results published to the capability report.
- Community/vendor adapters self-host their own conformance run and submit the signed report; first-party CI re-verifies on the nightly matrix.

#### Conformance badge
An adapter that passes earns a CI badge and a published capability report. This turns “supports any vector store” from a marketing claim into a verifiable contract — and gives the project a viral, vendor-friendly contribution loop (the same pattern that grew the OTel and MCP ecosystems).

---

## 10. Alerting, data model, API & cost

### 10.1 Policy & alerting
- **Threshold modes:** absolute, relative-to-baseline, and statistical (z-score / control-chart / EWMA) so noisy high-dimensional metrics do not spam alerts.
- **Healthy-change discrimination:** rate-of-change and cross-family reclassification per §7.3; distribution drift defaults to info severity, retrieval-quality regression to paging.
- **Severity tiers:** info / warn / critical, each with its own routing.
- **Suppression windows:** mute drift during known re-embedding or bulk-load jobs so expected change is not paged.
- **Sinks:** webhook, Slack, PagerDuty, email, OpenTelemetry events, or auto-filed GitHub/Jira issue.
- **Policy-as-code:** thresholds and routing live in versioned YAML alongside the store config.

### 10.2 Baseline-store data model

| Entity | Holds |
|---|---|
| `stores` | Registered targets, adapter id, connection config **(secret ref only, §7.2)**, active drift plan |
| `snapshots` | Versioned baselines: sampled vectors (optional) + reference statistics + frozen PCA basis |
| `query_sets` | Golden queries, optional labels for recall@k / nDCG, **corpus-version + label-health** |
| `runs` | Each monitoring execution: plan, sample size, timing, confidence, **MDE / power, read-units/cost** |
| `metrics` | Time series of every computed drift value per store/metric |
| `alerts` | Fired alerts, severity, acknowledgement, resolution |
| `policies` | Thresholds, suppression windows, routing (versioned) |

### 10.3 API surface (representative — REST + SSE)

```
POST   /stores                 register a store (adapter + config)
GET    /stores                 list stores + health + active plan
POST   /stores/:id/runs        trigger a monitoring run
GET    /stores/:id/metrics     drift time series (metric, range)
POST   /stores/:id/baseline    capture / promote a baseline
GET/POST /stores/:id/querysets manage golden query sets
GET/PUT  /stores/:id/policy    alert policy (YAML/JSON)
GET    /stores/:id/cost        estimated & actual read-unit/$ per run
GET    /events  (SSE)          live run + alert stream for the UI
```
*REST-only surface (GraphQL dropped from v0.1 — no consumer justified the added surface area). SSE carries the live UI stream.*

### 10.4 Cost model

Monitoring runs against the user's **production** store, and on metered stores (Pinecone, Vertex, Vectorize, Turbopuffer) every sample, query, and fetch is **billed to the user and competes with their real traffic**. For a "Datadog for your vector index" this is a top-three adoption objection, so cost is a first-class budget, not an afterthought:

- **Read-unit / $ budget per run**, alongside the records/bytes/wall-clock budgets — the scheduler will not exceed it; it degrades fidelity (smaller sample, fewer seed queries) and labels the run accordingly.
- **Cost surfaced at registration:** the connection form estimates per-run and monthly cost from the chosen cadence, sample size, and store pricing, *before* the operator commits — and ties directly to the MDE/power tradeoff (§8.5): higher fidelity costs more, shown explicitly.
- **Cadence-aware scheduling:** retrieval-quality checks (cheap, high value) can run frequently; full-distribution sampling (expensive) runs less often or only on change signals.
- **Actuals tracked** in `runs` and exposed at `GET /stores/:id/cost`, so cost is observable over time like any other metric.

---

## 11. UI / dashboard specification

The UI answers one question first — **“is any index misbehaving right now?”** — and then lets you drill into why. It is explicit about fidelity: every view labels the active drift plan, any sampling-confidence caveat, **and any under-powered run (§8.5)**, so an operator never mistakes a degraded or under-powered plan for a clean bill of health.

| View | What it shows |
|---|---|
| **Fleet overview** | Card per monitored store: health badge, active drift plan, last run, open alerts, and a drift sparkline. The “what’s going on right now” screen. |
| **Store detail / drift timeline** | Multi-metric time series with baseline bands and alert markers; toggle metric families on/off. |
| **Embedding-space view** | 2D projection overlaying baseline vs current as contour densities; cluster-movement is visible at a glance. **(FULL plan only; fit-on-baseline / transform-current — see note below.)** |
| **Per-dimension heatmap** | Drift score per dimension (PSI/share-drifted) to localize where the space moved. |
| **Retrieval-drift panel** | Per-query top-k overlap, RBO, and score-trend; recall@k / nDCG when labels exist. Works even when vectors are unreadable. **The headline panel.** |
| **Index-health panel** | Cardinality, latency percentiles, null/dup counts over time. |
| **Alerts & policy editor** | Threshold tuning with live preview against history; suppression windows; routing. |
| **Adapter registry** | Connected stores, detected vs declared capabilities, conformance status. |
| **Cost panel** | Per-store estimated vs actual read-units/$ per run and trend (§10.4). |

### The signature screen — and how to make it statistically honest
The embedding-space projection — baseline vs current overlaid as density contours, with drifting clusters highlighted — is the demo moment. It makes an abstract statistic (“MMD rose 0.18”) viscerally legible: you can *see* the corpus moving. Pair it with the per-dimension heatmap to go from “it moved” to “it moved *here*.”

**Critical caveat:** the overlay is only valid if both snapshots are projected through the **same fixed mapping**. VDM **fits the projection on the baseline and `transform()`s the current sample through that frozen basis** — never re-fits per snapshot, which would produce axis-incomparable plots that fabricate or hide movement.
- **Default: PCA** — a true linear `transform`, exactly reproducible, the honest choice for baseline-vs-current overlays.
- **Optional: parametric UMAP** — for richer cluster structure, but only via a baseline-fitted model applied to current data; standard (non-parametric) UMAP is reserved for single-snapshot exploration and never used for overlays.

---

## 12. Tech-stack options

Four coherent stacks, each rated for fit with this problem and for your existing toolchain (TypeScript/Node primary, Python secondary, pgvector/Supabase, OpenTelemetry). The hard constraint shaping all of them: vendor SDKs are JavaScript-strongest, but the drift math (MMD, domain-classifier, UMAP) is Python-strongest. Where a stack puts that seam determines its trade-offs.

### 12.1 Option A — TypeScript control plane + Python compute sidecar
**Conviction: HIGH — recommended.**
- **Why:** TypeScript owns the parts where its ecosystem is strongest (adapters against JS-first vendor SDKs, API, scheduling, UI), while a Python worker does the statistically heavy drift computation where its ecosystem is unbeatable. Clean control-plane / compute-plane seam.
- **Control plane:** Node 20+/TS, Hono or Fastify API, BullMQ (Redis) for jobs, Postgres + pgvector as the baseline store, React + Vite + Tailwind + shadcn/ui front end, visx + deck.gl/regl for the embedding scatter, Recharts for timelines.
- **Compute plane:** Python worker (FastAPI or Arq) using numpy / scipy / scikit-learn / umap-learn; optionally wrap Evidently or Alibi-Detect for proven detectors. OpenTelemetry across both.
- **Trade-off:** two languages and two runtimes to operate — acceptable, and the seam is clean and well-bounded.

### 12.2 Option B — Pure Python
**Conviction: HIGH for math-first; MEDIUM for product polish.**
- **Why:** Maximum statistical rigor with the least glue. Reuse Evidently / Alibi-Detect / NannyML directly; single language for all drift work; fastest path to credible, comparable detectors.
- **Stack:** FastAPI, Arq or Celery, Postgres or DuckDB, numpy/scipy/scikit-learn/umap; UI in Streamlit (fast to ship) or React (more polished); Plotly + Datashader for large scatter plots.
- **Trade-off:** vendor SDKs are JS-first, so some adapters are thinner in Python; Streamlit UIs are quick but not product-grade; the plugin DX is less aligned with your TS-centric tooling.

### 12.3 Option C — Pure TypeScript
**Conviction: MEDIUM.**
- **Why:** One language end-to-end, the best adapter DX (vendor SDKs are JS-strong), and the easiest contribution funnel for web developers — strong for OSS growth.
- **Stack:** Node/TS, Fastify, BullMQ, Postgres + pgvector, React UI; drift math in TS via ml-matrix / simple-statistics.
- **Trade-off:** you reimplement MMD, the domain-classifier, and UMAP in a JS ecosystem where they are second-class and less trusted. Distribution-drift fidelity suffers; this is the real cost.

### 12.4 Option D — Go core + Python ML service
**Conviction: MEDIUM-LOW — only at fleet scale.**
- **Why:** A Go collector gives high-throughput sampling and a single static binary for self-hosting large fleets; Python still does the math over gRPC.
- **Stack:** Go collectors/adapters + gRPC, Python ML service, Postgres, React UI.
- **Trade-off:** weakest fit for your current toolchain, most operational overhead, slowest iteration. Overkill until you are monitoring very large fleets of stores.

### 12.5 Recommendation
**Build Option A.**
It places genericity where the SDKs live (TypeScript adapters) and rigor where the math lives (a Python compute worker), and it slots directly into your existing pgvector / Supabase / OpenTelemetry muscle memory and your sfllm / MemoryLens TypeScript world. Start the compute worker by wrapping Evidently’s detectors so the math is trusted from day one; replace or extend with custom detectors only where you need to.

If you want the fastest possible proof-of-concept to validate the drift signal **before** investing in the adapter ecosystem, prototype in Option B (pure Python + Evidently) for a single store — pgvector — then port the architecture to Option A for the real, multi-store product.

---

## 13. Phased roadmap

### 13.1 Near-term — prove the signal (Phase 0–1)
- **Phase 0 — spike:** Pure-Python prototype against pgvector; centroid + MMD + domain-classifier drift on a fixed snapshot; **confirm the signal is real, legible, and that it distinguishes a known harmful shift from benign corpus growth (§7.3)** on a known corpus. *This is the de-risking gate — validate the signal before building the ecosystem.*
- **Phase 1 — MVP:** Option A skeleton. Adapter contract + conformance suite; first-party adapters for **pgvector, Qdrant, and Pinecone** (one FULL via scan, one FULL via native sample, one FULL-via-id-listing-with-query-seeded-fallback — exercising the full capability-negotiation path). FULL drift family A + **retrieval-quality family B (the headline)** + cardinality + latency. Fleet overview + drift timeline + retrieval-drift panel + embedding-space view (frozen-PCA). Cost budgeting + estimate-at-registration. Threshold alerting to webhook/Slack with healthy-change discrimination.

### 13.2 Long-term — ecosystem & retrieval depth (Phase 2–3)
- **Phase 2 — retrieval & breadth:** Query-set management with recall@k / nDCG and label-health; retrieval-drift depth; adapters for Weaviate, Milvus, Chroma, **Vertex (the true query-only proof point)**, Elasticsearch; statistics-only privacy baselines; control-chart alerting; OTel export; nightly conformance matrix for cloud-only stores.
- **Phase 3 — platform:** Managed cloud baselines and fleet alerting; RBAC, audit, and governance for enterprise; vendor-certified community adapters; auto-remediation hooks (trigger re-embedding jobs on drift).

#### OSS → cloud → enterprise path
Mirror the MemoryLens motion: Apache-2.0 core with first-party adapters builds GitHub traction and an adapter ecosystem; a hosted cloud adds managed baselines, fleet alerting, and the embedding-space UI as a service; enterprise adds governance, RBAC, and audit. The conformance suite is the flywheel — every certified adapter widens the moat and the addressable surface.

---

## 14. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Query-seeded sampling is biased toward dense regions | Never present it as an unbiased distribution; surface a sampling-confidence indicator and restrict it to retrieval/score-drift, not distribution claims. Prefer id-listing+fetch wherever IDs are listable (Pinecone/Vectorize) to avoid query bias entirely. |
| **Benign corpus growth trips drift alerts (false positives)** | Retrieval-quality-vs-labels as the primary alert; rate-of-change/control-chart detection; cross-family reclassification (cardinality-correlated change = ingestion); suppression windows; explicit champion rebaselining (§7.3). |
| **Bounded sampling lacks power to detect subtle drift** | Compute and surface minimum-detectable-effect/power per run; label under-powered runs; size sampling budget from a target effect at the store's dimensionality (§8.5). |
| Scanning/querying large stores is expensive (real $) | Hard sampling + **read-unit/$ budgets**; cost estimate at registration; cadence-aware scheduling; cheap retrieval checks frequent, expensive distribution sampling rare (§10.4). |
| Capabilities drift across SDK versions | Conformance suite in CI plus version-pinned adapters; declared capabilities verified, not trusted; nightly cloud-store matrix for non-emulatable stores (§9.2). |
| **Golden labels rot as the corpus changes** | Label-health monitoring; versioned query sets tied to corpus version; partial scoring that excludes stale labels (§8.2). |
| High-dimensional drift stats are noisy | Frozen-basis PCA reduction before per-dim tests; control-chart / EWMA thresholds; Benjamini–Hochberg correction; prefer MMD and classifier-AUC (with permutation null) over raw per-dim PSI (§8.5). |
| Stores that never return vectors | Honest QUERY/MINIMAL plans with explicit UI labeling; optional write-tap sidecar for fidelity where integration is possible (the one non-agnostic path). |
| Embeddings are sensitive (invertible) | Statistics-only baselines; encryption at rest; access control on the baseline store; configurable raw-vector retention. |
| **Production credentials grant read access to sensitive data** | Dedicated secret backend separate from results DB; least-privilege/read-only credentials; encryption at rest; audited access via OTel (§7.2). |
| Misleading embedding-space overlays | Fit-on-baseline / transform-current with a frozen basis; PCA default; parametric-only UMAP for overlays (§11). |
| “Yet another observability tool” fatigue | OTel-native so it composes with existing stacks; lead with the store-agnostic adapter + conformance + **retrieval-quality** angle no one else has. |

---

## 15. What makes this defensible

- **The retrieval-quality lens over a live connection** monitors what the user actually feels — worse answers — across an arbitrary production store, on a schedule, and works even when raw vectors are unreadable. This is the part no existing tool does and the part that stays valuable even though the underlying math is commodity.
- **The adapter contract + conformance suite** turn “any vector store” from a claim into a verifiable, vendor-extensible standard — the same ecosystem mechanic behind OpenTelemetry and MCP.
- **Capability negotiation + graceful degradation** mean VDM is useful against query-first stores (Vertex AI Vector Search) that defeat every DataFrame-based tool — and uses the *right* sampling strategy (id-listing+fetch) for stores like Pinecone that *can* be read but resist unbiased sampling, rather than treating them as black boxes.
- **Honest fidelity** — sampling-confidence, statistical-power/MDE, and cost are all first-class and surfaced in the UI — so operators trust a green light because they can see exactly what was (and wasn't) measured, and at what cost.

**Net:** the novelty is *not* the drift math — that is well understood, and v0.2 is explicit that it's the commodity layer. It is the **store-agnostic connector with honest capability negotiation, a conformance-certified adapter ecosystem, and a retrieval-quality lens**, delivered with honest fidelity and a visualization that makes drift legible. That combination does not exist in the open today, and it is strongest precisely on the query-first stores no DataFrame tool can reach.
