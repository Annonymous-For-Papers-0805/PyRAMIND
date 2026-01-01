# PyRAMIND

### Auditable Deterministic Pyramid Memory for Long-Running LLM Agents

> **Py**ramidal **R**etention with **A**rithmetic **M**emory for **In**ference-time **D**eterminism — a memory architecture for LLM agents in which **every** maintenance decision (write consolidation, reinforcement, forgetting, tier promotion, retrieval scoring) is a closed-form arithmetic operation over cached embeddings. The LLM is invoked only at the boundaries: optional entity tagging at ingest and answer generation at query.

Given fixed embeddings, cached boundary outputs, hyperparameters, and software, each maintenance transition is **replayable from logged inputs** — every retained or forgotten memory is reconstructable as an explicit arithmetic chain.

---

## TL;DR

| | |
|---|---|
| **Problem** | LLM-orchestrated memory (Mem0, Zep, A-MEM, HippoRAG, …) places model calls inside the write-manage-read loop, so replays depend on cached outputs, model revisions, prompt versions, SDK behaviour, and provider variance. |
| **Approach** | Push the LLM to the boundary. Keep ingest scoring, decay, promotion, forgetting, and retrieval ranking as closed-form arithmetic over cached embeddings, organised in a four-tier **Pyramid** (`raw → facts → themes → worldview`). |
| **Retention** | Each memory's strength is a `max`-ensemble of three psychology-grounded forgetting curves modulated by local semantic density. |
| **Result** | **0.93** accuracy on LongMemEval-S (matched-harness; 95% Wilson CI [0.91, 0.95]), zero LLM calls in the maintenance path, **bit-identical** memory states across seeds. Weaker on temporal / multi-hop LoCoMo (**0.62**) where explicit relational structure helps. |

---

## How memory strength is computed

For each memory `m` with reinforcement count `f`, age `t`, unit-norm embedding `e`, and local density `D`:

```
M_base = R · ln(f+1) · exp(−t / τD)                # Ebbinghaus exponential (Eq. 4)
M_pow  = R · ln(f+1) · (t+1)^(−1 / τD)             # Wickelgren–Wixted power law (Eq. 5)
M_bay  = π(R, f) · ln(f+1) · exp(−t / τD)          # ACT-R Bayesian rational analysis (Eq. 6)
M      = max(M_base, M_pow, M_bay)                 # ensemble (Eq. 7)
```

The local **density modulator** `D = max(ε_D, 1 − max cos(e, e'))` contracts the decay timescale for crowded memories and preserves the half-life of isolated ones, modelling retroactive interference.

The retention-weighted store centroid `c = norm(Σ M_m e_m / Σ M_m')` (Eq. 1) drives the relevance term `R = clip((cos(e, c)+1)/2, R_min, R_max)` (Eq. 2) and a small contextual drift `e ← norm((1−α)e + αc)` with `α = 0.05·max(0, cos(e, c))` (Eq. 12).

A three-step momentum trend rescales `M`; an entropy-adaptive prune threshold `θ_f' = θ_f · adapt(H_norm)` (Eqs. 9–11) keeps the store from collapsing during regime changes.

At query time, candidate memories are ranked by `s(q, m) = M · cos(e_q, e_m)` (Eq. 13). The top-k are passed to the answer generator. Themes and worldview vectors are **not** prompted directly — they shape retrieval through `c`, `R`, and the drift rule.

## How the Pyramid is built

| Tier | Built by | Promotion rule |
|---|---|---|
| `RAW` | every ingested turn, verbatim, with frozen embedding | always written; **never pruned** (ground-truth episodic store) |
| `FACTS` | entity-tagged facts (typed dates / names / amounts / topics prepended before embedding) | `M_m > θ_p` AND `f_m ≥ f_min` |
| `THEMES` | deterministic **k-means++** over fact embeddings, `k = min(k_max, |F|)` | rebuilt every N cycles |
| `WORLDVIEW` | top right singular vectors (**truncated SVD**) of the theme-centroid matrix | rebuilt with themes |

Every transition is linear algebra. No LLM summarisation.

---

## Headline results

Held-out, matched-harness numbers from the paper (Tables 2–4):

| Benchmark | PyRAMIND | 95% Wilson CI | Notes |
|---|---|---|---|
| LongMemEval-S (n=500) | **0.93** | [0.91, 0.95] | Best matched-harness peer 0.81 (Dense RAG) |
| LoCoMo held-out test (n=1400) | **0.62** | [0.59, 0.64] | Tied with Zep v3 (p=0.078) |
| LongMemEval-Oracle (n=500) | 0.97 R@5 / 0.53 QA | — | Retrieval-to-answer gap is judge-side; paraphrase-tolerant judge → 0.85 |
| HotpotQA distractor (n=100 dev) | 0.80 | — | Multi-hop stress test |

**Lifecycle cost** on held-out LoCoMo (Table 4): PyRAMIND ≈ **1.4k LLM tokens/turn**, ≈2.1 s end-to-end median latency, ≈8.4 maintenance writes/sec on a single embedding worker — **zero** LLM tokens in the maintenance path.

Reproductions of vendor systems (Mem0 OSS, Zep v3) and the full BM25/Dense RAG matched-harness Table 2 are described in the paper; this anonymous artifact ships the PyRAMIND core, the BM25 and Dense RAG baselines, the four benchmark runners, and the matched-harness orchestrator. Vendor SDKs are not bundled.

---

## Repository layout

```
src/
  pyramind_lib/                # the memory engine — closed-form, deterministic
    engine.py                    # MemoryEngine: record / cycle / query
    formulas.py                  # 13 retention functions; FORMULAS registry; FORMULA_INFO
    pyramid.py                   # raw → facts → themes (k-means++) → worldview (SVD)
    retrieval.py                 # cosine + top-k
    scoring.py                   # R, D, momentum
    entities.py                  # typed entity extraction (single LLM call)
    embeddings.py                # BGEEmbedder (paper default) + AzureEmbedder
    azure.py                     # rate-limited OpenAI-compatible client + disk cache
    innovations.py               # momentum, adaptive forget, contextual drift,
                                 #   cross-reinforcement, entity-tag storage
    config.py, types.py
  benchmarks/
    orchestrator.py              # single-cell run; emits per-question artifact + manifest
    sweep.py                     # (systems × benchmarks × seeds) Cartesian sweep
    judge.py                     # LLM-as-judge (gpt-4.1, T=0, content-hash cached)
    metrics.py                   # f1, recall@k, McNemar χ² + Yates, Wilson 95% CI,
                                 #   exact-binomial McNemar fallback
    aggregate.py                 # JSON → leaderboards, paired-correctness extraction
    baselines/
      pyramind.py                # PyRAMIND baseline
      bm25.py                    # Naive RAG (BM25 + top-k)         ← paper Table 2
      naive_rag.py               # Dense RAG (BGE/Azure + cosine)   ← alias "dense-rag"
      full_context.py            # full-haystack baseline
    runners/
      lme.py, locomo.py, hotpot.py
    datasets/
      lme_loader.py
      locomo_loader.py           # supports --split {all, dev, test} per paper §4
      hotpot_loader.py

scripts/
  download_data.sh             # fetch the 4 benchmarks into data/
  run_benchmark.sh             # parameterised sweep (env-var driven)
  check_status.sh              # progress check during long runs

results/                       # per-cell records: results/<bench>/<system>/seed-N.json
data/                          # downloaded datasets (created by download_data.sh)
.cache/                        # boundary-LLM response cache (created at runtime)
logs/                          # run logs (created at runtime)

requirements.txt               # python deps (numpy, scikit-learn, requests, rank-bm25)
.env.example                   # OpenAI-compatible credentials template
LICENSE                        # MIT
REPRODUCIBILITY.md             # step-by-step reproduction guide
```

---

## Quick start

```bash
# 1. install dependencies
python -m pip install -r requirements.txt
# (optional) paper-default BGE embeddings:
#     pip install sentence-transformers       # ~1.3 GB model on first use

# 2. configure boundary-LLM credentials (any OpenAI-compatible endpoint)
cp .env.example .env && $EDITOR .env

# 3. fetch all four benchmarks
./scripts/download_data.sh

# 4. run PyRAMIND on all four benchmarks (full splits, seed 42)
./scripts/run_benchmark.sh

# 5. aggregate per-cell artifacts into a leaderboard
PYTHONPATH=src python -m benchmarks.aggregate results/
```

**Smoke test (a few minutes, cached):**

```bash
LOCOMO_MAX=2 LME_MAX=10 LME_ORACLE_MAX=10 HOTPOT_MAX=10 \
    ./scripts/run_benchmark.sh
```

**Reproduce the paper's matched-harness Table 2 (LME-S + LoCoMo held-out test, all baselines, seed 42):**

```bash
SYSTEMS="bm25 naive-rag pyramind" \
BENCHMARKS="lme locomo" \
SPLIT="test" \
SEEDS="42" \
    ./scripts/run_benchmark.sh
```

**Formula ablations (Table 3 top half):**

```bash
for f in pyramind power-law bayesian-act-r ensemble ensemble-mean ensemble-median; do
    FORMULA=$f ./scripts/run_benchmark.sh
done
```

**Tier ablations (Table 3 bottom half)** — toggle via `--config-path config.json` overriding `enable_facts`, `enable_themes`, `enable_worldview` (see `REPRODUCIBILITY.md §5c`).

**Determinism check (paper §5):**

```bash
SEEDS="42 123 999" BENCHMARKS="locomo" SPLIT=test ./scripts/run_benchmark.sh
PYTHONPATH=src python -m benchmarks.aggregate results/   # all three seeds → identical accuracy
```

See `REPRODUCIBILITY.md` for full configuration, hardware notes, the BGE/Azure embedding backend choice, the LoCoMo dev/test split convention, and the determinism guarantees.

---

## Fixed hyperparameters

Tuned once on the LoCoMo development split and **frozen** for all held-out runs (Paper Table 1):

| Parameter | Value | Description |
|---|---|---|
| `embedding_backend` | `"bge"` | `BAAI/bge-large-en-v1.5` (1024-d fp32); `"azure"` available |
| `tau` | 40 | global timescale |
| `theta_promote` (θ_p) | 0.30 | promotion threshold raw → fact |
| `theta_forget` (θ_f) | 0.01 | base prune threshold |
| `dedup_threshold` (θ_dedup) | 0.97 | cosine cutoff for dedup |
| `warm_start_K` | 8 | first K writes use `D ≡ 1` |
| `eps_D` | 0.01 | density floor |
| `k_themes` (k_max) | 5 | maximum themes |
| `cross_top_k` (k_cross) | 3 | neighbours reinforced per ingest |
| `top_k_retrieval` | 5 | retrieval depth |
| momentum (↑, ↓, →) | (1.20, 0.85, 1.00) | 3-cycle trend multipliers |
| entropy thresholds / multipliers | 0.85 / 0.40 ; 1.5 / 0.7 / 1.0 | adaptive prune-threshold rescaling |
| seeds tested | 42, 123, 999 | identical results across all three |

All are exposed via `PyramindConfig` (`src/pyramind_lib/config.py`).

---

## Reproducibility & auditability

PyRAMIND's claim is **maintenance-path determinism**, not provider-portability. Given:

1. the ordered turn stream,
2. the same embedding checkpoint,
3. the same cached boundary-LLM outputs (entity tagger + answer generator + judge),
4. the same hyperparameters, and
5. the same software version,

the memory state after every cycle, the per-question retrieval, and the final accuracy are **bit-identical**. The orchestrator writes a per-run manifest (config hash, dataset SHA, git commit, requirements lockfile hash, prompt-template hash, Python/platform info) into every artifact so audits can confirm the artifacts above.

Statistical reporting matches the paper:
- **McNemar's χ² with Yates continuity correction** for paired matched-harness comparisons (`benchmarks.metrics.mcnemar_chi2`)
- **Wilson 95% score intervals** for unpaired accuracies (`benchmarks.metrics.wilson_ci`)
- The exact binomial McNemar variant is also exposed for transparency (`mcnemar_exact_test`)

---

## What's intentionally NOT here

- **Mem0 OSS / Zep v3 reproductions** — Table 2 of the paper reports those head-to-head numbers; the wrappers around their public SDKs are not part of this anonymous artifact (to keep dependencies light and the artifact provider-neutral). They can be added by wrapping the SDKs behind the `Baseline` interface in `src/benchmarks/baselines/`.
- **The paper itself** — this artifact is **code only**.
- **Author identity** — no names, affiliations, emails, provider-specific endpoints, or AI-tool attribution anywhere. The git history is a single anonymous commit at UTC.

---

## Citing

```bibtex
@inproceedings{pyramind_anon,
  title     = {{PyRAMIND}: Auditable Deterministic Pyramid Memory
               for Long-Running {LLM} Agents},
  author    = {Anonymous Author(s)},
  booktitle = {Under review},
  year      = {2026}
}
```

## License

MIT — see `LICENSE`.
