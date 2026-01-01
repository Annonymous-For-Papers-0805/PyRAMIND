# Reproducibility

This document describes how to reproduce the PyRAMIND results from a clean
machine. All sample sizes, seeds, and hyperparameters are parameterised — no
benchmark-specific constants are baked into the code.

## 1. Environment

- Python 3.10+ (tested on 3.10 and 3.11)
- An OpenAI-compatible chat + embedding endpoint (e.g. Azure OpenAI)
- ~5 GB free disk for the cache and the four downloaded benchmarks

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Credentials

```bash
cp .env.example .env
# fill in:
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_ENDPOINT
#   AZURE_OPENAI_API_VERSION
#   AZURE_OPENAI_CHAT_DEPLOYMENT     (e.g. gpt-4.1)
#   AZURE_OPENAI_EMBED_DEPLOYMENT    (e.g. text-embedding-3-large)
```

The Azure client (`src/pyramind_lib/azure.py`) honours rate-limit responses
with exponential backoff (12 attempts, up to 600 s/attempt, honours
`Retry-After` headers), and caches every response on disk under
`.cache/azure/` keyed by SHA-256 of `(deployment, payload)`. Reruns are free.

## 3. Datasets

```bash
./scripts/download_data.sh
```

Pulls the four benchmarks into `data/`:

| File                              | Source                                               |
|-----------------------------------|------------------------------------------------------|
| `locomo10.json`                   | snap-research/LoCoMo                                 |
| `longmemeval_s.json`              | xiaowu0162/LongMemEval (small split, 500 questions)  |
| `longmemeval_oracle.json`         | xiaowu0162/LongMemEval (oracle haystack)             |
| `hotpot_dev_distractor_v1.json`   | HotpotQA distractor dev                              |

If a dataset host is unreachable, point the loader at a locally-mirrored
copy by editing the `--data-paths` flag — the loader interface is in
`src/benchmarks/datasets/`.

**LoCoMo splits.** Per paper §4, LoCoMo's 10 conversations are partitioned
as `dev = first 3 conversations (~586 questions)` and
`test = remaining 7 conversations (~1400 questions)`. The split is
selected with `--split {dev,test,all}` (default `all`):

```bash
SPLIT=dev  ./scripts/run_benchmark.sh   # validation slice (~586 Qs)
SPLIT=test ./scripts/run_benchmark.sh   # held-out test (~1400 Qs)
SPLIT=all  ./scripts/run_benchmark.sh   # full 1986 Qs
```

The split is silently ignored by non-LoCoMo benchmarks.

## 4. Run

The single entrypoint is `scripts/run_benchmark.sh`. All sample sizes,
seeds, top-k, formula choice, and benchmark selection are passed via
environment variables — nothing is hardcoded.

```bash
# default: all four benchmarks, seed 42, top-k 5, formula "ensemble", full splits
./scripts/run_benchmark.sh

# subsample (e.g. quick smoke test)
LOCOMO_MAX=2 LME_MAX=10 LME_ORACLE_MAX=10 HOTPOT_MAX=10 \
    ./scripts/run_benchmark.sh

# a single benchmark, three seeds
BENCHMARKS="lme" SEEDS="42 123 999" ./scripts/run_benchmark.sh

# add the BM25 sparse Naive-RAG baseline (paper Table 2 reference)
SYSTEMS="bm25 naive-rag pyramind" ./scripts/run_benchmark.sh

# pick the LoCoMo split (default "all")
SPLIT=test ./scripts/run_benchmark.sh

# ablate the formula
FORMULA=pyramind        ./scripts/run_benchmark.sh   # canonical Ebbinghaus + log-freq
FORMULA=power-law       ./scripts/run_benchmark.sh   # Wickelgren–Wixted only
FORMULA=bayesian-act-r  ./scripts/run_benchmark.sh   # ACT-R Bayesian only
FORMULA=ensemble        ./scripts/run_benchmark.sh   # max(.,.,.) — paper default
FORMULA=ensemble-mean   ./scripts/run_benchmark.sh   # arithmetic mean ablation
FORMULA=ensemble-median ./scripts/run_benchmark.sh   # median ablation (robust)
```

Per-cell artifacts land at `results/<benchmark>/<system>/seed-<N>.json`
and contain every per-question record (predicted answer, ground truth,
verdict, F1, retrieved-id list, latency, judge reasoning) plus a config
hash for the run.

Reruns are idempotent: existing seed JSONs are skipped unless
`--force` is passed to the underlying sweep.

## 5. Aggregating

```bash
PYTHONPATH=src python -m benchmarks.aggregate results/
```

Prints accuracy, F1, mean latency, total cost, and recall@5 per cell.

## 5b. Embedding backend

`PyramindConfig.embedding_backend` selects the embedding model:

- `"bge"` (paper default) — `BAAI/bge-large-en-v1.5`, 1024-d fp32, local
  inference via `sentence-transformers`. Install with
  `pip install sentence-transformers` (kept out of the base requirements
  because the dependency stack is heavy).
- `"azure"` (cost-friendly alternative) — Azure `text-embedding-3-large`,
  3072-d. Reuses the existing `AzureClient` plus its on-disk cache, which
  is what every published sweep here used. Pick this if you want to avoid
  shipping the torch/sentence-transformers stack.

Both backends are exposed via `pyramind_lib.embeddings.make_embedder(...)`.
The engine itself currently still embeds through `AzureClient`; the backend
abstraction is the seam for paper-faithful runs and is exercised in the
embedding-ablation table.

## 5c. Ablations

Tier-ablation flags (which of the four Pyramid tiers the engine emits)
and the eight innovation flags A–H live on `PyramindConfig` in
`src/pyramind_lib/config.py`. Because `run_benchmark.sh` does not expose
every flag as an env var, tier/innovation ablations are passed through
the **`--config-path` JSON mechanism** that the sweep already understands
(`PyramindConfig.save(path)` / `PyramindConfig.load(path)`):

```bash
# 1. dump a default config to JSON, then edit the flag(s) you want to ablate
python -c "from pyramind_lib.config import PyramindConfig; \
    PyramindConfig(agent_role='benchmark').save('configs/no_themes.json')"
# manually flip enable_themes: true → false in configs/no_themes.json

# 2. run the sweep with that config
PYTHONPATH=src python -m benchmarks.sweep \
    --systems pyramind --benchmarks lme \
    --seeds 42 --top-k 5 --formula ensemble \
    --data-paths lme=data/longmemeval_s.json \
    --config-path configs/no_themes.json \
    --output-dir results/
```

The tier flags exposed today are `enable_facts`, `enable_themes`, and
`enable_worldview` (default all `true`). The eight innovation flags
(`enable_context_vector`, `enable_deterministic_R`, `enable_pyramid`,
`enable_momentum`, `enable_entropy_adaptive_forget`,
`enable_evolving_embeddings`, `enable_cross_reinforcement`,
`enable_entity_augmentation`) follow the same pattern.

## 6. Hyperparameters

Defaults live in `src/pyramind_lib/config.py` as `PyramindConfig`. The
values used in the paper (held fixed across all four benchmarks; no
per-dataset tuning):

| Parameter           | Value | Description                              |
|---------------------|-------|------------------------------------------|
| `tau`               | 40    | base half-life in cycles                 |
| `tau_multiplier`    | 8.0   | per-tier half-life multiplier            |
| `promote_threshold` | 0.30  | promote threshold (raw → fact), θ\_p     |
| `forget_threshold`  | 0.01  | forget threshold, θ\_f                   |
| `dedup_threshold`   | 0.97  | cosine similarity for dedup              |
| `warm_start_K`      | 8     | warm-start cycles before density kicks in|
| `R_min` / `R_max`   | 0.05 / 0.95 | R clamps (Eq. 2)                   |
| `eps_D`             | 0.01  | density floor (Eq. 3)                    |
| `max_theme_clusters`| 5     | k for k-means++ over fact embeddings     |
| `top_k_retrieval`   | 5     | retrieved memories per query             |
| `formula_name`      | ensemble | `max(M_pyramind, M_power, M_bayes)`   |
| `embedding_backend` | bge   | paper default; `azure` is the alternative|

All can be overridden via CLI flags (`--top-k`, `--formula`, …) or by
editing the dataclass (and passing it via `--config-path`, see §5c).

## 7. Determinism

- `temperature=0` for both the answerer and the judge
- The Pyramid uses k-means++ with a fixed seed (`random_state=seed`)
- The Azure response cache pins LLM responses across reruns
- Memory-strength arithmetic is closed-form (no Monte Carlo, no sampling)

Three seeds (42, 123, 999) on LoCoMo produced bit-identical accuracy
under this policy.

**Conditional determinism (paper §5).** The "bit-identical" claim holds
**conditional on**: (i) a fixed embedding checkpoint (BGE
`BAAI/bge-large-en-v1.5` at the pinned revision, or a stable Azure
`text-embedding-3-large` deployment); (ii) cached boundary outputs (the
`.cache/azure/` SHA-256-keyed response cache pins both extraction and
answer-generation LLM calls); (iii) fixed `PyramindConfig` parameters,
including all eight innovation flags and all three tier flags; and
(iv) fixed software (Python version, NumPy/scikit-learn versions in
`requirements.txt`, OS-level BLAS). Changing any of these can change the
floating-point outputs.

## 8. Hardware notes

The pipeline is single-threaded and IO-bound on the LLM endpoint. A full
sweep across all four benchmarks (full splits, one seed) takes a few hours
on a saturated PAYG endpoint and a few minutes on cached reruns. Memory
footprint stays under 2 GB.

**Embedding backend hardware cost.** The paper-default `bge` backend
runs `BAAI/bge-large-en-v1.5` locally and requires
`pip install sentence-transformers` plus a one-time ~1.3 GB model
download (cached under `~/.cache/huggingface/`). CPU inference is
adequate for the published sweep; a CUDA device speeds ingest by
roughly an order of magnitude. The `azure` backend (Azure
`text-embedding-3-large`, selected via `AZURE_OPENAI_EMBED_DEPLOYMENT`)
is the cost-/latency-friendly alternative: no model download, no torch
dependency, identical interface, and the on-disk response cache makes
reruns free.
