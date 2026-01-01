#!/usr/bin/env bash
# Run the PyRAMIND benchmark sweep across LoCoMo, LongMemEval-S,
# LongMemEval-Oracle, and HotpotQA. Writes per-seed JSON records under results/.
#
# Pre-requisites:
#   1. cp .env.example .env  &&  fill in your Azure / OpenAI credentials
#   2. ./scripts/download_data.sh        (fetches the 4 benchmarks into data/)
#   3. pip install -r requirements.txt
#
# Sample sizes default to "all" (no --max-items flag). To run a smaller subset
# pass an env var per benchmark before invoking the script, for example:
#
#   LOCOMO_MAX=2  LME_MAX=30  LME_ORACLE_MAX=30  HOTPOT_MAX=30 \
#       ./scripts/run_benchmark.sh
#
# Other knobs:
#   SEEDS         space-separated list of integer seeds  (default: 42)
#   TOP_K         retrieval top-k                        (default: 5)
#   FORMULA       pyramind | power-law | bayesian-act-r | ensemble |
#                 ensemble-mean | ensemble-median                  (default: ensemble)
#   SYSTEMS       space-separated systems
#                 (full-context | naive-rag | dense-rag | bm25 | pyramind)
#                                                        (default: pyramind)
#   PYTHON_BIN    interpreter to use                     (default: python)
#   BENCHMARKS    space-separated benchmark keys         (default: all four)
#   SPLIT         LoCoMo dev/test split (paper §4): dev|test|all   (default: all)

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi
: "${AZURE_OPENAI_API_KEY:?AZURE_OPENAI_API_KEY missing — copy .env.example to .env}"
: "${AZURE_OPENAI_ENDPOINT:?AZURE_OPENAI_ENDPOINT missing}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42}"
TOP_K="${TOP_K:-5}"
FORMULA="${FORMULA:-ensemble}"
SYSTEMS="${SYSTEMS:-pyramind}"
BENCHMARKS="${BENCHMARKS:-locomo lme lme-oracle hotpot}"
SPLIT="${SPLIT:-all}"

LOG="logs/run_benchmark.log"
mkdir -p logs results .cache/azure
echo "=== run_benchmark started $(date '+%F %T') ===" | tee -a "$LOG"

export PYTHONPATH="src"

# benchmark-key  →  data path,  optional max-items env var
declare -A DATA_PATH=(
    [locomo]="locomo=data/locomo10.json"
    [lme]="lme=data/longmemeval_s.json"
    [lme-oracle]="lme-oracle=data/longmemeval_oracle.json"
    [hotpot]="hotpot=data/hotpot_dev_distractor_v1.json"
)
declare -A MAX_VAR=(
    [locomo]="${LOCOMO_MAX:-}"
    [lme]="${LME_MAX:-}"
    [lme-oracle]="${LME_ORACLE_MAX:-}"
    [hotpot]="${HOTPOT_MAX:-}"
)

run_sweep () {
    local bench="$1"
    local label="$bench  ·  systems=$SYSTEMS  ·  formula=$FORMULA"
    local max_items="${MAX_VAR[$bench]}"
    local extra=()
    [[ -n "$max_items" ]] && extra+=( --max-items "$max_items" )

    echo "" | tee -a "$LOG"
    echo "=== $label  $(date '+%F %T') ===" | tee -a "$LOG"
    nice -n 19 "$PYTHON_BIN" -m benchmarks.sweep \
        --systems $SYSTEMS \
        --benchmarks "$bench" \
        --seeds $SEEDS \
        --top-k "$TOP_K" \
        --formula "$FORMULA" \
        --split "$SPLIT" \
        --data-paths "${DATA_PATH[$bench]}" \
        --azure-api-key "$AZURE_OPENAI_API_KEY" \
        --azure-endpoint "$AZURE_OPENAI_ENDPOINT" \
        --cache-dir .cache/azure \
        --output-dir results/ \
        "${extra[@]}" \
        2>&1 | tee -a "$LOG"
}

for b in $BENCHMARKS; do
    run_sweep "$b"
done

echo "" | tee -a "$LOG"
echo "=== run_benchmark done $(date '+%F %T') ===" | tee -a "$LOG"
