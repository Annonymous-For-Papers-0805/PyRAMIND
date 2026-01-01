#!/usr/bin/env bash
# Download the four public benchmarks used in the paper into ./data/.
#
# Datasets:
#   - LoCoMo (10 conversations, 1986 QAs)         https://github.com/snap-research/locomo
#   - LongMemEval-S (500 questions, ~115k tokens) https://github.com/xiaowu0162/LongMemEval
#   - LongMemEval-Oracle (same items, oracle haystack) — same repo as LongMemEval
#   - HotpotQA distractor dev (10 paragraphs/item) https://hotpotqa.github.io
#
# Each loader in src/benchmarks/datasets/ consumes the file paths produced
# below; run_benchmark.sh references the same paths.
#
# LoCoMo split convention (paper §4): the 10 conversations split as
#   dev  = first 3 conversations   (≈586  questions)
#   test = remaining 7 conversations (≈1400 questions)
# Selected at run time via `--split {dev,test,all}` (or `SPLIT=…` env var
# in run_benchmark.sh). The download is the same file either way.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
cd data

echo "[1/4] LoCoMo (locomo10.json)"
if [[ ! -s locomo10.json ]]; then
    curl -fSL -o locomo10.json \
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
fi

echo "[2/4] LongMemEval-S (longmemeval_s.json)"
if [[ ! -s longmemeval_s.json ]]; then
    curl -fSL -o longmemeval_s.json \
        "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s.json"
fi

echo "[3/4] LongMemEval-Oracle (longmemeval_oracle.json)"
if [[ ! -s longmemeval_oracle.json ]]; then
    curl -fSL -o longmemeval_oracle.json \
        "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle.json"
fi

echo "[4/4] HotpotQA distractor dev (hotpot_dev_distractor_v1.json)"
if [[ ! -s hotpot_dev_distractor_v1.json ]]; then
    curl -fSL -o hotpot_dev_distractor_v1.json \
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
fi

echo
echo "Done. Files in data/:"
ls -la *.json
