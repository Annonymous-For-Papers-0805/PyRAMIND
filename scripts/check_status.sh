#!/usr/bin/env bash
# Progress check for the sequential pipeline.
cd "$(dirname "$0")/.."

echo "=== screen ==="
screen -ls 2>/dev/null | grep "\.ibn-" || echo "  (none active)"

echo
echo "=== sweep progress (last 10 lines) ==="
grep -E "\[[0-9]+/[0-9]+\] (RUN|DONE|FAIL|SKIP)|cells →|summary:|^==|done in|wrote .*seed-" logs/run_all.log 2>/dev/null | tail -10

echo
echo "=== result cell counts ==="
for d in locomo lme; do
    if [[ -d "results/$d" ]]; then
        for s in "results/$d"/*; do
            [[ -d "$s" ]] || continue
            n=$(find "$s" -name "seed-*.json" 2>/dev/null | wc -l | tr -d ' ')
            echo "  $d/$(basename "$s"): $n"
        done
    fi
done

echo
echo "=== throughput / load ==="
echo "  files in last 5 min: $(find .cache/azure -name '*.json' -mmin -5 2>/dev/null | wc -l | tr -d ' ')"
echo "  load: $(sysctl -n vm.loadavg)"
echo "  skip count: $(grep 'EMBED-SKIP\|COMPLETE-SKIP' logs/run_all.log 2>/dev/null | wc -l | tr -d ' ')"
