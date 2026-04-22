#!/bin/bash
# Sequential stress test chain: validate -> tableone -> ward -> ecdf.
# Each stage runs its own harness invocation and writes a row to results.jsonl.
# Continues on failure so partial data still lands even if one stage OOMs.
#
# Usage (launch from Terminal.app, not Cursor):
#   cd /Users/dema/WD/CLIF-TableOne
#   nohup dev/stress_test/run_chain.sh > output/stress_test_chain.log 2>&1 &
#   echo "PID: $!"
#   # then close Terminal

set -u
cd /Users/dema/WD/CLIF-TableOne

rm -rf output/final output/intermediate output/stress_test
mkdir -p output

caffeinate -i -s &
CAF=$!
trap 'kill "$CAF" 2>/dev/null' EXIT

for stage in validate tableone ward ecdf; do
  echo "=== STAGE $stage START: $(date +%H:%M:%S) ==="
  uv run python dev/stress_test/stress_test.py --label "mac-$stage" --only "$stage"
  rc=$?
  echo "=== STAGE $stage END: $(date +%H:%M:%S) exit=$rc ==="
done

echo "=== ALL STAGES COMPLETE: $(date +%H:%M:%S) ==="
