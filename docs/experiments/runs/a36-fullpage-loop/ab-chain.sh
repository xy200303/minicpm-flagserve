#!/bin/bash
set -u
CASE='[[4096,1024,64,256]]'
run_baseline() {
  local run=$1
  if [ -f "/workspace/experiments/runs/$run/summary.csv" ]; then
    return 0
  fi
  PLUGIN_ROOT=/workspace/audit-plugin GEMS_ROOT=/workspace/a31-gems \
    /workspace/run_bi_case_v5.sh "$run" "$CASE"
}
run_candidate() {
  local run=$1
  if [ -f "/workspace/experiments/runs/$run/summary.csv" ]; then
    return 0
  fi
  PLUGIN_ROOT=/workspace/audit-plugin GEMS_ROOT=/workspace/a36-official-gems \
    /workspace/run_bi_case_v5.sh "$run" "$CASE"
}
run_baseline a36-4k-a1 && run_candidate a36-4k-b1 && \
  run_baseline a36-4k-a2 && run_candidate a36-4k-b2
