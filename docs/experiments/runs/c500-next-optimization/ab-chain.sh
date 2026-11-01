#!/bin/bash
set -u
while kill -0 16025 2>/dev/null; do
  sleep 10
done
if [ ! -f /data/experiments/runs/c500-current-4k-host/summary.csv ]; then
  echo "host profile did not complete" >&2
  exit 1
fi
CASE='[[4096,1024,64,256]]'
ENV_FILE=/data/c500-a37-env.sh /data/run_c500_case.sh c500-a37-4k-b1 "$CASE" && \
  /data/run_c500_case.sh c500-a37-4k-a2 "$CASE" && \
  ENV_FILE=/data/c500-a37-env.sh /data/run_c500_case.sh c500-a37-4k-b2 "$CASE"
