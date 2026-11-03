#!/bin/bash
set -u
CASE='[[4096,1024,64,256]]'
GEMS_ROOT=/workspace/a31-gems /workspace/run_bi_a36_case.sh a36-4k-a1 "$CASE" && \
GEMS_ROOT=/workspace/a36-gems /workspace/run_bi_a36_case.sh a36-4k-b1 "$CASE" && \
GEMS_ROOT=/workspace/a31-gems /workspace/run_bi_a36_case.sh a36-4k-a2 "$CASE" && \
GEMS_ROOT=/workspace/a36-gems /workspace/run_bi_a36_case.sh a36-4k-b2 "$CASE"
