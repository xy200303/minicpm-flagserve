#!/bin/bash
set -u
CASE='[[16384,1024,64,128]]'
ENV_FILE=/data/c500-a42-env.sh /data/run_c500_case.sh c500-a42-16k-b1 "$CASE" && \
  /data/run_c500_case.sh c500-a42-16k-a2 "$CASE" && \
  ENV_FILE=/data/c500-a42-env.sh /data/run_c500_case.sh c500-a42-16k-b2 "$CASE"
