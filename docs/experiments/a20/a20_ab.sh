#!/bin/bash
# A20 official A/B: interleaved stock (A) vs self-developed decode kernel (B).
set -u
run() {
  local id=$1 case=$2
  shift 2
  env "$@" /workspace/exp_run.sh "$id" "$case" 1
}

echo "=== 4k A/B start $(date) ==="
for tag in A1 B1 A2 B2; do
  if [ "${tag:0:1}" = "A" ]; then
    echo "--- $tag: stock path ---"
    run "A20_4k_$tag" '[[4096,1024,64,256]]' VLLM_FL_DISABLE_DECODE_FAST=1
  else
    echo "--- $tag: fast kernel ---"
    run "A20_4k_$tag" '[[4096,1024,64,256]]' VLLM_FL_DISABLE_DECODE_FAST=0
  fi
done

echo "=== 16k A/B start $(date) ==="
for tag in A1 B1; do
  if [ "${tag:0:1}" = "A" ]; then
    echo "--- $tag: stock path ---"
    run "A20_16k_$tag" '[[16384,1024,64,128]]' VLLM_FL_DISABLE_DECODE_FAST=1
  else
    echo "--- $tag: fast kernel ---"
    run "A20_16k_$tag" '[[16384,1024,64,128]]' VLLM_FL_DISABLE_DECODE_FAST=0
  fi
done
echo "=== all runs done $(date) ==="
