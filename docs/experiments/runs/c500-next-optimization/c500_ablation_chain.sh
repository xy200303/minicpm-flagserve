#!/bin/bash
set -u
CASE='[[4096,1024,64,256]]'
VLLM_FL_DISABLE_SAMPLER_FUSION=1 /data/run_c500_case.sh c500-ablate-no-sampler-4k "$CASE" && \
VLLM_FL_DISABLE_MCBLAS_MM=1 /data/run_c500_case.sh c500-ablate-no-mcblas-4k "$CASE" && \
VLLM_FL_DISABLE_LMHEAD_NN=1 /data/run_c500_case.sh c500-ablate-no-lmhead-nn-4k "$CASE"
