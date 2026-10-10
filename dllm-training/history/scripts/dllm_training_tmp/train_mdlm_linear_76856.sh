#!/bin/bash
set -euo pipefail
cd /home/js2723/PROJECTS/VUD
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
trap 'training_exit=$?; echo "MDLM launcher exit code: ${training_exit}"' EXIT

echo 'Checking CUDA training before launching linear seed 0.'
mdlm_cuda_smoke_dir=$(mktemp -d /home/js2723/PROJECTS/VUD/tmp/mdlm-cuda-smoke-XXXXXX)
.venv/bin/python dllm-training/train.py \
  --config dllm-training/configs/smoke.json \
  --output "$mdlm_cuda_smoke_dir" --device cuda

echo 'Starting full linear seed 0 training.'
.venv/bin/python dllm-training/train.py \
  --config dllm-training/configs/linear_vud.json \
  --output dllm-training/outputs/linear_seed0 \
  --device cuda
