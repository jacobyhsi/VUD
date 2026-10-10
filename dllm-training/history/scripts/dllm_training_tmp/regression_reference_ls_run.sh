#!/bin/bash
# Re-screen with the GP lengthscale range broadened to [0.1, 3]: gaps on GPU0, linear on GPU1 (1M GP draws).
cd /home/js2723/PROJECTS/VUD/dllm-training
export PYTHONUNBUFFERED=1 LS_MIN=0.1 GP_HYPER=1000000
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tmp/scripts/regression_reference.py gaps het_gp,mlp > tmp/logs/regression_reference_gaps_ls0.1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python tmp/scripts/regression_reference.py linear_regression het_gp,mlp > tmp/logs/regression_reference_linear_regression_ls0.1.log 2>&1 &
wait
