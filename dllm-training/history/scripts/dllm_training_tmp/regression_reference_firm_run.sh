#!/bin/bash
# Firmer exact check on linear_regression: 2 independent seeds x 1,000 MLP chunks (2M networks each; 10x the first run
# in total), 1M GP draws, lengthscale 0.1-3. Seed 1 on GPU0, seed 2 on GPU1; agreement between them shows the precision.
cd /home/js2723/PROJECTS/VUD/dllm-training
export PYTHONUNBUFFERED=1 LS_MIN=0.1 GP_HYPER=1000000 MLP_CHUNKS=1000
for s in 1 2; do
  CUDA_VISIBLE_DEVICES=$((s - 1)) SEED=$((s * 10)) TAG=_firm_seed$s ../.venv/bin/python tmp/scripts/regression_reference.py linear_regression het_gp,mlp \
    > tmp/logs/regression_reference_linear_regression_firm_seed$s.log 2>&1 &
done
wait
