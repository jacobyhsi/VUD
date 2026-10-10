#!/bin/bash
# VUD toy-regression vanilla on linear_regression with the Stage A MDLM,
# using the same arguments as the Dream-v0-Base-7B vanilla run.
set -eu
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1
export VUD_MDLM_DEVICE=cuda
exec .venv/bin/python run.py toy-regression \
  --model_name "mdlm:/home/js2723/PROJECTS/VUD/dllm-training/outputs/bayes_linear_vud_small/checkpoint.pt" \
  --dataset_name linear_regression \
  --x_range "{'x1': [-15, 15, 0.2]}" \
  --D_size 15 \
  --va_method vanilla \
  --model_temperature 1 \
  --icl_sample_seed 0 \
  --save_directory "MDLM-BayesLinear-1.3M/va_vanilla" \
  --resume 1
