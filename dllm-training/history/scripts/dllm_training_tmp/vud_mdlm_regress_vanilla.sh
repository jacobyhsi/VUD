#!/bin/bash
# VUD toy-regression vanilla with the general regression MDLM, matching the
# Dream-v0-Base-7B runs. Usage: vud_mdlm_regress_vanilla.sh <dataset> <folder> [extra args]
set -eu
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1
export VUD_MDLM_DEVICE=cuda
dataset=$1; folder=$2; shift 2
exec .venv/bin/python run.py toy-regression \
  --model_name "mdlm:/home/js2723/PROJECTS/VUD/dllm-training/outputs/${RUN:-bayes_regress_general}/checkpoint.pt" \
  --dataset_name "$dataset" \
  --va_method vanilla \
  --model_temperature 1 \
  --icl_sample_seed 0 \
  --save_directory "${NAME:-MDLM-BayesRegress-0.8M}/$folder" \
  --resume 1 "$@"
