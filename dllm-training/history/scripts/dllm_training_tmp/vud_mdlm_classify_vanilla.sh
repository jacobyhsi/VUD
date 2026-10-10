#!/bin/bash
# VUD toy-classification vanilla with a 2-D classification MDLM, matching the
# Dream-v0-Base-7B runs' arguments. Usage: vud_mdlm_classify_vanilla.sh <dataset> [extra args]
# Override the model with RUN (output folder under dllm-training/outputs) and NAME (results name).
set -eu
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1
export VUD_MDLM_DEVICE=cuda
# Pin to one GPU when GPU is set (Slurm exposes every GPU of the allocation).
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES=$GPU; fi
dataset=$1; shift
RUN=${RUN:-bayes_classify2d}
NAME=${NAME:-MDLM-BayesClassify2D-2.5M}
exec .venv/bin/python run.py toy-classification \
  --model_name "mdlm:/home/js2723/PROJECTS/VUD/dllm-training/outputs/$RUN/checkpoint.pt" \
  --dataset_name "$dataset" \
  --va_method vanilla \
  --model_temperature 1 \
  --icl_sample_seed 0 \
  --save_directory "$NAME/va_vanilla" \
  --resume 1 "$@"
