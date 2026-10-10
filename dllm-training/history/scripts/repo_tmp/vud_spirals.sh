#!/bin/bash
set -eu
cd /home/js2723/PROJECTS/VUD
export CUDA_VISIBLE_DEVICES="$1"
export PYTHONUNBUFFERED=1
method="$2"
folder="$3"
args=(
  run.py toy-classification
  --model_name Dream-org/Dream-v0-Base-7B
  --dataset_name spirals
  --D_size 200
  --x_range "{'x1': [-4, 4, 0.1], 'x2': [-4, 4, 0.1]}"
  --va_method "$method"
  --model_temperature 1
  --icl_sample_seed 0
  --save_directory "Dream-v0-Base-7B/$folder"
  --resume 1
)
echo "START $(date) host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES method=$method"
exec .venv/bin/python "${args[@]}"
