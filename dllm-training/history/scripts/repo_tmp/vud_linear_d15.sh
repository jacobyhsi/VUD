#!/bin/bash
set -eu
cd /home/js2723/PROJECTS/VUD
export CUDA_VISIBLE_DEVICES="$1"
export PYTHONUNBUFFERED=1
method="$2"
folder="$3"
exec .venv/bin/python run.py toy-regression \
  --model_name Dream-org/Dream-v0-Base-7B \
  --dataset_name linear_regression \
  --x_range "{'x1': [-15, 15, 0.2]}" \
  --D_size 15 \
  --va_method "$method" \
  --num_joint_samples 100 \
  --model_temperature 1 \
  --icl_sample_seed 0 \
  --save_directory "Dream-v0-Base-7B/$folder" \
  --resume 1
