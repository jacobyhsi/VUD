#!/bin/bash
set -eu
cd /home/js2723/PROJECTS/VUD
export CUDA_VISIBLE_DEVICES="$1"
export PYTHONUNBUFFERED=1
method="$2"
echo "START $(date -u) host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES method=$method"
set +e
.venv/bin/python run.py toy-classification \
  --model_name Dream-org/Dream-v0-Base-7B \
  --dataset_name moons_1 \
  --va_method "$method" \
  --model_temperature 1 \
  --icl_sample_seed 0 \
  --resume 1
run_status=$?
printf '%s\n' "$run_status" > "tmp/logs/moons1_${method}.exit"
echo "END $(date -u) status=$run_status"
exit "$run_status"
