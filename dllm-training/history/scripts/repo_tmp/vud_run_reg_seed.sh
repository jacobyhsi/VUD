#!/bin/bash
set -u
cd /home/js2723/PROJECTS/VUD
source /home/js2723/PROJECTS/VUD/.venv/bin/activate
export CUDA_VISIBLE_DEVICES="$1"
export PYTHONUNBUFFERED=1
SEED="$2"
METHOD="$3"

echo "START $(date) host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES method=$METHOD seed=$SEED"

if [ "$METHOD" = "vanilla" ]; then
  python run.py toy-regression \
    --model_name "Dream-org/Dream-v0-Base-7B" \
    --dataset_name gaps \
    --x_range "{'x': [-15, 15, 0.2]}" \
    --va_method vanilla \
    --model_temperature 1 \
    --icl_sample_seed "$SEED" \
    --save_directory Dream-v0-Base-7B/va_vanilla \
    --resume 1
else
  python run.py toy-regression \
    --model_name "Dream-org/Dream-v0-Base-7B" \
    --dataset_name gaps \
    --x_range "{'x': [-15, 15, 0.2]}" \
    --va_method parallel_pp \
    --num_joint_samples 100 \
    --model_temperature 1 \
    --icl_sample_seed "$SEED" \
    --save_directory Dream-v0-Base-7B/va_parallel_pp \
    --resume 1
fi

echo "FINISHED $(date) method=$METHOD seed=$SEED exit=$?"
exec bash
