#!/bin/bash
set -eu
cd /home/js2723/PROJECTS/VUD
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
.venv/bin/python -m unittest discover -s dllm-training -p 'test_*.py' -v
output=$(mktemp -d /home/js2723/PROJECTS/VUD/tmp/mdlm-smoke-XXXXXX)
echo "SMOKE_OUTPUT=$output"
.venv/bin/python dllm-training/train.py --config dllm-training/configs/smoke.json --output "$output" --device cpu --max-steps 2
.venv/bin/python dllm-training/train.py --config dllm-training/configs/smoke.json --output "$output" --device cpu --max-steps 4 --resume
echo "SUCCESS training and resume"
