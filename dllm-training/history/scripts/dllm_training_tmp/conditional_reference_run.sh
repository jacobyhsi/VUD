#!/bin/bash
# Full-size conditional reference: moons_1 on GPU0 (5x MLP networks for ESS), logistic_regression on GPU1 (shared with Dream).
cd /home/js2723/PROJECTS/VUD/dllm-training
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tmp/scripts/conditional_reference.py moons_1 1000000 2500 > tmp/logs/conditional_reference_moons_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python tmp/scripts/conditional_reference.py logistic_regression 1000000 500 > tmp/logs/conditional_reference_logistic_regression.log 2>&1 &
wait
