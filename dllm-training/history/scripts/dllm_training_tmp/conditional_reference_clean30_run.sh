#!/bin/bash
# Exact check of the controlled GP prior with 30% clean tasks (was 50%): moons_1 on GPU0, logistic_regression on GPU1.
cd /home/js2723/PROJECTS/VUD/dllm-training
export PYTHONUNBUFFERED=1 CLEAN_PROB=0.3
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tmp/scripts/conditional_reference.py moons_1 1000000 2500 > tmp/logs/conditional_reference_moons_1_clean0.3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python tmp/scripts/conditional_reference.py logistic_regression 1000000 500 > tmp/logs/conditional_reference_logistic_regression_clean0.3.log 2>&1 &
wait
