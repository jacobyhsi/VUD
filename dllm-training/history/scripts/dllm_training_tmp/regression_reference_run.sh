#!/bin/bash
# Exact-posterior screen of the regression prior (het GP + TabPFN v1 MLP, and each member alone): gaps on GPU0, linear on GPU1.
cd /home/js2723/PROJECTS/VUD/dllm-training
export PYTHONUNBUFFERED=1
run() { for members in het_gp,mlp mlp; do ../.venv/bin/python tmp/scripts/regression_reference.py $1 $members; done; }
CUDA_VISIBLE_DEVICES=0 run gaps > tmp/logs/regression_reference_gaps.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash -c "../.venv/bin/python tmp/scripts/regression_reference.py linear_regression het_gp; ../.venv/bin/python tmp/scripts/regression_reference.py linear_regression het_gp,mlp; ../.venv/bin/python tmp/scripts/regression_reference.py linear_regression mlp" > tmp/logs/regression_reference_linear_regression.log 2>&1 &
wait
