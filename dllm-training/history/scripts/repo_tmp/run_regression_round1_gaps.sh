#!/bin/bash
# Rerun of round 1's gaps VUD (its feature is named x, so it needs an explicit x_range), then the breakdown.
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1 VUD_MDLM_DEVICE=cuda CUDA_VISIBLE_DEVICES=0
NAME=MDLM-Regress-HetGP50-Geometry-0.8M
checkpoint=$(cat tmp/logs/regression_round1_selected.txt)
.venv/bin/python run.py toy-regression --model_name "mdlm:$checkpoint" --dataset_name gaps --x_range "{'x': [-15, 15, 0.2]}" \
  --va_method vanilla --model_temperature 1 --icl_sample_seed 0 --save_directory "$NAME/va_vanilla" --resume 1 \
  > tmp/logs/regression_round1_vud_gaps.log 2>&1
.venv/bin/python dllm-training/tmp/scripts/regression_breakdown.py Dream-v0-Base-7B MDLM-BayesRegressGP-0.8M $NAME \
  > tmp/logs/regression_round1_breakdown.log 2>&1
echo 'GAPS RERUN COMPLETE'
