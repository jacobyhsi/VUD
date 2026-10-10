#!/bin/bash
# Train the row-vector MDLM on the VUD linear distribution, then run the held-out evaluation.
cd /home/js2723/PROJECTS/VUD
PY=/home/js2723/PROJECTS/VUD/.venv/bin/python
LOG=dllm-training/tmp/logs/mdlm_diag
OUT=dllm-training/outputs/row_digits_linear_vud
$PY dllm-training/train.py --config dllm-training/configs/row_digits_linear_vud.json \
  --output $OUT --device cuda > $LOG/row_digits_linear_vud.log 2>&1 || exit 1
$PY dllm-training/evaluate.py --checkpoint $OUT/checkpoint.pt --output $OUT/evaluation_seed2026 \
  --device cuda --tasks 256 --draws 32 --seed 2026 > $LOG/row_digits_linear_vud_eval.log 2>&1
