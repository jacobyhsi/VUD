#!/bin/bash
cd /home/js2723/PROJECTS/VUD
PY=.venv/bin/python
OUT=dllm-training/outputs
LOG=dllm-training/tmp/logs/mdlm_diag
$PY dllm-training/continuous_baseline.py --config dllm-training/configs/continuous_icl_gd.json --output $OUT/continuous_icl_gd --device cuda > $LOG/continuous_icl_gd.log 2>&1 &
$PY dllm-training/train.py --config dllm-training/configs/digits_icl_gd.json --output $OUT/digits_icl_gd --device cuda > $LOG/digits_icl_gd.log 2>&1 &
$PY dllm-training/train.py --config dllm-training/configs/digits_1d_noiseless.json --output $OUT/digits_1d_noiseless --device cuda > $LOG/digits_1d_noiseless.log 2>&1 &
wait
echo ALL_DONE >> $LOG/full.done
