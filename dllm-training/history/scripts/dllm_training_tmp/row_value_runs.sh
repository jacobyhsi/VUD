#!/bin/bash
cd /home/js2723/PROJECTS/VUD
PY=/home/js2723/PROJECTS/VUD/.venv/bin/python
LOG=dllm-training/tmp/logs/mdlm_diag
OUT=dllm-training/outputs
$PY dllm-training/continuous_baseline.py --config dllm-training/configs/continuous_classic_icl_gd.json \
  --output $OUT/continuous_classic_icl_gd --device cuda > $LOG/continuous_classic_icl_gd.log 2>&1 &
for name in row_digits_1d_noiseless row_digits_icl_gd; do
  $PY dllm-training/train.py --config dllm-training/configs/$name.json \
    --output $OUT/$name --device cuda > $LOG/$name.log 2>&1 &
done
wait
echo ALL_DONE > $LOG/row_value.done
