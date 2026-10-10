#!/bin/bash
# Train a broad-prior MDLM, then evaluate against its Bayes oracle (regression or classification).
# Usage: bayes_linear_vud_run.sh <config name>   (default bayes_linear_vud)
NAME=${1:-bayes_linear_vud}
cd /home/js2723/PROJECTS/VUD
# Worker processes need a node-local temp dir; NFS cannot remove open files.
export TMPDIR=/tmp/js2723-mdlm-$NAME && mkdir -p $TMPDIR
# Pin to one GPU when GPU is set (Slurm exposes every GPU of the allocation).
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES=$GPU; fi
PY=/home/js2723/PROJECTS/VUD/.venv/bin/python
LOG=dllm-training/tmp/logs/mdlm_diag
OUT=dllm-training/outputs/$NAME
$PY dllm-training/train.py --config dllm-training/configs/$NAME.json \
  --output $OUT --device cuda --workers ${WORKERS:-8} > $LOG/$NAME.log 2>&1 || exit 1
if grep -q '"task": "regression_gp"' dllm-training/configs/$NAME.json; then
  (cd dllm-training && $PY evaluate_regression_general.py --checkpoint ../$OUT/checkpoint.pt \
    --output ../$OUT/evaluation_general --device cuda --seed 2026) > $LOG/${NAME}_eval.log 2>&1
elif grep -qE '"task": "classification_(mix|pfn)"' dllm-training/configs/$NAME.json; then
  (cd dllm-training && $PY evaluate_vud_classification.py --checkpoint ../$OUT/checkpoint.pt \
    --output ../$OUT/evaluation_vud_datasets --device cuda --seed 2026) > $LOG/${NAME}_eval.log 2>&1
elif grep -q '"task": "classification"' dllm-training/configs/$NAME.json; then
  $PY dllm-training/evaluate_classification.py --checkpoint $OUT/checkpoint.pt --output $OUT/evaluation_seed2026 \
    --device cuda --tasks 512 --seed 2026 > $LOG/${NAME}_eval.log 2>&1
else
  $PY dllm-training/evaluate.py --checkpoint $OUT/checkpoint.pt --output $OUT/evaluation_seed2026 \
    --device cuda --tasks 256 --draws 32 --seed 2026 > $LOG/${NAME}_eval.log 2>&1
fi
rm -rf $TMPDIR
