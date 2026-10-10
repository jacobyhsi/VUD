#!/bin/bash
# Re-score the generated TabPFN-original datasets (both halves) and summarise.
cd /home/js2723/PROJECTS/VUD/dllm-training
unset SLURM_PROCID LOCAL_RANK
export PYTHONUNBUFFERED=1
L=tmp/logs/tabpfn_original
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tmp/scripts/tabpfn_original_eval.py $L/datasets_0.npz $L/preds_0.csv 0 > $L/eval_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python tmp/scripts/tabpfn_original_eval.py $L/datasets_1.npz $L/preds_1.csv 500 > $L/eval_gpu1.log 2>&1 &
wait
../.venv/bin/python tmp/scripts/tabpfn_original_summary.py $L/preds_0.csv $L/preds_1.csv > $L/summary.log 2>&1
echo 'TABPFN ORIGINAL EVAL COMPLETE'
