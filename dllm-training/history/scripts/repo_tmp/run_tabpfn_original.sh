#!/bin/bash
# Score the MDLMs on 1000 datasets simulated with TabPFN v1's original prior code: generate on CPU, evaluate half per GPU, summarise.
cd /home/js2723/PROJECTS/VUD/dllm-training
unset SLURM_PROCID LOCAL_RANK
export PYTHONUNBUFFERED=1
L=tmp/logs/tabpfn_original
mkdir -p $L
CUDA_VISIBLE_DEVICES= ../.venv/bin/python tmp/scripts/tabpfn_original_generate.py 1000 $L/datasets.npz > $L/generate.log 2>&1 || exit 1
../.venv/bin/python -c "
import numpy as np; d = np.load('$L/datasets.npz')
for k, sl in enumerate((slice(0, 500), slice(500, 1000))):
    np.savez(f'$L/datasets_{k}.npz', x=d['x'][sl], y=d['y'][sl], n_context=d['n_context'][sl])"
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tmp/scripts/tabpfn_original_eval.py $L/datasets_0.npz $L/preds_0.csv > $L/eval_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python tmp/scripts/tabpfn_original_eval.py $L/datasets_1.npz $L/preds_1.csv > $L/eval_gpu1.log 2>&1 &
wait
../.venv/bin/python tmp/scripts/tabpfn_original_summary.py $L/preds_0.csv $L/preds_1.csv > $L/summary.log 2>&1
echo 'TABPFN ORIGINAL COMPLETE'
