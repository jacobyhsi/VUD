#!/bin/bash
# Simulated ground-truth comparison (classification, TabPFN v1 default generation): 100 tasks, 50 per GPU
# (both GPUs shared with the Qwen servers, which reserve 40% each), then the summary.
cd /home/js2723/PROJECTS/VUD/dllm-training
export PYTHONUNBUFFERED=1 PRIOR=tabpfn
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tmp/scripts/sim_groundtruth_classification.py 0 50 > ../tmp/logs/sim_groundtruth_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python tmp/scripts/sim_groundtruth_classification.py 50 100 > ../tmp/logs/sim_groundtruth_gpu1.log 2>&1 &
wait
../.venv/bin/python tmp/scripts/sim_groundtruth_summary.py tmp/logs/sim_groundtruth_classification_tabpfn > ../tmp/logs/sim_groundtruth_summary.log 2>&1
echo 'SIM GROUND TRUTH COMPLETE'
