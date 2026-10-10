#!/bin/bash
# TabPFN GP + TabPFN MLP (non-causal) bag: bank -> train 0.8M -> VUD vanilla on logistic_regression and moons_1.
# Usage: tabpfn_bag_pipeline.sh <tag> <bag> <results name> [bank GPU]. Every stage takes the least-loaded GPU when it
# starts, unless a bank GPU is given (memory alone misses jobs that have not allocated yet).
TAG=$1; BAG=$2; NAME=$3
cd /home/js2723/PROJECTS/VUD/dllm-training
least_loaded() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
BANK=data_banks/tabpfn_${TAG}_cls_3m
CUDA_VISIBLE_DEVICES=${4:-$(least_loaded)} PYTHONUNBUFFERED=1 TMPDIR=/tmp ../.venv/bin/python generate_pfn_bank.py --output $BANK \
  --label-prior-bag "$BAG" --device cuda --chunk 8192 > tmp/logs/tabpfn_${TAG}_bank.log 2>&1 || exit 1
cd /home/js2723/PROJECTS/VUD
GPU=$(least_loaded) WORKERS=8 bash dllm-training/tmp/scripts/bayes_linear_vud_run.sh tabpfn_${TAG}_classify2d || exit 1
export RUN=tabpfn_${TAG}_classify2d NAME
GPU=$(least_loaded) bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh logistic_regression --x_range "{'x1': [-15, 15, 0.2]}" \
  > dllm-training/tmp/logs/vud_mdlm_tabpfn_${TAG}_logistic.log 2>&1 &
sleep 90   # let the first job allocate before choosing the GPU for the second
GPU=$(least_loaded) bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh moons_1 > dllm-training/tmp/logs/vud_mdlm_tabpfn_${TAG}_moons1.log 2>&1 &
wait
