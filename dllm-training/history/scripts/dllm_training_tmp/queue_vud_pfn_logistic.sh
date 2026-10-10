#!/bin/bash
# After MDLM-PFNClassify2D training + evaluation, run VUD vanilla on logistic_regression (Dream's arguments).
cd /home/js2723/PROJECTS/VUD
while screen -ls | grep -q "\.pfn_classify\b"; do sleep 60; done
[ -f dllm-training/outputs/pfn_classify2d/checkpoint.pt ] || { echo "no checkpoint"; exit 1; }
GPU=0 RUN=pfn_classify2d NAME=MDLM-PFNClassify2D-0.8M srun --jobid=80896 --overlap --cpu-bind=none --export=ALL \
  bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh logistic_regression --x_range "{'x1': [-15, 15, 0.2]}" \
  > dllm-training/tmp/logs/vud_mdlm_pfn_logistic.log 2>&1
