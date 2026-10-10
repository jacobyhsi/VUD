#!/bin/bash
# TabPFN-prior MDLM: wait for the bank (screen tabpfn_bank, GPU1) -> train on GPU0 -> evaluate (GPU0)
# alongside VUD vanilla on logistic_regression (GPU0) and moons_1 (GPU1).
cd /home/js2723/PROJECTS/VUD/dllm-training
until [ -f data_banks/tabpfn_cls_3m/meta.json ]; do sleep 60; done
cd /home/js2723/PROJECTS/VUD
GPU=0 WORKERS=8 bash dllm-training/tmp/scripts/bayes_linear_vud_run.sh tabpfn_classify2d &
until [ -f dllm-training/outputs/tabpfn_classify2d/checkpoint.pt ] && ! pgrep -f "train.py --config dllm-training/configs/tabpfn_classify2d" > /dev/null; do sleep 60; done
export RUN=tabpfn_classify2d NAME=MDLM-TabPFNClassify2D-0.8M
GPU=0 bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh logistic_regression --x_range "{'x1': [-15, 15, 0.2]}" \
  > dllm-training/tmp/logs/vud_mdlm_tabpfn_logistic.log 2>&1 &
GPU=1 bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh moons_1 > dllm-training/tmp/logs/vud_mdlm_tabpfn_moons1.log 2>&1 &
wait
