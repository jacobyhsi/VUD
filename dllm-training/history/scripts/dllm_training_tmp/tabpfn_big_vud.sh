#!/bin/bash
# VUD vanilla with the 2.68M TabPFN MDLM: logistic_regression on GPU0, moons_1 on GPU1.
cd /home/js2723/PROJECTS/VUD
export RUN=tabpfn_classify2d_big NAME=MDLM-TabPFNClassify2D-2.7M
GPU=0 bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh logistic_regression --x_range "{'x1': [-15, 15, 0.2]}" \
  > dllm-training/tmp/logs/vud_mdlm_tabpfn_big_logistic.log 2>&1 &
GPU=1 bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh moons_1 > dllm-training/tmp/logs/vud_mdlm_tabpfn_big_moons1.log 2>&1 &
wait
