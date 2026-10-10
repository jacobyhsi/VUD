#!/bin/bash
# VUD vanilla with the 0.8M TabPFN MDLM trained to 60k steps, both datasets on GPU0 (GPU1 runs the 2.7M moons_1 job).
cd /home/js2723/PROJECTS/VUD
export RUN=tabpfn_classify2d_long NAME=MDLM-TabPFNClassify2D-0.8M-60k GPU=0
bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh logistic_regression --x_range "{'x1': [-15, 15, 0.2]}" \
  > dllm-training/tmp/logs/vud_mdlm_tabpfn_long_logistic.log 2>&1 &
bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh moons_1 > dllm-training/tmp/logs/vud_mdlm_tabpfn_long_moons1.log 2>&1 &
wait
