#!/bin/bash
# Train + VUD for an existing TabPFN bag bank (bank built by tabpfn_bag_pipeline.sh).
# Usage: tabpfn_bag_train_vud.sh <tag> <results name> <GPU>. Never edit a pipeline script while it runs (bash reads it lazily).
TAG=$1; NAME=$2; GPU=$3
cd /home/js2723/PROJECTS/VUD
[ -f dllm-training/data_banks/tabpfn_${TAG}_cls_3m/meta.json ] || { echo "no bank"; exit 1; }
GPU=$GPU WORKERS=8 bash dllm-training/tmp/scripts/bayes_linear_vud_run.sh tabpfn_${TAG}_classify2d || exit 1
export RUN=tabpfn_${TAG}_classify2d NAME GPU
bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh logistic_regression --x_range "{'x1': [-15, 15, 0.2]}" \
  > dllm-training/tmp/logs/vud_mdlm_tabpfn_${TAG}_logistic.log 2>&1 &
bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh moons_1 > dllm-training/tmp/logs/vud_mdlm_tabpfn_${TAG}_moons1.log 2>&1 &
wait
