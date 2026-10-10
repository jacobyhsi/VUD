#!/bin/bash
# Exact-Bayes screen of TabPFN's GP member (plus GP-Bernoulli and TabPFN MLP for reference) on both datasets; GPU from $GPU.
cd /home/js2723/PROJECTS/VUD/dllm-training
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 MODE1_MEMBERS=gp_bernoulli,tabpfn,tabpfn_gp MODE1_TAG=tabpfngp
for d in logistic_regression moons_1; do
  ../.venv/bin/python tmp/scripts/mode1_bag.py $d > tmp/logs/mode1_bag_${d}_tabpfngp.log 2>&1 &
done
wait
