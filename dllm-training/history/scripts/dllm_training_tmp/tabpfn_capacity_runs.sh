#!/bin/bash
# Bigger TabPFN MDLM (GPU0, fresh) and the 0.8M run continued to 60k steps (GPU1); usage: <big|long>
cd /home/js2723/PROJECTS/VUD
export TMPDIR=/tmp/js2723-mdlm-tabpfn-$1 && mkdir -p $TMPDIR
if [ "$1" = big ]; then GPU=0; EXTRA=""; else GPU=1; EXTRA="--resume"; fi
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python dllm-training/train.py --config dllm-training/configs/tabpfn_classify2d_$1.json \
  --output dllm-training/outputs/tabpfn_classify2d_$1 --device cuda --workers 8 $EXTRA \
  > dllm-training/tmp/logs/mdlm_diag/tabpfn_classify2d_$1.log 2>&1
rm -rf $TMPDIR
