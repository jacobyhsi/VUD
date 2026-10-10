#!/bin/bash
# Full TabPFN v1 MLP prior (non-causal + causal branches, 50/50 as the released is_causal choice):
# bank (both GPUs, joined) -> train 0.8M -> VUD vanilla on logistic_regression and moons_1.
# Every stage picks the least-loaded GPU when it starts. Runs inside an srun step on the allocation.
cd /home/js2723/PROJECTS/VUD/dllm-training
least_loaded() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
other() { [ "$1" = 0 ] && echo 1 || echo 0; }
BANK=data_banks/tabpfn_full_cls_3m
BANK_ARGS="--output $BANK --label-prior-bag tabpfn=0,tabpfn_causal=0 --device cuda --chunk 8192 --join"
FIRST=$(least_loaded)
CUDA_VISIBLE_DEVICES=$FIRST PYTHONUNBUFFERED=1 TMPDIR=/tmp ../.venv/bin/python generate_pfn_bank.py $BANK_ARGS \
  > tmp/logs/tabpfn_full_bank_gpu$FIRST.log 2>&1 &
until [ -d $BANK/claims ] && [ -f $BANK/labelling.npy ]; do sleep 2; done
SECOND=$(other $FIRST)
CUDA_VISIBLE_DEVICES=$SECOND PYTHONUNBUFFERED=1 TMPDIR=/tmp ../.venv/bin/python generate_pfn_bank.py $BANK_ARGS \
  > tmp/logs/tabpfn_full_bank_gpu$SECOND.log 2>&1 &
wait
[ -f $BANK/meta.json ] || { echo "bank failed"; exit 1; }
cd /home/js2723/PROJECTS/VUD
GPU=$(least_loaded) WORKERS=8 bash dllm-training/tmp/scripts/bayes_linear_vud_run.sh tabpfn_full_classify2d || exit 1
export RUN=tabpfn_full_classify2d NAME=MDLM-TabPFNFullClassify2D-0.8M
GPU=$(least_loaded) bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh logistic_regression --x_range "{'x1': [-15, 15, 0.2]}" \
  > dllm-training/tmp/logs/vud_mdlm_tabpfn_full_logistic.log 2>&1 &
sleep 90   # let the first job allocate before choosing the GPU for the second
GPU=$(least_loaded) bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh moons_1 > dllm-training/tmp/logs/vud_mdlm_tabpfn_full_moons1.log 2>&1 &
wait
