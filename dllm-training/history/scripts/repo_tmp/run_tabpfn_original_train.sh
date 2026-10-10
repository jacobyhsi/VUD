#!/bin/bash
# MDLM trained on TabPFN v1's ORIGINAL prior code: 3M-dataset bank (24 CPU shards, seeds 1-24; test set is seed 0),
# B6's recipe (0.8M DiT, 20k steps) on GPU 0, then score it on the 969 held-out test datasets and summarise.
cd /home/js2723/PROJECTS/VUD/dllm-training
unset SLURM_PROCID LOCAL_RANK
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
L=tmp/logs/tabpfn_original
SH=data_banks/tabpfn_original_shards
mkdir -p $SH
for s in $(seq 1 24); do
  [ -f $SH/shard_$s.npz ] || CUDA_VISIBLE_DEVICES= ../.venv/bin/python tmp/scripts/tabpfn_original_generate.py 125000 $SH/shard_$s.npz $s > $L/bank_shard_$s.log 2>&1 &
done
wait
../.venv/bin/python tmp/scripts/tabpfn_original_join_bank.py $SH data_banks/tabpfn_original_cls_3m > $L/bank_join.log 2>&1 || exit 1
cat $L/bank_join.log
rm -f $SH/shard_*.npz && rmdir $SH
OUT=outputs/tabpfn_original_classify2d
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python train.py --config configs/tabpfn_original_classify2d.json --output $OUT \
  --device cuda --workers 4 --worker-start-method spawn > $L/train.log 2>&1 || exit 1
CKPT=$(ls $OUT/checkpoint*.pt | sort | tail -1)
echo "trained: $CKPT"
for k in 0 1; do
  EVAL_MODELS=MDLM_tabpfn_original=$CKPT CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tmp/scripts/tabpfn_original_eval.py \
    $L/datasets_$k.npz $L/preds_new_$k.csv $((k * 500)) > $L/eval_new_$k.log 2>&1 || exit 1
done
../.venv/bin/python tmp/scripts/tabpfn_original_summary.py $L/preds_new_0.csv $L/preds_new_1.csv > $L/summary_new.log 2>&1
echo 'TABPFN ORIGINAL TRAIN COMPLETE'
