#!/bin/bash
# Setting 1: MDLM trained on TabPFN v1's notebook-default prior data -> VUD vanilla on the classification toys.
cd /home/js2723/PROJECTS/VUD
export CUDA_VISIBLE_DEVICES=0 VUD_MDLM_DEVICE=cuda
CKPT=/home/js2723/PROJECTS/VUD/dllm-training/outputs/tabpfn_original_classify2d/checkpoint.pt
for dataset in moons_1 moons_2 logistic_regression; do
  extra=(); [ "$dataset" = logistic_regression ] && extra=(--x_range "{'x1': [-15, 15, 0.2]}")
  .venv/bin/python run.py toy-classification --model_name "mdlm:$CKPT" --dataset_name "$dataset" --va_method vanilla \
    --model_temperature 1 --icl_sample_seed 0 --save_directory "MDLM-TabPFNNotebook-0.8M/va_vanilla" --resume 1 "${extra[@]}" \
    > tmp/logs/setting1_vud_$dataset.log 2>&1
done
echo 'SETTING 1 COMPLETE'
