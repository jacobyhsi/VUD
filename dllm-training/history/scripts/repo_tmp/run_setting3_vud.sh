#!/bin/bash
# Setting 3: TabPFN v1's released classifier used directly -> VUD vanilla on the classification toys.
# tabpfn:v1 = package defaults (3 ensemble members, logits / 0.8); tabpfn:v1-t1 = the same with the 0.8 temperature undone.
cd /home/js2723/PROJECTS/VUD
unset SLURM_PROCID LOCAL_RANK
export CUDA_VISIBLE_DEVICES=1 VUD_TABPFN_DEVICE=cuda
run_variant() {
  local model=$1 name=$2
  for dataset in moons_1 moons_2 logistic_regression; do
    extra=(); [ "$dataset" = logistic_regression ] && extra=(--x_range "{'x1': [-15, 15, 0.2]}")
    .venv/bin/python run.py toy-classification --model_name "$model" --dataset_name "$dataset" --va_method vanilla \
      --model_temperature 1 --icl_sample_seed 0 --save_directory "$name/va_vanilla" --resume 1 "${extra[@]}" \
      > tmp/logs/setting3_${name}_vud_$dataset.log 2>&1
  done
}
run_variant tabpfn:v1 TabPFN-v1 &
run_variant tabpfn:v1-t1 TabPFN-v1-T1 &
wait
echo 'SETTING 3 COMPLETE'
