#!/bin/bash
# Dream parallel_pp on logistic_regression for each n given, one after another.
# n=100 is the default and uses the default folder va_parallel_pp.
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
# Pin to one GPU when GPU is set; Slurm exposes every GPU of the allocation to the step.
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES=$GPU; fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
for n in "$@"; do
  if [ "$n" = 100 ]; then folder=Dream-v0-Base-7B/va_parallel_pp; else folder=Dream-v0-Base-7B/va_parallel_pp_n$n; fi
  echo "START n=$n $(date -u)"
  .venv/bin/python run.py toy-classification \
    --model_name Dream-org/Dream-v0-Base-7B \
    --dataset_name logistic_regression \
    --x_range "{'x1': [-15, 15, 0.2]}" \
    --va_method parallel_pp --num_joint_samples "$n" \
    --model_temperature 1 --icl_sample_seed 0 \
    --save_directory "$folder" --resume 1 > "tmp/logs/dream_logistic_pp_n$n.log" 2>&1
  echo "END n=$n status=$? $(date -u)"
done
