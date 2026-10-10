#!/bin/bash
# Run via screen + srun on the existing allocation. GPU1's Dream run is untouched.
set -euo pipefail
cd /home/js2723/PROJECTS/VUD
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 VUD_MDLM_DEVICE=cuda
export TMPDIR=/home/js2723/PROJECTS/VUD/tmp/tabpfn_geometry_runtime
mkdir -p "$TMPDIR" tmp/logs
bank=dllm-training/data_banks/tabpfn_episode_pairs_3m
.venv/bin/python dllm-training/generate_tabpfn_episode_bank.py --output "$bank" --datasets 3006464 --chunk 1024 --device cuda \
  > tmp/logs/tabpfn_geometry_bank.log 2>&1

# First measure existing checkpoints on the new held-out conditional tasks.
.venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py \
  --checkpoint dllm-training/outputs/tabpfn_classify2d/checkpoint.pt dllm-training/outputs/tabpfn_classify2d_long/checkpoint.pt dllm-training/outputs/tabpfn_classify2d_big/checkpoint.pt \
  --bank "$bank/geometry" --split test --tasks 1024 \
  --output dllm-training/outputs/tabpfn_episode_baselines/conditional_test \
  > tmp/logs/tabpfn_geometry_baseline_conditionals.log 2>&1

pids=()
for arm in geometry iid_control; do
  .venv/bin/python dllm-training/train.py \
    --config "dllm-training/configs/tabpfn_classify2d_${arm}.json" \
    --output "dllm-training/outputs/tabpfn_classify2d_${arm}" \
    --device cuda --workers 4 --worker-start-method spawn --keep-checkpoints \
    > "tmp/logs/tabpfn_${arm}_train.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ "$status" -ne 0 ]; then echo 'A training arm failed; inspect training logs'; exit 1; fi

for arm in geometry iid_control; do
  out="dllm-training/outputs/tabpfn_classify2d_${arm}"
  # Same validation tasks and rule for both arms; no final VUD context used.
  .venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py \
    --checkpoint "$out"/checkpoint_step_*.pt --bank "$bank/geometry" \
    --split validation --tasks 1024 --select --output "$out/conditional_validation" \
    > "tmp/logs/tabpfn_${arm}_conditional_validation.log" 2>&1
  read -r checkpoint < "$out/conditional_validation/selected_checkpoint.txt"
  for domain in iid geometry; do
    .venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py \
      --checkpoint "$checkpoint" --bank "$bank/$domain" --split test --tasks 1024 \
      --output "$out/conditional_test_$domain" \
      > "tmp/logs/tabpfn_${arm}_conditional_test_${domain}.log" 2>&1
  done
  .venv/bin/python dllm-training/evaluate_vud_classification.py \
    --checkpoint "$checkpoint" --output "$out/evaluation_vud_datasets" --device cuda \
    > "tmp/logs/tabpfn_${arm}_predictive_eval.log" 2>&1
  if [ "$arm" = geometry ]; then name=MDLM-TabPFNClassify2D-Geometry-0.8M; else name=MDLM-TabPFNClassify2D-IIDControl-0.8M; fi
  for dataset in logistic_regression moons_1; do
    extra=()
    if [ "$dataset" = logistic_regression ]; then extra=(--x_range "{'x1': [-15, 15, 0.2]}"); fi
    .venv/bin/python run.py toy-classification --model_name "mdlm:$checkpoint" \
      --dataset_name "$dataset" --va_method vanilla --model_temperature 1 \
      --icl_sample_seed 0 --save_directory "$name/va_vanilla" --resume 1 "${extra[@]}" \
      > "tmp/logs/tabpfn_${arm}_vud_${dataset}.log" 2>&1
  done
done
echo 'TABPFN GEOMETRY ABLATION AND EVALUATIONS COMPLETE'
