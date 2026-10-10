#!/bin/bash
# ControlledGP50 prior with paired VUD-geometry / IID-control episodes (your TabPFN geometry design, new prior).
# bank (GPU0) -> train geometry (GPU0) + iid_control (GPU1) -> validation checkpoint selection -> conditional tests ->
# predictive eval -> VUD on logistic_regression, moons_1, moons_2 -> score against the exact conditional reference.
# GPUs were chosen from nvidia-smi at launch (GPU0 idle, GPU1 holds the Dream spirals run). Do not edit while running.
set -uo pipefail
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1 VUD_MDLM_DEVICE=cuda TMPDIR=/home/js2723/PROJECTS/VUD/tmp/controlled_gp50_geometry_runtime
mkdir -p "$TMPDIR" tmp/logs
bank=dllm-training/data_banks/controlled_gp50_episode_pairs_3m
CUDA_VISIBLE_DEVICES=0 .venv/bin/python dllm-training/generate_tabpfn_episode_bank.py --output "$bank" --datasets 3006464 \
  --chunk 8192 --device cuda --controlled-gp-fraction 0.5 > tmp/logs/controlled_gp50_geometry_bank.log 2>&1 || { echo 'bank failed'; exit 1; }

declare -A GPU=([geometry]=0 [iid_control]=1)
declare -A NAME=([geometry]=MDLM-TabPFN-ControlledGP50-Geometry-0.8M [iid_control]=MDLM-TabPFN-ControlledGP50-IIDControl-0.8M)
arm_pipeline() {
  local arm=$1 out=dllm-training/outputs/tabpfn_controlled_gp50_$1
  export CUDA_VISIBLE_DEVICES=${GPU[$arm]}
  .venv/bin/python dllm-training/train.py --config dllm-training/configs/tabpfn_controlled_gp50_$arm.json --output "$out" \
    --device cuda --workers 4 --worker-start-method spawn --keep-checkpoints > tmp/logs/controlled_gp50_${arm}_train.log 2>&1 || return 1
  .venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py --checkpoint "$out"/checkpoint_step_*.pt --bank "$bank/geometry" \
    --split validation --tasks 1024 --select --output "$out/conditional_validation" > tmp/logs/controlled_gp50_${arm}_conditional_validation.log 2>&1 || return 1
  local checkpoint; read -r checkpoint < "$out/conditional_validation/selected_checkpoint.txt"
  for domain in iid geometry; do
    .venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py --checkpoint "$checkpoint" --bank "$bank/$domain" --split test \
      --tasks 1024 --output "$out/conditional_test_$domain" > tmp/logs/controlled_gp50_${arm}_conditional_test_$domain.log 2>&1
  done
  .venv/bin/python dllm-training/evaluate_vud_classification.py --checkpoint "$checkpoint" --output "$out/evaluation_vud_datasets" \
    --device cuda > tmp/logs/controlled_gp50_${arm}_predictive_eval.log 2>&1
  for dataset in logistic_regression moons_1 moons_2; do
    extra=(); [ "$dataset" = logistic_regression ] && extra=(--x_range "{'x1': [-15, 15, 0.2]}")
    .venv/bin/python run.py toy-classification --model_name "mdlm:$checkpoint" --dataset_name "$dataset" --va_method vanilla \
      --model_temperature 1 --icl_sample_seed 0 --save_directory "${NAME[$arm]}/va_vanilla" --resume 1 "${extra[@]}" \
      > tmp/logs/controlled_gp50_${arm}_vud_$dataset.log 2>&1
  done
}
arm_pipeline geometry & p1=$!
arm_pipeline iid_control & p2=$!
# Held-out moons_2 for the baseline ControlledGP50 checkpoint, on GPU0 while the arms train.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python run.py toy-classification \
  --model_name mdlm:/home/js2723/PROJECTS/VUD/dllm-training/outputs/tabpfn_controlled_gp50_classify2d/checkpoint.pt \
  --dataset_name moons_2 --va_method vanilla --model_temperature 1 --icl_sample_seed 0 \
  --save_directory MDLM-TabPFN-ControlledGP50-0.8M/va_vanilla --resume 1 > tmp/logs/controlled_gp50_baseline_vud_moons_2.log 2>&1
wait $p1; s1=$?; wait $p2; s2=$?
[ $s1 -eq 0 ] && [ $s2 -eq 0 ] || echo "An arm failed (geometry=$s1, iid_control=$s2); inspect its logs"
cd dllm-training
for d in moons_1 logistic_regression; do
  CUDA_VISIBLE_DEVICES=0 SCORE_ONLY=1 EXTRA_MODELS=${NAME[geometry]},${NAME[iid_control]} ../.venv/bin/python \
    tmp/scripts/conditional_reference.py $d > ../tmp/logs/controlled_gp50_geometry_reference_score_$d.log 2>&1
done
echo 'CONTROLLED GP50 GEOMETRY ABLATION COMPLETE'
