#!/bin/bash
# Round 2 on the ControlledGP50 prior, one change per arm vs the round-1 Geometry run:
#   full: near_query_prob 0.7 -> 1.0, broad_query_prob 0.5 -> 0.75   (train on GPU0)
#   1d:   one_d_prob 0.3 -> 0.5, geometry unchanged                  (train on GPU1, shared with Dream)
# Banks are built on GPU0 (idle at launch; bank generation next to Dream was ~4x slower before).
# Then: validation checkpoint selection -> conditional tests -> predictive eval -> VUD (LR, moons_1, moons_2) ->
# scores against the exact ControlledGP50 reference. Do not edit while running.
set -uo pipefail
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1 VUD_MDLM_DEVICE=cuda TMPDIR=/home/js2723/PROJECTS/VUD/tmp/controlled_gp50_round2_runtime
mkdir -p "$TMPDIR" tmp/logs
declare -A BANKARGS=([full]="--near-query-prob 1.0 --broad-query-prob 0.75" [1d]="--one-d-prob 0.5")
declare -A GPU=([full]=0 [1d]=1)
declare -A NAME=([full]=MDLM-TabPFN-ControlledGP50-GeometryFull-0.8M [1d]=MDLM-TabPFN-ControlledGP50-Geometry1D-0.8M)
for arm in full 1d; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python dllm-training/generate_tabpfn_episode_bank.py \
    --output dllm-training/data_banks/controlled_gp50_episode_pairs_${arm}_3m --datasets 3006464 --chunk 8192 --device cuda \
    --controlled-gp-fraction 0.5 ${BANKARGS[$arm]} > tmp/logs/controlled_gp50_round2_${arm}_bank.log 2>&1 &
done
wait
for arm in full 1d; do
  [ -f dllm-training/data_banks/controlled_gp50_episode_pairs_${arm}_3m/geometry/meta.json ] || { echo "bank $arm failed"; exit 1; }
done
arm_pipeline() {
  local arm=$1 out=dllm-training/outputs/tabpfn_controlled_gp50_geometry_$1
  local bank=dllm-training/data_banks/controlled_gp50_episode_pairs_$1_3m
  export CUDA_VISIBLE_DEVICES=${GPU[$arm]}
  .venv/bin/python dllm-training/train.py --config dllm-training/configs/tabpfn_controlled_gp50_geometry_$arm.json --output "$out" \
    --device cuda --workers 4 --worker-start-method spawn --keep-checkpoints > tmp/logs/controlled_gp50_round2_${arm}_train.log 2>&1 || return 1
  .venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py --checkpoint "$out"/checkpoint_step_*.pt --bank "$bank/geometry" \
    --split validation --tasks 1024 --select --output "$out/conditional_validation" > tmp/logs/controlled_gp50_round2_${arm}_conditional_validation.log 2>&1 || return 1
  local checkpoint; read -r checkpoint < "$out/conditional_validation/selected_checkpoint.txt"
  # Test on its own geometry and on round 1's geometry bank (a common yardstick across arms).
  .venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py --checkpoint "$checkpoint" --bank "$bank/geometry" --split test \
    --tasks 1024 --output "$out/conditional_test_own" > tmp/logs/controlled_gp50_round2_${arm}_conditional_test_own.log 2>&1
  .venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py --checkpoint "$checkpoint" \
    --bank dllm-training/data_banks/controlled_gp50_episode_pairs_3m/geometry --split test \
    --tasks 1024 --output "$out/conditional_test_round1" > tmp/logs/controlled_gp50_round2_${arm}_conditional_test_round1.log 2>&1
  .venv/bin/python dllm-training/evaluate_vud_classification.py --checkpoint "$checkpoint" --output "$out/evaluation_vud_datasets" \
    --device cuda > tmp/logs/controlled_gp50_round2_${arm}_predictive_eval.log 2>&1
  for dataset in logistic_regression moons_1 moons_2; do
    extra=(); [ "$dataset" = logistic_regression ] && extra=(--x_range "{'x1': [-15, 15, 0.2]}")
    .venv/bin/python run.py toy-classification --model_name "mdlm:$checkpoint" --dataset_name "$dataset" --va_method vanilla \
      --model_temperature 1 --icl_sample_seed 0 --save_directory "${NAME[$arm]}/va_vanilla" --resume 1 "${extra[@]}" \
      > tmp/logs/controlled_gp50_round2_${arm}_vud_$dataset.log 2>&1
  done
}
arm_pipeline full & p1=$!
arm_pipeline 1d & p2=$!
wait $p1; s1=$?; wait $p2; s2=$?
[ $s1 -eq 0 ] && [ $s2 -eq 0 ] || echo "An arm failed (full=$s1, 1d=$s2); inspect its logs"
cd dllm-training
for d in moons_1 logistic_regression; do
  CUDA_VISIBLE_DEVICES=0 SCORE_ONLY=1 EXTRA_MODELS=MDLM-TabPFN-ControlledGP50-Geometry-0.8M,${NAME[full]},${NAME[1d]} ../.venv/bin/python \
    tmp/scripts/conditional_reference.py $d > ../tmp/logs/controlled_gp50_round2_reference_score_$d.log 2>&1
done
echo 'CONTROLLED GP50 GEOMETRY ROUND 2 COMPLETE'
