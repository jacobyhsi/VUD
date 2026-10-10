#!/bin/bash
# Round 5, one change per arm vs the round-4 candidate (Geometry1D + layouts 0.5, 20k steps):
#   long:     resume the candidate (copied run) to 40k steps, same bank        -> GPU1
#   layout75: layout_prob 0.5 -> 0.75 (new bank), 20k steps                    -> GPU0
# Both GPUs idle at launch. Then per arm: validation selection -> conditional test -> predictive eval -> VUD
# (moons_1, moons_2, LR); finally reference scores, moons_1 far-field breakdown and calibration. Do not edit while running.
set -uo pipefail
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1 VUD_MDLM_DEVICE=cuda TMPDIR=/home/js2723/PROJECTS/VUD/tmp/round5_runtime
mkdir -p "$TMPDIR" tmp/logs
declare -A NAME=([long]=MDLM-TabPFN-ControlledGP50-Geometry1D-Layout-40k-0.8M [layout75]=MDLM-TabPFN-ControlledGP50-Geometry1D-Layout75-0.8M)
declare -A CONFIG=([long]=tabpfn_controlled_gp50_geometry_1d_layout_long [layout75]=tabpfn_controlled_gp50_geometry_1d_layout75)
declare -A BANK=([long]=dllm-training/data_banks/controlled_gp50_episode_pairs_1d_layout_3m [layout75]=dllm-training/data_banks/controlled_gp50_episode_pairs_1d_layout75_3m)
arm_pipeline() {
  local arm=$1 out=dllm-training/outputs/${CONFIG[$1]} bank=${BANK[$1]} resume=$3
  export CUDA_VISIBLE_DEVICES=$2
  .venv/bin/python dllm-training/train.py --config dllm-training/configs/${CONFIG[$arm]}.json --output "$out" $resume \
    --device cuda --workers 4 --worker-start-method spawn --keep-checkpoints > tmp/logs/round5_${arm}_train.log 2>&1 || return 1
  .venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py --checkpoint "$out"/checkpoint_step_*.pt --bank "$bank/geometry" \
    --split validation --tasks 1024 --select --output "$out/conditional_validation" > tmp/logs/round5_${arm}_conditional_validation.log 2>&1 || return 1
  local checkpoint; read -r checkpoint < "$out/conditional_validation/selected_checkpoint.txt"
  .venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py --checkpoint "$checkpoint" --bank "$bank/geometry" --split test \
    --tasks 1024 --output "$out/conditional_test_own" > tmp/logs/round5_${arm}_conditional_test_own.log 2>&1
  .venv/bin/python dllm-training/evaluate_vud_classification.py --checkpoint "$checkpoint" --output "$out/evaluation_vud_datasets" \
    --device cuda > tmp/logs/round5_${arm}_predictive_eval.log 2>&1
  for dataset in moons_1 moons_2 logistic_regression; do
    extra=(); [ "$dataset" = logistic_regression ] && extra=(--x_range "{'x1': [-15, 15, 0.2]}")
    .venv/bin/python run.py toy-classification --model_name "mdlm:$checkpoint" --dataset_name "$dataset" --va_method vanilla \
      --model_temperature 1 --icl_sample_seed 0 --save_directory "${NAME[$arm]}/va_vanilla" --resume 1 "${extra[@]}" \
      > tmp/logs/round5_${arm}_vud_$dataset.log 2>&1
  done
  echo "$checkpoint" > tmp/logs/round5_${arm}_selected.txt
}
arm_pipeline long 1 --resume & plong=$!
CUDA_VISIBLE_DEVICES=0 .venv/bin/python dllm-training/generate_tabpfn_episode_bank.py --output ${BANK[layout75]} --datasets 3006464 \
  --chunk 8192 --device cuda --controlled-gp-fraction 0.5 --one-d-prob 0.5 --layout-prob 0.75 > tmp/logs/round5_layout75_bank.log 2>&1 \
  && arm_pipeline layout75 0 "" & play=$!
wait $plong; s1=$?; wait $play; s2=$?
[ $s1 -eq 0 ] && [ $s2 -eq 0 ] || echo "An arm failed (long=$s1, layout75=$s2); inspect its logs"
cd dllm-training
CAND=MDLM-TabPFN-ControlledGP50-Geometry1D-Layout-0.8M
for d in moons_1 logistic_regression; do
  SCORE_ONLY=1 EXTRA_MODELS=$CAND,${NAME[long]},${NAME[layout75]} ../.venv/bin/python tmp/scripts/conditional_reference.py $d \
    > ../tmp/logs/round5_reference_score_$d.log 2>&1
done
../.venv/bin/python tmp/scripts/farfield_breakdown.py $CAND ${NAME[long]} ${NAME[layout75]} MDLM-BayesClassify2D-2.5M > ../tmp/logs/round5_breakdown.log 2>&1
args=(); for arm in long layout75; do [ -f ../tmp/logs/round5_${arm}_selected.txt ] && args+=("$arm=$(cat ../tmp/logs/round5_${arm}_selected.txt)"); done
../.venv/bin/python tmp/scripts/farfield_calibration.py "${args[@]}" > ../tmp/logs/round5_calibration.log 2>&1
echo 'ROUND 5 COMPLETE'
