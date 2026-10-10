#!/bin/bash
# Round 3, aimed at moons_1 far-field overconfidence; one change per arm vs the Geometry1D candidate:
#   gp75: prior 75% controlled GP / 25% TabPFN MLP (bank on GPU0, then train 0.8M on GPU1 next to Dream)
#   big:  2.7M model (6 x 192) on the candidate's own bank (train on GPU0 from the start)
# Then per arm: validation checkpoint selection -> conditional test -> predictive eval -> VUD (LR, moons_1, moons_2);
# finally reference scores and the moons_1 far-field breakdown. GPUs chosen from nvidia-smi at launch. Do not edit while running.
set -uo pipefail
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1 VUD_MDLM_DEVICE=cuda TMPDIR=/home/js2723/PROJECTS/VUD/tmp/farfield_round3_runtime
mkdir -p "$TMPDIR" tmp/logs
declare -A NAME=([gp75]=MDLM-TabPFN-ControlledGP75-Geometry1D-0.8M [big]=MDLM-TabPFN-ControlledGP50-Geometry1D-2.7M)
declare -A CONFIG=([gp75]=tabpfn_controlled_gp75_geometry_1d [big]=tabpfn_controlled_gp50_geometry_1d_big)
declare -A BANK=([gp75]=dllm-training/data_banks/controlled_gp75_episode_pairs_1d_3m [big]=dllm-training/data_banks/controlled_gp50_episode_pairs_1d_3m)
arm_pipeline() {
  local arm=$1 out=dllm-training/outputs/${CONFIG[$1]} bank=${BANK[$1]}
  export CUDA_VISIBLE_DEVICES=$2
  .venv/bin/python dllm-training/train.py --config dllm-training/configs/${CONFIG[$arm]}.json --output "$out" \
    --device cuda --workers 4 --worker-start-method spawn --keep-checkpoints > tmp/logs/farfield_round3_${arm}_train.log 2>&1 || return 1
  .venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py --checkpoint "$out"/checkpoint_step_*.pt --bank "$bank/geometry" \
    --split validation --tasks 1024 --select --output "$out/conditional_validation" > tmp/logs/farfield_round3_${arm}_conditional_validation.log 2>&1 || return 1
  local checkpoint; read -r checkpoint < "$out/conditional_validation/selected_checkpoint.txt"
  .venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py --checkpoint "$checkpoint" --bank "$bank/geometry" --split test \
    --tasks 1024 --output "$out/conditional_test_own" > tmp/logs/farfield_round3_${arm}_conditional_test_own.log 2>&1
  .venv/bin/python dllm-training/evaluate_vud_classification.py --checkpoint "$checkpoint" --output "$out/evaluation_vud_datasets" \
    --device cuda > tmp/logs/farfield_round3_${arm}_predictive_eval.log 2>&1
  for dataset in moons_1 moons_2 logistic_regression; do
    extra=(); [ "$dataset" = logistic_regression ] && extra=(--x_range "{'x1': [-15, 15, 0.2]}")
    .venv/bin/python run.py toy-classification --model_name "mdlm:$checkpoint" --dataset_name "$dataset" --va_method vanilla \
      --model_temperature 1 --icl_sample_seed 0 --save_directory "${NAME[$arm]}/va_vanilla" --resume 1 "${extra[@]}" \
      > tmp/logs/farfield_round3_${arm}_vud_$dataset.log 2>&1
  done
}
arm_pipeline big 0 & pbig=$!
CUDA_VISIBLE_DEVICES=0 .venv/bin/python dllm-training/generate_tabpfn_episode_bank.py --output ${BANK[gp75]} --datasets 3006464 \
  --chunk 8192 --device cuda --controlled-gp-fraction 0.75 --one-d-prob 0.5 > tmp/logs/farfield_round3_gp75_bank.log 2>&1 \
  && arm_pipeline gp75 1 & pgp=$!
wait $pbig; s1=$?; wait $pgp; s2=$?
[ $s1 -eq 0 ] && [ $s2 -eq 0 ] || echo "An arm failed (big=$s1, gp75=$s2); inspect its logs"
cd dllm-training
for d in moons_1 logistic_regression; do
  CUDA_VISIBLE_DEVICES=0 SCORE_ONLY=1 EXTRA_MODELS=MDLM-TabPFN-ControlledGP50-Geometry1D-0.8M,${NAME[gp75]},${NAME[big]} \
    ../.venv/bin/python tmp/scripts/conditional_reference.py $d > ../tmp/logs/farfield_round3_reference_score_$d.log 2>&1
done
../.venv/bin/python tmp/scripts/farfield_breakdown.py MDLM-TabPFN-ControlledGP50-Geometry1D-0.8M ${NAME[gp75]} ${NAME[big]} \
  MDLM-BayesClassify2D-2.5M > ../tmp/logs/farfield_round3_breakdown.log 2>&1
echo 'FAR-FIELD ROUND 3 COMPLETE'
