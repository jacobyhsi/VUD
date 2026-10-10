#!/bin/bash
# Round 4: the Geometry1D candidate's settings plus structured input layouts (layout_prob 0.5: clusters / random
# smooth curves / correlated Gaussians, standardized per task). The prior over functions is unchanged.
# bank -> train 0.8M -> validation selection -> conditional test -> predictive eval -> VUD (moons_1, moons_2, LR)
# -> reference scores, moons_1 far-field breakdown, far-field calibration. GPU0 (both GPUs idle at launch). Do not edit while running.
set -uo pipefail
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1 VUD_MDLM_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 TMPDIR=/home/js2723/PROJECTS/VUD/tmp/layout_round4_runtime
mkdir -p "$TMPDIR" tmp/logs
NAME=MDLM-TabPFN-ControlledGP50-Geometry1D-Layout-0.8M
CONFIG=tabpfn_controlled_gp50_geometry_1d_layout
bank=dllm-training/data_banks/controlled_gp50_episode_pairs_1d_layout_3m
out=dllm-training/outputs/$CONFIG
.venv/bin/python dllm-training/generate_tabpfn_episode_bank.py --output $bank --datasets 3006464 --chunk 8192 --device cuda \
  --controlled-gp-fraction 0.5 --one-d-prob 0.5 --layout-prob 0.5 > tmp/logs/layout_round4_bank.log 2>&1 || { echo 'bank failed'; exit 1; }
.venv/bin/python dllm-training/train.py --config dllm-training/configs/$CONFIG.json --output $out --device cuda --workers 4 \
  --worker-start-method spawn --keep-checkpoints > tmp/logs/layout_round4_train.log 2>&1 || { echo 'training failed'; exit 1; }
.venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py --checkpoint $out/checkpoint_step_*.pt --bank $bank/geometry \
  --split validation --tasks 1024 --select --output $out/conditional_validation > tmp/logs/layout_round4_conditional_validation.log 2>&1 || exit 1
read -r checkpoint < $out/conditional_validation/selected_checkpoint.txt
.venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py --checkpoint "$checkpoint" --bank $bank/geometry --split test \
  --tasks 1024 --output $out/conditional_test_own > tmp/logs/layout_round4_conditional_test_own.log 2>&1
.venv/bin/python dllm-training/evaluate_vud_classification.py --checkpoint "$checkpoint" --output $out/evaluation_vud_datasets \
  --device cuda > tmp/logs/layout_round4_predictive_eval.log 2>&1
for dataset in moons_1 moons_2 logistic_regression; do
  extra=(); [ "$dataset" = logistic_regression ] && extra=(--x_range "{'x1': [-15, 15, 0.2]}")
  .venv/bin/python run.py toy-classification --model_name "mdlm:$checkpoint" --dataset_name "$dataset" --va_method vanilla \
    --model_temperature 1 --icl_sample_seed 0 --save_directory "$NAME/va_vanilla" --resume 1 "${extra[@]}" \
    > tmp/logs/layout_round4_vud_$dataset.log 2>&1
done
cd dllm-training
for d in moons_1 logistic_regression; do
  SCORE_ONLY=1 EXTRA_MODELS=MDLM-TabPFN-ControlledGP50-Geometry1D-0.8M,$NAME ../.venv/bin/python tmp/scripts/conditional_reference.py $d \
    > ../tmp/logs/layout_round4_reference_score_$d.log 2>&1
done
../.venv/bin/python tmp/scripts/farfield_breakdown.py MDLM-TabPFN-ControlledGP50-Geometry1D-0.8M $NAME MDLM-BayesClassify2D-2.5M \
  > ../tmp/logs/layout_round4_breakdown.log 2>&1
../.venv/bin/python tmp/scripts/farfield_calibration.py "with layouts=$checkpoint" > ../tmp/logs/layout_round4_calibration.log 2>&1
echo 'LAYOUT ROUND 4 COMPLETE'
