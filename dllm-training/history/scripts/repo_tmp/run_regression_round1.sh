#!/bin/bash
# Regression round 1 (the classification recipe): 50% het GP (lengthscale 0.1-3) + 50% TabPFN v1 MLP in regression mode,
# geometry episodes + 75% structured layouts, 0.8M MDLM, 40k steps, x and y z-scored by the context.
# bank (GPU0) -> train (GPU0) -> pick the checkpoint with the lowest held-out validation loss ->
# VUD vanilla on linear_regression (GPU0) and gaps (GPU1) in parallel -> TU/AU/EU breakdown. Do not edit while running.
set -uo pipefail
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1 VUD_MDLM_DEVICE=cuda TMPDIR=/home/js2723/PROJECTS/VUD/tmp/regression_round1_runtime
mkdir -p "$TMPDIR" tmp/logs
NAME=MDLM-Regress-HetGP50-Geometry-0.8M
out=dllm-training/outputs/regression_hetgp50_geometry
CUDA_VISIBLE_DEVICES=0 .venv/bin/python dllm-training/generate_regression_episode_bank.py \
  --output dllm-training/data_banks/regression_hetgp50_episodes_3m --datasets 3006464 --chunk 8192 --device cuda \
  > tmp/logs/regression_round1_bank.log 2>&1 || { echo 'bank failed'; exit 1; }
CUDA_VISIBLE_DEVICES=0 .venv/bin/python dllm-training/train.py --config dllm-training/configs/regression_hetgp50_geometry.json \
  --output $out --device cuda --workers 4 --worker-start-method spawn --keep-checkpoints > tmp/logs/regression_round1_train.log 2>&1 \
  || { echo 'training failed'; exit 1; }
checkpoint=$(.venv/bin/python - <<PY
import json
rows = [json.loads(l) for l in open('$out/metrics.jsonl') if 'validation_loss' in l]
best = min(rows, key=lambda r: r['validation_loss'])
print(f"/home/js2723/PROJECTS/VUD/$out/checkpoint_step_{best['step']:06d}.pt")
PY
)
echo "$checkpoint" > tmp/logs/regression_round1_selected.txt
for spec in "linear_regression 0" "gaps 1"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$2 .venv/bin/python run.py toy-regression --model_name "mdlm:$checkpoint" --dataset_name $1 \
    --va_method vanilla --model_temperature 1 --icl_sample_seed 0 --save_directory "$NAME/va_vanilla" --resume 1 \
    > tmp/logs/regression_round1_vud_$1.log 2>&1 &
done
wait
.venv/bin/python dllm-training/tmp/scripts/regression_breakdown.py Dream-v0-Base-7B MDLM-BayesRegressGP-0.8M $NAME \
  > tmp/logs/regression_round1_breakdown.log 2>&1
echo 'REGRESSION ROUND 1 COMPLETE'
