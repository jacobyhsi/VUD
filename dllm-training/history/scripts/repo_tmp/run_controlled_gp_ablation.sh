#!/usr/bin/env bash
# Run inside screen + srun on the existing allocation. GPU1's Dream run is untouched.
set -euo pipefail
cd /home/js2723/PROJECTS/VUD
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 VUD_MDLM_DEVICE=cuda
export PYTHONPATH=/home/js2723/PROJECTS/VUD/dllm-training
runtime_dir=$(mktemp -d /home/js2723/PROJECTS/VUD/tmp/controlled-gp-runtime.XXXXXX)
export TMPDIR="$runtime_dir"
cleanup() {
  case "$runtime_dir" in
    /home/js2723/PROJECTS/VUD/tmp/controlled-gp-runtime.*) rm -r -- "$runtime_dir" ;;
    *) echo 'Unexpected runtime path; refusing cleanup' >&2 ;;
  esac
}
trap cleanup EXIT
mkdir -p tmp/logs
bank=dllm-training/data_banks/tabpfn_controlled_gp50_cls_3m
out=dllm-training/outputs/tabpfn_controlled_gp50_classify2d
name=MDLM-TabPFN-ControlledGP50-0.8M
for path in "$bank" "$out" results/toy_classification/moons_1/"$name" results/toy_classification/logistic_regression/"$name"; do
  if [ -e "$path" ]; then echo "Fresh experiment required; $path already exists"; exit 1; fi
done
echo 'STAGE: final controlled-prior tests'
.venv/bin/python -m unittest test_controlled_gp_prior -v > tmp/logs/controlled_gp_final_tests.log 2>&1
echo 'STAGE: generated-task and finite-particle posterior screening'
.venv/bin/python dllm-training/screen_controlled_gp_prior.py \
  --config dllm-training/configs/controlled_gp_prior.json --output tmp/controlled_gp_screen \
  --device cuda --particles 131072 --chunk 1024 \
  > tmp/logs/controlled_gp_screen.log 2>&1
echo 'STAGE: generate 3M-dataset matched mixture bank'
.venv/bin/python dllm-training/generate_pfn_bank.py \
  --output "$bank" --datasets 3000000 --chunk 8192 --device cuda \
  --label-prior-bag tabpfn=0,controlled_gp=0 \
  --controlled-gp-config dllm-training/configs/controlled_gp_prior.json \
  > tmp/logs/controlled_gp50_bank.log 2>&1
echo 'STAGE: train unchanged 0.8M model for 20k steps'
.venv/bin/python dllm-training/train.py \
  --config dllm-training/configs/tabpfn_controlled_gp50_classify2d.json \
  --output "$out" --device cuda --workers 8 --worker-start-method spawn --keep-checkpoints \
  > tmp/logs/controlled_gp50_train.log 2>&1
echo 'STAGE: held-out conditional validation/test (no checkpoint selection)'
for split in validation test; do
  .venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py \
    --checkpoint "$out/checkpoint.pt" --bank "$bank" --split "$split" --tasks 1024 \
    --output "$out/conditional_$split" --device cuda \
    > "tmp/logs/controlled_gp50_conditional_$split.log" 2>&1
done
echo 'STAGE: 40-context logistic/moons predictive evaluation'
.venv/bin/python dllm-training/evaluate_vud_classification.py \
  --checkpoint "$out/checkpoint.pt" --output "$out/evaluation_vud_datasets" --device cuda \
  > tmp/logs/controlled_gp50_predictive_eval.log 2>&1
echo 'STAGE: unchanged VUD vanilla results and plots'
for dataset in logistic_regression moons_1; do
  extra=(--D_size 30)
  if [ "$dataset" = logistic_regression ]; then extra=(--D_size 15 --x_range "{'x1': [-15, 15, 0.2]}"); fi
  .venv/bin/python run.py toy-classification --model_name "mdlm:$out/checkpoint.pt" \
    --dataset_name "$dataset" --va_method vanilla --model_temperature 1 \
    --icl_sample_seed 0 --save_directory "$name/va_vanilla" --resume 1 "${extra[@]}" \
    > "tmp/logs/controlled_gp50_vud_${dataset}.log" 2>&1
done
echo 'CONTROLLED GP TRAINING AND LOGISTIC/MOONS VUD EVALUATIONS COMPLETE'
