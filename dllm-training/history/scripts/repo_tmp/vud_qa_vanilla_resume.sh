#!/bin/bash
set -eu
cd /home/js2723/PROJECTS/VUD
export CUDA_VISIBLE_DEVICES="$1"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
qa_id="$2"
qa_ood="${3:-}"
qa_args=(--id "$qa_id" --model Dream-org/Dream-v0-Instruct-7B --va_method vanilla --seed 123 --num_seeds 5 --resume)
if [[ -n "$qa_ood" ]]; then
  qa_args+=(--ood "$qa_ood")
fi
exec .venv/bin/python run.py qa "${qa_args[@]}"
