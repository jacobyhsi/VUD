#!/bin/bash
set -eu
cd /home/js2723/PROJECTS/VUD
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == login* ]]; then
  echo 'This launcher must run on a Slurm compute node.' >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$1"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
qa_id="$2"
qa_ood="$3"
qa_log_dir=/home/js2723/tmp/vud-vanilla-qa-20260921
trap 'qa_rc=$?; printf "%s\n" "$qa_rc" > "$qa_log_dir/${qa_id}_${qa_ood}.exit"; echo "FINISHED $(date -Is) exit=$qa_rc"' EXIT
echo "START $(date -Is) host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES id=$qa_id ood=$qa_ood job=$SLURM_JOB_ID step=${SLURM_STEP_ID:-unknown}"
/home/js2723/PROJECTS/VUD/.venv/bin/python run.py qa \
  --id "$qa_id" --ood "$qa_ood" \
  --model Dream-org/Dream-v0-Instruct-7B \
  --va_method vanilla --seed 123 --num_seeds 5
