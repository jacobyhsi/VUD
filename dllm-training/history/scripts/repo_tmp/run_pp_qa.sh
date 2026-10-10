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
qa_ood="${3:-}"
qa_log_dir=/home/js2723/PROJECTS/VUD/tmp/logs/qa/parallel/20260926
mkdir -p "$qa_log_dir"
if [[ -n "$qa_ood" ]]; then
  qa_tag="${qa_id}_${qa_ood}"
else
  qa_tag="${qa_id}_ID"
fi
exec > "$qa_log_dir/${qa_tag}.log" 2>&1
trap 'qa_rc=$?; printf "%s\n" "$qa_rc" > "$qa_log_dir/${qa_tag}.exit"; echo "FINISHED $(date -Is) exit=$qa_rc"' EXIT
echo "START $(date -Is) host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES id=$qa_id ood=${qa_ood:-none} job=$SLURM_JOB_ID step=${SLURM_STEP_ID:-unknown}"
qa_cmd=(
  /home/js2723/PROJECTS/VUD/.venv/bin/python run.py qa
  --id "$qa_id"
  --model Dream-org/Dream-v0-Instruct-7B
  --va_method parallel_pp --num_joint_samples 100 --seed 123 --num_seeds 5 --resume
)
if [[ -n "$qa_ood" ]]; then
  qa_cmd+=(--ood "$qa_ood")
fi
"${qa_cmd[@]}"
