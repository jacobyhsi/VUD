#!/bin/bash
set -euo pipefail
cd /home/js2723/PROJECTS/VUD
export CUDA_VISIBLE_DEVICES=0
test_dir=$(mktemp -d /home/js2723/PROJECTS/VUD/tmp/tabpfn-geometry-test.XXXXXX)
export TMPDIR="$test_dir"
trap 'find "$test_dir" -depth -delete' EXIT
export PYTHONPATH=/home/js2723/PROJECTS/VUD/dllm-training
.venv/bin/python -m unittest test_tabpfn_episodes test_pfn_bank test_mdlm test_classification test_vud_adapter -v
.venv/bin/python dllm-training/generate_tabpfn_episode_bank.py --output "$test_dir/bank" --datasets 64 --chunk 32 --draws 4 --device cuda
.venv/bin/python -c 'import json,sys,pathlib; root=pathlib.Path(sys.argv[1]); c=json.loads(pathlib.Path("dllm-training/configs/tabpfn_classify2d_geometry.json").read_text()); c["data"].update(bank_path=str(root/"bank/geometry"),bank_holdout=20); c["training"].update(steps=4,batch_size=4,eval_every=2,eval_tasks=8); (root/"config.json").write_text(json.dumps(c))' "$test_dir"
.venv/bin/python dllm-training/train.py --config "$test_dir/config.json" --output "$test_dir/model" --device cuda --workers 2 --worker-start-method spawn --max-steps 2 --keep-checkpoints
.venv/bin/python dllm-training/train.py --config "$test_dir/config.json" --output "$test_dir/model" --device cuda --workers 2 --worker-start-method spawn --resume --keep-checkpoints
test -f "$test_dir/model/checkpoint_step_000002.pt"
test -f "$test_dir/model/checkpoint_step_000004.pt"
.venv/bin/python dllm-training/evaluate_tabpfn_conditionals.py --checkpoint "$test_dir/model/checkpoint_step_000002.pt" "$test_dir/model/checkpoint_step_000004.pt" --bank "$test_dir/bank/geometry" --holdout 20 --tasks 2 --output "$test_dir/conditionals" --device cuda --select
test -s "$test_dir/conditionals/selected_checkpoint.txt"
echo 'ALL GEOMETRY TESTS AND CUDA RESUME SMOKE PASSED'
