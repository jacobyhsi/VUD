#!/bin/bash
# Resume Dream spirals vanilla (D=200, 80x80 grid) with the original arguments.
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES=$GPU; fi
exec .venv/bin/python run.py toy-classification \
  --model_name Dream-org/Dream-v0-Base-7B \
  --dataset_name spirals --D_size 200 \
  --x_range "{'x1': [-4, 4, 0.1], 'x2': [-4, 4, 0.1]}" \
  --va_method vanilla --model_temperature 1 --icl_sample_seed 0 \
  --save_directory Dream-v0-Base-7B/va_vanilla --resume 1
