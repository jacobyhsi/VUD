#!/bin/bash
set -eu
cd /home/js2723/PROJECTS/VUD
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
.venv/bin/python -m unittest discover -s dllm-training -p 'test_*.py' -v
