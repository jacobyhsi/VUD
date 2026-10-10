#!/bin/bash
set -eu
cd /home/js2723/PROJECTS/VUD
export CUDA_VISIBLE_DEVICES=""
export VUD_MDLM_DEVICE=cpu
export OMP_NUM_THREADS=2
.venv/bin/python tmp/scripts/test_mdlm_vud.py /home/js2723/PROJECTS/VUD/tmp/mdlm-smoke-OKmVLC/checkpoint.pt
