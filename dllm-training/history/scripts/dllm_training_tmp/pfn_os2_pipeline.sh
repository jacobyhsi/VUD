#!/bin/bash
# BoTorch outputscale prior Gamma(2.0, 0.15): bank -> train (GPU 0) -> VUD-dataset evaluation -> VUD vanilla on logistic_regression.
cd /home/js2723/PROJECTS/VUD/dllm-training
export TMPDIR=/tmp/js2723-bank-os2 && mkdir -p $TMPDIR
../.venv/bin/python generate_pfn_bank.py --output data_banks/pfn_gp_cls_noise1e-3_os2_3m --datasets 3000000 \
  --workers 24 --noise 0.001 --outputscale-concentration 2.0 > tmp/logs/pfn_bank_noise1e-3_os2.log 2>&1 || exit 1
rm -rf $TMPDIR
cd /home/js2723/PROJECTS/VUD
GPU=0 WORKERS=8 bash dllm-training/tmp/scripts/bayes_linear_vud_run.sh pfn_classify2d_noise1e-3_os2 || exit 1
GPU=0 RUN=pfn_classify2d_noise1e-3_os2 NAME=MDLM-PFNClassify2D-BoTorchOS-0.8M bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh \
  logistic_regression --x_range "{'x1': [-15, 15, 0.2]}" > dllm-training/tmp/logs/vud_mdlm_pfn_os2_logistic.log 2>&1
