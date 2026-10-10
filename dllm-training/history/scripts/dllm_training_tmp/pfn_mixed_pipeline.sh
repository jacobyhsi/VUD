#!/bin/bash
# Mixed labels (50% paper median threshold, 50% sigmoid -> Bernoulli), BoTorchOS GP prior, noise 0.001:
# bank -> train (GPU 0) -> VUD-dataset evaluation -> VUD vanilla on logistic_regression and moons_1.
cd /home/js2723/PROJECTS/VUD/dllm-training
export TMPDIR=/tmp/js2723-bank-mixed && mkdir -p $TMPDIR
../.venv/bin/python generate_pfn_bank.py --output data_banks/pfn_gp_cls_noise1e-3_os2_mixed_3m --datasets 3000000 \
  --workers 24 --noise 0.001 --outputscale-concentration 2.0 --median-label-prob 0.5 > tmp/logs/pfn_bank_mixed.log 2>&1 || exit 1
rm -rf $TMPDIR
cd /home/js2723/PROJECTS/VUD
GPU=0 WORKERS=8 bash dllm-training/tmp/scripts/bayes_linear_vud_run.sh pfn_classify2d_noise1e-3_os2_mixed || exit 1
export GPU=0 RUN=pfn_classify2d_noise1e-3_os2_mixed NAME=MDLM-PFNClassify2D-MixedLabels-0.8M
bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh logistic_regression --x_range "{'x1': [-15, 15, 0.2]}" \
  > dllm-training/tmp/logs/vud_mdlm_pfn_mixed_logistic.log 2>&1
bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh moons_1 > dllm-training/tmp/logs/vud_mdlm_pfn_mixed_moons1.log 2>&1
