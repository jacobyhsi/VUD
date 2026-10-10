#!/bin/bash
# Train MDLM-PFNClassify2D with local contexts (GPU 0) -> VUD-dataset evaluation -> VUD vanilla on logistic_regression.
cd /home/js2723/PROJECTS/VUD
GPU=0 WORKERS=8 bash dllm-training/tmp/scripts/bayes_linear_vud_run.sh pfn_classify2d_noise1e-3_localctx || exit 1
GPU=0 RUN=pfn_classify2d_noise1e-3_localctx NAME=MDLM-PFNClassify2D-LocalCtx-0.8M bash dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh \
  logistic_regression --x_range "{'x1': [-15, 15, 0.2]}" > dllm-training/tmp/logs/vud_mdlm_pfn_localctx_logistic.log 2>&1
