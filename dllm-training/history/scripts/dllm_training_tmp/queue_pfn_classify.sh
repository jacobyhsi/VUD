#!/bin/bash
# Wait for the PFN bank to finish, then train MDLM-PFNClassify2D (GPU 0) and evaluate it on VUD datasets.
cd /home/js2723/PROJECTS/VUD
until [ -f dllm-training/data_banks/pfn_gp_cls_3m/meta.json ]; do
  screen -ls | grep -q pfn_bank || { echo "bank job ended without meta.json"; exit 1; }
  sleep 60
done
GPU=0 WORKERS=8 srun --jobid=80896 --overlap --cpu-bind=none --export=ALL bash dllm-training/tmp/scripts/bayes_linear_vud_run.sh pfn_classify2d
