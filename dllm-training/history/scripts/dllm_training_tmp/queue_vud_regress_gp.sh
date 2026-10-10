#!/bin/bash
# After MDLM-BayesRegressGP-0.8M training + evaluation finish, run VUD vanilla on
# linear_regression and gaps with default arguments (gaps D=30), default folders.
cd /home/js2723/PROJECTS/VUD
while screen -ls | grep -q "\.regress_gp"; do sleep 60; done
[ -f dllm-training/outputs/bayes_regress_gp/checkpoint.pt ] || exit 1
S=dllm-training/tmp/scripts/vud_mdlm_regress_vanilla.sh; L=dllm-training/tmp/logs
export RUN=bayes_regress_gp NAME=MDLM-BayesRegressGP-0.8M
screen -dmS vud_gp_linear bash -c "srun --jobid=76856 --overlap --cpu-bind=none --export=ALL bash $S linear_regression va_vanilla --x_range \"{'x1': [-15, 15, 0.2]}\" > $L/vud_mdlm_regress_gp_linear.log 2>&1"
screen -dmS vud_gp_gaps bash -c "srun --jobid=76856 --overlap --cpu-bind=none --export=ALL bash $S gaps va_vanilla --x_range \"{'x': [-15, 15, 0.2]}\" > $L/vud_mdlm_regress_gp_gaps.log 2>&1"
