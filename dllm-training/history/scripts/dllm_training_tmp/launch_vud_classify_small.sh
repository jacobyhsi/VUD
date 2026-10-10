#!/bin/bash
# Start VUD vanilla on logistic_regression and moons_1 with MDLM-BayesClassify2D-0.8M, each in its own screen.
cd /home/js2723/PROJECTS/VUD
S=dllm-training/tmp/scripts/vud_mdlm_classify_vanilla.sh
L=dllm-training/tmp/logs
E="RUN=bayes_classify2d_small NAME=MDLM-BayesClassify2D-0.8M"
screen -dmS vud_cls_lr_small bash -c "$E srun --jobid=76856 --overlap --cpu-bind=none --export=ALL bash $S logistic_regression --x_range \"{'x1': [-15, 15, 0.2]}\" > $L/vud_mdlm_classify_small_logistic.log 2>&1"
screen -dmS vud_cls_moons1_small bash -c "$E srun --jobid=76856 --overlap --cpu-bind=none --export=ALL bash $S moons_1 > $L/vud_mdlm_classify_small_moons1.log 2>&1"
screen -ls
