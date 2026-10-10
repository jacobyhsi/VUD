#!/bin/bash
# Model-size ablation on the Stage A linear prior (20k steps each), run one after another.
cd /home/js2723/PROJECTS/VUD
for name in ablate_linear_4x128 ablate_linear_4x128_cond32 ablate_linear_2x128 ablate_linear_4x64; do
  bash dllm-training/tmp/scripts/bayes_linear_vud_run.sh $name
done
