#!/bin/bash
# Skip moons_2 for settings 1-3 (the user asked for logistic_regression + moons_1 only): kill those run.py processes as
# they start; exit once the three setting pipelines are gone, then remove the partial moons_2 result folders.
cd /home/js2723/PROJECTS/VUD
while screen -ls | grep -q 'setting[123]'; do
  ps -eo pid,args | awk '$2 ~ /python$/ && /run.py toy-classification/ && /moons_2/ && /(TabPFNNotebook|TabPFNModel|TabPFN-v1)/ {print $1}' | xargs -r kill
  sleep 5
done
for name in MDLM-TabPFNNotebook-0.8M MDLM-TabPFNModel-0.8M TabPFN-v1 TabPFN-v1-T1; do
  rm -rf "results/toy_classification/moons_2/$name"
done
echo 'moons_2 skipped and cleaned'
