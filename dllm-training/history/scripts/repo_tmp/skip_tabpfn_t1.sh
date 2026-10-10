#!/bin/bash
# Drop the temperature-undone TabPFN variant (the user wants TabPFN v1 as published only): kill its run.py processes
# as they start until setting 3 ends, then remove its result folders.
cd /home/js2723/PROJECTS/VUD
while screen -ls | grep -q setting3; do
  ps -eo pid,args | awk '$2 ~ /python$/ && /run.py toy-classification/ && /tabpfn:v1-t1/ {print $1}' | xargs -r kill
  sleep 5
done
rm -rf results/toy_classification/*/TabPFN-v1-T1
echo 'TabPFN-v1-T1 dropped and cleaned'
