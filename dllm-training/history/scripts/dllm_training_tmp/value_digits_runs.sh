#!/bin/bash
cd /home/js2723/PROJECTS/VUD
PY=/home/js2723/PROJECTS/VUD/.venv/bin/python
LOG=dllm-training/tmp/logs/mdlm_diag
set -e
$PY -m unittest discover -s dllm-training -p 'test_*.py' > $LOG/value_digits_tests.log 2>&1
# Old checkpoints (no value embedding) must still load and sample through the VUD backend.
(cd dllm-training && $PY -c "
from small_mdlm.backend import NumericMDLM
b = NumericMDLM('outputs/linear_seed0/checkpoint.pt', 'cuda')
print('old checkpoint sample:', b.generate([[1.0], [2.0], [3.0]], [1.0, 2.0, float('nan')], [2], seed=0))
") >> $LOG/value_digits_tests.log 2>&1
set +e
for name in value_digits_1d_noiseless value_digits_icl_gd; do
  $PY dllm-training/train.py --config dllm-training/configs/$name.json \
    --output dllm-training/outputs/$name --device cuda > $LOG/$name.log 2>&1 &
done
wait
echo ALL_DONE > $LOG/value_digits.done
