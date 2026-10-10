"""CPU-only end-to-end dispatch and actual VUD runner smoke checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

root = Path('/home/js2723/PROJECTS/VUD')
sys.path.insert(0, str(root))
sys.path.insert(0, str(root/'dllm-training'))
from src.chat import chat_response_only, sample_joint_numbers, infer_number_n_masks
from src.prompt import ToyRegressionPrompt
from src.utils import extract

torch.set_num_threads(2)
checkpoint = Path(sys.argv[1]).resolve()
model = f'mdlm:{checkpoint}'
prompt = ToyRegressionPrompt(model_name=model)
context = pd.DataFrame({'note': ['x1 = -1.0', 'x1 = 1.0', 'x1 = 2.0'], 'label': [4., 2., 1.]})
for extra in (False, True):
    message = prompt.get_general_prompt(context, 'x1 = 0.0', 0,
        icl_z_note='x1 = 0.1' if extra else None, icl_u_label=3. if extra else None)
    output = chat_response_only(message, seed=0, model=model)
    assert np.isfinite(extract(output)), output
    print('single/conditional dispatch', extra, output, flush=True)
icl = prompt.note_label_df_to_icl_string(context, 0, 'x1 = 0.1', '###VUD_U_PLACEHOLDER###')
u, y, ids = sample_joint_numbers(icl, 'x1 = 0.0', seed=1, model=model)
assert np.isfinite([u, y]).all()
assert len(ids) == infer_number_n_masks([], model)
print('joint dispatch', u, y, flush=True)

# Exact optimizer/RNG/data-index resume check against uninterrupted training.
fresh = Path(tempfile.mkdtemp(prefix='mdlm-uninterrupted-', dir=root/'tmp'))
subprocess.run([sys.executable, 'dllm-training/train.py', '--config',
    'dllm-training/configs/smoke.json', '--output', str(fresh), '--device', 'cpu'], check=True)
resumed = torch.load(checkpoint, weights_only=True, map_location='cpu')
uninterrupted = torch.load(fresh/'checkpoint.pt', weights_only=True, map_location='cpu')
for key in resumed['model']:
    torch.testing.assert_close(resumed['model'][key], uninterrupted['model'][key], rtol=0, atol=0)
print('exact training resume PASS', flush=True)

for method in ('vanilla', 'parallel_pp'):
    # Normalized destination stays under this exact smoke-test folder, not
    # among scientific results. The runner prefixes results/task/dataset/.
    destination = f'../../../tmp/{checkpoint.parent.name}/vud_{method}'
    cmd = [sys.executable, 'run.py', 'toy-regression', '--model_name', model,
        '--dataset_name', 'linear_regression', '--D_size', '3',
        '--x_range', "{'x1': [0, 0.1, 0.2]}", '--num_z', '1',
        '--num_permutations', '3', '--num_outlier_pairs_to_remove', '0',
        '--std_method', 'default', '--num_joint_samples', '4', '--plot', '0',
        '--va_method', method, '--save_directory', destination]
    subprocess.run(cmd, check=True, timeout=120)
    files = list((checkpoint.parent/f'vud_{method}').rglob('results_test_x*.csv'))
    assert len(files) == 1, files
    table = pd.read_csv(files[0])
    assert table['va_method'].eq(method).all()
    assert np.isfinite(table[['Va', 'H[p(y|x,D)]']].to_numpy()).all()
    print(method, 'full VUD runner PASS', flush=True)
print('ALL INTEGRATION CHECKS PASSED (smoke weights only, not accuracy validation)', flush=True)
