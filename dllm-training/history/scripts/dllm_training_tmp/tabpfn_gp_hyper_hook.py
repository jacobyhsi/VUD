"""Record the GP hyperparameters TabPFN v1's ORIGINAL prior draws (hook on priors.fast_gp.get_batch; generation unchanged).

Runs tabpfn_original_generate.py's prior (same config) and saves every 1-feature GP draw's noise / outputscale /
lengthscale. Usage (from dllm-training/): python tmp/scripts/tabpfn_gp_hyper_hook.py <datasets> <out.json>
"""
import json
from pathlib import Path
import sys

N, OUT = sys.argv[1], sys.argv[2]
sys.argv = ['tabpfn_original_generate.py', N, '/dev/null.npz', '777']
src = open('generate_tabpfn_v1_prior_data.py').read()
setup, loop = src.split('config_sample = evaluate_hypers(config)')
ns = {'__name__': 'hook', '__file__': str(Path('generate_tabpfn_v1_prior_data.py').resolve())}
exec(compile(setup, 'generate', 'exec'), ns)
import tabpfn.priors.fast_gp as fast_gp                                      # noqa: E402  (the module model_builder uses)

draws, calls, original = [], [], fast_gp.get_batch


def recording_get_batch(batch_size, seq_len, num_features, *args, hyperparameters=None, **kwargs):
    calls.append((num_features, type(hyperparameters).__name__))
    if num_features == 1 and isinstance(hyperparameters, dict):
        draws.append({k: float(hyperparameters[k]) for k in ('noise', 'outputscale', 'lengthscale')} |
                     {'sampling': hyperparameters.get('sampling', 'uniform')})
    return original(batch_size, seq_len, num_features, *args, hyperparameters=hyperparameters, **kwargs)


fast_gp.get_batch = recording_get_batch
exec(compile('config_sample = evaluate_hypers(config)' + loop.split('np.savez')[0], 'generate', 'exec'), ns)
json.dump(draws, open(OUT, 'w'), indent=1)
print(f'{len(draws)} one-feature GP draws recorded; GP calls {len(calls)}, e.g. {calls[:5]}')
