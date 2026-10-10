"""Simulate classification datasets with TabPFN v1's ORIGINAL prior code (automl/TabPFN, tag v1.0.0, a9298ae0).

The prior is built exactly as in TabPFN's own notebook tabpfn/PriorFittingCustomPrior.ipynb (reload_config + the
notebook's overrides -> evaluate_hypers -> get_model(should_train=False) -> its prior data loader), i.e. the
prior bag of TabPFN's MLP/SCM prior (causal and non-causal) and GP prior with the differentiable hyperpriors.
The only changes are the ones the MDLM needs: 2 classes (max_num_classes = 2), at most 2 features (features used
~ U{1, 2}), sequence length 64, and no NaNs (already the notebook default). TabPFN returns x and y only (no true
p(y|x)). Runs on CPU. Needs TabPFN v1 (setup_tabpfn_v1.sh); see TABPFN_V1.md.
Usage: python generate_tabpfn_v1_prior_data.py <datasets> <out.npz> [seed]
(seed 0 = the test set; training-bank shards use seeds >= 1; join shards with join_tabpfn_v1_bank.py)
"""
import os
import sys
from pathlib import Path

# TabPFN's train() (which builds the prior loader) starts distributed training under SLURM with several visible
# GPUs; the prior is sampled on CPU, so hide both.
for _var in ('SLURM_PROCID', 'LOCAL_RANK'):
    os.environ.pop(_var, None)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np                                                           # noqa: E402
import torch                                                                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from small_mdlm.tabpfn_v1 import setup                                       # noqa: E402
setup(prior=True)
from scripts.model_configs import get_prior_config_causal, evaluate_hypers   # noqa: E402
from scripts.model_builder import get_model                                  # noqa: E402
from priors.utils import uniform_int_sampler_f                               # noqa: E402

N, OUT = int(sys.argv[1]), Path(sys.argv[2])
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SEQ, BATCH, MAX_FEATURES = 64, 64, 2
torch.manual_seed(SEED); np.random.seed(SEED)
import random; random.seed(SEED)                                                # TabPFN's priors also use `random`

# --- reload_config(config_type='causal') from the notebook, binary instead of multiclass
config = get_prior_config_causal(max_features=MAX_FEATURES)
config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
config['recompute_attn'] = True
config['max_num_classes'] = 2
config['num_classes'] = 2
config['balanced'] = False
# --- the notebook's overrides, unchanged
config['bptt_extra_samples'] = None
config['output_multiclass_ordered_p'] = 0.
del config['differentiable_hyperparameters']['output_multiclass_ordered_p']
config['multiclass_type'] = 'rank'
del config['differentiable_hyperparameters']['multiclass_type']
config['sampling'] = 'normal'
del config['differentiable_hyperparameters']['sampling']
config['pre_sample_causes'] = True
config['multiclass_loss_type'] = 'nono'
config['normalize_to_ranking'] = False
config['categorical_feature_p'] = .2
config['nan_prob_no_reason'] = .0
config['nan_prob_unknown_reason'] = .0
config['set_value_to_nan'] = .1
config['normalize_with_sqrt'] = False
config['new_mlp_per_example'] = True
config['prior_mlp_scale_weights_sqrt'] = True
config['batch_size_per_gp_sample'] = None
config['normalize_ignore_label_too'] = False
config['differentiable_hps_as_style'] = False
config['max_eval_pos'] = 1000                                                # replaced below for 64-point sequences
config['random_feature_rotation'] = True
config['rotate_normalized_labels'] = True
config['mix_activations'] = False
config['emsize'], config['nhead'], config['canonical_y_encoder'] = 512, 4, False
# --- sizes for this use (features used ~ U{1, 2}; short sequences)
config['num_features_used'] = {'uniform_int_sampler_f(1,max_features)': uniform_int_sampler_f(1, MAX_FEATURES)}
config['bptt'], config['batch_size'], config['aggregate_k_gradients'], config['num_steps'] = SEQ, BATCH, 1, N // BATCH + 1
config['epochs'], config['eval_positions'] = 1, [SEQ // 2]
# TabPFN's own train/test split sampler: context size ~ U{10..30}, the remaining 34-54 points are test points
# (the notebook's 1000 is beyond 64 points, which leaves no test part and fails TabPFN's class-compatibility check).
config['min_eval_pos'], config['max_eval_pos'] = 10, 31
config['total_available_time_in_s'] = None                                   # notebook values
config['train_mixed_precision'] = True
config['efficient_eval_masking'] = True

config_sample = evaluate_hypers(config)
model = get_model(config_sample, 'cpu', should_train=False, verbose=0)
loader = model[3]
xs, ys = [], []
ns = []
for (style, x, y), target, single_eval_pos in loader:
    # x: (SEQ, BATCH, features), target: (SEQ, BATCH) labels, -100 where TabPFN rejected the dataset
    x, y = x.cpu().numpy(), target.cpu().numpy()
    for b in range(x.shape[1]):
        if np.isfinite(x[:, b]).all() and set(np.unique(y[:, b])) <= {0.0, 1.0} and np.abs(x[:, b]).sum() > 0:
            xs.append(x[:, b, :MAX_FEATURES]); ys.append(y[:, b]); ns.append(single_eval_pos)
    if len(xs) >= N:
        break
xs, ys, ns = np.stack(xs[:N]), np.stack(ys[:N]), np.array(ns[:N])
np.savez(OUT, x=xs, y=ys, n_context=ns, config=str({k: str(v) for k, v in config_sample.items()}))
print(f'saved {len(xs)} datasets of {SEQ} points to {OUT}; features used per dataset: '
      f'{np.mean([(np.abs(x).sum(0) > 0).sum() for x in xs]):.2f} on average; class-1 share {ys.mean():.2f}; '
      f'context sizes {ns.min()}-{ns.max()}')
