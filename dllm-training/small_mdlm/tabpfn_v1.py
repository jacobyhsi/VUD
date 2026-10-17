"""Locate and import TabPFN v1 (automl/TabPFN, tag v1.0.0) for the TabPFN-v1 experiments.

The clone (code, prior and released checkpoint) lives in dllm-training/third_party/TabPFN, created by
setup_tabpfn_v1.sh; set TABPFN_V1_DIR to use another location. ConfigSpace, needed only to build TabPFN's prior,
is installed into dllm-training/third_party/pydeps so it does not touch the project environment.
"""
import os
from pathlib import Path
import sys
import typing

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT/'third_party'


def repo_dir():
    path = Path(os.environ.get('TABPFN_V1_DIR', THIRD_PARTY/'TabPFN'))
    if not (path/'tabpfn'/'models_diff').is_dir():
        raise FileNotFoundError(f'TabPFN v1 not found at {path}; run dllm-training/setup_tabpfn_v1.sh '
                                '(or set TABPFN_V1_DIR to a clone of automl/TabPFN at tag v1.0.0)')
    return path


def setup(prior=False):
    """Put TabPFN v1 on sys.path (plus ConfigSpace when building its prior) and patch it for current PyTorch.

    prior=True also exposes TabPFN's internal packages as top-level `scripts` / `priors`, as its notebooks do."""
    repo = repo_dir()
    paths = [repo] + ([repo/'tabpfn'] if prior else [])
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if prior and str(THIRD_PARTY/'pydeps') not in sys.path:
        sys.path.append(str(THIRD_PARTY/'pydeps'))
    import torch
    import torch.nn.modules.transformer as transformer
    # TabPFN v1 (2022) imports these names from torch.nn.modules.transformer; current PyTorch no longer re-exports
    # them. Its own code is left unchanged.
    for name in ('Optional', 'Tensor', 'Module', 'MultiheadAttention', 'Linear', 'Dropout', 'LayerNorm'):
        if not hasattr(transformer, name):
            setattr(transformer, name, getattr(typing, name, None) or getattr(torch.nn, name, None) or getattr(torch, name))
    # TabPFNClassifier calls check_X_y / check_array with force_all_finite, which scikit-learn renamed to
    # ensure_all_finite (1.6) and then removed; map the old keyword before TabPFN imports these functions.
    import inspect
    import sklearn.utils.validation as validation
    for name in ('check_X_y', 'check_array'):
        check = getattr(validation, name)
        if 'force_all_finite' not in inspect.signature(check).parameters:
            def renamed(*args, _check=check, force_all_finite=None, **kwargs):
                if force_all_finite is not None:
                    kwargs.setdefault('ensure_all_finite', force_all_finite)
                return _check(*args, **kwargs)
            setattr(validation, name, renamed)
    return repo
