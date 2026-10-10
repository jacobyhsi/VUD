"""Opt-in TabPFN v1 adapter for toy-classification vanilla VUD (the released classifier, used as-is).

Use --model_name tabpfn:v1 (package defaults: 3 ensemble members, logits / 0.8) or tabpfn:v1-t1 (the same,
with TabPFN's built-in 0.8 softmax temperature undone, i.e. softmax of the ensemble-averaged logits).
Needs TabPFN v1 (automl/TabPFN v1.0.0: code + checkpoint): run dllm-training/setup_tabpfn_v1.sh once, or set
TABPFN_V1_DIR. VUD_TABPFN_DEVICE picks the device (default cpu).
"""
from functools import lru_cache
import os
from pathlib import Path
import sys

import numpy as np

from src.small_mdlm_backend import features, rows

_TRAINING = Path(__file__).resolve().parents[1]/'dllm-training'
_TEMPERATURE = 0.8                     # transformer_predict hard-codes log(0.8) when the model has no style


def is_tabpfn(model):
    return model.startswith('tabpfn:')


@lru_cache(maxsize=2)
def load(model):
    if model not in ('tabpfn:v1', 'tabpfn:v1-t1'):
        raise ValueError('Expected tabpfn:v1 or tabpfn:v1-t1')
    if str(_TRAINING) not in sys.path:
        sys.path.insert(0, str(_TRAINING))
    from small_mdlm.tabpfn_v1 import setup
    repo = setup()
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device=os.environ.get('VUD_TABPFN_DEVICE', 'cpu'), base_path=str(repo/'tabpfn'))


def score_labels(message, label_keys, temperature, model):
    """Class probabilities for the query of a toy-classification prompt; returns (predicted label, {label: prob})."""
    context, separator, query = message.strip().rpartition('\n')
    if not separator:
        context, query = '', message.strip()
    x, y, missing = rows(context)
    if missing:
        raise ValueError('Joint label sampling is not supported for TabPFN')
    x, y = np.array(x, dtype=float), np.array(y).astype(int)
    xq = np.array([features(query.removesuffix('<output>').strip())], dtype=float)
    present = np.unique(y)
    if len(present) == 1:                 # TabPFN needs two classes in the context
        probs = {int(present[0]): 1.0}
    else:
        clf = load(model).fit(x, y, overwrite_warning=True)
        if model == 'tabpfn:v1':
            p = clf.predict_proba(xq)[0]
        else:                             # ensemble-averaged logits / 0.8 -> undo the temperature, then softmax
            logits = clf.predict_proba(xq, return_logits=True)[0] * _TEMPERATURE
            p = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
        probs = {int(c): float(v) for c, v in zip(clf.classes_, p)}
    result = {key: probs.get(int(key), 0.0) for key in label_keys}
    return max(result, key=result.get), result
