"""Opt-in GP classifier adapter for toy-classification vanilla VUD.

Use --model_name gp:/absolute/path/fit.json, where fit.json comes from dllm-training/fit_gp_classifier.py (the
amplitude and lengthscale fitted to a classification bank). Each query fits scikit-learn's GaussianProcessClassifier
(kernel ConstantKernel(amplitude) * RBF(lengthscale), both fixed, Laplace inference) to the prompt's context and
returns predict_proba at the query. Inputs are z-scored with the context rows, as the MDLM sees them.
"""
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np

from src.small_mdlm_backend import features, rows

_TRAINING = Path(__file__).resolve().parents[1]/'dllm-training'


def is_gp(model):
    return model.startswith('gp:')


@lru_cache(maxsize=4)
def load(model):
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    fit = json.loads(Path(model[3:]).read_text())
    return ConstantKernel(fit['amplitude'], 'fixed') * RBF(fit['lengthscale'], 'fixed')


def score_labels(message, label_keys, temperature, model):
    """Class probabilities for the query of a toy-classification prompt; returns (predicted label, {label: prob})."""
    from sklearn.gaussian_process import GaussianProcessClassifier
    if str(_TRAINING) not in sys.path:
        sys.path.insert(0, str(_TRAINING))
    from small_mdlm.episodes import normalize_by_context
    if temperature != 1:
        raise ValueError('The GP backend has no temperature; use --model_temperature 1')
    context, separator, query = message.strip().rpartition('\n')
    if not separator:
        context, query = '', message.strip()
    x, y, missing = rows(context)
    if missing:
        raise ValueError('Joint label sampling is not supported for the GP backend')
    y = np.array(y).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError('The GP classifier needs both classes in the context')
    z = normalize_by_context(np.array(x + [features(query.removesuffix('<output>').strip())], dtype=float),
                             list(range(len(x))))
    clf = GaussianProcessClassifier(load(model), optimizer=None).fit(z[:-1], y)
    probs = {int(c): float(p) for c, p in zip(clf.classes_, clf.predict_proba(z[-1:])[0])}
    result = {key: probs.get(int(key), 0.0) for key in label_keys}
    return max(result, key=result.get), result
