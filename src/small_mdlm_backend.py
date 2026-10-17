"""Opt-in numeric MDLM adapter: toy-regression vanilla/parallel_pp and toy-classification vanilla/parallel_pp/parallel_pp_exact.

Use --model_name mdlm:/absolute/path/checkpoint.pt. Never dispatches ordinary
Dream/QA/classification calls here. Load only trusted local checkpoints.
"""
from functools import lru_cache
import os
from pathlib import Path
import re
import sys


def is_small_mdlm(model):
    return model.startswith('mdlm:')


@lru_cache(maxsize=4)
def load(model):
    if not is_small_mdlm(model):
        raise ValueError('Expected mdlm:/path/to/checkpoint.pt')
    package = Path(__file__).resolve().parents[1]/'dllm-training'
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from small_mdlm.backend import NumericMDLM
    # CPU default avoids silently taking VRAM from running experiments.
    return NumericMDLM(model[5:], device=os.environ.get('VUD_MDLM_DEVICE', 'cpu'))


_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


def features(note):
    matches = re.findall(r'(?:x\d*)\s*=\s*(' + _NUMBER + r')', note)
    if not matches:
        raise ValueError(f'Unsupported numeric feature note: {note!r}')
    return [float(value) for value in matches]


def rows(icl):
    x, y, missing = [], [], []
    for line in icl.strip().splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'\s*(.*?)\s*<output>\s*(.*?)\s*</output>\s*', line)
        if not match:
            raise ValueError(f'Unsupported numeric context row: {line!r}')
        x.append(features(match[1]))
        if match[2] == '###VUD_U_PLACEHOLDER###':
            missing.append(len(y))
            y.append(float('nan'))
        else:
            y.append(float(match[2]))
    return x, y, missing


def generate_text(message, seed, temperature, model):
    context, separator, query = message.strip().rpartition('\n')
    if not separator:
        context, query = '', message.strip()
    query = query.removesuffix('<output>').strip()
    x, y, missing = rows(context)
    if missing:
        raise ValueError('Use joint sampling for masked auxiliary answers')
    x.append(features(query))
    y.append(float('nan'))
    value = load(model).generate(x, y, [len(y) - 1], seed, temperature)[0]
    # The VUD prompt already ends with <output>; return only its continuation.
    return f'{value} </output>'


def score_labels(message, label_keys, temperature, model):
    """Exact class probabilities for the query of a toy-classification prompt.

    Returns (predicted label text, {label_key: probability}) like chat()."""
    context, separator, query = message.strip().rpartition('\n')
    if not separator:
        context, query = '', message.strip()
    query = query.removesuffix('<output>').strip()
    x, y, missing = rows(context)
    if missing:
        raise ValueError('Joint label sampling is not supported for MDLM classification yet')
    x.append(features(query))
    y.append(float('nan'))
    backend = load(model)
    # A 2-D checkpoint reads 1-D tasks with the missing features set to 0,
    # exactly as its training data encodes them.
    width = backend.config['data']['input_size']
    if any(len(row) > width for row in x):
        raise ValueError(f'Prompt has more features than the checkpoint ({width})')
    x = [row + [0.0] * (width - len(row)) for row in x]
    probs = backend.label_probs(x, y, len(y) - 1, temperature)
    result = {key: float(probs[int(key)]) for key in label_keys}
    return max(result, key=result.get), result


def sample_joint_labels(icls, x_note, label_keys, seed, temperature, model):
    """Jointly sample (u, y) class labels for the auxiliary row and the query (parallel_pp).

    icls is a list of contexts (e.g. one permutation per sample); returns one (u, y) per context,
    sampled together in one batch."""
    backend = load(model)
    width = backend.config['data']['input_size']
    xs, ys, targets = [], [], []
    for icl in icls:
        x, y, missing = rows(icl)
        if len(missing) != 1:
            raise ValueError('Joint sampling requires exactly one auxiliary placeholder')
        x.append(features(x_note))
        y.append(float('nan'))
        if any(len(row) > width for row in x):
            raise ValueError(f'Prompt has more features than the checkpoint ({width})')
        targets.append([missing[0], len(y) - 1])
        xs.append([row + [0.0] * (width - len(row)) for row in x])
        ys.append(y)
    samples = backend.sample_labels(xs, ys, targets, seed, temperature)
    keys = {str(key) for key in label_keys}
    if any(str(label) not in keys for pair in samples for label in pair):
        raise ValueError(f'Sampled labels {samples} are outside {list(label_keys)}')
    return [(str(u), str(prediction)) for u, prediction in samples]


def joint_label_law(icls, x_note, label_keys, temperature, model):
    """Exact parallel joint over (u, y) labels for each context (parallel_pp_exact).

    Returns, per context, {'joint', 'u_first', 'y_first'}: each a dict {(u, y): probability}
    keyed by label_keys, the law of the same sampler that sample_joint_labels draws from."""
    backend = load(model)
    width = backend.config['data']['input_size']
    xs, ys, targets = [], [], []
    for icl in icls:
        x, y, missing = rows(icl)
        if len(missing) != 1:
            raise ValueError('Joint scoring requires exactly one auxiliary placeholder')
        x.append(features(x_note))
        y.append(float('nan'))
        if any(len(row) > width for row in x):
            raise ValueError(f'Prompt has more features than the checkpoint ({width})')
        targets.append([missing[0], len(y) - 1])
        xs.append([row + [0.0] * (width - len(row)) for row in x])
        ys.append(y)
    laws = backend.pair_label_law(xs, ys, targets, temperature)
    as_dict = lambda m: {(u, y): float(m[int(u), int(y)]) for u in label_keys for y in label_keys}
    return [{'joint': as_dict(law['joint']), 'u_first': as_dict(law['a_first']),
             'y_first': as_dict(law['b_first'])} for law in laws]


def sample_joint_numbers(icl, x_note, seed, temperature, model):
    x, y, missing = rows(icl)
    if len(missing) != 1:
        raise ValueError('Joint sampling requires exactly one auxiliary placeholder')
    x.append(features(x_note))
    y.append(float('nan'))
    backend = load(model)
    u, prediction = backend.generate(x, y, [missing[0], len(y) - 1], seed, temperature)
    return u, prediction, backend.tokenizer.number(u)


def infer_number_n_masks(model):
    return load(model).tokenizer.width
