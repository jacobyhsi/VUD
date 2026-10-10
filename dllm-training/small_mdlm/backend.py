"""Trusted local checkpoint -> structured single/joint numeric sampling."""
import numpy as np
import torch
from .episodes import normalize_by_context, normalize_targets
from .model import answer_probs, build_model, sample
from .tokenizer import NumberTokenizer


class NumericMDLM:
    def __init__(self, checkpoint, device='cpu'):
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved.get('format') != 'vud-numeric-mdlm-v1':
            raise ValueError('Not a numeric MDLM checkpoint')
        self.config = saved['config']
        self.tokenizer = NumberTokenizer(**self.config['tokenizer'])
        self.model = build_model(self.config['model'], self.tokenizer,
                                 self.config['data']['input_size']).to(self.device)
        self.model.load_state_dict(saved['model'])
        self.model.eval()

    def prepare_x(self, x, target_rows):
        """Apply the checkpoint's input preprocessing: PFN-bank models saw x
        z-scored with the context (non-target) rows, so do the same here."""
        x = np.asarray(x, dtype=float)
        if self.config['data'].get('task') in ('classification_pfn', 'regression_pfn'):
            observed = [r for r in range(len(x)) if r not in set(target_rows)]
            x = normalize_by_context(x, observed)
        return x

    def generate(self, x, y, target_rows, seed=0, temperature=1.0, steps=None):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.config['data']['input_size']:
            raise ValueError(f"Checkpoint expects {self.config['data']['input_size']} input features")
        if len(target_rows) > self.config['data']['max_queries']:
            raise ValueError('More queries than this checkpoint was configured for')
        y = np.asarray(y, dtype=float)
        scale = None
        if self.config['data'].get('task') == 'regression_pfn':
            # regression_pfn checkpoints saw y z-scored with the observed rows; map the answers back.
            observed = [r for r in range(len(y)) if r not in set(target_rows) and np.isfinite(y[r])]
            z, y_mean, y_std = normalize_targets(np.where(np.isfinite(y), y, 0.0), observed)
            y, scale = np.where(np.isfinite(y), z, y), (y_mean, y_std)
        e = self.tokenizer.encode(self.prepare_x(x, target_rows), y, target_rows, hide=True)
        ids, kinds, attention, values = self.tokenizer.collate([e], self.device)
        generated = sample(self.model, self.tokenizer, ids, kinds, attention,
                           seed=seed, temperature=temperature, values=values,
                           steps=steps or self.config['training']['sample_steps'])
        values = [self.tokenizer.decode_number(generated[0, e['spans'][row]].tolist()) for row in target_rows]
        return values if scale is None else [v * scale[1] + scale[0] for v in values]

    def label_probs(self, x, y, target_row, temperature=1.0):
        """Exact p(label | x, D) for one hidden row of a classification checkpoint."""
        if not self.tokenizer.labels:
            raise ValueError('Not a classification checkpoint')
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.config['data']['input_size']:
            raise ValueError(f"Checkpoint expects {self.config['data']['input_size']} input features")
        e = self.tokenizer.encode(self.prepare_x(x, [target_row]), y, [target_row], hide=True)
        ids, kinds, attention, values = self.tokenizer.collate([e], self.device)
        probs = answer_probs(self.model, self.tokenizer, ids, kinds, attention, values, temperature)
        (position,) = e['spans'][target_row]
        return probs[0, position, 12:12 + self.tokenizer.labels].double().cpu().numpy()
