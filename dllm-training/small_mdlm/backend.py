"""Trusted local checkpoint -> structured single/joint numeric sampling."""
import numpy as np
import torch
from .episodes import normalize_by_context, normalize_targets
from .model import answer_probs, build_model, canvas_probs, sample
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

    def sample_labels(self, xs, ys, target_rows, seed=0, temperature=1.0, steps=None):
        """Jointly sample class labels for the hidden rows of a batch of tasks from a classification
        checkpoint. target_rows[i] lists task i's hidden rows; returns their labels in that order.

        Each label is one token, so the MDLM reverse process reveals the rows in a random order
        and later labels are drawn conditional on the earlier ones. Tasks are sampled
        independently in one batch, all from a single generator seeded with seed."""
        if not self.tokenizer.labels:
            raise ValueError('Not a classification checkpoint')
        if max(len(rows) for rows in target_rows) > self.config['data']['max_queries']:
            raise ValueError('More queries than this checkpoint was configured for')
        encoded = []
        for x, y, rows in zip(xs, ys, target_rows, strict=True):
            x = np.asarray(x, dtype=float)
            if x.ndim != 2 or x.shape[1] != self.config['data']['input_size']:
                raise ValueError(f"Checkpoint expects {self.config['data']['input_size']} input features")
            encoded.append(self.tokenizer.encode(self.prepare_x(x, rows), y, rows, hide=True))
        ids, kinds, attention, values = self.tokenizer.collate(encoded, self.device)
        generated = sample(self.model, self.tokenizer, ids, kinds, attention,
                           seed=seed, temperature=temperature, values=values,
                           steps=steps or self.config['training']['sample_steps'])
        return [[int(generated[i, e['spans'][row][0]]) - 12 for row in rows]
                for i, (e, rows) in enumerate(zip(encoded, target_rows))]

    def pair_label_law(self, xs, ys, target_rows, temperature=1.0, steps=None):
        """Exact joint law of `sample_labels` for two hidden label rows (a, b) per task.

        The sampler reveals each label at an independent, uniformly random step of `steps`, so a
        is revealed first, b first, or both in the same step with probabilities (S-1)/2S, (S-1)/2S
        and 1/S. Five canvases per task (both masked, a revealed as each label, b revealed as each
        label, for binary labels) give every proposal distribution. Returns, per task, the K x K
        joints over (a, b) for the a-first and b-first orders and the full sampler mixture."""
        labels = self.tokenizer.labels
        if not labels:
            raise ValueError('Not a classification checkpoint')
        if any(len(rows) != 2 for rows in target_rows):
            raise ValueError('Exactly two hidden rows per task')
        encoded = []
        for x, y, rows in zip(xs, ys, target_rows, strict=True):
            x = np.asarray(x, dtype=float)
            if x.ndim != 2 or x.shape[1] != self.config['data']['input_size']:
                raise ValueError(f"Checkpoint expects {self.config['data']['input_size']} input features")
            encoded.append(self.tokenizer.encode(self.prepare_x(x, rows), y, rows, hide=True))
        ids, kinds, attention, values = self.tokenizer.collate(encoded, self.device)
        ids = ids.masked_fill(kinds > 0, self.tokenizer.mask_id)
        positions = [(e['spans'][a][0], e['spans'][b][0]) for e, (a, b) in zip(encoded, target_rows)]
        # Canvas order per task: both masked, a = 0..K-1, b = 0..K-1.
        canvases = []
        for i, (pa, pb) in enumerate(positions):
            canvases.append(ids[i])
            for pos in (pa, pb):
                for label in range(labels):
                    revealed = ids[i].clone()
                    revealed[pos] = 12 + label
                    canvases.append(revealed)
        per_task = 1 + 2 * labels
        repeat = lambda t: t.repeat_interleave(per_task, dim=0)
        probs = canvas_probs(self.model, self.tokenizer, torch.stack(canvases), repeat(kinds),
                             repeat(attention), repeat(values), temperature)
        probs = probs[..., 12:12 + labels].double()
        steps = steps or self.config['training']['sample_steps']
        out = []
        for i, (pa, pb) in enumerate(positions):
            block = probs[i * per_task:(i + 1) * per_task]
            qa, qb = block[0, pa], block[0, pb]
            qb_given_a = block[1:1 + labels, pb]            # row = a's label
            qa_given_b = block[1 + labels:, pa]             # row = b's label
            a_first = qa[:, None] * qb_given_a
            b_first = (qb[:, None] * qa_given_b).T          # index as [a, b]
            mixture = ((steps - 1) / (2 * steps)) * (a_first + b_first) + (qa[:, None] * qb[None, :]) / steps
            out.append({name: value.cpu().numpy() for name, value in
                        (('a_first', a_first), ('b_first', b_first), ('joint', mixture))})
        return out

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
