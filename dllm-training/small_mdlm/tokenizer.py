"""Fixed-width numeric tokens with explicit answer positions; no clipping.

labels=0 (default): answers are numbers (regression). labels=K>=2: answers
are one class-label token, reusing the digit symbols '0'..'K-1', so the
vocabulary (and old checkpoints) are unchanged.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass
class NumberTokenizer:
    integer_digits: int = 3
    decimals: int = 2
    labels: int = 0
    # Stable checkpoint vocabulary; no pretrained language vocabulary.
    symbols = ('[PAD]', '[MASK]', '[BOS]', '[EOS]', 'x', 'y', '=', ',', ';',
               '+', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    pad_id = 0
    mask_id = 1

    def __post_init__(self):
        if self.integer_digits < 1 or self.decimals < 1:
            raise ValueError('Positive integer_digits and decimals required')
        if self.labels == 1 or not 0 <= self.labels <= 10:
            raise ValueError('labels must be 0 (regression) or 2..10 classes')

    @property
    def answer_width(self):
        return 1 if self.labels else self.width

    def row_size(self, input_size):
        """Row-vector width: x, then y (one-hot over labels when classifying), then is_hidden."""
        return input_size + (self.labels or 1) + 1

    def label(self, value):
        if float(value) != int(value) or not 0 <= int(value) < self.labels:
            raise ValueError(f'Label {value} is not an integer in 0..{self.labels - 1}')
        return [self.symbols.index(str(int(value)))]

    def decode_label(self, token_id):
        return int(self.symbols[int(token_id)])

    @property
    def width(self):
        return 1 + self.integer_digits + 1 + self.decimals

    def number(self, value):
        if not math.isfinite(float(value)):
            raise ValueError('Cannot tokenize a non-finite number')
        value = round(float(value), self.decimals)
        if value == 0:
            value = 0.0
        text = f'{value:+0{self.width}.{self.decimals}f}'
        if len(text) != self.width:
            raise ValueError(f'Number {value} exceeds tokenizer range; no clipping is performed')
        return [self.symbols.index(c) for c in text]

    def decode_number(self, ids):
        if len(ids) != self.width:
            raise ValueError('Wrong number of numeric tokens')
        return float(''.join(self.symbols[int(i)] for i in ids))

    def encode(self, x, y, target_rows, hide=False):
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 2 or y.shape != (len(x),) or len(x) < 1:
            raise ValueError('Expected x=(rows, features), y=(rows,)')
        targets = set(target_rows)
        if not targets or len(targets) != len(target_rows) or not targets <= set(range(len(x))):
            raise ValueError('Target rows must be unique valid row indices')
        ids, kinds, values, token_rows, spans = [2], [0], [0.0], [-1], {}
        row = -1
        def append(tokens, token_kinds=None, value=0.0):
            ids.extend(tokens)
            kinds.extend(token_kinds or [0] * len(tokens))
            values.extend([value] * len(tokens))
            token_rows.extend([row] * len(tokens))
        for row, features in enumerate(x):
            append([4, 6])
            for j, value in enumerate(features):
                if j:
                    append([7])
                append(self.number(value), value=round(float(value), self.decimals))
            append([8, 5, 6])
            start = len(ids)
            # At inference never read a hidden answer, even if y contains NaN.
            hidden_value = 0 if hide and row in targets else y[row]
            if self.labels:
                answer, grammar = self.label(hidden_value), [3]
            else:
                answer = self.number(hidden_value)
                grammar = [1] + [2] * self.integer_digits + [0] + [2] * self.decimals
            if row in targets:
                # Hidden answers carry value 0 so the target never leaks through it.
                append(answer, grammar)
            else:
                append(answer, value=round(float(y[row]), self.decimals))
            if row in targets:
                spans[row] = list(range(start, start + self.answer_width))
            append([8])
        row = -1
        append([3])
        ids = torch.tensor(ids, dtype=torch.long)
        kinds = torch.tensor(kinds, dtype=torch.long)
        if hide:
            ids[kinds > 0] = self.mask_id
        # Column 0: the value of the number this token belongs to (0 for
        # hidden answers). Columns 1..: the whole row as (x, y, is_hidden_row),
        # with y=0 on hidden rows, so every token of row j sees (x_j, y_j),
        # like the paper's constructed tokens. For classification y is a
        # one-hot over labels. Special tokens are all zero.
        d = x.shape[1]
        row_table = np.zeros((len(x) + 1, self.row_size(d)))
        for r in range(len(x)):
            row_table[r, :d] = np.round(x[r].astype(float), self.decimals)
            if r in targets:
                row_table[r, -1] = 1
            elif self.labels:
                row_table[r, d + int(y[r])] = 1
            else:
                row_table[r, d] = round(float(y[r]), self.decimals)
        values = np.column_stack([values, row_table[token_rows]])
        values = torch.tensor(values, dtype=torch.float32)
        return {'ids': ids, 'kinds': kinds, 'values': values, 'spans': spans}

    def collate(self, examples, device='cpu'):
        length = max(len(e['ids']) for e in examples)
        ids = torch.full((len(examples), length), self.pad_id, dtype=torch.long)
        kinds = torch.zeros_like(ids)
        values = torch.zeros(*ids.shape, examples[0]['values'].shape[-1], dtype=torch.float32)
        for i, e in enumerate(examples):
            ids[i, :len(e['ids'])] = e['ids']
            kinds[i, :len(e['ids'])] = e['kinds']
            values[i, :len(e['ids'])] = e['values']
        return ids.to(device), kinds.to(device), (ids != self.pad_id).to(device), values.to(device)

    def constrain(self, logits, kinds):
        # Fixed numeric grammar used in BOTH training and sampling, not rejection
        # sampling. Decimal points and separators remain observed structure.
        vocab = torch.arange(len(self.symbols), device=logits.device)
        allowed = ((kinds[..., None] == 1) & ((vocab == 9) | (vocab == 10)))
        allowed |= (kinds[..., None] == 2) & (vocab >= 12)
        allowed |= (kinds[..., None] == 3) & (vocab >= 12) & (vocab < 12 + self.labels)
        allowed |= kinds[..., None] == 0
        allowed[..., self.mask_id] = False
        return logits.masked_fill(~allowed, -1e9)
