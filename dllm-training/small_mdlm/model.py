"""MDLM DiT backbone and conditional absorbing-mask objective/sampler.

Equations follow MDLM's SUBS parameterization and ancestral DDPM update.
We use mask probability t, no explicit time conditioning (sigma=0), and clamp
all observed numeric/context tokens. See PROVENANCE.md for differences.
"""
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from .vendor.dit import DIT


class ValueDIT(DIT):
    """DiT that also embeds real values next to the digit symbols.

    value_embedding 'scalar' (or true): every token of an observed number gets
    (value / value_scale) times a learned vector (xVal-style).
    value_embedding 'row': every token of row j gets W (x_j, y_j, is_hidden) /
    value_scale, the paper's constructed token, with y_j = 0 on hidden rows.
    Either way +v and -v point in opposite directions and attention can
    multiply signed values; answers are still predicted as digit tokens.
    """

    def __init__(self, config, vocab_size, row_size=None):
        super().__init__(config, vocab_size)
        hidden = self.config.model.hidden_size
        self.value_scale = float(self.config.model.value_scale)
        self.row_mode = self.config.model.value_embedding == 'row'
        # Same scale as the symbol-embedding initialization.
        bound = hidden ** -0.5
        if self.row_mode:
            if row_size is None:
                raise ValueError('Row value embedding needs the input size')
            self.row_embed = nn.Parameter(torch.empty(row_size, hidden).uniform_(-bound, bound))
        else:
            self.value_embed = nn.Parameter(torch.empty(hidden).uniform_(-bound, bound))

    def forward(self, indices, sigma, attention_mask=None, values=None):
        x = self.vocab_embed(indices)
        if values is not None:
            if self.row_mode:
                x = x + (values[..., 1:] / self.value_scale) @ self.row_embed
            else:
                x = x + (values[..., :1] / self.value_scale) * self.value_embed
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)
        for block in self.blocks:
            x = block(x, rotary_cos_sin, c, attention_mask=attention_mask)
        return self.output_layer(x, c)


def build_model(config, tokenizer, input_size=None):
    # cond_dim is the width of the DiT's timestep-conditioning input. We always
    # pass sigma=0 (as MDLM does with time_conditioning=False), so the input is
    # constant and carries no information; cond_dim=0 (the default) removes it
    # and leaves each block's adaLN as learned per-block scale/shift/gate
    # vectors (its bias), still zero-initialized. Older configs set it explicitly.
    config = dict(config)
    config.setdefault('cond_dim', 0)
    hidden, heads = config['hidden_size'], config['n_heads']
    if hidden % heads or (hidden // heads) % 2:
        raise ValueError('Head dimension must be an even integer')
    if config.get('value_embedding'):
        if not config.get('value_scale', 0) > 0:
            raise ValueError('value_embedding requires a positive value_scale')
        if config['value_embedding'] not in (True, 'scalar', 'row'):
            raise ValueError(f"Unknown value_embedding {config['value_embedding']!r}")
        row_size = None if input_size is None else tokenizer.row_size(input_size)
        return _quiet(lambda: ValueDIT({'model': dict(config, scale_by_sigma=False)},
                                       len(tokenizer.symbols), row_size), config)
    return _quiet(lambda: DIT({'model': dict(config, scale_by_sigma=False)}, len(tokenizer.symbols)), config)


def _quiet(construct, config):
    """Build the model; with cond_dim=0 PyTorch warns that initializing empty layers is a no-op."""
    with warnings.catch_warnings():
        if config['cond_dim'] == 0:
            warnings.filterwarnings('ignore', message='Initializing zero-element tensors is a no-op')
        return construct()


def _logits(model, tokenizer, x, kinds, attention, values):
    sigma = torch.zeros(len(x), device=x.device)
    if isinstance(model, ValueDIT):
        logits = model(x, sigma, attention_mask=attention, values=values)
    else:
        logits = model(x, sigma, attention_mask=attention)
    return tokenizer.constrain(logits.float(), kinds)


def masked_loss(model, tokenizer, ids, kinds, attention, generator, values=None):
    batch = ids.shape[0]
    # Uniform t on [eps,1); do not force a mask or drop empty-mask episodes,
    # which would change the objective. Each episode is normalized by eligible
    # answer-token count, NOT the realized number of masked tokens.
    t = 1e-3 + (1 - 1e-3) * torch.rand(batch, device=ids.device, generator=generator)
    masked = (torch.rand(ids.shape, device=ids.device, generator=generator) < t[:, None]) & (kinds > 0)
    noisy = ids.masked_fill(masked, tokenizer.mask_id)
    logits = _logits(model, tokenizer, noisy, kinds, attention, values)
    ce = F.cross_entropy(logits.transpose(1, 2), ids, reduction='none')
    denominators = (kinds > 0).sum(-1).clamp_min(1)
    return ((ce * masked).sum(-1) / t / denominators).mean()


@torch.no_grad()
def all_masked_losses(model, tokenizer, ids, kinds, attention, values=None):
    """Per-token cross-entropy with every answer token masked (the first reverse step).

    Diagnostic only: sign/leading-digit losses fall below their marginal
    entropy only if the model reads the context.
    """
    noisy = ids.masked_fill(kinds > 0, tokenizer.mask_id)
    logits = _logits(model, tokenizer, noisy, kinds, attention, values)
    return F.cross_entropy(logits.transpose(1, 2), ids, reduction='none')


@torch.no_grad()
def answer_probs(model, tokenizer, ids, kinds, attention, values=None, temperature=1.0):
    """Predictive distribution at every answer token with all answers masked.

    For a one-token class label this is the exact p(y | x, D); returns
    (batch, length, vocab) probabilities, zero outside each token's grammar.
    """
    noisy = ids.masked_fill(kinds > 0, tokenizer.mask_id)
    logits = _logits(model, tokenizer, noisy, kinds, attention, values)
    return (logits / temperature).softmax(-1)


@torch.no_grad()
def canvas_probs(model, tokenizer, ids, kinds, attention, values=None, temperature=1.0):
    """Proposal distribution at every answer token of a partly revealed canvas.

    ids holds mask_id at still-hidden answer tokens and the revealed token elsewhere, exactly as
    in `sample`, which draws its proposals from these probabilities."""
    logits = _logits(model, tokenizer, ids, kinds, attention, values)
    return (logits / temperature).softmax(-1)


@torch.no_grad()
def sample(model, tokenizer, ids, kinds, attention, seed=0, steps=32, temperature=1.0, values=None):
    if steps < 1 or not 0 < temperature < float('inf'):
        raise ValueError('Positive steps and finite positive temperature required')
    generator = torch.Generator(device=ids.device).manual_seed(seed)
    x = ids.clone()
    x[kinds > 0] = tokenizer.mask_id
    was_training = model.training
    model.eval()
    try:
        for remaining in range(steps, 0, -1):
            masked = x == tokenizer.mask_id
            if not masked.any():
                break
            logits = _logits(model, tokenizer, x, kinds, attention, values)
            probs = (logits[masked] / temperature).softmax(-1)
            proposals = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            # q(unmask between t and s | masked at t)=(t-s)/t.
            reveal = torch.rand(proposals.shape, device=x.device, generator=generator) < 1 / remaining
            current = x[masked]
            x[masked] = torch.where(reveal, proposals, current)
        if (x == tokenizer.mask_id).any():
            raise RuntimeError('Sampler left unresolved answer tokens')
        return x
    finally:
        model.train(was_training)
