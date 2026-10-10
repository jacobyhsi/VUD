"""Step 1 sanity check: von Oswald et al.'s ICL regression with real-valued tokens.

Same DiT backbone as the MDLM with real-valued tokens (x, y, is_y, is_query).
token_layout 'constructed' (the paper's main setup) gives one token (x_j, y_j)
per demonstration; 'classic' (their normal_token_construct notebook) puts x_j
and y_j in neighbouring tokens, so the model must first learn to pair them.
A linear head reads the query position and is trained with MSE on fresh tasks
each step. If this does not learn, the problem is the backbone/optimizer, not
digit tokens or diffusion.

Validation reports MSE against zero, one GD step from W0=0 with the best
learning rate (the paper's comparison), least squares, and a shuffled-label
control of the trained model.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from small_mdlm.vendor.dit import DIT


class ContinuousICL(nn.Module):
    def __init__(self, model_config, input_size, token_layout='constructed'):
        super().__init__()
        if token_layout not in ('constructed', 'classic'):
            raise ValueError(f'Unknown token_layout {token_layout!r}')
        self.token_layout = token_layout
        self.dit = DIT({'model': dict(model_config, scale_by_sigma=False)}, 1)
        # Replace the token table with a linear embedding of (x, y, is_y, is_query).
        self.dit.vocab_embed = nn.Linear(input_size + 3, model_config['hidden_size'])

    def forward(self, x_ctx, y_ctx, x_query):
        batch, n, d = x_ctx.shape
        zeros = x_ctx.new_zeros(batch, n, 1)
        if self.token_layout == 'constructed':
            context = torch.cat([x_ctx, y_ctx[..., None], zeros, zeros], -1)
        else:
            x_tokens = torch.cat([x_ctx, zeros, zeros, zeros], -1)
            y_tokens = torch.cat([x_ctx.new_zeros(batch, n, d), y_ctx[..., None], zeros + 1, zeros], -1)
            # Interleave x_1, y_1, x_2, y_2, ...
            context = torch.stack([x_tokens, y_tokens], 2).reshape(batch, 2 * n, d + 3)
        query = torch.cat([x_query[:, None], x_ctx.new_zeros(batch, 1, 2), x_ctx.new_ones(batch, 1, 1)], -1)
        out = self.dit(torch.cat([context, query], 1), x_ctx.new_zeros(batch))
        return out[:, -1, 0]


def sample_tasks(data, seed, split, index, batch):
    """Vectorized create_reg_data distribution; see data.py for the reference version."""
    split_id = {'train': 0, 'validation': 1}[split]
    rng = np.random.default_rng(np.random.SeedSequence([seed, split_id, index]))
    d, n, r = data['input_size'], data['context_size'], data['input_range']
    w = rng.normal(0, data['weight_scale'], (batch, d))
    x = rng.uniform(-r / 2, r / 2, (batch, n + 1, d))
    y = np.einsum('bnd,bd->bn', x, w) + rng.normal(0, data['noise_std'], (batch, n + 1))
    return x[:, :-1], y[:, :-1], x[:, -1], y[:, -1]


def baselines(x_ctx, y_ctx, x_q, y_q):
    n = x_ctx.shape[1]
    # One GD step from W0=0 on L(W)=1/(2N) sum (W x_i - y_i)^2, best eta in closed form.
    step = np.einsum('bn,bnd,bd->b', y_ctx, x_ctx, x_q) / n
    eta = (step @ y_q) / (step @ step)
    ols = np.array([x_q[i] @ np.linalg.lstsq(x_ctx[i], y_ctx[i], rcond=None)[0]
                    for i in range(len(x_q))])
    return {'zero_mse': float(np.mean(y_q ** 2)),
            'gd1_mse': float(np.mean((eta * step - y_q) ** 2)), 'gd1_eta': float(eta),
            'least_squares_mse': float(np.mean((ols - y_q) ** 2))}


@torch.no_grad()
def validate(model, tensors, reference, device):
    model.eval()
    x_ctx, y_ctx, x_q, y_q = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in tensors)
    pred = model(x_ctx, y_ctx, x_q)
    shuffled = y_ctx[:, torch.randperm(y_ctx.shape[1], generator=torch.Generator().manual_seed(0))]
    shuffled_pred = model(x_ctx, shuffled, x_q)
    model.train()
    return {'validation_mse': ((pred - y_q) ** 2).mean().item(),
            'shuffled_mse': ((shuffled_pred - y_q) ** 2).mean().item(), **reference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/continuous_icl_gd.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data, training = config['data'], config['training']
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output is nonempty; choose a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(config, indent=2) + '\n')
    device = torch.device(args.device)
    torch.manual_seed(training['seed'])
    model = ContinuousICL(config['model'], data['input_size'],
                          data.get('token_layout', 'constructed')).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['lr'], weight_decay=0.0)
    val = sample_tasks(data, training['seed'], 'validation', 0, training['eval_tasks'])
    reference = baselines(*val)
    print(f'parameters={sum(p.numel() for p in model.parameters())}; device={device}; {reference}', flush=True)
    started = time.monotonic()
    for step in range(1, training['steps'] + 1):
        batch = sample_tasks(data, training['seed'], 'train', step, training['batch_size'])
        x_ctx, y_ctx, x_q, y_q = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in batch)
        loss = ((model(x_ctx, y_ctx, x_q) - y_q) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % training['eval_every'] == 0 or step == training['steps']:
            metrics = {'step': step, 'train_mse': loss.item(),
                       'elapsed_seconds': time.monotonic() - started,
                       **validate(model, val, reference, device)}
            print(json.dumps(metrics), flush=True)
            with (args.output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            torch.save({'config': config, 'step': step, 'model': model.state_dict()},
                       args.output/'checkpoint.pt')


if __name__ == '__main__':
    main()
