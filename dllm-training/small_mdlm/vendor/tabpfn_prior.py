"""TabPFN (v1) MLP prior for binary classification, non-causal branch.

Adapted from automl/TabPFN (Apache-2.0), tag v1.0.0 (a9298ae0): tabpfn/priors/mlp.py
(get_batch, non-causal path), tabpfn/priors/differentiable_prior.py (meta samplers) and
tabpfn/priors/flexible_categorical.py (MulticlassRank / MulticlassValue with two classes).
Hyperpriors are those stored in the released checkpoint
(models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt, 'differentiable_hyperparameters'):
  num_layers   2 + round(TN(mu, mu*s)),  log mu ~ U(log 1, log 6)
  hidden       4 + round(TN(mu, mu*s)),  log mu ~ U(log 5, log 130)
  noise_std    TN(mu, mu*s),             log mu ~ U(log 1e-4, log 0.3)
  init_std     TN(mu, mu*s),             log mu ~ U(log 0.01, log 10)
  (TN = normal truncated to [0, 1e6]; log s ~ U(log 0.01, log 1) for each)
  dropout      0.9 * Beta(b, k), b, k ~ U(0.1, 5);  block_wise_dropout, pre_sample_weights,
               pre_sample_causes ~ fair coins
  activations  per layer from softmax([1, w1, w2, w3]) over (Tanh, Identity, ReLU, ELU), w ~ U(-5, 6)
  labels       class 1 iff y > threshold, y normalized over the dataset; threshold = y at a random
               point ('rank') or ~ N(0, 1) ('value'), equally likely
Causal branch (sample_causal_classification_datasets, is_causal in the release): num_causes =
1 + round(TN(mu, mu*s)), log mu ~ U(log 1, log 12); causes feed Linear(num_causes, h) -> (num_layers - 1) x
[act, Linear(h, h), +noise] with h >= 1 + 2d and no output layer; x and y are hidden nodes picked at
random (y the last node when y_is_effect; the d + 1 nodes contiguous when in_clique; x sorted when
sort_features; each a fair coin).
Network: Linear(d, h) -> (num_layers - 1) x [act, Linear(h, h), +noise] -> [act, Linear(h, 1), +noise];
weight matrices ~ N(0, init_std^2) scaled for dropout as in the release (prior_mlp_scale_weights_sqrt),
biases keep PyTorch's default Linear init.
Local changes: one hyperparameter draw per dataset (TabPFN draws one per batch, so the per-dataset
prior is the same); batched networks with widths capped at 256 and depth at 10 (larger draws are
clamped); continuous inputs only (sampling 'normal'; the 'mixed' categorical/zipf inputs, missing
values are left out; the causal branch is sampled separately). The third activation in
the checkpoint is an anonymous lambda; ReLU (the v1.0.0 config's choice) is used.
"""
import math

import torch

MAX_HIDDEN, MAX_LAYERS, MAX_CAUSES = 256, 10, 16
ACTIVATIONS = (torch.tanh, lambda a: a, torch.relu, torch.nn.functional.elu)


def _u(count, lo, hi, g, dev):
    return lo + (hi - lo) * torch.rand(count, device=dev, generator=g)


def _trunc_norm_log_scaled(count, min_mean, max_mean, g, dev):
    """TabPFN's meta_trunc_norm_log_scaled: normal(mu, mu * s) truncated to [0, inf) by rejection."""
    mu = torch.exp(_u(count, math.log(min_mean), math.log(max_mean), g, dev))
    sd = mu * torch.exp(_u(count, math.log(0.01), math.log(1.0), g, dev))
    out = torch.full((count,), -1.0, device=dev)
    while (out < 0).any():
        draw = mu + sd * torch.randn(count, device=dev, generator=g)
        out = torch.where(out < 0, draw, out)
    return out


def _block_mask(rows, cols, n_rows, n_cols, blocks, dev):
    """Block-diagonal masks (count, rows, cols) for true sizes n_rows x n_cols and `blocks` blocks."""
    w = (n_rows // blocks).clamp(min=1)[:, None, None]
    h = (n_cols // blocks).clamp(min=1)[:, None, None]
    r = torch.arange(rows, device=dev)[None, :, None]
    c = torch.arange(cols, device=dev)[None, None, :]
    br, bc = r // w, c // h
    return ((br == bc) & (br < blocks[:, None, None])).float()


def _weights(count, rows, cols, n_rows, n_cols, init_std, dropout, block_wise, first, g, dev):
    """One weight matrix per network in torch's (out, in) layout, masked to its true size."""
    real = ((torch.arange(rows, device=dev)[None, :, None] < n_rows[:, None, None])
            & (torch.arange(cols, device=dev)[None, None, :] < n_cols[:, None, None])).float()
    z = torch.randn(count, rows, cols, device=dev, generator=g)
    # Block-wise: random 1..ceil(sqrt(min dim)) blocks, std scaled by keep_prob^(1/2).
    top = torch.ceil(torch.sqrt(torch.minimum(n_rows, n_cols).float()))
    blocks = (1 + torch.floor(torch.rand(count, device=dev, generator=g) * top)).long()
    bmask = _block_mask(rows, cols, n_rows, n_cols, blocks, dev) * real
    keep = (bmask.sum((1, 2)) / (n_rows * n_cols).float()).clamp(min=1e-6)
    block_w = z * (init_std / keep.sqrt())[:, None, None] * bmask
    # Element dropout (none in the first layer): std / (1 - p^(1/2)), then Bernoulli(1 - p) mask.
    p = torch.zeros_like(dropout) if first else dropout.clamp(max=0.99)
    drop = (torch.rand(count, rows, cols, device=dev, generator=g) < (1 - p)[:, None, None]).float()
    drop_w = z * (init_std / (1 - p.sqrt()))[:, None, None] * drop * real
    return torch.where(block_wise[:, None, None], block_w, drop_w)


def _bias(count, size, n_out, fan_in, g, dev):
    """PyTorch's default Linear bias init U(-1/sqrt(fan_in), 1/sqrt(fan_in)), masked to n_out units."""
    bound = (1 / fan_in.float().sqrt())[:, None]
    b = (2 * torch.rand(count, size, device=dev, generator=g) - 1) * bound
    return b * (torch.arange(size, device=dev)[None] < n_out[:, None]).float()


def sample_networks(count, num_features, generator=None, device='cpu', causes=None, min_hidden=None):
    """Batched networks. causes (count,) gives per-network input sizes (causal branch; inputs padded
    to MAX_CAUSES), and min_hidden a per-network lower bound on the width."""
    g, dev = generator, device
    layers = (2 + torch.round(_trunc_norm_log_scaled(count, 1, 6, g, dev))).long().clamp(max=MAX_LAYERS)
    hidden = (4 + torch.round(_trunc_norm_log_scaled(count, 5, 130, g, dev))).long().clamp(max=MAX_HIDDEN)
    if min_hidden is not None:
        hidden = torch.maximum(hidden, min_hidden)
    noise_std = _trunc_norm_log_scaled(count, 1e-4, 0.3, g, dev)
    init_std = _trunc_norm_log_scaled(count, 0.01, 10.0, g, dev)
    # Beta(b, k) as G1 / (G1 + G2) so the generator drives it.
    g1 = torch._standard_gamma(_u(count, 0.1, 5.0, g, dev), generator=g)
    g2 = torch._standard_gamma(_u(count, 0.1, 5.0, g, dev), generator=g)
    dropout = 0.9 * g1 / (g1 + g2).clamp(min=1e-30)
    coin = lambda: torch.rand(count, device=dev, generator=g) < 0.5
    block_wise, pre_sample_weights, pre_sample_causes, rank = coin(), coin(), coin(), coin()
    act_logits = torch.cat([torch.ones(count, 1, device=dev), _u(count * 3, -5.0, 6.0, g, dev).view(count, 3)], 1)
    H, d = MAX_HIDDEN, num_features if causes is None else MAX_CAUSES
    hidden_layers = MAX_LAYERS - 1
    ones = torch.ones(count, dtype=torch.long, device=dev)
    n_in = ones * d if causes is None else causes
    net = {'layers': layers, 'hidden': hidden, 'noise_std': noise_std, 'init_std': init_std,
           'dropout': dropout, 'rank': rank, 'value_z': torch.randn(count, device=dev, generator=g)}
    net['w0'] = _weights(count, H, d, hidden, n_in, init_std, dropout, block_wise, True, g, dev)
    net['b0'] = _bias(count, H, hidden, n_in, g, dev)
    net['w'] = torch.stack([_weights(count, H, H, hidden, hidden, init_std, dropout, block_wise, False, g, dev)
                            for _ in range(hidden_layers)], 1)
    net['b'] = torch.stack([_bias(count, H, hidden, hidden, g, dev) for _ in range(hidden_layers)], 1)
    net['w_out'] = _weights(count, 1, H, ones, hidden, init_std, dropout, block_wise, False, g, dev)
    net['b_out'] = _bias(count, 1, ones, hidden, g, dev)
    # Noise std per unit: |N(0, noise_std)| when pre_sample_weights, else noise_std. Slot -1 is the output layer's.
    per_unit = torch.abs(torch.randn(count, hidden_layers + 1, H, device=dev, generator=g)) * noise_std[:, None, None]
    net['noise'] = torch.where(pre_sample_weights[:, None, None], per_unit, noise_std[:, None, None].expand_as(per_unit))
    net['act'] = torch.multinomial(torch.softmax(act_logits, 1), MAX_LAYERS, replacement=True, generator=g)
    # Inputs: N(mean_i, |std_i|) with mean_i ~ N(0, 1), std_i = |N(0, 1) * mean_i| when pre_sample_causes.
    mean = torch.randn(count, d, device=dev, generator=g)
    std = torch.abs(torch.randn(count, d, device=dev, generator=g) * mean)
    net['x_mean'] = torch.where(pre_sample_causes[:, None], mean, torch.zeros_like(mean))
    net['x_std'] = torch.where(pre_sample_causes[:, None], std, torch.ones_like(std))
    return net


def _activate(a, kind):
    out = torch.zeros_like(a)
    for i, f in enumerate(ACTIVATIONS):
        sel = kind == i
        if sel.any():
            out[sel] = f(a[sel])
    return out


def sample_inputs(net, n, generator=None):
    count, d = net['x_mean'].shape
    z = torch.randn(count, n, d, device=net['x_mean'].device, generator=generator)
    return net['x_mean'][:, None] + net['x_std'][:, None] * z


def forward(net, x, generator=None):
    """x (count, N, d) in the network's input scale -> noisy outputs (count, N)."""
    noise = lambda a, layer: a + net['noise'][:, layer, None, :a.shape[-1]] * torch.randn(
        a.shape, device=a.device, dtype=a.dtype, generator=generator)
    a = x @ net['w0'].transpose(1, 2) + net['b0'][:, None]
    for layer in range(MAX_LAYERS - 1):
        live = (layer < net['layers'] - 1)[:, None, None]
        nxt = noise(_activate(a, net['act'][:, layer]) @ net['w'][:, layer].transpose(1, 2) + net['b'][:, layer, None], layer)
        a = torch.where(live, nxt, a)
    last = (net['layers'] - 1).clamp(max=MAX_LAYERS - 1)
    out = _activate(a, net['act'].gather(1, last[:, None]).squeeze(1)) @ net['w_out'].transpose(1, 2) + net['b_out'][:, None]
    out = out + net['noise'][:, -1, None, :1] * torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
    return out.squeeze(-1)


def threshold(net, out, generator=None):
    """Threshold on the raw output for the dataset `out` (count, N): y normalized, then 'rank' or 'value'."""
    mean, std = out.mean(1), out.std(1) + 1e-6
    count, n = out.shape
    pick = torch.randint(n, (count,), device=out.device, generator=generator)
    rank_t = out.gather(1, pick[:, None]).squeeze(1)
    return torch.where(net['rank'], rank_t, mean + std * net['value_z'])


def label_probs(net, x, t, draws=64, generator=None):
    hits = sum((forward(net, x, generator) > t[:, None]).float() for _ in range(draws))
    return hits / draws


@torch.no_grad()
def sample_classification_datasets(count, seq_len, num_features, draws=64, generator=None, device='cpu'):
    """x (count, seq_len, d), latent = logit of the true P(y=1), and labels y (count, seq_len).

    Networks whose outputs overflow (rare) are redrawn.
    """
    x = torch.zeros(count, seq_len, num_features, device=device)
    latent, y = torch.zeros(count, seq_len, device=device), torch.zeros(count, seq_len, device=device)
    todo = torch.arange(count, device=device)
    while len(todo):
        net = sample_networks(len(todo), num_features, generator, device)
        xs = sample_inputs(net, seq_len, generator)
        out = forward(net, xs, generator)
        t = threshold(net, out, generator)
        # Clipped at half a draw, so the logit stays finite where every pass agrees.
        p = label_probs(net, xs, t, draws, generator).clamp(0.5 / draws, 1 - 0.5 / draws)
        ok = torch.isfinite(out).all(1) & torch.isfinite(xs).all(2).all(1)
        x[todo[ok]], y[todo[ok]] = xs[ok], (out[ok] > t[ok, None]).float()
        latent[todo[ok]] = torch.log(p[ok] / (1 - p[ok]))
        todo = todo[~ok]
    return x, latent, y


def hidden_nodes(net, x, generator=None):
    """Every hidden block's (noisy) output: (count, N, MAX_LAYERS - 1, MAX_HIDDEN); blocks past a
    network's depth are computed but never selected."""
    noise = lambda a, layer: a + net['noise'][:, layer, None, :] * torch.randn(
        a.shape, device=a.device, dtype=a.dtype, generator=generator)
    a = x @ net['w0'].transpose(1, 2) + net['b0'][:, None]
    outs = []
    for layer in range(MAX_LAYERS - 1):
        a = noise(_activate(a, net['act'][:, layer]) @ net['w'][:, layer].transpose(1, 2) + net['b'][:, layer, None], layer)
        outs.append(a)
    return torch.stack(outs, 2)


def _remove_outliers_and_normalize(x, n_sigma=4):
    """tabpfn/utils.py remove_outliers (log-compress values beyond n_sigma robust std) then
    normalize_data (z-score, clip at +-100), over each dataset's points. The causal branch's hidden
    nodes can reach ~1e13, which these steps tame; the non-causal inputs are Gaussian, where they are
    near-identity, so that branch keeps raw inputs (and existing banks stay reproducible)."""
    x = x.double()
    mean, std = x.mean(1, keepdim=True), x.std(1, keepdim=True)
    inside = (x - mean).abs() <= n_sigma * std
    k = inside.sum(1, keepdim=True).clamp(min=2)
    mean = (x * inside).sum(1, keepdim=True) / k
    std = (((x - mean) ** 2) * inside).sum(1, keepdim=True).div(k - 1).sqrt()
    lower, upper = mean - n_sigma * std, mean + n_sigma * std
    x = torch.maximum(-torch.log1p(x.abs()) + lower, x)
    x = torch.minimum(torch.log1p(x.abs()) + upper, x)
    x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
    return x.clamp(-100, 100).float()


def _causal_once(count, seq_len, d, g, dev):
    causes = (1 + torch.round(_trunc_norm_log_scaled(count, 1, 12, g, dev))).long().clamp(max=MAX_CAUSES)
    net = sample_networks(count, d, g, dev, causes=causes,
                          min_hidden=torch.full((count,), 1 + 2 * d, dtype=torch.long, device=dev))
    coin = lambda: torch.rand(count, device=dev, generator=g) < 0.5
    y_is_effect, in_clique, sort_features = coin(), coin(), coin()
    nodes = hidden_nodes(net, sample_inputs(net, seq_len, g), g).flatten(2)     # (count, N, L*H)
    h, real = net['hidden'], (net['layers'] - 1) * net['hidden']                 # real nodes, TabPFN's flat order
    span = (MAX_LAYERS - 1) * MAX_HIDDEN
    # Not in a clique: a random permutation of nodes 0 .. real - 2 (randperm(nodes - 1) in the release).
    keys = torch.rand(count, span, device=dev, generator=g)
    keys[torch.arange(span, device=dev)[None] >= (real - 1)[:, None]] = 2.0
    spread = keys.argsort(1)[:, :1 + d]
    # In a clique: d + 1 consecutive nodes from a random start, in random order.
    start = torch.floor(torch.rand(count, device=dev, generator=g) * (real - d).float()).long()
    clique = start[:, None] + torch.rand(count, 1 + d, device=dev, generator=g).argsort(1)
    pick = torch.where(in_clique[:, None], clique, spread)
    y_flat = torch.where(y_is_effect, real - 1, pick[:, 0])
    x_flat = pick[:, 1:]
    x_flat = torch.where(sort_features[:, None], x_flat.sort(1).values, x_flat)
    padded = lambda k: (k // h[:, None]) * MAX_HIDDEN + k % h[:, None]           # flat real index -> padded column
    y_raw = nodes.gather(2, padded(y_flat[:, None])[:, None].expand(-1, seq_len, -1)).squeeze(2)
    x = nodes.gather(2, padded(x_flat)[:, None].expand(-1, seq_len, -1))
    t = threshold(net, y_raw, g)
    ok = torch.isfinite(y_raw).all(1) & torch.isfinite(x).all(2).all(1)
    return _remove_outliers_and_normalize(torch.nan_to_num(x)), (y_raw > t[:, None]).float(), ok


@torch.no_grad()
def sample_causal_classification_datasets(count, seq_len, num_features, draws=None, generator=None, device='cpu'):
    """TabPFN's causal branch: x and y are hidden nodes of a random network fed by random causes.

    p(y | x) has no closed form (the causes are latent), so the returned latent is the observed label
    saturated at +-30. Networks whose nodes overflow (rare) are redrawn. `draws` is unused.
    """
    x = torch.zeros(count, seq_len, num_features, device=device)
    y = torch.zeros(count, seq_len, device=device)
    todo = torch.arange(count, device=device)
    while len(todo):
        xs, ys, ok = _causal_once(len(todo), seq_len, num_features, generator, device)
        x[todo[ok]], y[todo[ok]] = xs[ok], ys[ok]
        todo = todo[~ok]
    return x, 30.0 * (2 * y - 1), y


@torch.no_grad()
def sample_gp_classification_datasets(count, seq_len, num_features, draws=None, generator=None, device='cpu'):
    """TabPFN v1's GP member (tabpfn/priors/fast_gp.py with the released checkpoint's hyperpriors).

    x ~ N(0, 1)^d (the release's 'normal'/'mixed' sampling is not 'uniform'); f ~ GP(0, outputscale *
    RBF(lengthscale)) with outputscale, lengthscale ~ meta_trunc_norm_log_scaled(1e-5, 10); observed
    y = f + N(0, noise), noise variance in {1e-5, 1e-4, 1e-2} with weights softmax([1, w1, w2]),
    w ~ U(-3, 5); class 1 iff y > the rank/value threshold. The returned latent is the logit of the true
    P(y = 1 | f) = Phi((f - t) / sqrt(noise)), clipped to [1e-6, 1 - 1e-6]. `draws` is unused.
    """
    g, dev = generator, device
    ls = _trunc_norm_log_scaled(count, 1e-5, 10.0, g, dev).double().clamp(min=1e-6)
    os_ = _trunc_norm_log_scaled(count, 1e-5, 10.0, g, dev).double().clamp(min=1e-8)
    logits = torch.cat([torch.ones(count, 1, device=dev), _u(count * 2, -3.0, 5.0, g, dev).view(count, 2)], 1)
    noise = torch.tensor([1e-5, 1e-4, 1e-2], device=dev, dtype=torch.float64)[
        torch.multinomial(torch.softmax(logits, 1), 1, generator=g).squeeze(1)]
    x = torch.randn(count, seq_len, num_features, device=dev, generator=g).double()
    K = os_[:, None, None] * torch.exp(-0.5 * ((x[:, :, None] - x[:, None]) ** 2).sum(-1) / ls[:, None, None] ** 2)
    K = K + (1e-6 * os_)[:, None, None] * torch.eye(seq_len, device=dev, dtype=torch.float64)
    f = (torch.linalg.cholesky(K) @ torch.randn(count, seq_len, 1, device=dev, generator=g).double()).squeeze(-1)
    sd = noise.sqrt()[:, None]
    y_obs = f + sd * torch.randn(count, seq_len, device=dev, generator=g).double()
    net = {'rank': torch.rand(count, device=dev, generator=g) < 0.5, 'value_z': torch.randn(count, device=dev, generator=g)}
    t = threshold(net, y_obs, g)
    p = torch.special.ndtr((f - t[:, None]) / sd).clamp(1e-6, 1 - 1e-6)
    return x.float(), torch.log(p / (1 - p)).float(), (y_obs > t[:, None]).float()
