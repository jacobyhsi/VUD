"""Exact uncertainty decomposition of the fitted GP classifier (setting 4), without VUD.

With the GP's posterior over the latent function f given the context D, at each test x:
    TU = H[E_f sigmoid(f(x))],   AU = E_f H[sigmoid(f(x))]  (the conditional entropy H[Y | f]),   EU = TU - AU,
in bits. Two posteriors over f are used:
- laplace: the Laplace approximation that scikit-learn's GaussianProcessClassifier (and so VUD on the GP) uses;
- mc: the exact posterior, by elliptical slice sampling (Murray et al., 2010) of f at the context points, then the
  GP's exact Gaussian conditional f(x) | f(context) at each test x, integrated by Gauss-Hermite quadrature.
The kernel is the fitted one (fit_gp_classifier.py) on inputs z-scored with the context, exactly as in the VUD backend.
Writes <results dir>/exact_decomposition.csv with both decompositions, the true aleatoric H[true p] and the VUD
values per test x, and prints near / far summaries (near: inside the context range in 1-D, within 0.35 of a context
point in 2-D). The MC run (thinned by 10) is split in two halves of chains to report its Monte Carlo error.
Usage: python gp_exact_decomposition.py <fit.json> <VUD results dir of the GP run> <dataset> [--samples=4000] [--chains=64]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_vud_classification import true_p1                     # noqa: E402
from small_mdlm.episodes import normalize_by_context                 # noqa: E402

NODES, WEIGHTS = np.polynomial.hermite_e.hermegauss(60)
WEIGHTS = WEIGHTS / WEIGHTS.sum()
THIN = 10


def entropy_bits(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def gaussian_terms(mean, var):
    """E sigmoid(f) and E H[sigmoid(f)] for f ~ N(mean, var), elementwise."""
    f = mean[..., None] + np.sqrt(np.maximum(var, 0))[..., None] * NODES
    p = 1 / (1 + np.exp(-f))
    return p @ WEIGHTS, entropy_bits(p) @ WEIGHTS


def elliptical_slice(L, y, chains, samples, burn, rng):
    """Samples of f at the context points from the exact posterior N(0, K) * prod Bernoulli(y | sigmoid f)."""
    loglik = lambda f: -np.logaddexp(0, -(2 * y - 1) * f).sum(-1)
    f = rng.standard_normal((chains, len(y))) @ L.T
    out = []
    for it in range(burn + samples):
        nu = rng.standard_normal(f.shape) @ L.T
        threshold = loglik(f) + np.log(rng.uniform(size=chains))
        theta = rng.uniform(0, 2 * np.pi, chains)
        lo, hi = theta - 2 * np.pi, theta.copy()
        todo = np.ones(chains, bool)
        new = f.copy()
        while todo.any():
            cand = f[todo] * np.cos(theta[todo])[:, None] + nu[todo] * np.sin(theta[todo])[:, None]
            ok = loglik(cand) > threshold[todo]
            idx = np.flatnonzero(todo)
            new[idx[ok]] = cand[ok]
            todo[idx[ok]] = False
            shrink = idx[~ok]
            lo[shrink] = np.where(theta[shrink] < 0, theta[shrink], lo[shrink])
            hi[shrink] = np.where(theta[shrink] >= 0, theta[shrink], hi[shrink])
            theta[shrink] = rng.uniform(lo[shrink], hi[shrink])
        f = new
        if it >= burn:
            out.append(f.copy())
    return np.stack(out, 1)                                           # (chains, samples, n)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    fit, run_dir, dataset = json.loads(Path(args[0]).read_text()), Path(args[1]), args[2]
    samples, chains = int(opts.get('samples', 4000)), int(opts.get('chains', 64))
    cols = ['x1'] if dataset == 'logistic_regression' else ['x1', 'x2']
    context = pd.read_csv(run_dir/'D_test.csv')
    vud = pd.read_csv(run_dir/'results_test.csv')
    xq = vud[[f'x_{c}' for c in cols]].drop_duplicates().to_numpy(float)
    cx, cy = context[cols].to_numpy(float), context['label'].to_numpy(int)
    z = normalize_by_context(np.vstack([cx, xq]), list(range(len(cx))))
    zc, zq = z[:len(cx)], z[len(cx):]
    kernel = ConstantKernel(fit['amplitude'], 'fixed') * RBF(fit['lengthscale'], 'fixed')

    # Laplace posterior, as in scikit-learn's predict_proba.
    est = GaussianProcessClassifier(kernel, optimizer=None).fit(zc, cy).base_estimator_
    k_cq = est.kernel_(est.X_train_, zq)
    mean = k_cq.T @ (est.y_train_ - est.pi_)
    var = est.kernel_.diag(zq) - (np.linalg.solve(est.L_, est.W_sr_[:, None] * k_cq) ** 2).sum(0)
    p_lap, h_lap = gaussian_terms(mean, var)
    tu_lap, au_lap = entropy_bits(p_lap), h_lap

    # Exact posterior by elliptical slice sampling, then the exact Gaussian conditional at each test x.
    K = kernel(zc) + 1e-6 * fit['amplitude'] * np.eye(len(zc))
    factor = cho_factor(K, lower=True)
    f_c = elliptical_slice(np.tril(factor[0]), cy, chains, samples, burn=1000, rng=np.random.default_rng(0))
    k_cq = kernel(zc, zq)
    solve = cho_solve(factor, k_cq)                                    # K^-1 k(c, q)
    cond_var = kernel.diag(zq) - (k_cq * solve).sum(0)
    f_c = f_c[:, ::THIN]                                               # successive slice-sampling draws are correlated
    halves = []
    for part in (f_c[: chains // 2], f_c[chains // 2:], f_c):
        flat = part.reshape(-1, part.shape[-1])
        p_sum, h_sum = np.zeros(len(cond_var)), np.zeros(len(cond_var))
        for i in range(0, len(flat), 512):                            # chunks keep the quadrature in memory
            p, h = gaussian_terms(flat[i:i + 512] @ solve, np.broadcast_to(cond_var, (len(flat[i:i + 512]), len(cond_var))))
            p_sum += p.sum(0); h_sum += h.sum(0)
        halves.append((p_sum / len(flat), h_sum / len(flat)))
    (p_a, h_a), (p_b, h_b), (p_mc, au_mc) = halves
    tu_mc = entropy_bits(p_mc)

    true_au = entropy_bits(true_p1(dataset, xq))
    near = ((xq[:, 0] >= cx[:, 0].min()) & (xq[:, 0] <= cx[:, 0].max())) if len(cols) == 1 \
        else np.linalg.norm(xq[:, None] - cx[None], axis=-1).min(1) < .35
    selected = []
    for key, g in vud.groupby([f'x_{c}' for c in cols], sort=False):
        cand = g.loc[g.kl_pyx_pyxz.isin(g.kl_pyx_pyxz.nsmallest(5))]
        h = g['H[p(y|x,D)]'].iloc[0]
        selected.append((h, cand.Va.min() if len(cols) == 2 else min(cand.Va.min(), h)))
    vud_tu, vud_va = np.array(selected).T

    out = pd.DataFrame({**{f'x_{c}': xq[:, i] for i, c in enumerate(cols)}, 'near': near, 'true_AU': true_au,
                        'p1_laplace': p_lap, 'TU_laplace': tu_lap, 'AU_laplace': au_lap, 'EU_laplace': tu_lap - au_lap,
                        'p1_mc': p_mc, 'TU_mc': tu_mc, 'AU_mc': au_mc, 'EU_mc': tu_mc - au_mc,
                        'TU_vud': vud_tu, 'AU_vud': vud_va, 'EU_vud': vud_tu - vud_va})
    out.to_csv(run_dir/'exact_decomposition.csv', index=False)
    print(f'{dataset}: MC error (two halves of {chains} chains x {samples}): max |dTU| '
          f'{np.abs(entropy_bits(p_a) - entropy_bits(p_b)).max():.3f}, max |dAU| {np.abs(h_a - h_b).max():.3f}')
    print(f'{"":20s} {"TU near / far":>15s} {"AU near / far":>15s} {"EU near / far":>15s} {"AU err near / far":>18s}')
    for name in ('mc', 'laplace', 'vud'):
        tu, au = out[f'TU_{name}'].to_numpy(), out[f'AU_{name}'].to_numpy()
        err = np.abs(au - true_au)
        f = lambda v: f'{v[near].mean():.2f} / {v[~near].mean():.2f}'
        label = {'mc': 'GP exact (MC)', 'laplace': 'GP exact (Laplace)', 'vud': 'VUD on the GP'}[name]
        print(f'{label:20s} {f(tu):>15s} {f(au):>15s} {f(tu - au):>15s} {f(err):>18s}')
    print(f'{"true AU":20s} {"":>15s} {f(true_au):>15s}')
    print('saved', run_dir/'exact_decomposition.csv')


if __name__ == '__main__':
    main()
