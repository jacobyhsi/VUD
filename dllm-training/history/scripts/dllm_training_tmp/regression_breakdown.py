"""Regression VUD results vs the true noise (nats): TU, Va (VUD's min-Va among the 5 lowest-KL z, capped at TU)
and Ve = TU - Va, by region: on the data, in gaps between data clusters, and beyond the data.
True aleatoric = entropy of the generating p(y|x): linear: N(-x+3, 2^2); gaps: mixture over modes weighted
by each mode's x-density (noise 0.1 / 0.1 / 2), computed numerically."""
import json, sys
import numpy as np, pandas as pd
ROOT = '/home/js2723/PROJECTS/VUD'
GAPS = json.load(open(f'{ROOT}/datasets_toy_regression/gaps/info.json'))


def true_entropy(ds, x):
    if ds == 'linear_regression':
        return np.full(len(x), 0.5 * np.log(2 * np.pi * np.e * 4.0))
    w = np.array(GAPS['num_features_per_mode'], float)
    m, s = np.array(GAPS['mode_means']), np.array(GAPS['mode_stds'])
    c, b, n = np.array(GAPS['mode_coeffs']), np.array(GAPS['mode_biases']), np.array(GAPS['noise_stds'])
    ys = np.linspace(-30, 30, 6001); dy = ys[1] - ys[0]
    out = []
    for xi in x:
        logp = np.log(w) - 0.5 * ((xi - m) / s) ** 2 - np.log(s)
        p = np.exp(logp - logp.max()); p /= p.sum()
        dens = (p[:, None] * np.exp(-0.5 * ((ys[None] - (c * xi + b)[:, None]) / n[:, None]) ** 2) / (n[:, None] * np.sqrt(2 * np.pi))).sum(0)
        dens = np.clip(dens, 1e-300, None)
        out.append(-(dens * np.log(dens)).sum() * dy)
    return np.array(out)


if __name__ == '__main__':
    for ds in ('linear_regression', 'gaps'):
        print(f'== {ds}  (nats; true aleatoric on data: ' + ('2.11' if ds == 'linear_regression' else '-0.88 at the two tight clusters, 2.11 at x~5') + ')')
        print(f'   {"model":28s} | region | TU    | Va (true)      | |Va-true| | Ve')
        for m in sys.argv[1:] or ['Dream-v0-Base-7B', 'MDLM-BayesRegress-0.8M', 'MDLM-BayesRegressGP-0.8M', 'MDLM-BayesLinear-1.3M']:
            f = f'{ROOT}/results/toy_regression/{ds}/{m}/va_vanilla/seed_0/results_test.csv'
            try:
                df = pd.read_csv(f)
            except FileNotFoundError:
                continue
            cx = pd.read_csv(f.replace('results_test', 'D_test')).iloc[:, 0].to_numpy(float)
            xcol = [c for c in df.columns if c.startswith('x_') and c != 'x_note'][0]
            rows = []
            for x, g in df.groupby(xcol):
                tu = g['H[p(y|x,D)]'].iloc[0]
                cand = g.loc[g.kl_pyx_pyxz.isin(g.kl_pyx_pyxz.nsmallest(5))]
                rows.append((x, tu, min(cand.Va.min(), tu)))
            r = pd.DataFrame(rows, columns=['x', 'TU', 'Va'])
            r['true'] = true_entropy(ds, r.x.to_numpy())
            d = np.abs(r.x.to_numpy()[:, None] - cx[None]).min(1)
            inside = (r.x >= cx.min()) & (r.x <= cx.max())
            regions = {'on data': d <= 0.5, 'gaps': inside & (d > 1.0), 'beyond': ~inside & (d > 1.0)}
            for name, mask in regions.items():
                mask = np.asarray(mask)
                if mask.any():
                    s = r[mask]
                    print(f'   {m:28s} | {name:7s}| {s.TU.mean():5.2f} | {s.Va.mean():5.2f} ({s.true.mean():5.2f}) | {np.abs(s.Va - s.true).mean():9.2f} | {(s.TU - s.Va).mean():.2f}')
