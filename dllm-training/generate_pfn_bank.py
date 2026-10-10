"""Pre-generate classification datasets from PFN-lineage priors.

Sampling a GP per dataset is too slow to do inside the training loop, so this
writes a fixed bank that small_mdlm.episodes ('classification_pfn') draws from.
Each dataset has seq_len points; a share of them uses only x1 (x2 = 0), which
is how 1-D tasks reach a 2-D checkpoint. Chunks are seeded by their index, so
the bank is reproducible and independent of the number of workers.

Output directory: x.npy (N, seq_len, 2) float32, latent.npy (N, seq_len)
float32 (sigmoid(latent) = true p(y=1)), y.npy (N, seq_len) int8, dims.npy (N,)
int8, and meta.json.

--label-prior-bag picks each dataset's prior from a PFN-style prior bag
(small_mdlm/vendor/pfn_prior_bag.py) with softmax(logit) weights:
'median' (paper, Section 6: class 1 above the dataset's median, deterministic)
and 'bernoulli' (released code: sigmoid -> Bernoulli) label a GP sample;
'bnn' (paper's tabular BNN prior), 'tabpfn' and 'tabpfn_causal' (TabPFN v1 MLP prior,
non-causal and causal branches) and 'tabpfn_gp' (TabPFN v1 GP member) draw their own
inputs, latents and labels. Median
datasets store a saturated latent (+-30) so sigmoid(latent) is the true label
probability; BNN/TabPFN latents are the logit of a 64-pass Monte Carlo estimate;
TabPFN-causal latents are the observed labels saturated (+-30), since p(y | x) has
no closed form there; TabPFN-GP latents are the exact logit of P(y = 1 | f). labelling.npy holds pfn_prior_bag.CODES
(0 Bernoulli, 1 median, 2 BNN, 3 TabPFN, 4 TabPFN causal, 5 TabPFN GP). --median-label-prob p is the older
spelling of the two-member GP bag with P(median) = p. --device cuda runs the
chunks one after another on the GPU (fast for BNN/TabPFN; GP members still
sample per dataset), the default runs them on CPU workers. --join lets several
processes (e.g. one per GPU) build one bank: the first creates the files, every
process claims chunks one at a time (claims/ files created with O_EXCL), so a
faster GPU takes more of them, and whichever process writes the last chunk
writes meta.json.

The opt-in 'controlled_gp' member is a local smooth/noise-controlled GP
adaptation, NOT TabPFN v1. --controlled-gp-config supplies its JSON settings;
the complete settings and signal variance are recorded in bank metadata.
Existing priors ignore those settings and retain their random-number streams.
"""
import os
import argparse
from dataclasses import asdict
import json
from multiprocessing import Pool
from pathlib import Path
import warnings

import numpy as np

CHUNK = 256  # datasets per seeded chunk; --chunk changes it (e.g. 8192 on a GPU)


def make_chunk(args):
    # The old nine-item calling convention is retained for existing scripts/tests.
    index, seq_len, one_d_prob, seed, noise, outputscale_concentration, bag, device, chunk = args[:9]
    controlled_settings = args[9] if len(args) == 10 else {}
    import torch
    warnings.filterwarnings('ignore')
    from small_mdlm.vendor import pfn_mlp_prior, tabpfn_prior
    from small_mdlm.vendor.controlled_gp_prior import ControlledGPConfig, sample_classification_datasets as controlled_sampler
    from small_mdlm.vendor.pfn_prior_bag import CODES, GP_MEMBERS, apply_labels, assign
    from small_mdlm.vendor.pfn_priors import sample_classification_datasets
    torch.manual_seed(seed * 1_000_003 + index)
    if device == 'cpu':
        torch.set_num_threads(1)
    mlp_generator = torch.Generator(device=device).manual_seed(seed * 1_000_003 + index)
    rng = np.random.default_rng([seed, index])
    one_d = rng.random(chunk) < one_d_prob
    members = assign(rng, chunk, bag)
    gp = np.isin(members, GP_MEMBERS)
    x = np.zeros((chunk, seq_len, 2), np.float32)
    latent, y = np.zeros((chunk, seq_len), np.float32), np.zeros((chunk, seq_len), np.int8)
    for dims, mask in ((1, one_d), (2, ~one_d)):
        if (mask & gp).any():
            sel = mask & gp
            hps = {} if noise is None else {'noise': noise}
            if outputscale_concentration is not None:
                hps['outputscale_concentration'] = outputscale_concentration
            xs, f, ys = sample_classification_datasets(int(sel.sum()), seq_len, dims, hps)
            x[sel, :, :dims], latent[sel], y[sel] = xs.numpy(), f.numpy(), ys.numpy()
        for name, sampler in (('bnn', pfn_mlp_prior.sample_classification_datasets),
                              ('tabpfn', tabpfn_prior.sample_classification_datasets),
                              ('tabpfn_causal', tabpfn_prior.sample_causal_classification_datasets),
                              ('tabpfn_gp', tabpfn_prior.sample_gp_classification_datasets)):
            sel = mask & (members == name)
            if sel.any():
                xs, f, ys = sampler(int(sel.sum()), seq_len, dims, generator=mlp_generator, device=device)
                x[sel, :, :dims], latent[sel], y[sel] = xs.cpu().numpy(), f.cpu().numpy(), ys.cpu().numpy()
        sel = mask & (members == 'controlled_gp')
        if sel.any():
            xs, f, ys = controlled_sampler(int(sel.sum()), seq_len, dims, generator=mlp_generator,
                                           device=device, config=ControlledGPConfig(**controlled_settings))
            x[sel, :, :dims], latent[sel], y[sel] = xs.cpu().numpy(), f.cpu().numpy(), ys.cpu().numpy()
    apply_labels(members, latent, y)
    codes = np.array([CODES[m] for m in members], np.int8)
    return index, x, latent, y, np.where(one_d, 1, 2).astype(np.int8), codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', type=int, default=3_000_000)
    parser.add_argument('--seq-len', type=int, default=34)
    parser.add_argument('--one-d-prob', type=float, default=0.3)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise', type=float, default=None,
                        help='Fix the latent noise variance (paper classification value: 0.001); default samples it')
    parser.add_argument('--outputscale-concentration', type=float, default=None,
                        help='Outputscale Gamma concentration (PFN code: 0.5; BoTorch default: 2.0); rate stays 0.15')
    parser.add_argument('--label-prior-bag', default=None,
                        help="Label priors and logits, e.g. 'median=0,bernoulli=0' (50/50); default 'bernoulli=0'")
    parser.add_argument('--median-label-prob', type=float, default=None,
                        help='Older spelling: two-member bag with P(median) = this value')
    parser.add_argument('--device', default='cpu', help="'cpu' (worker pool) or a CUDA device (sequential)")
    parser.add_argument('--chunk', type=int, default=CHUNK,
                        help='Datasets per seeded chunk (the bank depends on it); large values suit --device cuda')
    parser.add_argument('--join', action='store_true',
                        help='Share the work with other --join processes on the same output (start one first)')
    parser.add_argument('--controlled-gp-config', type=Path,
                        help='JSON settings for the opt-in controlled_gp bag member; does not change tabpfn_gp')
    args = parser.parse_args()
    from small_mdlm.vendor.pfn_prior_bag import CODES, GP_MEMBERS, member_probabilities, parse_weights
    if args.label_prior_bag and args.median_label_prob is not None:
        raise ValueError('Use either --label-prior-bag or --median-label-prob')
    if args.median_label_prob is not None:
        p = min(max(args.median_label_prob, 1e-12), 1 - 1e-12)
        bag = {'median': float(np.log(p)), 'bernoulli': float(np.log(1 - p))}
    else:
        bag = parse_weights(args.label_prior_bag or 'bernoulli=0')
    from small_mdlm.vendor.controlled_gp_prior import ControlledGPConfig
    controlled = ControlledGPConfig(**(json.loads(args.controlled_gp_config.read_text())
                                       if args.controlled_gp_config else {}))
    if args.controlled_gp_config and 'controlled_gp' not in bag:
        raise ValueError('--controlled-gp-config requires a controlled_gp bag member')
    if args.datasets < 1 or args.seq_len < 2 or args.chunk < 1 or args.workers < 1 or not 0 <= args.one_d_prob <= 1:
        raise ValueError('Positive dataset/chunk/worker counts and a valid dimensional probability required')
    if not args.join and args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh bank directory; existing arrays are never overwritten')
    chunks = -(-args.datasets // args.chunk)
    total = chunks * args.chunk
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'meta.json').exists():
        raise ValueError('Bank already complete; choose a fresh output directory')
    claims, written = args.output/'claims', args.output/'written'
    first = not args.join or not claims.exists()
    claims.mkdir(exist_ok=True); written.mkdir(exist_ok=True)
    # The first process creates the files; joining processes open them for writing.
    arrays = {name: np.lib.format.open_memmap(args.output/f'{name}.npy', 'w+' if first else 'r+', dtype, shape)
              for name, dtype, shape in (('x', np.float32, (total, args.seq_len, 2)),
                                         ('latent', np.float32, (total, args.seq_len)),
                                         ('y', np.int8, (total, args.seq_len)),
                                         ('dims', np.int8, (total,)),
                                         ('labelling', np.int8, (total,)))}
    jobs = [(i, args.seq_len, args.one_d_prob, args.seed, args.noise, args.outputscale_concentration, bag, args.device,
             args.chunk, asdict(controlled)) for i in range(chunks)]

    def claimed(jobs):
        for job in jobs:
            try:
                os.close(os.open(claims/str(job[0]), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            yield job

    with Pool(args.workers if args.device == 'cpu' else 1) as pool:
        results = (pool.imap_unordered(make_chunk, claimed(jobs)) if args.device == 'cpu'
                   else map(make_chunk, claimed(jobs)))
        for done, (i, x, latent, y, dims, labelling) in enumerate(results, 1):
            rows = slice(i * args.chunk, (i + 1) * args.chunk)
            arrays['x'][rows], arrays['latent'][rows], arrays['y'][rows] = x, latent, y
            arrays['dims'][rows], arrays['labelling'][rows] = dims, labelling
            for array in arrays.values():
                array.flush()
            (written/str(i)).touch()
            if done % max(1, 51_200 // args.chunk) == 0:
                print(f'{len(os.listdir(written))}/{chunks} chunks written ({done} by this process)', flush=True)
    if len(os.listdir(written)) < chunks:
        print(f'This process is done; other processes are finishing the bank', flush=True)
        return
    for folder in (claims, written):
        for f in folder.iterdir():
            f.unlink()
        folder.rmdir()
    names, probs = member_probabilities(bag)
    meta = {'prior': 'PFN-lineage prior bag (small_mdlm/vendor/pfn_prior_bag.py)',
            'datasets': total, 'seq_len': args.seq_len, 'one_d_prob': args.one_d_prob, 'seed': args.seed,
            'chunk': args.chunk, 'label_prior_bag': dict(zip(names, probs.tolist())), 'labelling_codes': CODES}
    if set(names) & set(GP_MEMBERS):
        meta['gp'] = {'hyperpriors': 'small_mdlm/vendor/pfn_priors.py defaults',
                      'latent_noise_variance': args.noise if args.noise is not None else 'sampled (Gamma(1.1, 0.05))',
                      'outputscale_prior': f'Gamma({args.outputscale_concentration or 0.5}, 0.15)'}
    if 'controlled_gp' in names:
        meta['controlled_gp'] = {'implementation': 'small_mdlm/vendor/controlled_gp_prior.py',
                                  'adaptation': 'local controlled prior, not the released TabPFN prior',
                                  'settings': asdict(controlled), 'signal_variance': 1.0,
                                  'input_distribution': 'IID N(0,1); legacy context-only normalization at episode loading',
                                  'threshold': 'N(0, threshold_std^2), independent of points/observation noise'}
    (args.output/'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Bank complete: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
