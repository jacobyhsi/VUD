"""Generate a fixed-layout regression episode bank (small_mdlm/regression_episodes.py) for task 'regression_pfn'.

Run on a GPU. Refuses nonempty outputs; meta.json is written only after all arrays are flushed.
Arrays: x (N, context_max + 2, 1), y, mean, std (N, context_max + 2), context_size, query_count, gp_task,
broad_query, near_query (N,).
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from small_mdlm.regression_episodes import make_regression_chunk
from small_mdlm.tabpfn_episodes import GeometryConfig
from small_mdlm.vendor.regression_priors import HetGPConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', type=int, default=3_006_464)
    parser.add_argument('--chunk', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gp-fraction', type=float, default=0.5, help='Share of het-GP tasks (rest: TabPFN v1 MLP)')
    parser.add_argument('--near-query-prob', type=float, default=GeometryConfig.near_query_prob)
    parser.add_argument('--broad-query-prob', type=float, default=GeometryConfig.broad_query_prob)
    parser.add_argument('--layout-prob', type=float, default=0.75)
    parser.add_argument('--ls-min', type=float, default=HetGPConfig.ls_min, help='Lower end of the GP function lengthscale range')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh output directory; existing banks are never overwritten')
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = GeometryConfig(near_query_prob=args.near_query_prob, broad_query_prob=args.broad_query_prob, layout_prob=args.layout_prob)
    gp_cfg = HetGPConfig(ls_min=args.ls_min)
    arrays, started = {}, time.monotonic()
    for start in range(0, args.datasets, args.chunk):
        chunk = make_regression_chunk(start // args.chunk, min(args.chunk, args.datasets - start), args.seed, cfg, args.device,
                                      args.gp_fraction, gp_cfg)
        if not arrays:
            arrays = {k: np.lib.format.open_memmap(args.output/f'{k}.npy', 'w+', v.dtype, (args.datasets, *v.shape[1:]))
                      for k, v in chunk.items()}
        for k, v in chunk.items():
            arrays[k][start:start + len(v)] = v
        if (start // args.chunk + 1) % 20 == 0 or start + args.chunk >= args.datasets:
            print(json.dumps({'datasets': min(start + args.chunk, args.datasets), 'total': args.datasets,
                              'elapsed_seconds': time.monotonic() - started}), flush=True)
    for v in arrays.values():
        v.flush()
    meta = {'format': 'regression-fixed-episodes-v1', 'datasets': args.datasets, 'seq_len': cfg.context_max + 2,
            'seed': args.seed, 'chunk': args.chunk, 'geometry': asdict(cfg), 'gp_fraction': args.gp_fraction,
            'het_gp': asdict(gp_cfg),
            'prior': f'{args.gp_fraction:g} het GP (small_mdlm/vendor/regression_priors.py) + '
                     f'{1 - args.gp_fraction:g} TabPFN v1 MLP prior in regression mode (non-causal)'}
    (args.output/'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Regression bank complete: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
