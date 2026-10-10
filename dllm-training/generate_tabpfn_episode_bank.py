"""Generate paired TabPFN IID/control and VUD-geometry episode banks safely.

Run on an allocated GPU. Refuses nonempty output roots; meta.json is written
only after all arrays have been flushed. Each bank has identical task IDs,
dimensions and episode sizes, making its held-out split paired with the other.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from small_mdlm.tabpfn_episodes import GeometryConfig, make_paired_chunk
from small_mdlm.vendor.controlled_gp_prior import ControlledGPConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', type=int, default=3_006_464)
    parser.add_argument('--chunk', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--draws', type=int, default=64)
    parser.add_argument('--near-query-prob', type=float, default=GeometryConfig.near_query_prob,
                        help='Share of geometry episodes whose auxiliary sits next to the query')
    parser.add_argument('--broad-query-prob', type=float, default=GeometryConfig.broad_query_prob,
                        help='Share of geometry episodes whose query is placed far from the context')
    parser.add_argument('--one-d-prob', type=float, default=GeometryConfig.one_d_prob, help='Share of 1-D tasks')
    parser.add_argument('--layout-prob', type=float, default=GeometryConfig.layout_prob,
                        help='Share of tasks whose inputs follow a structured layout (clusters, curve, correlated)')
    parser.add_argument('--controlled-gp-fraction', type=float, default=0.0,
                        help='Share of tasks from the controlled GP prior (rest: TabPFN MLP); 0 = TabPFN only, as before')
    parser.add_argument('--controlled-gp-config', type=Path, default=Path(__file__).parent/'configs/controlled_gp_prior.json')
    args = parser.parse_args()
    gp_cfg = ControlledGPConfig(**json.loads(args.controlled_gp_config.read_text()))
    if args.datasets < 1 or args.chunk < 1:
        raise ValueError('Positive dataset and chunk counts required')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh output root; existing banks are never overwritten')
    torch.set_num_threads(2)
    cfg = GeometryConfig(draws=args.draws, near_query_prob=args.near_query_prob,
                         broad_query_prob=args.broad_query_prob, one_d_prob=args.one_d_prob, layout_prob=args.layout_prob)
    arrays = {}
    started = time.monotonic()
    for start in range(0, args.datasets, args.chunk):
        result = make_paired_chunk(start // args.chunk, min(args.chunk, args.datasets-start), args.seed, cfg, args.device,
                                   args.controlled_gp_fraction, gp_cfg)
        if not arrays:
            for arm, chunk in result.items():
                root = args.output / arm
                root.mkdir(parents=True)
                arrays[arm] = {key: np.lib.format.open_memmap(root/f'{key}.npy', 'w+', value.dtype, (args.datasets, *value.shape[1:]))
                               for key, value in chunk.items()}
        for arm, chunk in result.items():
            for key, value in chunk.items():
                arrays[arm][key][start:start+len(value)] = value
        if start == 0 or (start // args.chunk + 1) % 20 == 0 or start + args.chunk >= args.datasets:
            for bank in arrays.values():
                for value in bank.values():
                    value.flush()
            print(json.dumps({'datasets_per_arm': min(start+args.chunk, args.datasets),
                              'total_per_arm': args.datasets, 'elapsed_seconds': time.monotonic()-started}), flush=True)
    for arm in arrays:
        meta = {'format': 'tabpfn-fixed-episodes-v1', 'arm': arm, 'geometry': asdict(cfg),
                'datasets': args.datasets, 'seq_len': cfg.context_max+2,
                'seed': args.seed, 'chunk': args.chunk,
                'threshold': 'fixed from 34 IID calibration points; extra locations share the threshold',
                'prior': 'unchanged small_mdlm/vendor/tabpfn_prior.py network/noise hyperpriors'}
        if args.controlled_gp_fraction > 0:
            meta['prior'] = (f'{args.controlled_gp_fraction:g} controlled GP (small_mdlm/vendor/controlled_gp_prior.py) + '
                             f'{1 - args.controlled_gp_fraction:g} TabPFN MLP (unchanged tabpfn_prior.py hyperpriors)')
            meta['controlled_gp'] = {'fraction': args.controlled_gp_fraction, 'settings': asdict(gp_cfg),
                                     'threshold': 'independent of the data; labels at extra locations are exact'}
        (args.output/arm/'meta.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(f'Paired banks complete: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
