"""Framework-independent linear-regression episodes for masked-DLM training.

Distribution follows create_reg_data in von Oswald et al.'s implementation:
https://github.com/google-research/self-organising-systems/blob/master/
transformers_learn_icl_by_gd/src/data.py
This is an independent NumPy implementation, not a bitwise JAX reproduction.
"""

import argparse
from dataclasses import dataclass
import json

import numpy as np


@dataclass(frozen=True)
class RegressionConfig:
    input_size: int = 10
    context_size: int = 10
    input_range: float = 1.0
    weight_scale: float = 1.0

    def __post_init__(self):
        for name in ('input_size', 'context_size'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not np.isfinite(self.input_range) or self.input_range <= 0:
            raise ValueError('input_range must be finite and positive')
        if not np.isfinite(self.weight_scale) or self.weight_scale < 0:
            raise ValueError('weight_scale must be finite and nonnegative')


@dataclass(frozen=True)
class RegressionTask:
    context_x: np.ndarray  # (context_size, input_size)
    context_y: np.ndarray  # (context_size,)
    query_x: np.ndarray    # (1, input_size)
    query_y: np.ndarray    # (1,); supervision only
    weights: np.ndarray    # (input_size,); diagnostics only

    def model_inputs(self):
        """Return only observable data, with no query labels or task weights."""
        return {name: getattr(self, name).copy()
                for name in ('context_x', 'context_y', 'query_x')}

    def constructed_tokens(self):
        """Paper-style numeric tokens; zero is a placeholder, NOT a DLM mask ID."""
        context = np.column_stack((self.context_x, self.context_y))
        query = np.column_stack((self.query_x, np.zeros(1)))
        return np.concatenate((context, query), axis=0)


class LinearRegressionTasks:
    """Index-addressable tasks, reproducible regardless of access order.

    Train/validation/test use separate deterministic random streams. This
    isolates splits but does not guarantee distinct values for degenerate
    distributions (e.g. weight_scale=0). No global NumPy RNG state is changed.
    """

    SPLITS = {'train': 0, 'validation': 1, 'test': 2}

    def __init__(self, config=None, seed=0, split='train'):
        self.config = config if config is not None else RegressionConfig()
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if split not in self.SPLITS:
            raise ValueError(f'split must be one of {tuple(self.SPLITS)}')
        self.seed = seed
        self.split = split

    def sample(self, task_index):
        if (isinstance(task_index, bool) or not isinstance(task_index, int)
                or task_index < 0):
            raise ValueError('task_index must be a nonnegative integer')
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, self.SPLITS[self.split], task_index]))
        cfg = self.config
        weights = rng.normal(size=cfg.input_size) * cfg.weight_scale
        context_x = rng.uniform(-cfg.input_range / 2, cfg.input_range / 2,
                                size=(cfg.context_size, cfg.input_size))
        query_x = rng.uniform(-cfg.input_range / 2, cfg.input_range / 2,
                              size=(1, cfg.input_size))
        return RegressionTask(context_x, context_x @ weights, query_x,
                              query_x @ weights, weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-size', type=int, default=10)
    parser.add_argument('--context-size', type=int, default=10)
    parser.add_argument('--input-range', type=float, default=1.0)
    parser.add_argument('--weight-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', choices=tuple(LinearRegressionTasks.SPLITS), default='train')
    parser.add_argument('--task-index', type=int, default=0)
    args = parser.parse_args()
    config = RegressionConfig(args.input_size, args.context_size,
                              args.input_range, args.weight_scale)
    task = LinearRegressionTasks(config, args.seed, args.split).sample(args.task_index)
    print(json.dumps({
        'model_inputs': {k: v.tolist() for k, v in task.model_inputs().items()},
        'query_y_supervision_only': task.query_y.tolist(),
        'weights_diagnostics_only': task.weights.tolist(),
    }, indent=2))


if __name__ == '__main__':
    main()
