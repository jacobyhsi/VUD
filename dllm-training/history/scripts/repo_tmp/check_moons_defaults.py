import ast
from pathlib import Path

# Execute only the CLI setup, avoiding model imports or loading.
path = Path('src/experiments/toy_classification.py')
source = path.read_text()
compile(source, str(path), 'exec')
cli = source[source.index('parser = argparse.ArgumentParser'):source.index('@dataclass')]
import argparse
scope = {'argparse': argparse}
exec(cli, scope)
parse = scope['parse_args']
expected = {
    'moons_1': (30, {'x1': [-1.5, 2.5, 0.2], 'x2': [-1.5, 2.5, 0.2]}),
    'moons_2': (30, {'x1': [-3.0, 3.5, 0.2], 'x2': [-2.5, 3.0, 0.2]}),
    'spirals': (200, {'x1': [-4, 4, 0.1], 'x2': [-4, 4, 0.1]}),
    'logistic_regression': (15, {'x1': [-12, 12, 0.2]}),
}
for dataset, (size, grid) in expected.items():
    args = parse(['--dataset_name', dataset])
    assert args.D_size == size
    assert ast.literal_eval(args.x_range) == grid
    custom = "{'x1': [0, 1, 0.1], 'x2': [0, 2, 0.1]}"
    args = parse(['--dataset_name', dataset, '--D_size', '7', '--x_range', custom])
    assert args.D_size == 7 and args.x_range == custom
    assert parse(['--dataset_name', dataset, '--D_size', '7']).x_range == str(grid)
    assert parse(['--dataset_name', dataset, '--x_range', custom]).D_size == size
    print(dataset, 'defaults and overrides PASS', flush=True)
print('PASS: source compiles; all defaults and independent overrides verified', flush=True)
