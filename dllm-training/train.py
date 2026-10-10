"""Train a randomly initialized conditional MDLM on online regression tasks."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from small_mdlm.episodes import EpisodeConfig, episode
from small_mdlm.model import all_masked_losses, answer_probs, build_model, masked_loss, sample
from small_mdlm.tokenizer import NumberTokenizer

FORMAT = 'vud-numeric-mdlm-v1'


def _worker_init(_):
    # Tensor IPC starts a resource-sharing Unix socket in TMPDIR. Close it
    # before multiprocessing removes that directory; NFS cannot unlink an
    # open socket cleanly and otherwise leaves busy .nfs files at shutdown.
    from multiprocessing import resource_sharer, util
    util.Finalize(None, resource_sharer.stop, exitpriority=10)


def examples(config, tokenizer, seed, start, count, split):
    result, tasks = [], []
    for index in range(start, start + count):
        task = episode(config, seed, index, split)
        result.append(tokenizer.encode(task['x'], task['y'], task['target_rows']))
        tasks.append(task)
    return result, tasks


class TrainBatches(torch.utils.data.Dataset):
    """Batch for training step s = tasks [s*B, (s+1)*B), so workers and resume stay deterministic."""

    def __init__(self, data_config, tokenizer, seed, batch_size, steps):
        self.data_config, self.tokenizer = data_config, tokenizer
        self.seed, self.batch_size, self.steps = seed, batch_size, steps

    def __len__(self):
        return self.steps

    def __getitem__(self, step):
        batch, _ = examples(self.data_config, self.tokenizer, self.seed,
                            step * self.batch_size, self.batch_size, 'train')
        return self.tokenizer.collate(batch)


def _validate_labels(model, tokenizer, examples_, tasks, ids, kinds, attention, values, loss):
    """Classification: exact p(y | x, D) at each hidden label vs the true p and the sampled label."""
    probs = answer_probs(model, tokenizer, ids, kinds, attention, values).cpu()
    label_ids = torch.arange(12, 12 + tokenizer.labels)
    log_loss, correct, distance, true_log_loss = [], [], [], []
    for i, (example, task) in enumerate(zip(examples_, tasks)):
        for row, (position,) in example['spans'].items():
            p = probs[i, position, label_ids].double().numpy()
            label, p_true = int(task['y'][row]), np.zeros(tokenizer.labels)
            p_true[:task['probs'].shape[1]] = task['probs'][row]
            log_loss.append(-np.log(max(p[label], 1e-12)))
            correct.append(p.argmax() == label)
            # Total-variation distance; for two classes this is |p1 - p1_true|.
            distance.append(0.5 * np.abs(p - p_true).sum())
            true_log_loss.append(-np.log(max(p_true[label], 1e-12)))
    model.train()
    return {'validation_loss': loss, 'label_log_loss': float(np.mean(log_loss)),
            'true_p_log_loss': float(np.mean(true_log_loss)), 'accuracy': float(np.mean(correct)),
            'abs_error_vs_true_p': float(np.mean(distance))}


@torch.no_grad()
def validate(model, tokenizer, data_config, training, device):
    model.eval()
    examples_, tasks = examples(data_config, tokenizer, training['seed'], 0,
                               training['eval_tasks'], 'validation')
    ids, kinds, attention, values = tokenizer.collate(examples_, device)
    gen = torch.Generator(device=device).manual_seed(100000 + training['seed'])
    loss = masked_loss(model, tokenizer, ids, kinds, attention, gen, values).item()
    if tokenizer.labels:
        return _validate_labels(model, tokenizer, examples_, tasks, ids, kinds, attention, values, loss)
    generated = sample(model, tokenizer, ids, kinds, attention,
                       seed=100001 + training['seed'], steps=training['sample_steps'], values=values)
    position_ce = all_masked_losses(model, tokenizer, ids, kinds, attention, values).cpu()
    names = (['sign'] + [f'int{k}' for k in range(tokenizer.integer_digits)]
             + ['point'] + [f'dec{k}' for k in range(tokenizer.decimals)])
    by_position = {name: [] for name in names if name != 'point'}
    errors, rounding, zero, context_mean, ols = [], [], [], [], []
    for i, (example, task) in enumerate(zip(examples_, tasks)):
        # Baselines see the same rounded observations as the tokenizer.
        observed = [r for r in range(len(task['y'])) if r not in example['spans']]
        x = np.round(task['x'], tokenizer.decimals)
        y_obs = np.round(task['y'][observed], tokenizer.decimals)
        design = x[observed]
        if data_config.bias_scale > 0:
            design = np.column_stack([design, np.ones(len(observed))])
        coef = np.linalg.lstsq(design, y_obs, rcond=None)[0]
        for row, positions in example['spans'].items():
            target = task['y'][row]
            pred = tokenizer.decode_number(generated[i, positions].tolist())
            errors.append((pred - target) ** 2)
            rounding.append((tokenizer.decode_number(tokenizer.number(target)) - target) ** 2)
            query = np.append(x[row], 1) if data_config.bias_scale > 0 else x[row]
            ols.append((query @ coef - target) ** 2)
            zero.append(target ** 2)
            context_mean.append((y_obs.mean() - target) ** 2)
            for name, position in zip(names, positions):
                if name != 'point':
                    by_position[name].append(position_ce[i, position].item())
    model.train()
    return {'validation_loss': loss, 'sample_mse': float(np.mean(errors)),
            'zero_mse': float(np.mean(zero)), 'context_mean_mse': float(np.mean(context_mean)),
            'least_squares_mse': float(np.mean(ols)),
            'target_quantization_mse': float(np.mean(rounding)),
            **{f'ce_all_masked_{k}': float(np.mean(v)) for k, v in by_position.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/linear_vud.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--keep-checkpoints', action='store_true', help='Also retain each evaluated checkpoint for held-out model selection')
    parser.add_argument('--max-steps', type=int, help='Optional training-step limit, useful for smoke tests/resume checks')
    parser.add_argument('--workers', type=int, default=0, help='CPU processes that build batches ahead of the GPU (does not change the data)')
    parser.add_argument('--worker-start-method', choices=['spawn', 'fork', 'forkserver'], default=None,
                        help='Optional loader process context; spawn avoids inheriting NFS-backed IPC socket handles')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    training = config['training']
    if min(training[k] for k in ('steps', 'batch_size', 'eval_every', 'eval_tasks', 'sample_steps')) < 1:
        raise ValueError('Training counts must be positive')
    limit = args.max_steps if args.max_steps is not None else training['steps']
    if limit < 1:
        raise ValueError('max-steps must be positive')
    checkpoint_path = args.output/'checkpoint.pt'
    if args.resume:
        if not checkpoint_path.is_file():
            raise ValueError('No checkpoint to resume')
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output is nonempty; use --resume or choose a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(training['seed'])
    tokenizer = NumberTokenizer(**config['tokenizer'])
    data_config = EpisodeConfig(**config['data'])
    model = build_model(config['model'], tokenizer, data_config.input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['lr'], weight_decay=0.0)
    generator = torch.Generator(device=device).manual_seed(training['seed'] + 12345)
    step = 0
    if args.resume:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if saved['format'] != FORMAT or saved['config'] != config:
            raise ValueError('Checkpoint configuration/format mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator_state'].cpu())
        torch.set_rng_state(saved['torch_state'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_state'].cpu(), device)
        step = saved['step']
    else:
        (args.output/'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'MDLM parameters={sum(p.numel() for p in model.parameters())}; device={device}; start_step={step}', flush=True)
    print('Checkpoint is an experimental synthetic-regression model, not a validated Dream replacement.', flush=True)
    # training.precision: 'fp32' (default), 'tf32' (TensorFloat-32 matmuls,
    # float32 elsewhere) or 'bf16' (autocast; learned noticeably slower on the
    # broad linear prior). Weights, optimizer state and the loss stay float32.
    precision = training.get('precision', 'fp32')
    if precision not in ('fp32', 'tf32', 'bf16'):
        raise ValueError(f'Unknown precision {precision!r}')
    torch.backends.cuda.matmul.allow_tf32 = precision == 'tf32'
    torch.backends.cudnn.allow_tf32 = precision == 'tf32'
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == 'bf16')
    started = time.monotonic()
    model.train()
    batches = torch.utils.data.DataLoader(
        TrainBatches(data_config, tokenizer, training['seed'], training['batch_size'], limit),
        batch_size=None, sampler=range(step, limit), num_workers=args.workers,
        multiprocessing_context=args.worker_start_method if args.workers else None,
        worker_init_fn=_worker_init if args.workers else None,
        pin_memory=device.type == 'cuda', prefetch_factor=4 if args.workers else None)
    for batch in batches:
        ids, kinds, attention, values = (t.to(device, non_blocking=True) for t in batch)
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            loss = masked_loss(model, tokenizer, ids, kinds, attention, generator, values)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % training['eval_every'] == 0 or step == limit:
            with autocast:
                validation = validate(model, tokenizer, data_config, training, device)
            metrics = {'step': step, 'train_loss': loss.item(),
                       'elapsed_seconds': time.monotonic() - started, **validation}
            print(json.dumps(metrics), flush=True)
            with (args.output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            saved = {'format': FORMAT, 'config': config, 'step': step,
                     'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'generator_state': generator.get_state(), 'torch_state': torch.get_rng_state(),
                     'cuda_state': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
            temporary = args.output/'checkpoint.pt.tmp'
            torch.save(saved, temporary)
            temporary.replace(checkpoint_path)
            if args.keep_checkpoints:
                # checkpoint.pt is replaced, not modified in place; hard links
                # preserve snapshots without another serialization or copy.
                import os
                os.link(checkpoint_path, args.output/f'checkpoint_step_{step:06d}.pt')
    print(f'Checkpoint: {checkpoint_path.resolve()}', flush=True)


if __name__ == '__main__':
    main()
