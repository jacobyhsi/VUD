# Running Scripts

The project runs on two clusters. Check `hostname` to see which one you are on.

## Original cluster (standing GPU job)

Since we are on a login node, we have to

1) Start a screen session
2) Check the jobid using squeue
3) srun --jobid=<> --overlap --pty bash to use the gpu
4) Run whatever script I told you about
5) Once the screen is done running the script, kill the screen

## hx2 cluster (hostname `hx2a04login*`)

We work from a login node (SLURM, account `hx2-users`). There is no standing GPU job; get GPUs
in one of two ways:

- **`interactive-gpu`** (QoS `interactive`): 1 job per user, at most 1 H200, 16 CPUs, 64G memory and 8 h wall.
  Use it for tests, debugging and short VUD runs:
  `salloc -p interactive-gpu --gres=gpu:h200:1 -c 16 --mem=64G -t 04:00:00 -J <name> --no-shell`
- **`gpu`** via `sbatch` (QoS `normal`): up to 12 H200s per user and 3 days wall (`--qos=overlimit`: 16 GPUs, 7 days).
  Use it for training, bank generation and long VUD sweeps. The queue is often long.

To run inside an allocation you hold:

1) Start a screen session
2) Check the jobid using squeue
3) srun --jobid=<> --overlap --pty bash (or `srun --jobid=<> --overlap bash <script>`) to use the gpu
4) Run whatever script I told you about
5) Once the screen is done running the script, kill the screen, and release a `salloc` job with `scancel` when done

## Both clusters

Feel free to start as many jobs as you can simultaneously unless there is no VRAM. Run one job per GPU: give each
independent run (dataset, arm, checkpoint) its own GPU. On the original cluster, spread the runs over the job's GPUs
(check `nvidia-smi` first). On hx2, use the interactive GPU for one and `sbatch -p gpu --gres=gpu:h200:1` for the
others when the `gpu` partition has free GPUs (check `AllocTRES` with `scontrol show node`). Only once no more GPUs can
be had, stack the remaining runs on the GPUs you hold (`srun --overlap`), balanced across them.

# Python Environment

On hx2: a uv venv at the repo root (`.venv`, git-ignored), using a uv-managed Python 3.12, so no modules are needed at runtime.
uv itself is installed standalone in `~/.local/bin` (on PATH via `~/.bashrc`; the `uv` module also works):

```bash
uv venv .venv --python 3.12 --python-preference only-managed
VIRTUAL_ENV=$PWD/.venv uv pip install -r requirements.txt pip einops
bash dllm-training/setup_tabpfn_v1.sh   # with .venv/bin first on PATH
```

Install on the login node (it has internet). Call `.venv/bin/python` explicitly in launchers.

# Temporary Files

- Keep temporary files inside this repository's `tmp/` directory, not `/home/js2723/tmp/` or other home-directory locations.
- Use `tmp/scripts/` for temporary launchers and analysis scripts, and `tmp/logs/` for logs. Reuse existing subdirectories where appropriate.
- Keep these temporary files ignored by Git unless the user explicitly asks to commit them.
- Please make sure to clean up and testing / smoke code.

# Project History

Before starting new work, read `dllm-training/history/EXPERIMENT_HISTORY.md`: everything tried so far
(what worked and what didn't, with numbers), the adopted classification/regression checkpoints, and the
operational lessons. Supporting material:
- `dllm-training/history/claude_notes/`: the assistant's working rules and lessons (copy into the
  Claude Code memory folder on a new machine)
- `dllm-training/history/checkpoints/`: key trained models
- `dllm-training/history/scripts/`: the original launchers and analysis scripts (reference only; paths are
  from the old cluster)
- `dllm-training/history/transcripts/`: full assistant transcripts (gzipped, emails redacted)
- `dllm-training/TABPFN_V1.md`: the TabPFN v1 experiments
- `results/`: VUD outputs for every model and dataset
