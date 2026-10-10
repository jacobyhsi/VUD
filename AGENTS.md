# Running Scripts

Since we are on a login node, we have to

1) Start a screen session
2) Check the jobid using squeue
3) srun --jobid=<> --overlap --pty bash to use the gpu
4) Run whatever script I told you about
5) Once the screen is done running the script, kill the screen

Feel free to start as many jobs as you can simultaneously unless there is no VRAM. Balance the jobs out across available GPUs.

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
