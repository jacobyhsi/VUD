---
name: hx2-cluster-setup
description: "The project moved (2026-10-11) to the hx2 SLURM cluster: no standing GPU job; salloc on interactive-gpu for short work, sbatch on gpu for long jobs; uv .venv"
metadata:
  node_type: memory
  type: project
  originSessionId: 920fc3dc-533c-4b78-bf45-37c47d58a281
  modified: 2026-10-11T15:48:53.191Z
---

Since 2026-10-11 the project lives at `/hx2-weka/home/js2723/PROJECTS/VUD` on the hx2 cluster. Since 2026-10-14 the user connects to login node hx2a04login03 (pinned in their SSH config; it reboots for maintenance every 28 days, Wednesdays about 03:00), not login02. The old cluster's standing job 80896 (2×H200, everything run with `srun --jobid=80896 --overlap`) no longer exists.

- **Short work** (tests, debugging, single VUD runs): `salloc -p interactive-gpu --gres=gpu:h200:1 -c 16 --mem=64G -t 04:00:00 --no-shell`, then `srun --jobid=<id> --overlap` from a screen session. The interactive QoS allows only 1 job per user, with 1 GPU, 16 CPUs, 64G memory (asking for more is rejected with QOSMaxMemoryPerJob) and 8 h wall. `scancel` it when done.
- **Long work** (training, bank generation, VUD sweeps): `sbatch -p gpu` with QoS normal: ≤12 H200s per user, 3 days wall (`overlimit`: 16 GPUs, 7 days). The queue is often long, ~300 pending.
- **Python:** repo `.venv` built with uv (standalone uv 0.13.0 in `~/.local/bin`, installed 2026-10-12; the `uv/0.10.9-GCCcore-14.3.0` module also works) + a uv-managed Python 3.12 in `~/.local/share/uv/python` (torch 2.13.0+cu130, vllm 0.31.0, transformers 5.17.0). The login and compute nodes both have internet access. There is no `nvidia-smi` on the login node; run it through srun.
- **If SLURM commands fail** with `fetch_config: DNS SRV lookup failed`, the login node has lost its SLURM config (seen on hx2a04login02 after its 2026-10-14 reboot: `sackd` inactive, no `_slurmctld._tcp` DNS record). Run SLURM commands through another login node, e.g. `ssh hx2a04login01 'squeue -u $USER'`. Login01 and login03 worked. The filesystem is shared, but a screen session lives on whichever node started it.

**Why:** the user confirmed the sbatch / interactive-gpu split on 2026-10-11, and AGENTS.md records it.
**How to apply:** a pipeline longer than about 8 h can't use interactive-gpu, so write it as an sbatch script that chains its steps ([[chain-follow-up-steps]]). Check GPU load inside the allocation ([[check-gpu-before-launch]]).
