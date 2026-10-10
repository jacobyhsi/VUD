---
name: check-gpu-before-launch
description: "Run nvidia-smi (via srun on the job) before every launch and put new jobs on the idle GPU; the user's long Dream runs often occupy one GPU"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ca6189b9-787a-42df-9a07-b2f20dcffb95
  modified: 2026-10-02T13:35:33.619Z
---

Before launching any GPU job, check current load with `srun --jobid=<job> --overlap nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` and put new work on the least-loaded GPU. Don't hard-code GPU numbers in launch scripts from an earlier plan.

**Why:** on 2026-10-02 the user found MDLM VUD jobs stacked on GPU1, alongside their long-running Dream spirals job, while GPU0 sat free. The user said: "can u make sure u always check gpu availability first? gpu0 is free but they are stacked on gpu1". The launch scripts had fixed GPU assignments.
**How to apply:** take the GPU as a parameter in launchers, and choose it from a fresh nvidia-smi check at launch time; within a pipeline, re-check between stages. Related: [[chain-follow-up-steps]].
