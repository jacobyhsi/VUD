---
name: chain-follow-up-steps
description: "Queue every follow-up step (checks, eval, VUD runs) in the same launcher script as a long job so work continues while the user is away"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ca6189b9-787a-42df-9a07-b2f20dcffb95
  modified: 2026-10-02T13:08:19.918Z
---

When launching long GPU jobs (training, bank generation), put all of the follow-up steps in the same screen/launcher script: e.g. train → mode-1 check → VUD on every candidate checkpoint, spread over both GPUs. Don't plan to "check and decide" by hand afterwards. If a choice is needed, run all the options; VUD runs are cheap (~25 min).

**Why:** on 2026-10-02 two trainings finished around 06:00–07:00 UTC, but the VUD runs only started at 12:50, when the user came back and asked "why didnt it run while i was gone". Background waits time out after at most 2 h, and Claude only acts on a new message or a task notification, so steps that weren't chained never ran.
**How to apply:** for any job longer than about 1 h, write the pipeline script (as `dllm-training/tmp/scripts/tabpfn_pipeline.sh` did) before launching. Related: [[dllm-training-tmp-location]].
