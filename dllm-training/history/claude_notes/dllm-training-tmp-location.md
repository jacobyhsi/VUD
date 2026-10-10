---
name: dllm-training-tmp-location
description: "Temp files for dllm-training work go in dllm-training/tmp/, not the repo-root tmp/"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ca6189b9-787a-42df-9a07-b2f20dcffb95
  modified: 2026-09-27T04:18:36.191Z
---

Temporary files made for `dllm-training/` work (launch scripts, logs, reference clones, smoke outputs) go in `dllm-training/tmp/{scripts,logs,refs,smoke}`, not the repo-root `tmp/` that AGENTS.md describes for general VUD work.

**Why:** The user asked to keep dllm-training scratch files together with that subproject (2026-09-27).

**How to apply:** For any dllm-training run or analysis, write launchers to `dllm-training/tmp/scripts/` and logs to `dllm-training/tmp/logs/`. Git already ignores it through the root `.gitignore` `tmp/` rule. Still follow AGENTS.md's screen + srun workflow and clean up smoke/test artifacts afterwards.
