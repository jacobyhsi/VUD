#!/bin/bash
# Original-repo setup on this branch: Qwen2.5-14B served by vLLM (as run_llm.sh on the toy_examples branch), one server
# per GPU (ports 8000/8001), VUD vanilla on all toy datasets except spirals, with the same arguments as the Dream/MDLM runs.
# Servers are stopped at the end. Results: results/toy_{classification,regression}/<dataset>/Qwen2.5-14B/va_vanilla.
set -uo pipefail
cd /home/js2723/PROJECTS/VUD
export PYTHONUNBUFFERED=1
# vLLM needs ninja on PATH; the PyTorch sampler avoids compiling FlashInfer's sampling kernel at startup.
export PATH=/home/js2723/PROJECTS/VUD/.venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/vllm serve "Qwen/Qwen2.5-14B" --dtype auto --tensor-parallel-size 1 --max_model_len 8000 \
    --gpu-memory-utilization 0.95 --port $((8000 + gpu)) > tmp/logs/qwen_vllm_gpu$gpu.log 2>&1 &
  echo $! > tmp/logs/qwen_vllm_gpu$gpu.pid
done
for port in 8000 8001; do
  until curl -s localhost:$port/v1/models > /dev/null; do sleep 10; done
  echo "server on $port ready"
done
run() {  # <toy-classification|toy-regression> <dataset> <port> [extra args]
  local kind=$1 dataset=$2 port=$3; shift 3
  .venv/bin/python run.py $kind --model_name Qwen/Qwen2.5-14B --model_port $port --dataset_name $dataset --va_method vanilla \
    --model_temperature 1 --icl_sample_seed 0 --resume 1 "$@" > tmp/logs/qwen_${dataset}.log 2>&1
}
pids=()
run toy-classification logistic_regression 8000 --x_range "{'x1': [-15, 15, 0.2]}" & pids+=($!)
run toy-classification moons_1 8000 & pids+=($!)
run toy-regression linear_regression 8000 & pids+=($!)
run toy-classification moons_2 8001 & pids+=($!)
run toy-regression gaps 8001 --x_range "{'x': [-15, 15, 0.2]}" & pids+=($!)
wait "${pids[@]}"   # only the experiment runs, not the servers
for gpu in 0 1; do kill $(cat tmp/logs/qwen_vllm_gpu$gpu.pid) 2>/dev/null; done
echo 'QWEN TOY RUNS COMPLETE'
