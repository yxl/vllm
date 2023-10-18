#!/bin/bash 
set -x

export PYTHONPATH=$PYTHONPATH:/home/vllm/vllm
export CUDA_VISIBLE_DEVICES=4
export WORLD_SIZE=1

python3 vllm/entrypoints/openai/api_server.py \
   --host 172.26.1.45 \
   --port 8001 \
   --served-model-name qwen-7b-chat \
   --model /home/vllm/model/model-qwen-7b-chat-hf \
   --trust-remote-code \
   --tokenizer-mode auto \
   --max-num-batched-tokens 8192 \
   --tensor-parallel-size 1

