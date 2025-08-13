#!/bin/bash

# Runs Gemma 3 4B with FP8 quant in a Docker container
# Assumes $MODEL_DIR is set to the directory containing the model files

docker run --rm -it \
  --name gemma3-4b \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -v "${MODEL_DIR}:/models:ro" \
  vllm/vllm-openai:latest \
  --model /models/google/gemma-3-4b-it \
  --quantization fp8 \
  --served-model-name google/gemma-3-4b-it
