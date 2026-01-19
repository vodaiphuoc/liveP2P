#!/usr/bin/env bash
set -e

exec ./llama-server \
  --hf-repo "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --threads "$THREADS" \
  --batch-size "$BATCH_SIZE" \
  --ctx-size "$CTX_SIZE" \
  --mlock \
  --n-gpu-layers 0\
  --verbosity 3\
  --chat-template none\
  --models-dir "$VOLUME"