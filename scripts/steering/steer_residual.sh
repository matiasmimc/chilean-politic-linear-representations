#!/bin/bash
# Steering residual con cPCA sobre prediccion de votos. GPU 1, un modelo por vez.
# Uso: bash scripts/steering/steer_residual.sh [modelo ...]
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(meta-llama/Llama-3.1-8B-Instruct google/gemma-4-E4B-it zai-org/GLM-4-9B-0414)
for M in "${MODELS[@]}"; do
  for METHOD in cpca cpca_90var cpca_topk; do
    echo "=== $M | $METHOD ==="
    uv run python scripts/steering/predecir_votos_steered.py +steering_method=$METHOD +target=residual +model=$M || echo "FALLO $M $METHOD"
  done
done
echo "TERMINADO"
