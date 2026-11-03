#!/bin/bash
# Extrae residual stream (politicos + fondo neutral) y calcula cPCA residual.
# Uso: CUDA_VISIBLE_DEVICES=1 bash scripts/extracion_activaciones/extraer_residual_todo.sh [modelo ...]
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(meta-llama/Llama-3.1-8B-Instruct google/gemma-4-E4B-it zai-org/GLM-4-9B-0414)

for M in "${MODELS[@]}"; do
  echo "=== $M: politicos (residual) ==="
  uv run python scripts/extracion_activaciones/extraer_activaciones.py --model "$M" --target residual || { echo "FALLO politicos $M"; continue; }
  echo "=== $M: neutral (residual) ==="
  uv run python scripts/extracion_activaciones/extraer_activaciones_neutrales.py --model "$M" --target residual || { echo "FALLO neutral $M"; continue; }
  echo "=== $M: cPCA residual ==="
  uv run python scripts/no-supervisado/analisis_cpca.py +target=residual +model="$M" || echo "FALLO cPCA $M"
done
echo "TERMINADO"
