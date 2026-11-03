#!/bin/bash
# Activaciones para los controles de probing (GPU 1): sin partido en el prompt, y solo-nombres en residual.
# Idempotente: extraer_activaciones.py salta lo ya extraido.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
X="uv run python scripts/extracion_activaciones/extraer_activaciones.py"
for M in meta-llama/Llama-3.1-8B-Instruct google/gemma-4-E4B-it zai-org/GLM-4-9B-0414; do
  echo "##### [$(date +%H:%M)] $M"
  $X --model $M --target residual --names-only                 # con partido, solo nombres, residual
  $X --model $M --names-only                                   # GLM atencion solo-nombres (si falta)
  for T in attention residual; do
    $X --model $M --target $T --no-party                       # intervenciones, sin partido
    $X --model $M --target $T --no-party --names-only          # solo nombres, sin partido
  done
done
echo "EXTRACCION TERMINADA $(date)"
