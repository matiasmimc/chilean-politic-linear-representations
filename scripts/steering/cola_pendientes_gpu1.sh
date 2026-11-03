#!/bin/bash
# Pendientes restantes, solo GPU 1, secuencial. Uso: bash scripts/steering/cola_pendientes_gpu1.sh
cd "$(dirname "$0")/../.."
LL=meta-llama/Llama-3.1-8B-Instruct; GL=zai-org/GLM-4-9B-0414
run() {
  echo "##### [$(date +%H:%M)] $1 $2"
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/steering/predecir_votos_steered.py +target=residual +model=$1 +steering_method=$2 +task_mode=divided +min_frac_contra=0.25
}
run $GL random_k16_s0; run $LL cpca_k1; run $LL cpca_k4; run $LL cpca_k64
echo "COLA TERMINADA $(date)"
uv run python scripts/steering/resumen_resultados.py
uv run python scripts/steering/bootstrap_controles.py divided
