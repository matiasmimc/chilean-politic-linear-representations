#!/bin/bash
# Steering con activaciones de atencion (o_proj por cabeza), tareas divididas, GPU 1, secuencial.
# Uso: bash scripts/steering/cola_atencion.sh
cd "$(dirname "$0")/../.."
LL=meta-llama/Llama-3.1-8B-Instruct; GE=google/gemma-4-E4B-it; GL=zai-org/GLM-4-9B-0414
run() {
  echo; echo "##### [$(date +%H:%M)] ATENCION $1 $2"
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/steering/predecir_votos_steered.py +model=$1 +steering_method=$2 +task_mode=divided +min_frac_contra=0.25
}
run $LL cpca
for s in 0 1 2; do run $LL random_k16_s$s; done
run $GE cpca; run $GE random_k16_s0
run $GL cpca; run $GL random_k16_s0
echo "COLA TERMINADA $(date)"
uv run python scripts/steering/resumen_resultados.py atencion
uv run python scripts/steering/bootstrap_controles.py divided atencion
