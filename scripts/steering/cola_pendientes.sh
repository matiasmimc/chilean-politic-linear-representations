#!/bin/bash
# Pendientes de la cola nocturna, dos colas en paralelo (GPU 0 y GPU 1). Uso: bash scripts/steering/cola_pendientes.sh
cd "$(dirname "$0")/../.."
LL=meta-llama/Llama-3.1-8B-Instruct; GE=google/gemma-4-E4B-it; GL=zai-org/GLM-4-9B-0414
D="+task_mode=divided +min_frac_contra=0.25"
run() { # gpu modelo metodo
  echo "##### [$(date +%H:%M)] GPU$1 $2 $3"
  CUDA_VISIBLE_DEVICES=$1 uv run python scripts/steering/predecir_votos_steered.py +target=residual +model=$2 +steering_method=$3 $D
}
( run 1 $GE cpca; run 1 $GE random_k16_s0; run 1 $LL cpca_k1; run 1 $LL cpca_k64; echo "COLA GPU1 TERMINADA $(date)" ) > logs/cola_pendientes_gpu1.log 2>&1 &
( run 0 $GL cpca; run 0 $GL random_k16_s0; run 0 $LL cpca_k4; echo "COLA GPU0 TERMINADA $(date)" ) > logs/cola_pendientes_gpu0.log 2>&1 &
wait
uv run python scripts/steering/resumen_resultados.py > logs/resumen_pendientes.log 2>&1
