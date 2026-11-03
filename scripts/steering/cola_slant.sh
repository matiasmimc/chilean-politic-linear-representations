#!/bin/bash
# Political slant (GPU 1, secuencial, idempotente): generacion por modelo y luego juez Gemma-3-27B (4 bits).
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
G="uv run python scripts/steering/intervencion_modelo.py +target=residual"
NP="paths.results_dir=results/replication/v3_noparty"
gen() { # modelo
  M=$1
  echo "##### [$(date +%H:%M)] generacion $M"
  $G +model=$M +steering_method=cpca
  $G +model=$M +steering_method=ridge
  $G +model=$M +steering_method=cpca_np $NP
  $G +model=$M +steering_method=ridge_np $NP
  for s in 0 1 2; do $G +model=$M +steering_method=random_k16_s$s; done
}
judge() { # filtro_modelo
  for m in cpca ridge cpca_np ridge_np random_k16_s0 random_k16_s1 random_k16_s2; do
    echo "##### [$(date +%H:%M)] juez $1 $m"
    uv run python scripts/steering/evaluar_respuestas.py --folder steering_v2/residual/$m --model $1
  done
}
gen meta-llama/Llama-3.1-8B-Instruct
judge Llama
gen google/gemma-4-E4B-it
judge gemma
echo "COLA SLANT TERMINADA $(date)"
