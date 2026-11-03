#!/bin/bash
# Prueba rapida de slant con direcciones CAA con control de tema (GPU 1): caatopic_k4 vs control aleatorio, Llama y Gemma.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
LL=meta-llama/Llama-3.1-8B-Instruct; GE=google/gemma-4-E4B-it
G="uv run python scripts/steering/intervencion_modelo.py +target=residual +out_tag=_wide +n_samples=2"
A='+alphas=[-2,-1,-0.5,0,0.5,1,2]'
for M in $LL $GE; do
  echo "##### [$(date +%H:%M)] generacion $M"
  for m in caatopic_k4 rndcaatopic_k4_s0; do $G +model=$M +steering_method=$m "$A"; done
done
echo "##### [$(date +%H:%M)] filtro de colapso"
uv run python scripts/steering/colapso_texto.py results/steering_v2/residual/caatopic_k4_wide results/steering_v2/residual/rndcaatopic_k4_s0_wide
for N in caatopic_k4 rndcaatopic_k4_s0; do
  for J in Llama gemma; do
    echo "##### [$(date +%H:%M)] juez $N $J"
    uv run python scripts/steering/evaluar_respuestas.py --folder steering_v2/residual/${N}_wide_sel --model $J
  done
done
echo "COLA CAATOPIC TERMINADA $(date)"
