#!/bin/bash
# Slant con direcciones CAA (contraste derecha-izquierda de intervenciones de los extremos de P6). GPU 1.
# Requiere: results/caa/<modelo>_caa_directions.pkl (extraer_intervenciones_caa.py + construir_direcciones_caa.py).
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
LL=meta-llama/Llama-3.1-8B-Instruct; GE=google/gemma-4-E4B-it
G="uv run python scripts/steering/intervencion_modelo.py +target=residual +out_tag=_wide"
A='+alphas=[-2,-1,-0.5,-0.25,0,0.25,0.5,1,2]'
for M in $LL $GE; do
  echo "##### [$(date +%H:%M)] generacion CAA $M"
  for m in caa_k1 caa_k4 rndcaa_k1_s0 rndcaa_k4_s0 rndcaa_k4_s1; do $G +model=$M +steering_method=$m "$A"; done
done
echo "##### [$(date +%H:%M)] filtro de colapso"
uv run python scripts/steering/colapso_texto.py results/steering_v2/residual/caa_k1_wide results/steering_v2/residual/caa_k4_wide results/steering_v2/residual/rndcaa_k1_s0_wide results/steering_v2/residual/rndcaa_k4_s0_wide results/steering_v2/residual/rndcaa_k4_s1_wide
for N in caa_k1 caa_k4 rndcaa_k1_s0 rndcaa_k4_s0 rndcaa_k4_s1; do
  for J in Llama gemma; do
    echo "##### [$(date +%H:%M)] juez $N $J"
    uv run python scripts/steering/evaluar_respuestas.py --folder steering_v2/residual/${N}_wide_sel --model $J
  done
done
echo "COLA CAA TERMINADA $(date)"
