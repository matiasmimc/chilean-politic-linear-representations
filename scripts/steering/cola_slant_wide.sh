#!/bin/bash
# Slant con grillas amplias de alpha (residual y atencion), GPU 1: generar -> filtrar colapso -> juez solo en rango util.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
LL=meta-llama/Llama-3.1-8B-Instruct; GE=google/gemma-4-E4B-it
NP="paths.results_dir=results/replication/v3_noparty"
G="uv run python scripts/steering/intervencion_modelo.py +out_tag=_wide +force=false"
AR='+alphas=[-16,-8,-4,-2,-1,0,1,2,4,8,16]'      # ridge / aleatorio
AK='+alphas=[-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8]'    # cPCA con k bajo
AA='+alphas=[-16,-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8,16]'  # atencion
for M in $LL $GE; do
  echo "##### [$(date +%H:%M)] generacion residual $M"
  $G +target=residual +model=$M +steering_method=ridge "$AR"
  $G +target=residual +model=$M +steering_method=ridge_np "$AR" $NP
  for s in 0 1; do $G +target=residual +model=$M +steering_method=random_k16_s$s "$AR"; done
  for k in 1 4; do
    $G +target=residual +model=$M +steering_method=cpca_k$k "$AK"
    $G +target=residual +model=$M +steering_method=cpca_np_k$k "$AK" $NP
  done
  echo "##### [$(date +%H:%M)] generacion atencion $M"
  $G +target=attention +model=$M +steering_method=ridge "$AA"
  for m in cpca cpca_k1 cpca_k4 random_k16_s0 random_k16_s1 random_k16_s2; do
    $G +target=attention +model=$M +steering_method=$m "$AA"
  done
done
echo "##### [$(date +%H:%M)] filtro de colapso"
uv run python scripts/steering/colapso_texto.py results/steering_v2/residual/*_wide results/steering_v2/attention/*_wide
for T in residual attention; do
  for D in results/steering_v2/$T/*_wide_sel; do
    N=$(basename $D)
    for J in Llama gemma; do
      echo "##### [$(date +%H:%M)] juez $T/$N $J"
      uv run python scripts/steering/evaluar_respuestas.py --folder steering_v2/$T/$N --model $J
    done
  done
done
echo "COLA SLANT WIDE TERMINADA $(date)"
