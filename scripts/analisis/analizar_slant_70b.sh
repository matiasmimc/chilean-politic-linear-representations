#!/bin/bash
# Analisis de slant para los 70B (Llama-3.1-70B-Instruct y LatamGPT) una vez copiados results/slant_v3_l70b_{instruct,latam}.
# Corre los dos clasificadores entrenados en texto real (TF-IDF, embeddings e5) y la dosis-respuesta con logit y ambos clasificadores.
# Uso: CUDA_VISIBLE_DEVICES=1 bash scripts/analisis/analizar_slant_70b.sh   (el juez LLM se corre aparte con juez_rapido.py)
set -e
cd "$(dirname "$0")/../.."
for R in slant_v3_l70b_instruct slant_v3_l70b_latam; do
  [ -f results/$R/dosis/intervention_raw.parquet ] || { echo "falta results/$R"; exit 1; }
done
uv run python scripts/analisis/clasificador_texto_real.py | grep -E "CV|dentro|l70b"
uv run python scripts/analisis/clasificador_embeddings.py | grep -E "alpha elegido|l70b|TERMINADO"
for R in slant_v3_l70b_instruct:l70b_instruct slant_v3_l70b_latam:l70b_latam; do
  root=${R%%:*}; m=${R##*:}
  echo "########## $root: logit"
  uv run python scripts/analisis/slant_v3_analisis.py --medida slant_logit --root $root --model $m | sed -n '/metodo/,$p'
  for suf in clf clfemb; do
    echo "########## $root: $suf"
    uv run python scripts/analisis/slant_v3_analisis.py --medida score_postura --root $root --model ${m}__$suf | sed -n '/metodo/,$p'
  done
done 2>&1 | tee logs/analisis_slant_70b.log
