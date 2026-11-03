#!/bin/bash
# Cola de fin de semana (GPU 1, secuencial, idempotente: salta lo ya terminado). Uso: bash scripts/steering/cola_fin_de_semana.sh
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
LL=meta-llama/Llama-3.1-8B-Instruct; GE=google/gemma-4-E4B-it; GL=zai-org/GLM-4-9B-0414
ST="uv run python scripts/steering/predecir_votos_steered.py"
D="+task_mode=divided +min_frac_contra=0.25"
done_() { # modo(residual|atencion) modelo metodo
  uv run python - "$@" <<'P'
import json,sys
mode,model,method=sys.argv[1:]
p="results/voting_predictions/residual/divided/optimal_alphas.json" if mode=="residual" else "results/voting_predictions/divided/optimal_alphas.json"
try: d=json.load(open(p)); sys.exit(0 if method in d.get(model.replace('/','_'),{}) else 1)
except Exception: sys.exit(1)
P
}
run() { # modo modelo metodo
  done_ $1 $2 $3 && { echo "## salto $1 $2 $3 (ya hecho)"; return; }
  echo; echo "##### [$(date +%H:%M)] $1 $2 $3"
  if [ "$1" = residual ]; then $ST +target=residual +model=$2 +steering_method=$3 $D; else $ST +model=$2 +steering_method=$3 $D; fi
}

# 1) barrido de contrast_alpha (Llama, residual)
for ca in 0.5 1 5; do
  echo "##### [$(date +%H:%M)] cPCA residual Llama contrast_alpha=$ca"
  [ -f results/cpca_analysis/residual/meta-llama_Llama-3.1-8B-Instruct_cpca_analysis_global_ca$ca.pkl ] || \
    uv run python scripts/no-supervisado/analisis_cpca.py +target=residual +model=$LL params.contrast_alpha=$ca +cpca_tag=_ca$ca
  run residual $LL cpca_ca$ca
done
# 2) GLM atencion: extraer features y correr
echo "##### [$(date +%H:%M)] extraer atencion GLM"
uv run python scripts/extracion_activaciones/extraer_activaciones.py --model $GL
run atencion $GL cpca; run atencion $GL random_k16_s0
# 3) mas semillas de control, residual
for s in 3 4; do run residual $LL random_k16_s$s; done
for s in 1 2; do run residual $GE random_k16_s$s; run residual $GL random_k16_s$s; done
# 4) mas semillas de control, atencion
for s in 1 2; do run atencion $GE random_k16_s$s; done
echo "COLA TERMINADA $(date)"
uv run python scripts/steering/resumen_resultados.py
uv run python scripts/steering/bootstrap_controles.py divided
uv run python scripts/steering/resumen_resultados.py atencion
uv run python scripts/steering/bootstrap_controles.py divided atencion
