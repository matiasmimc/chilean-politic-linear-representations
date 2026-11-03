#!/bin/bash
# cPCA y steering con activaciones SIN partido en el prompt (GPU 1, secuencial, idempotente).
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
LL=meta-llama/Llama-3.1-8B-Instruct; GE=google/gemma-4-E4B-it; GL=zai-org/GLM-4-9B-0414
NP="paths.results_dir=results/replication/v3_noparty"
ST="uv run python scripts/steering/predecir_votos_steered.py +target=residual +task_mode=divided +min_frac_contra=0.25 $NP"
done_() {
  uv run python - "$1" "$2" <<'P'
import json,sys
model,method=sys.argv[1:]
try: d=json.load(open("results/voting_predictions/residual/divided/optimal_alphas.json")); sys.exit(0 if method in d.get(model.replace('/','_'),{}) else 1)
except Exception: sys.exit(1)
P
}
run() { done_ $1 $2 && { echo "## salto $1 $2"; return; }; echo; echo "##### [$(date +%H:%M)] $1 $2"; $ST +model=$1 +steering_method=$2; }

for M in $LL $GE $GL; do
  S=$(echo $M | tr / _)
  [ -f results/replication/v3_noparty/residual/${S}_neutral_features.pkl ] || { echo "##### [$(date +%H:%M)] neutral sin partido $M"; uv run python scripts/extracion_activaciones/extraer_activaciones_neutrales.py --model $M --target residual --no-party; }
  [ -f results/cpca_analysis/residual/${S}_cpca_analysis_global_noparty.pkl ] || { echo "##### [$(date +%H:%M)] cPCA sin partido $M"; uv run python scripts/no-supervisado/analisis_cpca.py +target=residual +model=$M $NP +cpca_tag=_noparty; }
done

run $LL cpca_np
for s in 0 1 2; do run $LL random_np_k16_s$s; done
run $GE cpca_np; run $GE random_np_k16_s0
run $GL cpca_np; run $GL random_np_k16_s0
echo "COLA TERMINADA $(date)"
uv run python scripts/steering/resumen_resultados.py
uv run python scripts/steering/bootstrap_controles.py divided
