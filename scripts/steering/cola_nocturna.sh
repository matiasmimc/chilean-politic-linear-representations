#!/bin/bash
# Cola nocturna (GPU 1, secuencial; un fallo no detiene la cola). Uso: bash scripts/steering/cola_nocturna.sh
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
LL=meta-llama/Llama-3.1-8B-Instruct; GE=google/gemma-4-E4B-it; GL=zai-org/GLM-4-9B-0414
S="uv run python scripts/steering/predecir_votos_steered.py +target=residual"
D="+task_mode=divided +min_frac_contra=0.25"
step() { echo; echo "##### [$(date +%H:%M)] $*"; }

step "A Llama all cpca";            $S +model=$LL +steering_method=cpca
step "B Llama divided cpca";        $S +model=$LL +steering_method=cpca $D
for s in 0 1 2; do step "C Llama divided random s$s"; $S +model=$LL +steering_method=random_k16_s$s $D; done
step "C Llama all random s0";       $S +model=$LL +steering_method=random_k16_s0
step "D causal Llama";              uv run python scripts/interpretabilidad/validacion_causal_logit_lens.py +model=$LL
step "D causal Gemma";              uv run python scripts/interpretabilidad/validacion_causal_logit_lens.py +model=$GE
step "E Gemma divided cpca";        $S +model=$GE +steering_method=cpca $D
step "E GLM divided cpca";          $S +model=$GL +steering_method=cpca $D
for k in 1 4 64; do step "F Llama divided cpca_k$k"; $S +model=$LL +steering_method=cpca_k$k $D; done
step "G Gemma divided random s0";   $S +model=$GE +steering_method=random_k16_s0 $D
step "G GLM divided random s0";     $S +model=$GL +steering_method=random_k16_s0 $D
step "RESUMEN";                     uv run python scripts/steering/resumen_resultados.py
echo "COLA TERMINADA $(date)"
