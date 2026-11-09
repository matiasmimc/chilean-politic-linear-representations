"""Medida slant_logit (clasificador del propio modelo, en espanol) sobre los textos de steering_v2 de Chile (ridge/cPCA/aleatorias).

Permite comparar con `slant_us.py` con la misma medida: los resultados de Chile solo tenian el juez. Los textos y alphas ya
existen (results/steering_v2/residual/<metodo>/intervention_raw.parquet); aqui solo se puntuan (GPU, minutos).

Uso: CUDA_VISIBLE_DEVICES=1 uv run python scripts/steering/slant_logit_chile_v2.py
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablacion_direcciones import layer_modules  # noqa: E402
from slant_v3 import Steerer, slant_logits  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ROOT = BASE_DIR / "results/steering_v2/residual"
METHODS = ["ridge", "cpca", "ridge_wide", "cpca_k1_wide", "random_k16_s0", "random_k16_s1", "random_k16_s2",
           "random_k16_s0_wide", "random_k16_s1_wide", "random_k16_s2_wide"]
MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "google/gemma-4-E4B-it", "zai-org/GLM-4-9B-0414"]


def main():
    out_dir = BASE_DIR / "results/slant_chile_logit"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        data = {m: pd.read_parquet(ROOT / m / "intervention_raw.parquet") for m in METHODS if (ROOT / m).exists()}
        data = {m: d[d.model == name] for m, d in data.items() if (d.model == name).any()}
        if not data:
            continue
        tok = AutoTokenizer.from_pretrained(name, cache_dir=str(BASE_DIR / "model"), use_fast=False)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name, cache_dir=str(BASE_DIR / "model"), low_cpu_mem_usage=True,
                                                     torch_dtype=torch.float16, device_map="auto").eval()
        st = Steerer(layer_modules(model))
        rows = []
        for m, d in data.items():
            d = d.copy()
            d["slant_logit"] = slant_logits(model, tok, st, d.text.tolist(), bs=16)
            d["method"] = m
            rows.append(d)
            print(name, m, len(d), flush=True)
        pd.concat(rows).to_parquet(out_dir / f"{name.replace('/', '_')}.parquet")
        del model
        torch.cuda.empty_cache()
    print("CHILE LOGIT TERMINADO", flush=True)


if __name__ == "__main__":
    main()
