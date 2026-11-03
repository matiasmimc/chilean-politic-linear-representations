"""Activaciones medias (residual, todas las capas) de las intervenciones parlamentarias reales, por diputado.

Para cada intervencion se arma: [chat template: "Escribe una declaracion parlamentaria."] + texto como respuesta del
asistente; se promedia el residual (hidden_states de cada capa) sobre los tokens del texto. Luego se promedia por
diputado -> [N, L, d]. Sirve para construir direcciones de contraste (izquierda vs derecha) tipo CAA.

Uso: CUDA_VISIBLE_DEVICES=1 uv run python scripts/extracion_activaciones/extraer_intervenciones_caa.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_CACHE = BASE_DIR / "model"
OUT_DIR = BASE_DIR / "results/caa"
INSTRUCCION = "Escribe una declaración parlamentaria."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=350, help="tokens maximos por intervencion")
    ap.add_argument("--per-statement", action="store_true",
                    help="guardar la activacion media de CADA intervencion (para controlar por tema), en float16")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    safe = args.model.replace("/", "_")
    out = OUT_DIR / f"{safe}_{'statement_acts' if args.per_statement else 'mp_statement_means'}.pkl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        print(f"Ya existe {out}")
        return

    df = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=str(MODEL_CACHE))
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=str(MODEL_CACHE), low_cpu_mem_usage=True,
                                                 torch_dtype=torch.bfloat16, device_map="auto").eval()
    prefix = tok.apply_chat_template([{"role": "user", "content": INSTRUCCION}], tokenize=False, add_generation_prompt=True)
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]

    means, counts = [], []
    stmt_acts, stmt_mp, stmt_k = [], [], []
    for mp_i, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc="Diputados"):
        acc, n = None, 0
        for text in row["intervenciones"]:
            ids = tok(str(text), add_special_tokens=False)["input_ids"][:args.max_tokens]
            if len(ids) < 20:
                continue
            inp = torch.tensor([prefix_ids + ids], device=model.device)
            with torch.no_grad():
                hs = model(inp, output_hidden_states=True).hidden_states[1:]
            h = torch.stack([x[0, len(prefix_ids):].float().mean(0) for x in hs]).cpu().numpy()  # [L, d]
            acc = h if acc is None else acc + h
            if args.per_statement:
                stmt_acts.append(h.astype(np.float16)); stmt_mp.append(mp_i); stmt_k.append(n)
            n += 1
        means.append(acc / max(n, 1) if acc is not None else None)
        counts.append(n)
    if args.per_statement:
        pickle.dump({"acts": np.stack(stmt_acts), "mp_idx": np.array(stmt_mp), "stmt_k": np.array(stmt_k),
                     "p6": df["dim_ideology_P6"].astype(float).values}, open(out, "wb"))
        print(f"✅ {out} {np.stack(stmt_acts).shape}")
        return
    ok = [m is not None for m in means]
    shape = next(m for m in means if m is not None).shape
    means = np.stack([m if m is not None else np.full(shape, np.nan, dtype=np.float32) for m in means])
    pickle.dump({"means": means.astype(np.float32), "counts": np.array(counts),
                 "p6": df["dim_ideology_P6"].astype(float).values, "party": df["party_1_P6"].values,
                 "names": df["Name"].values}, open(out, "wb"))
    print(f"✅ {out} {means.shape}, {sum(ok)} diputados con intervenciones")


if __name__ == "__main__":
    main()
