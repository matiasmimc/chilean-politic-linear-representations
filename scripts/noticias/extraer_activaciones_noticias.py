"""Activaciones medias (residual, todas las capas) de articulos de noticias, en el MISMO espacio que las declaraciones.

Igual que extraer_intervenciones_caa.py --per-statement: [chat template: "Escribe una declaracion parlamentaria."] + texto como
respuesta del asistente; se promedia el residual de cada capa sobre los tokens del texto (primeros --max-tokens). Asi las
direcciones aprendidas con declaraciones (ridge, CAA, cPCA) se aplican a las noticias sin cambio de formato.
Con --neutral se extraen los 100 textos neutrales (data/intervenciones_neutrales.parquet) como fondo del cPCA.

El corpus debe tener una columna de texto y una de casa de noticias (csv, parquet o jsonl).
Uso: CUDA_VISIBLE_DEVICES=1 uv run python scripts/noticias/extraer_activaciones_noticias.py --model meta-llama/Llama-3.1-8B-Instruct \
       --corpus data/noticias/corpus.parquet --col-texto texto --col-casa casa [--max-por-casa 400]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "results/noticias"
INSTRUCCION = "Escribe una declaración parlamentaria."


def load_corpus(path):
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".jsonl", ".json"):
        return pd.read_json(p, lines=p.suffix == ".jsonl")
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--col-texto", default="texto")
    ap.add_argument("--col-casa", default="casa")
    ap.add_argument("--max-por-casa", type=int, default=400, help="articulos por casa (muestra aleatoria con semilla 0); 0 = todos")
    ap.add_argument("--max-tokens", type=int, default=350)
    ap.add_argument("--neutral", action="store_true", help="extraer los textos neutrales (fondo del cPCA) en vez del corpus")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    safe = args.model.replace("/", "_")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{safe}_{'neutral' if args.neutral else 'noticias'}_acts.pkl"
    if out.exists() and not args.force:
        print(f"Ya existe {out}")
        return

    if args.neutral:
        df = pd.read_parquet(BASE_DIR / "data/intervenciones_neutrales.parquet").rename(columns={"intervenciones": "texto"})
        df["casa"] = "neutral"
    else:
        df = load_corpus(args.corpus).rename(columns={args.col_texto: "texto", args.col_casa: "casa"})
        df = df.dropna(subset=["texto", "casa"]).drop_duplicates("texto")
        if args.max_por_casa:
            df = df.groupby("casa", group_keys=False).apply(lambda g: g.sample(min(len(g), args.max_por_casa), random_state=0))
        df = df.reset_index(drop=True)
        print(df.casa.value_counts().to_string(), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path or args.model, cache_dir=str(BASE_DIR / "model"))
    kw = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "torch_dtype": torch.float16} if args.load_8bit \
        else {"torch_dtype": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(args.model_path or args.model, cache_dir=str(BASE_DIR / "model"),
                                                 low_cpu_mem_usage=True, device_map="auto", **kw).eval()
    prefix = tok.apply_chat_template([{"role": "user", "content": INSTRUCCION}], tokenize=False, add_generation_prompt=True)
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]

    acts, keep = [], []
    for i, text in tqdm(enumerate(df.texto.astype(str)), total=len(df), desc="Textos"):
        ids = tok(text, add_special_tokens=False)["input_ids"][:args.max_tokens]
        if len(ids) < 20:
            continue
        inp = torch.tensor([prefix_ids + ids], device=model.device)
        with torch.no_grad():
            hs = model(inp, output_hidden_states=True).hidden_states[1:]
        acts.append(torch.stack([x[0, len(prefix_ids):].float().mean(0) for x in hs]).cpu().numpy().astype(np.float16))
        keep.append(i)
    meta = df.iloc[keep][[c for c in ("id", "casa", "tema", "fecha", "texto") if c in df.columns]].reset_index(drop=True)
    pickle.dump({"acts": np.stack(acts), "meta": meta}, open(out, "wb"))
    print(f"✅ {out} {np.stack(acts).shape}")


if __name__ == "__main__":
    main()
