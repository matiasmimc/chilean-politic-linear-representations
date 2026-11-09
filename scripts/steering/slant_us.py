"""Political slant con direcciones de congresistas de EE.UU. (ridge y cPCA sobre DW-NOMINATE), GPU.

Replica en ingles el diseno de `intervencion_modelo.py` (Chile): direcciones ridge (top-16 capas por Spearman en CV) y cPCA
(C_T - 2 C_B, fondo = textos neutrales) calculadas sobre las activaciones residuales del prompt solo con NOMBRE de 156 congresistas
(78 D + 78 R; etiqueta DW-NOMINATE dim1), y sumadas al ultimo token de cada paso de generacion con magnitud
alpha * std(proyeccion de los congresistas). 9 temas traducidos, 5 muestras por (tema, alpha), 200 tokens.
Controles: 3 direcciones aleatorias en las mismas capas que ridge (random_s0..2).
Medida: slant_logit = logit(right) - logit(left) del mismo modelo (sin steering) sobre el texto, con prompt en ingles.

Salida: results/slant_us/<modelo>/intervention_raw.parquet (model, method, alpha, topic, sample, text, slant_logit, distinct2).
Uso: CUDA_VISIBLE_DEVICES=1 uv run python scripts/steering/slant_us.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablacion_direcciones import first_id, layer_modules  # noqa: E402
from intervencion_modelo import ridge_components  # noqa: E402
from slant_v3 import Steerer, distinct2, generate  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = BASE_DIR / "results/slant_us"
TOPICS = ['abortion', 'immigration', 'gun rights', 'same-sex marriage', 'LGBTQ+ rights', 'climate change',
          'trust in science', 'social media regulation', 'freedom of speech']
PROMPT_TEXTO = ("Read the following political statement and answer with a single word whether the speaker is on the left or "
                "on the right of the US political spectrum (answer \"left\" or \"right\").\n\nStatement:\n{texto}")


def quick_ridge(X, y, lam_rel=0.3):
    Xc, yc = X - X.mean(0), y - y.mean()
    G = Xc @ Xc.T
    lam = lam_rel * np.trace(G) / X.shape[1]
    w = Xc.T @ np.linalg.solve(G + lam * np.eye(len(G)), yc)
    return w / np.linalg.norm(w)


def cpca_components(T, B, labels, k, contrast=2.0, per_layer=8):
    """Top-k autovectores de C_T - contrast*C_B entre todas las capas (por autovalor), signo alineado con ridge sobre labels."""
    comps = []
    for l in range(T.shape[1]):
        XT, XB = np.nan_to_num(T[:, l, :]).astype(np.float64), np.nan_to_num(B[:, l, :]).astype(np.float64)
        C = np.cov(XT - XT.mean(0), rowvar=False) - contrast * np.cov(XB - XB.mean(0), rowvar=False)
        d = C.shape[0]
        vals, vecs = eigh(C, subset_by_index=[d - per_layer, d - 1])
        w = quick_ridge(XT, labels)
        for v, e in zip(vecs.T[::-1], vals[::-1]):
            if e > 0:
                comps.append((float(e), l, (v if v @ w >= 0 else -v).astype(np.float32)))
    comps.sort(key=lambda c: -c[0])
    by_layer = {}
    for _, l, v in comps[:k]:
        by_layer.setdefault(l, []).append(v / np.linalg.norm(v))
    return by_layer


def to_shifts(by_layer, feats, alpha):
    out = {}
    for l, vecs in by_layer.items():
        s = [float(np.std(feats[:, l, :] @ v)) for v in vecs]
        out[l] = sum(alpha * si * v for si, v in zip(s, vecs))
    return out


@torch.no_grad()
def slant_logits(model, tok, st, texts, bs=16):
    st.set({})
    ids = [first_id(tok, "left"), first_id(tok, "right"), first_id(tok, "Left"), first_id(tok, "Right")]
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT_TEXTO.format(texto=t[:1500])}], tokenize=False,
                                       add_generation_prompt=True) for t in texts]
    order = np.argsort([len(p) for p in prompts])
    s = np.zeros(len(prompts), dtype=np.float32)
    for i in range(0, len(prompts), bs):
        idx = order[i:i + bs]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
        lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), -1)[:, ids].cpu().numpy()
        s[idx] = np.logaddexp(lp[:, 1], lp[:, 3]) - np.logaddexp(lp[:, 0], lp[:, 2])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-rand", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    safe = args.model.replace("/", "_")
    out_dir = OUT_ROOT / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    rdir = BASE_DIR / "results/us_analysis/residual"
    feats, labels = pickle.load(open(rdir / f"{safe}_politician_features.pkl", "rb"))
    feats = np.nan_to_num(feats.astype(np.float32))
    neutral = pickle.load(open(rdir / f"{safe}_neutral_features.pkl", "rb"))[0].astype(np.float32)

    dirs = {}
    dirs["ridge"], _ = ridge_components(feats, labels, 16)
    dirs["cpca_k1"] = cpca_components(feats, neutral, labels, 1)
    dirs["cpca_k16"] = cpca_components(feats, neutral, labels, 16)
    for sd in range(args.n_rand):
        rng = np.random.RandomState(sd)
        dirs[f"random_s{sd}"] = {l: [(lambda v: v / np.linalg.norm(v))(rng.randn(feats.shape[2]).astype(np.float32))
                                     for _ in vs] for l, vs in dirs["ridge"].items()}
    print({m: {l: len(v) for l, v in d.items()} for m, d in dirs.items()}, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=str(BASE_DIR / "model"), use_fast=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=str(BASE_DIR / "model"), low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float16, device_map="auto").eval()
    layers = layer_modules(model)
    st = Steerer(layers)
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"Write a statement about {t}."}], tokenize=False,
                                       add_generation_prompt=True) + f"Regarding {t}, I believe " for t in TOPICS]
    alphas = [0.0, 4.0] if args.smoke else args.alphas
    rows = []
    for method, by_layer in dirs.items():
        for alpha in tqdm(alphas, desc=method):
            if alpha == 0 and method != "ridge":
                continue  # alpha = 0 no depende de la direccion; se genera una vez como base (en ridge)
            st.set({l: v for l, v in to_shifts(by_layer, feats, alpha).items()})
            for s in range(1 if args.smoke else args.n_samples):
                st.coef = torch.ones(len(prompts))
                texts = generate(model, tok, st, prompts, np.ones(len(prompts), np.float32), args.max_new_tokens,
                                 bs=len(prompts), seed=s)
                for t, tx in zip(TOPICS, texts):
                    rows.append(dict(model=args.model, method=method if alpha != 0 else "base", alpha=alpha, topic=t, sample=s,
                                     text=tx, distinct2=distinct2(tx)))
        df = pd.DataFrame(rows)
        df["slant_logit"] = slant_logits(model, tok, st, df.text.tolist())
        df.to_parquet(out_dir / ("smoke.parquet" if args.smoke else "intervention_raw.parquet"))
        print(f"{method} listo: {len(df)} textos", flush=True)
    print("SLANT_US TERMINADO", flush=True)


if __name__ == "__main__":
    main()
