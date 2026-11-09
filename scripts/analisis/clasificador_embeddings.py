"""Igual que clasificador_texto_real.py pero con embeddings de oraciones (multilingual-e5-large) en vez de TF-IDF (GPU 1, ~minutos).

Descarta que el efecto medido sea solo vocabulario. Ridge sobre el embedding (normalizado) de cada declaracion real -> P6;
validacion por diputado; puntua los textos generados y guarda intervention_evaluated_<modelo>__clfemb.parquet.

Uso: CUDA_VISIBLE_DEVICES=1 uv run python scripts/analisis/clasificador_embeddings.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoTokenizer

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/analisis"))
from clasificador_texto_real import EXPS, ROOTS, clean  # noqa: E402

ENC = "intfloat/multilingual-e5-large"


@torch.no_grad()
def embed(tok, model, texts, bs=64, max_len=256):
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), np.float32)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok(["query: " + texts[j] for j in idx], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(model.device)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1)
        out[idx] = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
    return out


def main():
    tok = AutoTokenizer.from_pretrained(ENC, cache_dir=str(BASE / "model"))
    model = AutoModel.from_pretrained(ENC, cache_dir=str(BASE / "model"), torch_dtype=torch.float16).cuda().eval()
    mp = pd.read_parquet(BASE / "data/congreso_2022_2026_20intervenciones.parquet")
    rows = [(i, clean(t), float(mp.dim_ideology_P6.iloc[i])) for i in range(len(mp)) for t in mp.intervenciones.iloc[i]
            if len(str(t)) > 100]
    idx, txt, y = (np.array(c) for c in zip(*rows))
    y = y.astype(float)
    ok = ~np.isnan(y)
    idx, txt, y = idx[ok], txt[ok], y[ok]
    X = embed(tok, model, list(txt))
    print(f"{len(txt)} declaraciones reales, emb {X.shape}", flush=True)

    best = None
    for alpha in (1.0, 3.0, 10.0, 30.0):
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, idx):
            pred[te] = Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[te])
        lo, hi = np.percentile(y, [33, 67])
        ext = (y <= lo) | (y >= hi)
        mpp, mpy = pd.Series(pred).groupby(idx).mean(), pd.Series(y).groupby(idx).mean()
        party = mp.party_1_P6.fillna("IND").values[idx]
        dp = pred - pd.Series(pred).groupby(party).transform("mean").values
        dy = y - pd.Series(y).groupby(party).transform("mean").values
        r = (spearmanr(pred, y)[0], roc_auc_score(y[ext] >= hi, pred[ext]), spearmanr(mpp, mpy)[0], spearmanr(dp, dy)[0])
        print(f"alpha={alpha}: Spearman decl {r[0]:.3f}; AUC terciles {r[1]:.3f}; por diputado {r[2]:.3f}; dentro de partido "
              f"{r[3]:.3f}", flush=True)
        if best is None or r[1] > best[0]:
            best = (r[1], alpha)
    print("alpha elegido (mejor AUC en CV):", best[1], flush=True)
    m = Ridge(alpha=best[1]).fit(X, y)

    for root, name in ROOTS.items():
        for exp in EXPS:
            f = BASE / "results" / root / exp / "intervention_raw.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(f)
            s = m.predict(embed(tok, model, [clean(t) for t in d.text]))
            d["score_postura"] = (s - s.mean()) / s.std()
            d["score_coherence"] = 5
            d.to_parquet(f.parent / f"intervention_evaluated_{name}__clfemb.parquet")
            print(f"{root}/{exp}: {len(d)} textos", flush=True)
    print("CLFEMB TERMINADO", flush=True)


if __name__ == "__main__":
    main()
