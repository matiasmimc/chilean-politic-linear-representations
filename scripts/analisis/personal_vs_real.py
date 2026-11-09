"""¿El steering personal produce textos que se parecen a las declaraciones reales de ESE diputado? (GPU 1, minutos)

Para cada texto generado (diputado a, partido P, tema, metodo) se calcula la similitud coseno (embeddings e5) con el centroide de
las declaraciones reales de cada diputado b de su partido. Dos medidas, por metodo y modelo:
  ventaja_propia : sim(texto_a, centroide_a) - media de sim(texto_a, centroide_b) sobre los otros b de P (identidad)
  proximidad     : correlacion de Spearman, sobre b en P, entre sim(texto_a, centroide_b) y -|z_a - z_b| (z = P6 dentro de partido)
                   (¿se parece mas a diputados ideologicamente cercanos?)
Promedio por diputado (8 temas), IC95% por bootstrap de diputados; nulo de la ventaja_propia permutando las etiquetas de los
centroides dentro de partido. Solo partidos con >= 3 diputados.

Uso: CUDA_VISIBLE_DEVICES=1 uv run python scripts/analisis/personal_vs_real.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/analisis"))
from clasificador_embeddings import ENC, embed  # noqa: E402
from clasificador_texto_real import ROOTS, clean  # noqa: E402

OUT = BASE / "results/personal_vs_real"


def stats(E_txt, mpi, centroids, party, zw, valid_mp):
    """Por texto: ventaja propia y proximidad."""
    adv, prox = np.full(len(E_txt), np.nan), np.full(len(E_txt), np.nan)
    for p in np.unique(party[valid_mp]):
        mems = np.array([m for m in valid_mp if party[m] == p])
        if len(mems) < 3:
            continue
        C = centroids[mems]
        for i in np.where(np.isin(mpi, mems))[0]:
            s = C @ E_txt[i]
            a = np.where(mems == mpi[i])[0][0]
            adv[i] = s[a] - np.delete(s, a).mean()
            o = np.delete(np.arange(len(mems)), a)
            prox[i] = spearmanr(s[o], -np.abs(zw[mems[o]] - zw[mpi[i]]))[0]
    return adv, prox


def perm_null(E_txt, mpi, centroids, party, zw, valid_mp, n=200, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        C2 = centroids.copy()
        for p in np.unique(party[valid_mp]):
            mems = np.array([m for m in valid_mp if party[m] == p])
            C2[mems] = centroids[rng.permutation(mems)]
        a, _ = stats(E_txt, mpi, C2, party, zw, valid_mp)
        out.append(np.nanmean(a))
    return np.array(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ENC, cache_dir=str(BASE / "model"))
    model = AutoModel.from_pretrained(ENC, cache_dir=str(BASE / "model"), torch_dtype=torch.float16).cuda().eval()
    mp = pd.read_parquet(BASE / "data/congreso_2022_2026_20intervenciones.parquet")
    party = mp.party_1_P6.fillna("IND").values
    p6 = mp.dim_ideology_P6.astype(float).values
    zw = p6 - pd.Series(p6).groupby(party).transform("mean").values
    rows = [(i, clean(t)) for i in range(len(mp)) for t in mp.intervenciones.iloc[i] if len(str(t)) > 100]
    ridx, rtxt = np.array([r[0] for r in rows]), [r[1] for r in rows]
    R = embed(tok, model, rtxt)
    centroids = np.zeros((len(mp), R.shape[1]), np.float32)
    for m in np.unique(ridx):
        c = R[ridx == m].mean(0)
        centroids[m] = c / np.linalg.norm(c)
    valid_mp = np.array([m for m in np.unique(ridx) if not np.isnan(p6[m])])
    res = []
    for root, name in ROOTS.items():
        f = BASE / "results" / root / "personal" / "intervention_raw.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        E = embed(tok, model, [clean(t) for t in d.text])
        mpi = d.mp_idx.values
        adv, prox = stats(E, mpi, centroids, party, zw, valid_mp)
        d["adv"], d["prox"] = adv, prox
        for meth, g in d.dropna(subset=["adv"]).groupby("method"):
            per = g.groupby("mp_idx")[["adv", "prox"]].mean()
            rng = np.random.RandomState(0)
            bs = [per.iloc[rng.randint(0, len(per), len(per))].mean().values for _ in range(1000)]
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            gg = d[d.method == meth]
            null = perm_null(E[(d.method == meth).values], gg.mp_idx.values, centroids, party, zw, valid_mp, n=100)
            res.append(dict(modelo=name, metodo=meth, n_mp=len(per), ventaja_propia=per.adv.mean(), adv_lo=lo[0], adv_hi=hi[0],
                            nulo_p95=float(np.percentile(null, 95)), proximidad=per.prox.mean(), prox_lo=lo[1], prox_hi=hi[1]))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[-1].items()}, flush=True)
    pd.DataFrame(res).to_csv(OUT / "resumen.csv", index=False)
    print("PERSONAL_VS_REAL TERMINADO", flush=True)


if __name__ == "__main__":
    main()
