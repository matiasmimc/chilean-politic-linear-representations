"""Medida independiente de postura para los textos generados: clasificador entrenado en declaraciones REALES (CPU).

Entrena un ridge sobre TF-IDF (palabras 1-2gramas) de las 20 intervenciones de cada diputado para predecir su P6, sin menciones
a partidos/gobierno (mismo filtro POL que probes_control_tema). Validacion: GroupKFold por diputado (Spearman a nivel de
declaracion y AUC derecha vs izquierda en terciles extremos). Luego puntua todos los textos generados de slant_v3*, y guarda
intervention_evaluated_<modelo>__clf.parquet (score_postura = puntaje estandarizado, score_coherence = 5) para que
slant_v3_analisis.py lo lea con --medida score_postura --model <modelo>__clf.

Uso: uv run python scripts/analisis/clasificador_texto_real.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/probing"))
from probes_control_tema import POL  # noqa: E402

ROOTS = {"slant_v3": "Llama", "slant_v3_gemma": "gemma", "slant_v3_glm": "glm", "slant_v3_l70b_instruct": "l70b_instruct",
         "slant_v3_l70b_latam": "l70b_latam",
         **{f"slant_v3c_{t}_k{k}": m for t, m in (("ll", "Llama"), ("ge", "gemma"), ("gl", "glm")) for k in (1, 4, 8)},
         **{f"slant_v3c_{t}_k{k}_eq": m for t, m in (("ll", "Llama"), ("ge", "gemma"), ("gl", "glm")) for k in (1, 2, 8)}}
EXPS = ["dosis", "positivo", "personal", "personal_sub"]


def clean(t):
    return POL.sub(" ", str(t))


def fit(texts, y):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True, strip_accents="unicode")
    X = vec.fit_transform(texts)
    return vec, Ridge(alpha=3.0).fit(X, y)


def main():
    mp = pd.read_parquet(BASE / "data/congreso_2022_2026_20intervenciones.parquet")
    rows = [(i, clean(t), float(mp.dim_ideology_P6.iloc[i])) for i in range(len(mp)) for t in mp.intervenciones.iloc[i]
            if len(str(t)) > 100]
    idx, txt, y = (np.array(c) for c in zip(*rows))
    y = y.astype(float)
    ok = ~np.isnan(y)
    idx, txt, y = idx[ok], txt[ok], y[ok]
    print(f"{len(txt)} declaraciones reales de {len(np.unique(idx))} diputados", flush=True)

    # validacion cruzada por diputado
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(txt, y, idx):
        vec, m = fit(txt[tr], y[tr])
        pred[te] = m.predict(vec.transform(txt[te]))
    lo, hi = np.percentile(y, [33, 67])
    ext = (y <= lo) | (y >= hi)
    mp_pred = pd.Series(pred).groupby(idx).mean()
    mp_y = pd.Series(y).groupby(idx).mean()
    print(f"CV por diputado: Spearman declaracion = {spearmanr(pred, y)[0]:.3f}; AUC terciles extremos = "
          f"{roc_auc_score(y[ext] >= hi, pred[ext]):.3f}; Spearman por diputado (media de 20) = "
          f"{spearmanr(mp_pred, mp_y)[0]:.3f}", flush=True)
    party = mp.party_1_P6.fillna("IND").values
    dentro = pd.Series(pred - pd.Series(pred).groupby(party[idx]).transform("mean").values)
    ydentro = pd.Series(y - pd.Series(y).groupby(party[idx]).transform("mean").values)
    print(f"  dentro de partido (declaracion): Spearman = {spearmanr(dentro, ydentro)[0]:.3f}", flush=True)

    vec, m = fit(txt, y)
    for root, name in ROOTS.items():
        for exp in EXPS:
            f = BASE / "results" / root / exp / "intervention_raw.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(f)
            s = m.predict(vec.transform([clean(t) for t in d.text]))
            d["score_postura"] = (s - s.mean()) / s.std()
            d["score_coherence"] = 5
            out = f.parent / f"intervention_evaluated_{name}__clf.parquet"
            d.to_parquet(out)
            print(f"{root}/{exp}: {len(d)} textos -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
