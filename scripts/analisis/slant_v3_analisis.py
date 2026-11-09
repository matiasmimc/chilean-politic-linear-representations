"""Analisis de slant v3 (CPU): dosis-respuesta, control positivo y version personal.

Medidas: `slant_logit` (siempre) y `score_postura` (si el juez ya evaluo; filtro de coherencia >= 4).
Dosis: pendiente OLS de la medida sobre alpha (efectos fijos por tema) con IC bootstrap por tema; las 3 direcciones aleatorias
dan la distribucion de referencia. Personal: pendiente de la medida sobre z_within (efectos fijos por tema) con IC bootstrap por
diputado; Spearman parcial.

Uso: uv run python scripts/analisis/slant_v3_analisis.py [--medida slant_logit|score_postura]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent.parent / "results/slant_v3"
MODEL = "Llama"


def load(exp, medida):
    model = MODEL
    ev = BASE / exp / f"intervention_evaluated_{model}.parquet"
    d = pd.read_parquet(ev if ev.exists() and medida == "score_postura" else BASE / exp / "intervention_raw.parquet")
    if medida == "score_postura":
        d = d[pd.to_numeric(d.score_coherence, errors="coerce") >= 4]
        d["y"] = pd.to_numeric(d.score_postura, errors="coerce")
    else:
        d["y"] = d.slant_logit
    return d.dropna(subset=["y"]).copy()


def slope(x, y, g):
    xd = x - pd.Series(x).groupby(g).transform("mean").values
    yd = y - pd.Series(y).groupby(g).transform("mean").values
    return float((xd * yd).sum() / ((xd ** 2).sum() + 1e-12))


def boot(x, y, g, units, n=500, seed=0):
    rng = np.random.RandomState(seed)
    u = np.unique(units)
    by = {k: np.where(units == k)[0] for k in u}
    out = []
    for _ in range(n):
        idx = np.concatenate([by[k] for k in rng.choice(u, len(u))])
        out.append(slope(x[idx], y[idx], g[idx]))
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medida", default="slant_logit", choices=["slant_logit", "score_postura"])
    ap.add_argument("--root", default="slant_v3")
    ap.add_argument("--model", default="Llama", help="sufijo del archivo evaluado (Llama, gemma)")
    a = ap.parse_args()
    global BASE, MODEL
    BASE, MODEL = BASE.parent / a.root, a.model
    pd.set_option("display.width", 220)
    print(f"=== medida: {a.medida} ===")

    d = load("dosis", a.medida)
    base = d[d.method == "base"]
    print(f"\n-- DOSIS (n={len(d)}; media por alpha) --")
    print(d.assign(m=d.method.where(d.method == "base", d.method)).groupby(["method", "alpha"]).y.mean().unstack().round(2))
    print("distinct2 medio por metodo/alpha (colapso si <0.5):")
    print(d.groupby(["method", "alpha"]).distinct2.mean().unstack().round(2).to_string())
    rows = []
    for m in [x for x in d.method.unique() if x != "base"]:
        s = pd.concat([d[d.method == m], base])
        x, y, g, u = s.alpha.values.astype(float), s.y.values, s.topic.values, s.topic.values
        lo, hi = boot(x, y, g, u)
        rows.append(dict(metodo=m, n=len(s), pendiente_por_alpha=slope(x, y, g), ic_lo=lo, ic_hi=hi,
                         rho=spearmanr(x - pd.Series(x).groupby(g).transform("mean").values,
                                       y - pd.Series(y).groupby(g).transform("mean").values).statistic))
    r = pd.DataFrame(rows)
    print(r.round(3).to_string(index=False))

    p = load("positivo", a.medida)
    print("\n-- CONTROL POSITIVO (prompt de izquierda / derecha, sin steering) --")
    print(p.groupby("method").y.agg(["mean", "std", "count"]).round(2))
    if p.method.nunique() == 2:
        print("AUC derecha vs izquierda:", round(roc_auc_score(p.method.str.contains("derecha"), p.y), 3))

    for exp in ("personal_sub", "personal"):
        if not (BASE / exp / "intervention_raw.parquet").exists() or (a.medida == "score_postura" and exp == "personal"):
            continue
        q = load(exp, a.medida)
        print(f"\n-- PERSONAL [{exp}] (pendiente de la medida sobre z dentro de partido; n por metodo ~{len(q)//4}) --")
        rows = []
        for m, s in q.groupby("method"):
            x, y, g, u = s.z_within.values, s.y.values, s.topic.values, s.mp_idx.values
            lo, hi = boot(x, y, g, u)
            yd = y - pd.Series(y).groupby(g).transform("mean").values
            rows.append(dict(metodo=m, n=len(s), pendiente=slope(x, y, g), ic_lo=lo, ic_hi=hi,
                             spearman=spearmanr(x, yd).statistic))
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
