"""Metrica automatica de colapso de texto para elegir el rango de alpha antes de usar el juez.

distinct-2 = bigramas de palabras unicos / bigramas totales. Un (modelo, alpha) se considera NO colapsado si su
distinct-2 medio >= UMBRAL * distinct-2 medio en alpha=0 (mismo modelo y metodo) y los textos tienen >= 5 palabras.
Escribe, por carpeta, `<carpeta>_sel/intervention_raw.parquet` solo con los alphas no colapsados (+ alpha=0)
y `colapso_por_alpha.csv`.

Uso: uv run python scripts/steering/colapso_texto.py results/steering_v2/residual/ridge_wide [...]
"""
import sys
from pathlib import Path

import pandas as pd

UMBRAL = 0.7


def distinct2(text):
    w = str(text).lower().split()
    if len(w) < 5:
        return 0.0
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / len(bg)


for folder in map(Path, sys.argv[1:]):
    raw = folder / "intervention_raw.parquet"
    if not raw.exists():
        print(f"[salto] {raw}")
        continue
    df = pd.read_parquet(raw)
    df["distinct2"] = df.text.map(distinct2)
    g = df.groupby(["model", "alpha"]).distinct2.mean().reset_index()
    base = g[g.alpha == 0].set_index("model").distinct2
    g["rel"] = g.apply(lambda r: r.distinct2 / base[r.model] if r.model in base else float("nan"), axis=1)
    g["ok"] = (g.rel >= UMBRAL) | (g.alpha == 0)
    g.to_csv(folder / "colapso_por_alpha.csv", index=False)
    keep = df.merge(g[g.ok][["model", "alpha"]], on=["model", "alpha"])
    sel = folder.parent / (folder.name + "_sel")
    sel.mkdir(exist_ok=True)
    keep.drop(columns="distinct2").to_parquet(sel / "intervention_raw.parquet")
    for m, x in g.groupby("model"):
        ok = x[x.ok].alpha.tolist()
        print(f"{folder.name} | {m.split('/')[-1][:14]:14s} alphas ok: {ok} ({len(keep[keep.model == m])} textos)")
