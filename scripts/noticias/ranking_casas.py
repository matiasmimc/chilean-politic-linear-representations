"""Ranking de casas de noticias por posicion ideologica (para lectura cualitativa; sin etiqueta externa) (CPU).

Lee results/noticias/<modelo>/scores_articulos.parquet (ideologia_casas.py). Por casa y metodo: promedio de articulos, convertido a z
entre casas (media 0, mayor = derecha) para poder promediar escalas distintas; consenso = media de z entre metodos y modelos.
IC95 del consenso por bootstrap de articulos (dentro de casa). Ademas: acuerdo entre metodos/modelos (Spearman entre rankings) y
variante con control de tema (resid_tema). Salidas en results/noticias/ranking/: ranking.csv, ranking.md, acuerdo.csv, ranking.png.

Uso: uv run python scripts/noticias/ranking_casas.py [--metodos ridge caatopic cpca_knn] [--min-art 300]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ROOT = BASE_DIR / "results/noticias"
MODELOS = {"Llama": "meta-llama_Llama-3.1-8B-Instruct", "Gemma": "google_gemma-4-E4B-it", "GLM": "zai-org_GLM-4-9B-0414"}


def zmeans(df, cols):
    m = df.groupby("casa")[cols].mean()
    return (m - m.mean()) / m.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metodos", nargs="+", default=["ridge", "caatopic", "cpca_knn"])
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    out = ROOT / "ranking"
    out.mkdir(exist_ok=True)

    frames = {}
    for mod, safe in MODELOS.items():
        p = ROOT / safe / "scores_articulos.parquet"
        if p.exists():
            frames[mod] = pd.read_parquet(p)
    assert frames, "no hay scores_articulos.parquet"
    casas = sorted(set.intersection(*[set(f.casa) for f in frames.values()]))
    cols = {}
    for mod, f in frames.items():
        f = f[f.casa.isin(casas)].sort_values("id").reset_index(drop=True) if "id" in f else f[f.casa.isin(casas)]
        frames[mod] = f
        for var in ("base", "resid_tema"):
            for m in args.metodos + ["tfidf"]:
                if f"{var}|{m}" in f:
                    cols[(mod, var, m)] = f"{var}|{m}"

    # puntaje z por casa para cada (modelo, variante, metodo)
    Z = pd.concat({k: zmeans(frames[k[0]], [c])[c] for k, c in cols.items()}, axis=1)
    Z.columns = [f"{a}|{b}|{c}" for a, b, c in Z.columns]
    prim = [c for c in Z.columns if c.split("|")[2] != "tfidf" and c.split("|")[1] == "base"]
    ctrl = [c for c in Z.columns if c.split("|")[2] != "tfidf" and c.split("|")[1] == "resid_tema"]

    # bootstrap de articulos: consenso de los metodos principales
    rng = np.random.RandomState(0)
    idx = {mod: frames[mod].groupby("casa").indices for mod in frames}
    boots = []
    for _ in range(args.n_boot):
        zs = []
        for (mod, var, m), c in cols.items():
            if var != "base" or m == "tfidf":
                continue
            f = frames[mod]
            means = pd.Series({k: f[c].values[rng.choice(v, len(v))].mean() for k, v in idx[mod].items()})
            zs.append((means - means.mean()) / means.std())
        boots.append(pd.concat(zs, axis=1).mean(1))
    B = pd.concat(boots, axis=1)

    n_art = frames[next(iter(frames))].groupby("casa").size()
    res = pd.DataFrame({"consenso": Z[prim].mean(1), "ic_lo": B.quantile(0.025, axis=1), "ic_hi": B.quantile(0.975, axis=1),
                        "consenso_resid_tema": Z[ctrl].mean(1) if ctrl else np.nan,
                        "desacuerdo_sd": Z[prim].std(1), "n_articulos": n_art}).sort_values("consenso")
    res["rango"] = np.arange(1, len(res) + 1)
    res = res.join(Z[[c for c in Z.columns if c.split("|")[2] == "tfidf"]].mean(1).rename("tfidf_z"))
    res.round(3).to_csv(out / "ranking.csv")
    Z.round(3).loc[res.index].to_csv(out / "puntaje_z_por_metodo.csv")

    # acuerdo entre metodos/modelos
    ac = Z.rank().corr()
    ac.round(2).to_csv(out / "acuerdo.csv")
    cross = ac.loc[prim, prim].values[np.triu_indices(len(prim), 1)]
    print(f"Acuerdo (Spearman) entre {len(prim)} combinaciones modelo×metodo: mediana {np.median(cross):.2f}, min {cross.min():.2f}, max {cross.max():.2f}")

    with open(out / "ranking.md", "w") as fh:
        fh.write("| # | casa | consenso z [IC95] | control tema | desacuerdo | n |\n|---|---|---|---|---|---|\n")
        for c, r in res.iterrows():
            fh.write(f"| {r.rango:.0f} | {c} | {r.consenso:+.2f} [{r.ic_lo:+.2f}, {r.ic_hi:+.2f}] | {r.consenso_resid_tema:+.2f} | "
                     f"{r.desacuerdo_sd:.2f} | {r.n_articulos:.0f} |\n")
    print(open(out / "ranking.md").read())

    fig, ax = plt.subplots(figsize=(7, 0.28 * len(res) + 1.5))
    y = np.arange(len(res))
    ax.errorbar(res.consenso, y, xerr=[res.consenso - res.ic_lo, res.ic_hi - res.consenso], fmt="o", color="#1f4e79", ms=4, lw=1)
    ax.scatter(res.consenso_resid_tema, y, marker="x", color="#c0392b", s=25, label="con control de tema")
    ax.set_yticks(y)
    ax.set_yticklabels(res.index, fontsize=8)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("← izquierda        puntaje z (consenso de metodos y modelos)        derecha →")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "ranking.png", dpi=150)


if __name__ == "__main__":
    main()
