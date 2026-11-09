"""Muestra del corpus de noticias para el experimento de ideologia por casa (CPU).

Solo articulos con keyword en el titulo (`any_in_title`, ver data/noticias/LEEME.md), casas con >= --min-casa articulos y muestra
equilibrada: hasta --por-casa articulos por casa, repartidos de forma pareja entre temas (para que las casas no difieran en
la mezcla de temas). texto = titulo + cuerpo. Salida: data/noticias/muestra.parquet (casa, tema, fecha, texto, id).

Uso: uv run python scripts/noticias/muestra_noticias.py [--por-casa 400] [--min-casa 300]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC = BASE_DIR / "data/noticias/noticias_chile_2022_2026_keywords.parquet"
EXCLUIR = {"prensafutbol.cl", "redgol.cl", "espn.cl", "servel.cl", "tribunalcalificador.cl", "puntoticket.com", "itseller.cl",
           "madboxpc.com", "ohmygeek.net", "plataformaarquitectura.cl", "america-retail.com", "portalportuario.cl", "eha.cl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-casa", type=int, default=400)
    ap.add_argument("--min-casa", type=int, default=300)
    ap.add_argument("--min-chars", type=int, default=500)
    args = ap.parse_args()
    cols = ["id", "publish_date", "source", "title", "body", "any_in_title"]
    kw = [c for c in pd.read_parquet(SRC).columns if c.startswith("title_")]
    d = pd.read_parquet(SRC, columns=cols + kw)
    d = d[d.any_in_title.fillna(False).astype(bool) & ~d.source.isin(EXCLUIR) & (d.body.str.len() >= args.min_chars)].copy()
    # tema = primera keyword del titulo (las muy genericas, al final, solo si no hay otra)
    orden = [k for k in kw if k not in ("title_congreso", "title_partidos", "title_seguridad", "title_ciencia")] + \
            ["title_congreso", "title_partidos", "title_seguridad", "title_ciencia"]
    tema = np.full(len(d), "", dtype=object)
    for k in reversed(orden):
        tema[d[k].fillna(False).astype(bool).values] = k[6:]
    d["tema"] = tema
    cnt = d.source.value_counts()
    casas = cnt[cnt >= args.min_casa].index
    d = d[d.source.isin(casas)]
    print(f"{len(casas)} casas con >= {args.min_casa} articulos:\n{cnt[casas].to_string()}")
    rng = np.random.RandomState(0)
    out = []
    for c, g in d.groupby("source"):
        # reparto parejo entre temas: se recorre en rondas, un articulo por tema
        pools = {t: list(rng.permutation(len(x))) for t, x in g.groupby("tema").indices.items()}
        idx = {t: v for t, v in g.groupby("tema").indices.items()}
        pick = []
        while len(pick) < min(args.por_casa, len(g)) and any(pools.values()):
            for t in list(pools):
                if pools[t] and len(pick) < args.por_casa:
                    pick.append(idx[t][pools[t].pop()])
        out.append(g.iloc[pick])
    m = pd.concat(out)
    m = pd.DataFrame({"id": m.id, "casa": m.source, "tema": m.tema, "fecha": m.publish_date,
                      "texto": (m.title.str.strip() + "\n\n" + m.body.str.strip())}).reset_index(drop=True)
    m.to_parquet(BASE_DIR / "data/noticias/muestra.parquet", index=False)
    print(f"\n{len(m)} articulos, {m.casa.nunique()} casas")
    print(pd.crosstab(m.casa, m.tema).sum().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
