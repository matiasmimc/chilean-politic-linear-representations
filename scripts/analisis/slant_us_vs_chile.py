"""Dosis-respuesta del slant con direcciones ridge/cPCA: EE.UU. (DW-NOMINATE, ingles) frente a Chile (P6, espanol) (CPU).

Misma medida (slant_logit del propio modelo, izquierda/derecha), mismo alpha (desviaciones estandar de la proyeccion de los
politicos). Por pais, modelo y familia de direccion: pendiente por unidad de alpha con efectos fijos por tema y IC95% por
bootstrap de temas, sobre textos no colapsados (distinct-2 >= 0.5) y |alpha| <= R (R = 2, 4, 8); Spearman; y % de textos
colapsados por |alpha|.

Uso: uv run python scripts/analisis/slant_us_vs_chile.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/analisis"))
from slant_v3_analisis import boot, slope  # noqa: E402

MODELS = {"meta-llama_Llama-3.1-8B-Instruct": "Llama", "google_gemma-4-E4B-it": "Gemma", "zai-org_GLM-4-9B-0414": "GLM"}
FAMILIES = {  # familia -> (metodos de EE.UU., metodos de Chile)
    "ridge": (["ridge"], ["ridge", "ridge_wide"]),
    "cpca_k1": (["cpca_k1"], ["cpca_k1_wide"]),
    "cpca_k16": (["cpca_k16"], ["cpca"]),
    "aleatorias": ([f"random_s{i}" for i in range(3)],
                   [f"random_k16_s{i}{w}" for i in range(3) for w in ("", "_wide")]),
}


def distinct2(t):
    w = t.split()
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / max(len(bg), 1)


def load():
    out = {}
    for safe, name in MODELS.items():
        f = BASE / f"results/slant_us/{safe}/intervention_raw.parquet"
        if f.exists():
            out[("EE.UU.", name)] = pd.read_parquet(f)
        f = BASE / f"results/slant_chile_logit/{safe}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            d["distinct2"] = d.text.map(distinct2)
            out[("Chile", name)] = d
    return out


def main():
    pd.set_option("display.width", 220)
    rows = []
    for (country, model), d in load().items():
        for fam, (us_m, cl_m) in FAMILIES.items():
            sub = d[d.method.isin(us_m if country == "EE.UU." else cl_m)]
            if country == "EE.UU.":  # el alpha = 0 de EE.UU. se guarda como 'base': se agrega a cada familia
                sub = pd.concat([sub, d[d.method == "base"]])
            for R in (2, 4, 8):
                s = sub[(sub.alpha.abs() <= R)]
                n_all = len(s)
                s = s[s.distinct2 >= 0.5]
                if len(s) < 50 or s.alpha.nunique() < 3:
                    continue
                x, y, g = s.alpha.values.astype(float), s.slant_logit.values, s.topic.values
                b = slope(x, y, g)
                lo, hi = boot(x, y, g, g)
                rows.append(dict(pais=country, modelo=model, familia=fam, R=R, n=len(s), colapsados_pct=100 * (1 - len(s) / n_all),
                                 pendiente=b, lo=lo, hi=hi, spearman=spearmanr(x, y)[0]))
    r = pd.DataFrame(rows)
    out = BASE / "results/slant_us_vs_chile"
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "resumen.csv", index=False)
    for R in (2, 4, 8):
        print(f"\n=== |alpha| <= {R} (textos no colapsados) ===")
        print(r[r.R == R].drop(columns="R").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
