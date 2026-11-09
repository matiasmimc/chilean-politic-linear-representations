"""Consolidacion del efecto de caatopic sobre el slant (CPU): 10 direcciones aleatorias, mas muestras y sensibilidad a capas.

Para cada modelo x numero de capas (k=1,4,8) x medida (logit del propio modelo, clasificador TF-IDF y e5 entrenados en texto real):
pendiente por unidad de alpha con efectos fijos de tema (IC bootstrap por tema), solo textos con distinct-2 >= 0.5, para caatopic
y para cada direccion aleatoria. `p_aleat` = fraccion de aleatorias con pendiente >= la de caatopic (en el sentido de caatopic).
Salida: results/slant_caatopic/resumen.csv y aleatorias.csv.

Uso: uv run python scripts/analisis/caatopic_consolidacion.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/analisis"))
from slant_v3_analisis import boot, slope  # noqa: E402

MODELOS = {"ll": "Llama", "ge": "gemma", "gl": "glm"}
MEDIDAS = {"logit": None, "clf": "__clf", "clfemb": "__clfemb"}


def load(root, name, medida):
    d = pd.read_parquet(BASE / "results" / root / "dosis" / "intervention_raw.parquet")
    if MEDIDAS[medida]:
        e = pd.read_parquet(BASE / "results" / root / "dosis" / f"intervention_evaluated_{name}{MEDIDAS[medida]}.parquet")
        d["y"] = e.score_postura.values
    else:
        d["y"] = d.slant_logit.values
    return d[(d.distinct2 >= 0.5)].dropna(subset=["y"])


def stats(d, m, base):
    s = pd.concat([d[d.method == m], base])
    x, y, g = s.alpha.values.astype(float), s.y.values, s.topic.values
    return slope(x, y, g), s, x, y, g


def main():
    out = BASE / "results/slant_caatopic"
    out.mkdir(parents=True, exist_ok=True)
    rows, rr = [], []
    for t, name in MODELOS.items():
        for k, eq in [(1, ""), (4, ""), (8, ""), (1, "_eq"), (2, "_eq"), (8, "_eq")]:
            root = f"slant_v3c_{t}_k{k}{eq}"
            if not (BASE / "results" / root / "dosis" / "intervention_raw.parquet").exists():
                continue
            for medida in MEDIDAS:
                try:
                    d = load(root, name, medida)
                except FileNotFoundError:
                    continue
                base = d[d.method == "base"]
                b, s, x, y, g = stats(d, "caatopic", base)
                lo, hi = boot(x, y, g, g)
                rnd = {}
                for m in sorted(x for x in d.method.unique() if x.startswith("random")):
                    rnd[m] = stats(d, m, base)[0]
                    rr.append(dict(modelo=name, k_capas=k, medida=medida, direccion=m, pendiente=rnd[m]))
                v = np.array(list(rnd.values()))
                half = {}
                for sgn, lab in ((1, "alpha>0"), (-1, "alpha<0")):
                    h = pd.concat([d[(d.method == "caatopic") & (np.sign(d.alpha) == sgn)], base])
                    half[lab] = slope(h.alpha.values.astype(float), h.y.values, h.topic.values)
                rows.append(dict(modelo=name, k_capas=k, dosis_igualada=bool(eq), medida=medida, n=len(s), pendiente=b, ic_lo=lo, ic_hi=hi,
                                 rho=spearmanr(x, y - pd.Series(y).groupby(g).transform("mean").values).statistic,
                                 pend_alfa_pos=half["alpha>0"], pend_alfa_neg=half["alpha<0"],
                                 n_aleat=len(v), aleat_media=v.mean(), aleat_min=v.min(), aleat_max=v.max(),
                                 aleat_abs_max=np.abs(v).max(), p_aleat=float((np.sum(v * np.sign(b) >= b * np.sign(b)) + 1) / (len(v) + 1)),
                                 distinct2_pct_excl=float(100 * (1 - len(d) / len(pd.read_parquet(BASE / "results" / root / "dosis" / "intervention_raw.parquet"))))))
                print({k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(out / "resumen.csv", index=False)
    pd.DataFrame(rr).to_csv(out / "aleatorias.csv", index=False)


if __name__ == "__main__":
    main()
