"""A3 (CPU): ¿de donde viene la ganancia de cPCA en la prediccion de votos? Desglose por partido y proyecto.

Usa los parquets de `results/voting_predictions/residual/divided/` (split test, alpha=0 frente a alpha optimo del
mismo archivo) y compara cPCA con las direcciones aleatorias.

Salidas (results/desglose_ganancia_cpca/):
  por_partido.csv, por_proyecto.csv : delta de exactitud balanceada, tasas de "EN CONTRA" predichas/reales
  resumen.csv : por metodo, delta total, cambios de voto (flips), exactitud de los flips, y consistencia de los
                flips dentro de celdas (partido x proyecto): si los flips van todos en una direccion en la celda,
                el efecto es a nivel de partido/proyecto y no entre diputados.

Uso: uv run python scripts/analisis/desglose_ganancia_cpca.py [--models ...]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PRED_DIR = BASE_DIR / "results/voting_predictions/residual/divided"
OUT_DIR = BASE_DIR / "results/desglose_ganancia_cpca"
MODELS = ["meta-llama_Llama-3.1-8B-Instruct", "google_gemma-4-E4B-it", "zai-org_GLM-4-9B-0414"]
EC, AF = "EN CONTRA", "A FAVOR"


def bal_acc(df):
    r = [(df.voto_predicho[df.voto_real == c] == c).mean() for c in (EC, AF) if (df.voto_real == c).any()]
    return 100 * float(np.mean(r)) if r else np.nan


def load_pair(path):
    d = pd.read_parquet(path)
    d = d[d.split == "test"]
    a0, a1 = d[d.alpha == 0.0], d[d.alpha != 0.0]
    if a1.empty:
        return None
    key = ["mp_uid", "ley"]
    m = a0.merge(a1, on=key, suffixes=("_0", "_1"))
    return m, float(a1.alpha.iloc[0])


def per_group(m, col):
    rows = []
    for g, s in m.groupby(col):
        b = s.rename(columns={"voto_real_0": "voto_real", "voto_predicho_0": "voto_predicho"})[["voto_real", "voto_predicho"]]
        c = s.rename(columns={"voto_real_0": "voto_real", "voto_predicho_1": "voto_predicho"})[["voto_real", "voto_predicho"]]
        rows.append({col: g, "n": len(s), "real_en_contra": 100 * (s.voto_real_0 == EC).mean(),
                     "pred_ec_base": 100 * (s.voto_predicho_0 == EC).mean(),
                     "pred_ec_steer": 100 * (s.voto_predicho_1 == EC).mean(),
                     "ba_base": bal_acc(b), "ba_steer": bal_acc(c),
                     "delta_acierto_pp": 100 * ((s.voto_predicho_1 == s.voto_real_0).mean() - (s.voto_predicho_0 == s.voto_real_0).mean())})
    return pd.DataFrame(rows)


def summarize(m, party_col, name, alpha):
    ba0 = bal_acc(m.rename(columns={"voto_real_0": "voto_real", "voto_predicho_0": "voto_predicho"}))
    ba1 = bal_acc(m.rename(columns={"voto_real_0": "voto_real", "voto_predicho_1": "voto_predicho"}))
    flip = m.voto_predicho_0 != m.voto_predicho_1
    f = m[flip].copy()
    f["dir"] = np.where(f.voto_predicho_1 == EC, 1, -1)       # +1: pasa a EN CONTRA
    f["flip_ok"] = (f.voto_predicho_1 == f.voto_real_0)
    f["flip_was_ok"] = (f.voto_predicho_0 == f.voto_real_0)
    cells = f.groupby([party_col, "ley"]).dir.agg(["sum", "count"])
    cons = float((cells["sum"].abs().sum()) / cells["count"].sum()) if len(cells) else np.nan
    # consistencia esperada si las direcciones fueran independientes (mismo n por celda, p=media global)
    p = (f.dir == 1).mean() if len(f) else 0.5
    exp = float(np.mean([abs(2 * np.random.RandomState(i).binomial(n, p, 2000) - n).mean() / n
                         for i, n in enumerate(cells["count"].values)])) if len(cells) else np.nan
    # componente entre celdas del cambio en tasa de EN CONTRA predicha
    d = (m.voto_predicho_1 == EC).astype(float) - (m.voto_predicho_0 == EC).astype(float)
    cell_mean = d.groupby([m[party_col], m.ley]).transform("mean")
    return dict(metodo=name, alpha=alpha, n_test=len(m), ba_base=ba0, ba_steer=ba1, delta_ba=ba1 - ba0,
                flips_pct=100 * flip.mean(), flip_a_ec_pct=100 * (f.dir == 1).mean() if len(f) else np.nan,
                acc_flips_antes=100 * f.flip_was_ok.mean() if len(f) else np.nan,
                acc_flips_despues=100 * f.flip_ok.mean() if len(f) else np.nan,
                consist_flips_celda=cons, consist_esperada_indep=exp,
                var_entre_celdas_pct=100 * float(cell_mean.var() / d.var()) if d.var() > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summ, byp, byb = [], [], []
    for model in args.models:
        for path in sorted(PRED_DIR.glob(f"voting_predictions_{model}_*.parquet")):
            method = re.sub(rf"voting_predictions_{re.escape(model)}_|\.parquet", "", path.name)
            r = load_pair(path)
            if r is None:
                continue
            m, alpha = r
            m = m.rename(columns={"partido_0": "partido"})
            s = summarize(m, "partido", method, alpha)
            s["modelo"] = model
            summ.append(s)
            if method in ("cpca", "cpca_np"):
                for df, lst, col in ((per_group(m, "partido"), byp, "partido"), (per_group(m, "ley"), byb, "ley")):
                    df["modelo"], df["metodo"], df["alpha"] = model, method, alpha
                    lst.append(df)
    pd.DataFrame(summ).to_csv(OUT_DIR / "resumen.csv", index=False)
    pd.concat(byp).to_csv(OUT_DIR / "por_partido.csv", index=False)
    pd.concat(byb).to_csv(OUT_DIR / "por_proyecto.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summ).round(2).to_string())
    for model in args.models:
        t = pd.concat(byp)
        t = t[(t.modelo == model) & (t.metodo == "cpca")].sort_values("n", ascending=False)
        print(f"\n== {model}: por partido (cpca) ==\n", t.drop(columns=["modelo", "metodo"]).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
