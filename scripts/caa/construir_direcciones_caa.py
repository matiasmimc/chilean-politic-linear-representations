"""Direcciones de contraste (derecha - izquierda) por capa, a partir de las intervenciones de los extremos de P6.

  extremos : cuartil inferior de P6 (izquierda) y superior (derecha), cada diputado pesa igual (media por diputado).
  direccion: media(derecha) - media(izquierda) por capa; `diff_norm` = norma de la diferencia (escala natural de alpha).
Validacion (sin usar los diputados evaluados para construir la direccion):
  auc_cv      : CV 5-fold sobre diputados extremos; AUC de la proyeccion media de cada diputado retenido.
  rho_medio   : Spearman(proyeccion media, P6) en los diputados del 50% central (nunca usados al construir).
La seleccion de capas para steering usa solo auc_cv.

Uso: uv run python scripts/caa/construir_direcciones_caa.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "results/caa"


def direction(means, y):
    d = means[y == 1].mean(0) - means[y == 0].mean(0)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--q", type=float, default=0.25)
    args = ap.parse_args()
    safe = args.model.replace("/", "_")
    d = pickle.load(open(OUT_DIR / f"{safe}_mp_statement_means.pkl", "rb"))
    means, p6 = d["means"], d["p6"]
    valid = ~np.isnan(means).any(axis=(1, 2))
    lo, hi = np.quantile(p6[valid], [args.q, 1 - args.q])
    ext = valid & ((p6 <= lo) | (p6 >= hi))
    mid = valid & ~ext
    y_ext = (p6[ext] >= hi).astype(int)
    X_ext, X_mid = means[ext], means[mid]
    L = means.shape[1]
    print(f"{safe}: extremos {ext.sum()} (izq {np.sum(y_ext == 0)}, der {np.sum(y_ext == 1)}), centrales {mid.sum()}")

    rows, out = [], {}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X_ext[:, 0], y_ext))
    for l in range(L):
        v = direction(X_ext[:, l], y_ext)
        norm = float(np.linalg.norm(v))
        u = v / norm
        scores_cv = np.zeros(len(y_ext))
        for tr, te in folds:
            ucv = direction(X_ext[tr, l], y_ext[tr])
            ucv /= np.linalg.norm(ucv)
            scores_cv[te] = X_ext[te, l] @ ucv
        auc = roc_auc_score(y_ext, scores_cv)
        rho = spearmanr(X_mid[:, l] @ u, p6[mid]).statistic
        rows.append({"layer": l, "auc_cv": auc, "rho_medio": rho, "diff_norm": norm})
        out[l] = {"vector": u.astype(np.float32), "diff_norm": norm, "auc_cv": float(auc), "rho_medio": float(rho)}
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"{safe}_caa_validacion.csv", index=False)
    pickle.dump({"layers": out, "q": args.q, "n_ext": int(ext.sum()), "n_mid": int(mid.sum())},
                open(OUT_DIR / f"{safe}_caa_directions.pkl", "wb"))
    top = df.sort_values("auc_cv", ascending=False).head(8)
    print(top.round(3).to_string(index=False))
    print(f"rho_medio: mediana {df.rho_medio.median():.2f}, max {df.rho_medio.max():.2f} (capa {int(df.loc[df.rho_medio.idxmax(), 'layer'])})")


if __name__ == "__main__":
    main()
