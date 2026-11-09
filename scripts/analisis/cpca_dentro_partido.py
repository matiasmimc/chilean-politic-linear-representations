"""A1 (CPU): ¿cPCA ve ideologia DENTRO del partido?

Compara, por capa, direcciones label-free (cPCA con fondo neutral, PCA, PCA intra-partido, cPCA intra-partido
con fondo neutral, proyecciones aleatorias) frente al objetivo P6/b-call menos la media del partido.
Las direcciones no usan etiquetas; solo el ridge (k proyecciones -> objetivo) se ajusta con CV 5x5.
Nulo por permutacion de P6 dentro de cada partido, sobre el maximo en capas x k de cada metodo.

Uso: OMP_NUM_THREADS=1 uv run python scripts/analisis/cpca_dentro_partido.py [--models ...] [--conditions v2 v3_noparty]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import difflib
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REP = BASE_DIR / "results/replication"
OUT_DIR = BASE_DIR / "results/cpca_dentro_partido"
CONDITIONS = {"v2": REP / "v2" / "residual", "v3_noparty": REP / "v3_noparty" / "residual"}
MODELS = ["meta-llama_Llama-3.1-8B-Instruct", "google_gemma-4-E4B-it", "zai-org_GLM-4-9B-0414"]
KS = (1, 4, 16)
ALPHAS = np.logspace(-2, 5, 8)
CONTRAST_ALPHA = 2.0
KMAX = max(KS)
METHODS = ["cpca_neutral", "pca", "pca_within", "cpca_within_neutral", "random"]


def spear(a, b):
    if np.std(b) < 1e-12 or np.std(a) < 1e-12:
        return 0.0
    s = spearmanr(a, b).statistic
    return 0.0 if np.isnan(s) else float(s)


def load_bcall(names, path):
    df = pd.read_csv(path)
    clean = {leg.replace("_", " "): leg for leg in df["legislator"]}
    d1 = dict(zip(df["legislator"], df["d1"]))
    out = []
    for n in names:
        m = difflib.get_close_matches(n, list(clean), n=1, cutoff=0.6)
        out.append(d1[clean[m[0]]] if m else np.nan)
    return np.array(out, dtype=float)


def top_eigvecs(C, k):
    d = C.shape[0]
    _, v = eigh(C, subset_by_index=[d - k, d - 1])
    return v[:, ::-1]


def cov(X):
    Xc = X - X.mean(0)
    return Xc.T @ Xc / (len(X) - 1)


def layer_projections(XT, XB, parties, seed=0):
    """Dict metodo -> Z [N, KMAX] con las proyecciones de XT sobre las KMAX direcciones del metodo."""
    XT = XT.astype(np.float64)
    XB = XB.astype(np.float64)
    Xc = XT - XT.mean(0)
    pm = pd.DataFrame(XT).groupby(parties).transform("mean").values
    Xw = XT - pm
    CT, CB, CW = cov(XT), cov(XB), cov(Xw)
    out = {}
    out["cpca_neutral"] = Xc @ top_eigvecs(CT - CONTRAST_ALPHA * CB, KMAX)
    out["pca"] = Xc @ top_eigvecs(CT, KMAX)
    out["pca_within"] = Xc @ top_eigvecs(CW, KMAX)
    out["cpca_within_neutral"] = Xc @ top_eigvecs(CW - CONTRAST_ALPHA * CB, KMAX)
    rng = np.random.RandomState(seed)
    R = rng.randn(XT.shape[1], KMAX)
    out["random"] = Xc @ (R / np.linalg.norm(R, axis=0))
    return out


def oof_spearman(Z, y, seeds):
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    sc = []
    for s in seeds:
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=s).split(Z):
            pred[te] = RidgeCV(alphas=ALPHAS).fit(Z[tr], y[tr]).predict(Z[te])
        sc.append(spear(y, pred))
    return float(np.mean(sc))


def layer_job(layer, XT, XB, parties, y_w, seeds):
    proj = layer_projections(XT, XB, parties)
    rows = []
    for m, Z in proj.items():
        for k in KS:
            rows.append(dict(capa=layer, metodo=m, k=k, spearman=oof_spearman(Z[:, :k], y_w, seeds)))
    return rows, proj


def perm_null(projs, parties, y, n_perm, seed=0):
    """Nulo del maximo por metodo (capas x k), permutando y dentro de cada partido."""
    rng = np.random.RandomState(seed)
    idx_by_party = [np.where(parties == p)[0] for p in np.unique(parties)]
    nulls = {m: np.zeros(n_perm) for m in METHODS}
    for p in range(n_perm):
        yp = y.copy()
        for idx in idx_by_party:
            yp[idx] = y[rng.permutation(idx)]
        ypw = yp - pd.Series(yp).groupby(parties).transform("mean").values
        for m in METHODS:
            best = -1
            for proj in projs:
                for k in KS:
                    best = max(best, oof_spearman(proj[m][:, :k], ypw, [0]))
            nulls[m][p] = best
    return nulls


def ridge_full_job(layer, X, y_w, seeds):
    X = X.astype(np.float64)
    X = (X - X.mean(0)) / (X.std() + 1e-8)
    return dict(capa=layer, metodo="ridge_full", k=0, spearman=oof_spearman(X, y_w, seeds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--targets", nargs="+", default=["p6", "bcall"])
    ap.add_argument("--n-rep", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=12)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mp = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    parties_all = mp["party_1_P6"].fillna("IND").values
    targets_all = {"p6": mp["dim_ideology_P6"].astype(float).values}
    bpath = BASE_DIR / "data/bcall_data_chile/CHL-bcall-scores-2025.csv"
    if bpath.exists():
        targets_all["bcall"] = load_bcall(mp["Name"].tolist(), bpath)
    seeds = list(range(args.n_rep))

    for cond in args.conditions:
        for model in args.models:
            out_csv = OUT_DIR / f"capas_{cond}_{model}.csv"
            if out_csv.exists():
                print(f"[salto] {out_csv}")
                continue
            fdir = CONDITIONS[cond]
            XT_all, _ = pickle.load(open(fdir / f"{model}_politician_features.pkl", "rb"))
            XB, _ = pickle.load(open(fdir / f"{model}_neutral_features.pkl", "rb"))
            XT_all, XB = np.nan_to_num(XT_all.astype(np.float32)), np.nan_to_num(XB.astype(np.float32))
            L = min(XT_all.shape[1], XB.shape[1])
            all_rows = []
            for tname in args.targets:
                y_all = targets_all[tname]
                ok = ~np.isnan(y_all)
                XT, parties, y = XT_all[ok], parties_all[ok], y_all[ok]
                y_w = y - pd.Series(y).groupby(parties).transform("mean").values
                print(f"## {cond} {model} {tname} N={ok.sum()} capas={L}", flush=True)
                res = Parallel(n_jobs=args.n_jobs)(
                    delayed(layer_job)(l, XT[:, l], XB[:, l], parties, y_w, seeds) for l in range(L))
                ridge_rows = Parallel(n_jobs=args.n_jobs)(
                    delayed(ridge_full_job)(l, XT[:, l], y_w, seeds) for l in range(L))
                rows = [r for rr, _ in res for r in rr] + ridge_rows
                projs = [p for _, p in res]
                nulls = perm_null(projs, parties, y, args.n_perm)
                df = pd.DataFrame(rows)
                df["target"] = tname
                # p de permutacion del maximo por metodo (el max observado usa los mismos seeds[0] que el nulo)
                for m in METHODS:
                    sub = df[df.metodo == m]
                    obs0 = max(oof_spearman(projs[r.capa][m][:, :r.k], y_w, [0]) for r in sub.itertuples()
                               if r.spearman >= sub.spearman.max() - 1e-12)
                    df.loc[df.metodo == m, "p_perm_max"] = float((1 + np.sum(nulls[m] >= obs0)) / (1 + args.n_perm))
                    df.loc[df.metodo == m, "nulo_max_p95"] = float(np.percentile(nulls[m], 95))
                all_rows.append(df)
                best = df.groupby("metodo").spearman.max().sort_values(ascending=False)
                print(best.to_string(), flush=True)
            pd.concat(all_rows).to_csv(out_csv, index=False)
            print(f"-> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
