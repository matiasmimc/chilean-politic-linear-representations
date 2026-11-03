"""Probes Ridge robustos (CPU) sobre activaciones ya extraidas.

Mejoras respecto de `scripts/entrenar_probes.py` (2 folds, alpha=1 fijo, maximo sobre cabezas en el mismo CV):
  * alpha por RidgeCV (LOO interno) y CV 5-fold repetida.
  * seleccion de capa/cabeza ANIDADA: la unidad se elige con CV interna dentro del entrenamiento del fold externo.
  * nulo por permutacion del maximo sobre unidades (corrige el sesgo de seleccion).
  * IC95% por bootstrap sobre parlamentarios.
  * esquemas que separan ideologia de identidad de partido:
      random         : folds aleatorios (como antes)
      leave_party_out: los partidos del test no estan en el entrenamiento
      within_party   : objetivo = P6 - media del partido (ideologia dentro del partido)
  * baseline trivial: media de P6 por partido.

Uso: OMP_NUM_THREADS=1 uv run python scripts/probing/probes_robustos.py [--models ...] [--conditions ...]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import difflib
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REP = BASE_DIR / "results/replication"
OUT_DIR = BASE_DIR / "results/probing_robusto"
ALPHAS = np.logspace(-2, 5, 8)

CONDITIONS = {  # nombre -> directorio base (atencion en la raiz, residual en /residual)
    "v2": REP / "v2",
    "names_only": REP / "v2_names_only",
    "noparty": REP / "v3_noparty",
    "noparty_names_only": REP / "v3_noparty_names_only",
}
MODELS = ["meta-llama_Llama-3.1-8B-Instruct", "google_gemma-4-E4B-it", "zai-org_GLM-4-9B-0414"]


def spear(a, b):
    if np.std(b) < 1e-12:
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


def make_folds(scheme, parties, n, seed, k=5):
    """Lista de (train, test). leave_party_out reparte partidos completos entre folds."""
    rng = np.random.RandomState(seed)
    if scheme == "leave_party_out":
        uniq = list(np.unique(parties))
        rng.shuffle(uniq)
        counts = np.zeros(k)
        fold_of = {}
        for p in sorted(uniq, key=lambda p: -np.sum(parties == p)):  # greedy balanceado
            f = int(np.argmin(counts))
            fold_of[p] = f
            counts[f] += np.sum(parties == p)
        fold_id = np.array([fold_of[p] for p in parties])
        return [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(k) if (fold_id == f).any()]
    return [(tr, te) for tr, te in KFold(k, shuffle=True, random_state=seed).split(np.arange(n))]


def inner_folds(scheme, parties, tr, seed):
    sub = make_folds(scheme, parties[tr], len(tr), seed, k=4)
    return [(tr[a], tr[b]) for a, b in sub]


def fit_predict(X, y, tr, te):
    return RidgeCV(alphas=ALPHAS).fit(X[tr], y[tr]).predict(X[te])


def unit_job(X, y, parties, scheme, seeds):
    """Para una unidad: predicciones OOF externas y score interno por (repeticion, fold)."""
    X = (X - X.mean(0)) / (X.std() + 1e-8)
    outer = np.zeros((len(seeds), len(y)))
    inner_scores = []
    for r, seed in enumerate(seeds):
        row = []
        for tr, te in make_folds(scheme, parties, len(y), seed):
            outer[r, te] = fit_predict(X, y, tr, te)
            inn = np.zeros(len(y))
            for itr, ite in inner_folds(scheme, parties, tr, seed):
                inn[ite] = fit_predict(X, y, itr, ite)
            row.append(spear(y[tr], inn[tr]) if scheme != "leave_party_out" else spear(y[tr], inn[tr]))
        inner_scores.append(row)
    return outer, inner_scores


def perm_job(X, y, parties, scheme, seed, n_perm):
    X = (X - X.mean(0)) / (X.std() + 1e-8)
    folds = make_folds(scheme, parties, len(y), seed)
    rng = np.random.RandomState(1234)
    out = np.zeros(n_perm)
    for p in range(n_perm):
        yp = rng.permutation(y)
        pred = np.zeros(len(y))
        for tr, te in folds:
            pred[te] = fit_predict(X, yp, tr, te)
        out[p] = spear(yp, pred)
    return out


def bootstrap_ci(y, pred, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    vals = [spear(y[idx], pred[idx]) for idx in (rng.randint(0, len(y), len(y)) for _ in range(n_boot))]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def get_units(features):
    if features.ndim == 4:
        _, L, H, _ = features.shape
        return [((l, h), features[:, l, h, :]) for l in range(L) for h in range(H)]
    return [((l,), features[:, l, :]) for l in range(features.shape[1])]


def run_one(features, y, parties, scheme, n_rep, n_perm, n_jobs):
    n = len(y)
    seeds = list(range(n_rep))
    y_target = y.copy()
    if scheme == "within_party":
        pm = pd.Series(y).groupby(parties).transform("mean").values
        y_target = y - pm
    base_scheme = "random" if scheme == "within_party" else scheme
    units = get_units(features)
    res = Parallel(n_jobs=n_jobs)(delayed(unit_job)(X, y_target, parties, base_scheme, seeds) for _, X in units)
    outer = np.stack([r[0] for r in res])               # [U, R, N]
    inner = np.array([r[1] for r in res], dtype=float)  # [U, R, K]
    unit_scores = np.array([[spear(y_target, outer[u, r]) for r in range(n_rep)] for u in range(len(units))])
    mean_scores = unit_scores.mean(1)
    best = int(np.argmax(mean_scores))
    rows = []

    # anidado: por (repeticion, fold) elegir la unidad con mayor score interno
    nested_pred = np.zeros((n_rep, n))
    for r, seed in enumerate(seeds):
        for k, (_, te) in enumerate(make_folds(base_scheme, parties, n, seed)):
            u = int(np.argmax(inner[:, r, k]))
            nested_pred[r, te] = outer[u, r, te]
    nested_scores = [spear(y_target, nested_pred[r]) for r in range(n_rep)]
    ci_n = bootstrap_ci(y_target, nested_pred.mean(0))

    best_pred = outer[best].mean(0)
    ci_b = bootstrap_ci(y_target, best_pred)
    p_perm = np.nan
    if n_perm > 0:
        null = np.array(Parallel(n_jobs=n_jobs)(
            delayed(perm_job)(X, y_target, parties, base_scheme, seeds[0], n_perm) for _, X in units))
        null_max = null.max(0)
        p_perm = float((1 + np.sum(null_max >= unit_scores[best, 0])) / (1 + n_perm))

    rows.append(dict(metodo="naive_max", unidad=str(units[best][0]), spearman=float(mean_scores[best]),
                     ci_lo=ci_b[0], ci_hi=ci_b[1], p_perm=p_perm))
    rows.append(dict(metodo="nested", unidad="seleccion interna", spearman=float(np.mean(nested_scores)),
                     ci_lo=ci_n[0], ci_hi=ci_n[1], p_perm=np.nan))
    rows.append(dict(metodo="mediana_unidades", unidad="", spearman=float(np.median(mean_scores)),
                     ci_lo=np.nan, ci_hi=np.nan, p_perm=np.nan))
    rows.append(dict(metodo="top16_media", unidad="", spearman=float(np.sort(mean_scores)[-16:].mean()),
                     ci_lo=np.nan, ci_hi=np.nan, p_perm=np.nan))
    return rows


def party_baseline(y, parties, n_rep):
    sc = []
    for seed in range(n_rep):
        pred = np.zeros(len(y))
        for tr, te in make_folds("random", parties, len(y), seed):
            m = pd.Series(y[tr]).groupby(parties[tr]).mean()
            pred[te] = pd.Series(parties[te]).map(m).fillna(np.mean(y[tr])).values
        sc.append(spear(y, pred))
    return float(np.mean(sc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--reprs", nargs="+", default=["residual", "attention"])
    ap.add_argument("--targets", nargs="+", default=["p6", "bcall"])
    ap.add_argument("--schemes", nargs="+", default=["random", "leave_party_out", "within_party"])
    ap.add_argument("--n-rep", type=int, default=5)
    ap.add_argument("--n-perm-attention", type=int, default=20)
    ap.add_argument("--n-perm-residual", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=14)
    ap.add_argument("--out", default=str(OUT_DIR / "resultados.csv"))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mp = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    parties_all = mp["party_1_P6"].fillna("IND").values
    targets_all = {"p6": mp["dim_ideology_P6"].astype(float).values}
    bpath = BASE_DIR / "data/bcall_data_chile/CHL-bcall-scores-2025.csv"
    if "bcall" in args.targets and bpath.exists():
        targets_all["bcall"] = load_bcall(mp["Name"].tolist(), bpath)

    done = set()
    out_path = Path(args.out)
    all_rows = []
    if out_path.exists():
        prev = pd.read_csv(out_path)
        all_rows = prev.to_dict("records")
        done = set(zip(prev.condicion, prev.modelo, prev.repr, prev.target, prev.esquema))

    for cond in args.conditions:
        for model in args.models:
            for rep in args.reprs:
                fdir = CONDITIONS[cond] / ("residual" if rep == "residual" else "")
                fpath = fdir / f"{model}_politician_features.pkl"
                if not fpath.exists():
                    print(f"[salto] falta {fpath}")
                    continue
                features, _ = pickle.load(open(fpath, "rb"))
                features = np.nan_to_num(features.astype(np.float32))
                for tname in args.targets:
                    if tname not in targets_all:
                        continue
                    y_all = targets_all[tname]
                    ok = ~np.isnan(y_all)
                    for scheme in args.schemes:
                        key = (cond, model, rep, tname, scheme)
                        if key in done:
                            continue
                        n_perm = args.n_perm_residual if rep == "residual" else args.n_perm_attention
                        print(f"## {key} (N={ok.sum()}, unidades={len(get_units(features[ok]))})", flush=True)
                        rows = run_one(features[ok], y_all[ok], parties_all[ok], scheme, args.n_rep, n_perm, args.n_jobs)
                        for r in rows:
                            r.update(condicion=cond, modelo=model, repr=rep, target=tname, esquema=scheme, n=int(ok.sum()))
                        if scheme == "random":
                            rows.append(dict(condicion=cond, modelo=model, repr=rep, target=tname, esquema=scheme,
                                             metodo="party_mean_baseline", unidad="", spearman=party_baseline(y_all[ok], parties_all[ok], args.n_rep),
                                             ci_lo=np.nan, ci_hi=np.nan, p_perm=np.nan, n=int(ok.sum())))
                        all_rows.extend(rows)
                        pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
