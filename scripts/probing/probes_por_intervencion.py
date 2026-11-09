"""Probes ridge por intervencion (CPU): ideologia del diputado a partir de CADA declaracion, con CV agrupada.

Entrena sobre declaraciones individuales (~3100) y evalua agregando la prediccion por diputado (media).
Fuentes:
  anon     : results/caa/<modelo>_statement_acts.pkl (solo el texto, residual medio sobre tokens; sin nombre ni partido)
  decl_v2  : results/replication/v2_por_declaracion/residual (prompt con nombre y partido, ultimo token)
Objetivos por diputado: p6, bcall_d1, bcall_d2, gov_apoyo (fraccion de A FAVOR en proyectos "Mensaje" divididos).
Esquemas (los folds son de diputados completos, nunca de declaraciones):
  random : 5 folds aleatorios de diputados
  lpo    : leave-party-out (partidos completos por fold)
  within : objetivo y prediccion centrados por partido (solo partidos con >= 2 diputados)
Metrica: Spearman diputado a diputado de la prediccion media vs el objetivo. Capa y alpha: se reporta el maximo y su p de
permutacion (nulo del maximo sobre capas x alpha; las permutaciones reasignan el objetivo entre diputados, dentro del
partido en `within`). La SVD por (capa, fold) se calcula una vez y sirve para todas las permutaciones.

Uso: OMP_NUM_THREADS=1 uv run python scripts/probing/probes_por_intervencion.py --source anon --models ...
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import difflib
import pickle
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import svd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts/analisis"))
OUT_DIR = BASE_DIR / "results/probing_por_intervencion"
MODELS = ["meta-llama_Llama-3.1-8B-Instruct", "google_gemma-4-E4B-it"]
REL_ALPHAS = np.array([1e-2, 1e-1, 1.0, 10.0, 100.0])  # multiplos de mean(s^2)
SCHEMES = ["random", "lpo", "within"]


def load_bcall(names, path, col):
    df = pd.read_csv(path)
    clean = {leg.replace("_", " "): leg for leg in df["legislator"]}
    val = dict(zip(df["legislator"], df[col]))
    out = []
    for n in names:
        m = difflib.get_close_matches(n, list(clean), n=1, cutoff=0.6)
        out.append(val[clean[m[0]]] if m else np.nan)
    return np.array(out, dtype=float)


def gov_support(uid):
    """Fraccion de A FAVOR por diputado en proyectos de iniciativa 'Mensaje' con >= 25% de votos en contra."""
    with sqlite3.connect(BASE_DIR / "data/parlamento.db") as conn:
        rows = conn.execute("""
            SELECT sv.bill_id, v.mp_uid, v.voto FROM sesiones_votacion sv
            JOIN votos_parlamentario v ON sv.sesion_votacion_id = v.sesion_votacion_id
            JOIN bills b ON b.bill_id = sv.bill_id
            WHERE b.iniciativa = 'Mensaje' AND v.voto IN ('A Favor', 'En Contra')
              AND sv.bill_id IN (SELECT bill_id FROM sesiones_votacion GROUP BY bill_id
                                 HAVING AVG(1.0 * en_contra_total / NULLIF(a_favor_total + en_contra_total, 0)) >= 0.25)
        """).fetchall()
    df = pd.DataFrame(rows, columns=["bill", "mp", "voto"])
    g = df.groupby(["mp", "bill"]).voto.agg(f=lambda s: (s == "A Favor").mean()).reset_index()
    g["f"] = (g.f > 0.5).astype(float)
    per = g.groupby("mp").f.agg(["mean", "count"])
    per = per[per["count"] >= 5]["mean"]
    print(f"gov_apoyo: {len(per)} diputados con >=5 proyectos Mensaje divididos "
          f"({g.bill.nunique()} proyectos)", flush=True)
    return np.array([per.get(u, np.nan) for u in uid], dtype=float)


def make_folds(unit_party, valid, scheme, k=5, seed=0):
    """Lista de arrays de indices de diputados (test) sobre los indices `valid`."""
    rng = np.random.RandomState(seed)
    idx = np.where(valid)[0]
    if scheme == "lpo":
        parties = list(np.unique(unit_party[idx]))
        rng.shuffle(parties)
        counts, fold_of = np.zeros(k), {}
        for p in sorted(parties, key=lambda p: -np.sum(unit_party[idx] == p)):
            f = int(np.argmin(counts))
            fold_of[p] = f
            counts[f] += np.sum(unit_party[idx] == p)
        fid = np.array([fold_of[p] for p in unit_party[idx]])
    else:
        fid = rng.permutation(len(idx)) % k
    return [idx[fid == f] for f in range(k) if (fid == f).any()]


def rank_corr_cols(A, y):
    """Spearman de cada columna de A [n, P] con cada columna de y [n, P] (mismas dimensiones)."""
    Ra, Ry = rankdata(A, axis=0), rankdata(y, axis=0)
    Ra -= Ra.mean(0)
    Ry -= Ry.mean(0)
    return (Ra * Ry).sum(0) / (np.sqrt((Ra ** 2).sum(0) * (Ry ** 2).sum(0)) + 1e-12)


def job(X, mp_of, valid_mp, party_mp, y_cols, scheme_folds, within_mask, scheme):
    """Una capa, un objetivo, un esquema. y_cols [n_mp, P] (columna 0 = real). Devuelve scores [A, P]."""
    n_mp = len(party_mp)
    stmt_ok = valid_mp[mp_of]
    P = y_cols.shape[1]
    oof = np.zeros((len(REL_ALPHAS), len(mp_of), P), dtype=np.float32)
    Y = y_cols[mp_of]  # [S, P]
    for te_mp in scheme_folds:
        te = np.isin(mp_of, te_mp) & stmt_ok
        tr = ~np.isin(mp_of, te_mp) & stmt_ok
        mu = X[tr].mean(0)
        U, s, Vt = svd(X[tr] - mu, full_matrices=False, lapack_driver="gesdd")
        proj = (X[te] - mu) @ Vt.T
        ytr = Y[tr]
        ym = ytr.mean(0)
        UtY = U.T @ (ytr - ym)
        for a, c in enumerate(REL_ALPHAS):
            lam = c * np.mean(s ** 2)
            oof[a][te] = proj @ ((s / (s ** 2 + lam))[:, None] * UtY) + ym
    scores = np.zeros((len(REL_ALPHAS), P))
    cnt = np.bincount(mp_of[stmt_ok], minlength=n_mp).astype(float)
    ok_mp = valid_mp & (cnt > 0)
    for a in range(len(REL_ALPHAS)):
        agg = np.zeros((n_mp, P))
        np.add.at(agg, mp_of[stmt_ok], oof[a][stmt_ok])
        agg = agg[ok_mp] / cnt[ok_mp][:, None]
        yy = y_cols[ok_mp]
        if scheme == "within":
            pm = pd.DataFrame(agg).groupby(party_mp[ok_mp]).transform("mean").values
            keep = within_mask[ok_mp]
            scores[a] = rank_corr_cols((agg - pm)[keep], yy[keep])
        else:
            scores[a] = rank_corr_cols(agg, yy)
    return scores


def target_columns(y, party, valid, scheme, n_perm, seed=0):
    """[n_mp, 1+n_perm] con el objetivo real y sus permutaciones (dentro del partido en `within`)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    cols = [y.copy()]
    idx = np.where(valid)[0]
    groups = [idx[party[idx] == p] for p in np.unique(party[idx])] if scheme == "within" else [idx]
    for _ in range(n_perm):
        yp = y.copy()
        for g in groups:
            yp[g] = y[rng.permutation(g)]
        cols.append(yp)
    Y = np.stack(cols, 1)
    if scheme == "within":
        for j in range(Y.shape[1]):
            m = pd.Series(np.where(valid, Y[:, j], np.nan)).groupby(party).transform("mean").values
            Y[:, j] = Y[:, j] - m
    return np.nan_to_num(Y)


def load_source(source, model, mp):
    if source == "anon":
        d = pickle.load(open(BASE_DIR / f"results/caa/{model}_statement_acts.pkl", "rb"))
        return d["acts"], d["mp_idx"]
    fdir = BASE_DIR / "results/replication/v2_por_declaracion/residual"
    feats, _ = pickle.load(open(fdir / f"{model}_politician_features.pkl", "rb"))
    meta = pickle.load(open(fdir / f"{model}_politicians_metadata.pkl", "rb"))["metadata"]
    name2i = {n: i for i, n in enumerate(mp["Name"])}
    return feats.astype(np.float16), np.array([name2i[m["name"]] for m in meta])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["anon", "decl_v2"], default="anon")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--targets", nargs="+", default=["p6", "bcall_d1", "bcall_d2", "gov_apoyo"])
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=12)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mp = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    party = mp["party_1_P6"].fillna("IND").values
    from cpca_features_votos import map_mps
    uid = map_mps(mp["Name"].tolist())
    bpath = BASE_DIR / "data/bcall_data_chile/CHL-bcall-scores-2025.csv"
    targets = {"p6": mp["dim_ideology_P6"].astype(float).values,
               "bcall_d1": load_bcall(mp["Name"], bpath, "d1"), "bcall_d2": load_bcall(mp["Name"], bpath, "d2"),
               "gov_apoyo": gov_support(uid)}

    for model in args.models:
        acts, mp_of = load_source(args.source, model, mp)
        L = acts.shape[1]
        print(f"## {args.source} {model}: {acts.shape} diputados={len(np.unique(mp_of))}", flush=True)
        # reindexar diputados a 0..n-1 sobre los 156 (mp_of ya indexa el parquet)
        rows, null_rows = [], []
        for tname in args.targets:
            y = targets[tname]
            has = np.zeros(len(mp), bool)
            has[np.unique(mp_of)] = True
            valid = ~np.isnan(y) & has
            cnt_party = pd.Series(party[valid]).value_counts()
            within_mask = valid & np.array([cnt_party.get(p, 0) >= 2 for p in party])
            for scheme in SCHEMES:
                folds = make_folds(party, valid, "lpo" if scheme == "lpo" else "random")
                Y = target_columns(y, party, valid, scheme, args.n_perm)
                res = Parallel(n_jobs=args.n_jobs)(
                    delayed(job)(acts[:, l].astype(np.float32), mp_of, valid, party, Y, folds, within_mask, scheme)
                    for l in range(L))
                S = np.stack(res)  # [L, A, P]
                obs = S[:, :, 0]
                null_max = S[:, :, 1:].reshape(-1, S.shape[2] - 1).max(0)
                li, ai = np.unravel_index(np.argmax(obs), obs.shape)
                p = float((1 + np.sum(null_max >= obs[li, ai])) / (1 + args.n_perm))
                print(f"{tname:10s} {scheme:7s} n={int(valid.sum())} max={obs[li, ai]:+.3f} capa={li} "
                      f"alpha_rel={REL_ALPHAS[ai]} p={p:.3f} nulo95={np.percentile(null_max, 95):.3f}", flush=True)
                for l in range(L):
                    for a in range(len(REL_ALPHAS)):
                        rows.append(dict(fuente=args.source, modelo=model, target=tname, esquema=scheme,
                                         capa=l, alpha_rel=REL_ALPHAS[a], spearman=obs[l, a]))
                null_rows.append(dict(fuente=args.source, modelo=model, target=tname, esquema=scheme,
                                      n_mp=int(valid.sum()), max_spearman=obs[li, ai], capa=li,
                                      alpha_rel=REL_ALPHAS[ai], p_perm=p, nulo_p95=float(np.percentile(null_max, 95))))
                pd.DataFrame(rows).to_csv(OUT_DIR / f"capas_{args.source}_{model}.csv", index=False)
                pd.DataFrame(null_rows).to_csv(OUT_DIR / f"resumen_{args.source}_{model}.csv", index=False)


if __name__ == "__main__":
    main()
