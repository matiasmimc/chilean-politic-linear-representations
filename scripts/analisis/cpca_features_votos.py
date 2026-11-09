"""A2 (CPU): ¿las proyecciones cPCA aportan al voto mas alla del partido? (sin steering)

Features por diputado: proyecciones de sus activaciones sobre las k direcciones cPCA top (las mismas
que usa el steering de votos: `all_components` ordenados por autovalor) frente a direcciones aleatorias
en las mismas capas (N_RAND semillas) y frente a P6 (oraculo).

  leave-bill-out : logistica sobre [partido] vs [z] vs [partido + z] (sin informacion del proyecto)
  leave-MP-out   : base = proyecto + partido x proyecto (el "techo"); + z x proyecto (pendiente por proyecto)

Metricas: exactitud balanceada, AUC agrupado, AUC medio dentro de proyecto.
Tareas: proyectos divididos (frac. media de votos en contra >= 0.25), todos los diputados disponibles.

Uso: OMP_NUM_THREADS=1 uv run python scripts/analisis/cpca_features_votos.py [--models ...]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import difflib
import pickle
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data/parlamento.db"
REP = BASE_DIR / "results/replication"
CPCA_DIR = BASE_DIR / "results/cpca_analysis/residual"
OUT_DIR = BASE_DIR / "results/cpca_features_votos"
MODELS = ["meta-llama_Llama-3.1-8B-Instruct", "google_gemma-4-E4B-it", "zai-org_GLM-4-9B-0414"]
SOURCES = {  # nombre -> (directorio de features, sufijo del pkl de cPCA)
    "cpca": (REP / "v2" / "residual", ""),
    "cpca_np": (REP / "v3_noparty" / "residual", "_noparty"),
}
MIN_FRAC_CONTRA = 0.25
KS = (1, 4, 16)


def build_votes():
    """DataFrame (mp_uid, bill_id, y) con y=1 si EN CONTRA; voto mayoritario por proyecto, sin empates."""
    with sqlite3.connect(DB_PATH) as conn:
        frac = dict(conn.execute("""SELECT bill_id, AVG(1.0 * en_contra_total / NULLIF(a_favor_total + en_contra_total, 0))
                                    FROM sesiones_votacion GROUP BY bill_id""").fetchall())
        rows = conn.execute("""SELECT sv.bill_id, v.mp_uid, v.voto FROM sesiones_votacion sv
                               JOIN votos_parlamentario v ON sv.sesion_votacion_id = v.sesion_votacion_id
                               JOIN bills b ON b.bill_id = sv.bill_id
                               WHERE v.voto IN ('A Favor', 'En Contra') AND b.resumen IS NOT NULL
                                 AND length(b.resumen) > 50""").fetchall()
    df = pd.DataFrame(rows, columns=["bill_id", "mp_uid", "voto"])
    df = df[df.bill_id.map(lambda b: (frac.get(b) or 0) >= MIN_FRAC_CONTRA)]
    g = df.groupby(["mp_uid", "bill_id"]).voto.agg(
        n_contra=lambda s: (s == "En Contra").sum(), n_favor=lambda s: (s == "A Favor").sum()).reset_index()
    g = g[g.n_contra != g.n_favor]
    g["y"] = (g.n_contra > g.n_favor).astype(int)
    return g[["mp_uid", "bill_id", "y"]]


def map_mps(names):
    with sqlite3.connect(DB_PATH) as conn:
        dim = pd.read_sql("SELECT mp_uid, nombre_completo FROM dim_parlamentario", conn)
    choices = dict(zip(dim.nombre_completo, dim.mp_uid))
    out = []
    for n in names:
        if n in choices:
            out.append(choices[n])
            continue
        m = difflib.get_close_matches(n, list(choices), n=1, cutoff=0.8)
        out.append(choices[m[0]] if m else -1)
    return np.array(out)


def projections(X, comps, k):
    """X [N, L, D]; comps ordenadas por autovalor. Devuelve Z [N, k] y la lista de capas."""
    Z, layers = [], []
    for c in comps[:k]:
        Z.append(X[:, c["layer"], :] @ c["vector"])
        layers.append(c["layer"])
    return np.stack(Z, 1), layers


def random_projections(X, layers, seed):
    rng = np.random.RandomState(seed)
    Z = []
    for l in layers:
        r = rng.randn(X.shape[2])
        Z.append(X[:, l, :] @ (r / np.linalg.norm(r)))
    return np.stack(Z, 1)


def zscore(Z):
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-12)


def metrics(y, p, bill):
    pred = (p >= 0.5).astype(int)
    rec = [np.mean(pred[y == c] == c) for c in (0, 1) if (y == c).any()]
    out = dict(bal_acc=100 * float(np.mean(rec)), auc=float(roc_auc_score(y, p)))
    aucs = [roc_auc_score(y[bill == b], p[bill == b]) for b in np.unique(bill)
            if len(np.unique(y[bill == b])) == 2]
    out["auc_within_bill"] = float(np.mean(aucs))
    return out


def cv_logit(X, y, groups, bill, n_splits, C):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return metrics(y, p, bill)


def onehot(*cols):
    keys = pd.Series(["|".join(map(str, r)) for r in zip(*cols)])
    return OneHotEncoder(handle_unknown="ignore").fit_transform(keys.values.reshape(-1, 1))


def eval_features(Zmp, mp_index, party, bill, y, groups_bill, groups_mp):
    """Zmp [N_mp, k] z-score por diputado. Devuelve metricas por esquema."""
    Z = zscore(Zmp)[mp_index]
    P = onehot(party)
    res = {}
    res["lbo_party"] = cv_logit(P.tocsr(), y, groups_bill, bill, 11, 1.0)
    res["lbo_z"] = cv_logit(Z, y, groups_bill, bill, 11, 1.0)
    res["lbo_party+z"] = cv_logit(sparse.hstack([P, sparse.csr_matrix(Z)]).tocsr(), y, groups_bill, bill, 11, 1.0)
    B0 = sparse.hstack([onehot(bill), onehot(party, bill)]).tocsr()
    res["lmo_bill+party_x_bill"] = cv_logit(B0, y, groups_mp, bill, 10, 1.0)
    # pendiente por proyecto: z (k cols) x one-hot(bill)
    Bh = onehot(bill).tocsr()
    inter = sparse.hstack([Bh.multiply(Z[:, [j]]) for j in range(Z.shape[1])]).tocsr()
    res["lmo_B0+z_x_bill"] = cv_logit(sparse.hstack([B0, inter]).tocsr(), y, groups_mp, bill, 10, 0.3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--n-rand", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=12)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mp = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    uid = map_mps(mp["Name"].tolist())
    print(f"MPs sin mapear: {(uid < 0).sum()} de {len(uid)}", flush=True)
    votes = build_votes()
    votes = votes[votes.mp_uid.isin(uid)]
    uid2idx = {u: i for i, u in enumerate(uid) if u >= 0}
    votes = votes[votes.mp_uid.map(lambda u: u in uid2idx)].reset_index(drop=True)
    mp_index = votes.mp_uid.map(uid2idx).values
    party = mp["party_1_P6"].fillna("IND").values[mp_index]
    bill, y = votes.bill_id.values, votes.y.values
    p6 = mp["dim_ideology_P6"].astype(float).values
    print(f"pares={len(votes)} MPs={votes.mp_uid.nunique()} proyectos={votes.bill_id.nunique()} "
          f"frac_en_contra={y.mean():.3f}", flush=True)
    # grupos: GroupKFold por proyecto / por diputado
    g_bill, g_mp = bill, votes.mp_uid.values

    rows = []
    for model in args.models:
        for src, (fdir, suffix) in SOURCES.items():
            X, _ = pickle.load(open(fdir / f"{model}_politician_features.pkl", "rb"))
            X = np.nan_to_num(X.astype(np.float32))
            comps = pickle.load(open(CPCA_DIR / f"{model}_cpca_analysis_global{suffix}.pkl", "rb"))["all_components"]
            p6_mp = np.nan_to_num(p6, nan=np.nanmean(p6))[:, None]
            base = dict(modelo=model, fuente=src)
            for name, Zmp in (("p6_oraculo", p6_mp), ):
                for k, v in eval_features(Zmp, mp_index, party, bill, y, g_bill, g_mp).items():
                    rows.append({**base, "features": name, "k": 1, "esquema": k, **v})
            for k in KS:
                Zc, layers = projections(X, comps, k)
                print(f"## {model} {src} k={k} capas={sorted(set(layers))}", flush=True)
                for esq, v in eval_features(Zc, mp_index, party, bill, y, g_bill, g_mp).items():
                    rows.append({**base, "features": "cpca", "k": k, "esquema": esq, **v})
                ctrl = Parallel(n_jobs=args.n_jobs)(
                    delayed(eval_features)(random_projections(X, layers, s), mp_index, party, bill, y, g_bill, g_mp)
                    for s in range(args.n_rand))
                for esq in ctrl[0]:
                    for s, c in enumerate(ctrl):
                        rows.append({**base, "features": "random", "k": k, "esquema": esq, "semilla": s, **c[esq]})
            pd.DataFrame(rows).to_csv(OUT_DIR / "resultados.csv", index=False)
    print(f"-> {OUT_DIR / 'resultados.csv'}")


if __name__ == "__main__":
    main()
