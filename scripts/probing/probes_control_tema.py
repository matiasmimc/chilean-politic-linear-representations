"""Control de tema/lexico para los probes por declaracion anonimas (CPU).

Pregunta: la senal ideologica (sobre todo dentro de partido) que lee un ridge sobre las activaciones de una declaracion,
¿es mas que tema y vocabulario? Variantes (mismos folds, objetivos, esquemas y metrica que probes_por_intervencion.py):
  base       : activaciones tal cual (replica)
  lexico     : en vez de activaciones, TF-IDF -> SVD (n_lex dims). Cota de lo que se explica solo con vocabulario.
  resid_tema : activaciones con la parte lineal explicada por el tema (TF-IDF -> SVD k_tema dims) restada, por capa.
               El tema se estima sin mirar el objetivo.
  sin_pol    : solo declaraciones que no nombran partidos, coaliciones, gobierno/oposicion ni al presidente.
Si `lexico` iguala a `base` dentro de partido, la senal es vocabulario; si `resid_tema`/`sin_pol` la mantienen, no lo es.

Uso: OMP_NUM_THREADS=1 uv run python scripts/probing/probes_control_tema.py --models ...
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts/probing"))
sys.path.insert(0, str(BASE_DIR / "scripts/analisis"))
import probes_por_intervencion as P  # noqa: E402
from cpca_features_votos import map_mps  # noqa: E402

OUT_DIR = BASE_DIR / "results/probing_control_tema"
POL = re.compile(r"\b(gobierno|oposici[oó]n|oficialismo|presidente (de la rep[uú]blica|boric|pi[nñ]era)|boric|pi[nñ]era|"
                 r"chile vamos|chile podemos|apruebo dignidad|unidad por chile|frente amplio|republicano|"
                 r"udi|rn|renovaci[oó]n nacional|dem[oó]crata cristian|dc|pc|comunista|socialista|ppd|"
                 r"partido|bancada|coalici[oó]n|derecha|izquierda|centroizquierda|centroderecha|ministr[oa])\b", re.I)


def statement_texts(mp, d):
    return [str(mp.intervenciones.iloc[m][k]) for m, k in zip(d["mp_idx"], d["stmt_k"])]


def residualize(acts, Z):
    """Resta de cada capa la regresion lineal sobre Z (con intercepto). acts [S, L, d] f16 -> f16."""
    Zc = np.hstack([np.ones((len(Z), 1)), Z]).astype(np.float64)
    Q, _ = np.linalg.qr(Zc)
    out = np.empty_like(acts)
    for l in range(acts.shape[1]):
        A = acts[:, l].astype(np.float32)
        out[:, l] = (A - Q.astype(np.float32) @ (Q.T.astype(np.float32) @ A)).astype(np.float16)
    return out


def run(acts, mp_of, mp, party, targets, tag, model, schemes, n_perm, n_jobs, rows):
    L = acts.shape[1]
    has = np.zeros(len(mp), bool)
    has[np.unique(mp_of)] = True
    for tname, y in targets.items():
        valid = ~np.isnan(y) & has
        cnt_party = pd.Series(party[valid]).value_counts()
        within_mask = valid & np.array([cnt_party.get(p, 0) >= 2 for p in party])
        for scheme in schemes:
            folds = P.make_folds(party, valid, "lpo" if scheme == "lpo" else "random")
            Y = P.target_columns(y, party, valid, scheme, n_perm)
            res = Parallel(n_jobs=n_jobs)(
                delayed(P.job)(acts[:, l].astype(np.float32), mp_of, valid, party, Y, folds, within_mask, scheme)
                for l in range(L))
            S = np.stack(res)
            obs = S[:, :, 0]
            null_max = S[:, :, 1:].reshape(-1, S.shape[2] - 1).max(0)
            li, ai = np.unravel_index(np.argmax(obs), obs.shape)
            p = float((1 + np.sum(null_max >= obs[li, ai])) / (1 + n_perm))
            print(f"{tag:11s} {tname:9s} {scheme:7s} n_mp={int(valid.sum())} n_decl={len(mp_of)} max={obs[li, ai]:+.3f} "
                  f"capa={li} a={P.REL_ALPHAS[ai]} p={p:.3f} nulo95={np.percentile(null_max, 95):.3f} "
                  f"medio={obs.mean():+.3f}", flush=True)
            rows.append(dict(modelo=model, variante=tag, target=tname, esquema=scheme, n_mp=int(valid.sum()),
                             n_decl=len(mp_of), max_spearman=obs[li, ai], capa=li, alpha_rel=P.REL_ALPHAS[ai],
                             p_perm=p, nulo_p95=float(np.percentile(null_max, 95)), media_capas_alphas=float(obs.mean())))
            pd.DataFrame(rows).to_csv(OUT_DIR / f"resumen_{model}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=P.MODELS)
    ap.add_argument("--targets", nargs="+", default=["p6", "bcall_d1", "gov_apoyo"])
    ap.add_argument("--schemes", nargs="+", default=["lpo", "within"])
    ap.add_argument("--variants", nargs="+", default=["base", "lexico", "resid_tema", "sin_pol"])
    ap.add_argument("--n-lex", type=int, default=300)
    ap.add_argument("--k-tema", type=int, default=50)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=12)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mp = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    party = mp["party_1_P6"].fillna("IND").values
    uid = map_mps(mp["Name"].tolist())
    bpath = BASE_DIR / "data/bcall_data_chile/CHL-bcall-scores-2025.csv"
    all_t = {"p6": mp["dim_ideology_P6"].astype(float).values, "bcall_d1": P.load_bcall(mp["Name"], bpath, "d1"),
             "bcall_d2": P.load_bcall(mp["Name"], bpath, "d2"), "gov_apoyo": P.gov_support(uid)}
    targets = {t: all_t[t] for t in args.targets}

    for model in args.models:
        d = pickle.load(open(BASE_DIR / f"results/caa/{model}_statement_acts.pkl", "rb"))
        acts, mp_of = d["acts"], d["mp_idx"]
        texts = statement_texts(mp, d)
        assert len(texts) == len(acts)
        tfidf = TfidfVectorizer(min_df=3, max_df=0.5, sublinear_tf=True, strip_accents="unicode").fit_transform(texts)
        pol = np.array([bool(POL.search(t)) for t in texts])
        print(f"## {model} {acts.shape}; mencionan partido/gobierno: {pol.mean():.1%}", flush=True)
        rows = []
        for v in args.variants:
            if v == "base":
                A, m = acts, mp_of
            elif v == "lexico":
                A = TruncatedSVD(args.n_lex, random_state=0).fit_transform(tfidf).astype(np.float16)[:, None, :]
                m = mp_of
            elif v == "resid_tema":
                Z = TruncatedSVD(args.k_tema, random_state=0).fit_transform(tfidf)
                A, m = residualize(acts, Z), mp_of
            elif v == "sin_pol":
                A, m = acts[~pol], mp_of[~pol]
            run(A, m, mp, party, targets, v, model, args.schemes, args.n_perm, args.n_jobs, rows)


if __name__ == "__main__":
    main()
