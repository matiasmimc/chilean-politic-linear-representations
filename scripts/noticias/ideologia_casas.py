"""¿Las representaciones aprendidas con diputados ordenan las casas de noticias por ideologia? (CPU)

Entrenamiento: declaraciones de diputados (results/caa/<m>_statement_acts.pkl, P6 como etiqueta). Aplicacion: activaciones de
articulos (extraer_activaciones_noticias.py), mismo formato. El puntaje de una casa es el promedio de los puntajes de sus articulos.
Metodos (la capa/hiperparametros se eligen solo con diputados, validacion cruzada agrupada por diputado, nunca con las casas):
  ridge       : ridge activaciones -> P6, top-n_capas capas.
  caa/caatopic: proyeccion sobre la direccion de contraste derecha-izquierda (results/caa/*_directions.pkl).
  cpca_ridge  : subespacio cPCA (C_T - alpha*C_B, fondo = textos neutrales) y ridge sobre las proyecciones (calibra signo y pesos).
  cpca_knn    : en el subespacio cPCA, P6 medio de las k declaraciones de diputados mas cercanas (coseno), activaciones recentradas.
  tfidf       : ridge sobre TF-IDF de las declaraciones (cota de lo que se explica con vocabulario).
  aleatoria   : nulo = direcciones aleatorias en las mismas capas (|rho| con las etiquetas).
Evaluacion por casa: Spearman/Pearson con el puntaje externo (data/noticias/casas_puntaje.csv: casa,puntaje; mayor = derecha) con IC
por bootstrap de articulos y p de permutacion de etiquetas; sin etiquetas igual se reporta fiabilidad (split-half) y eta2
(parte de la varianza entre casas). Variante resid_tema resta de las activaciones la parte lineal explicada por el tema (TF-IDF->SVD).

Uso: uv run python scripts/noticias/ideologia_casas.py --model meta-llama/Llama-3.1-8B-Instruct [--variantes base resid_tema]
"""
import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts/probing"))
OUT_DIR = BASE_DIR / "results/noticias"
ALPHAS = np.logspace(-2, 5, 8)


def prep(S, N, Bg, l):
    """Estandariza con la media de las declaraciones y una desviacion global (como ridge_components)."""
    mu = S[:, l].astype(np.float32).mean(0)
    sd = S[:, l].astype(np.float32).std() + 1e-8
    f = lambda A: (A[:, l].astype(np.float32) - mu) / sd
    return f(S), f(N), (f(Bg) if Bg is not None else None)


def cv_pred(X, y, g, model_fn, k=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(k).split(X, y, g):
        p[te] = model_fn().fit(X[tr], y[tr]).predict(X[te])
    return p


def dep_spearman(pred, y, g):
    d = pd.DataFrame({"p": pred, "y": y, "g": g}).groupby("g").mean()
    return spearmanr(d.p, d.y).statistic


def cpca_basis(Xs, Xb, alpha, k):
    Ct = np.cov(Xs, rowvar=False)
    Cb = np.cov(Xb, rowvar=False)
    _, V = eigh(Ct - alpha * Cb, subset_by_index=[Xs.shape[1] - k, Xs.shape[1] - 1])
    return V[:, ::-1].astype(np.float32)


def knn_pred(Zq, Zref, yref, k):
    Zq = Zq / (np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-8)
    Zref = Zref / (np.linalg.norm(Zref, axis=1, keepdims=True) + 1e-8)
    out = np.zeros(len(Zq))
    for i in range(0, len(Zq), 2000):
        sim = Zq[i:i + 2000] @ Zref.T
        out[i:i + 2000] = yref[np.argpartition(-sim, k, axis=1)[:, :k]].mean(1)
    return out


def residualize(A, Z):
    Q, _ = np.linalg.qr(np.hstack([np.ones((len(Z), 1)), Z]))
    out = np.empty_like(A)
    for l in range(A.shape[1]):
        X = A[:, l].astype(np.float32)
        out[:, l] = (X - Q.astype(np.float32) @ (Q.T.astype(np.float32) @ X)).astype(np.float16)
    return out


def outlet_table(score, casas):
    d = pd.DataFrame({"s": score, "casa": casas}).groupby("casa").s.agg(["mean", "count", "sem"])
    return d


def eta2(score, casas):
    d = pd.DataFrame({"s": score, "c": casas})
    return float(((d.groupby("c").s.transform("mean") - d.s.mean()) ** 2).sum() / ((d.s - d.s.mean()) ** 2).sum())


def split_half(score, casas, n=50, seed=0):
    rng = np.random.RandomState(seed)
    d = pd.DataFrame({"s": score, "c": casas})
    r = []
    for _ in range(n):
        h = rng.rand(len(d)) < 0.5
        a, b = d[h].groupby("c").s.mean(), d[~h].groupby("c").s.mean()
        r.append(spearmanr(a, b.reindex(a.index)).statistic)
    return float(np.nanmean(r))


def evaluate(score, casas, labels, n_boot=500, n_perm=10000, seed=0):
    """Spearman/Pearson entre el puntaje medio por casa y la etiqueta externa; IC bootstrap de articulos; p de permutacion."""
    rng = np.random.RandomState(seed)
    d = pd.DataFrame({"s": score, "c": casas})
    lab = labels.reindex(sorted(d.c.unique())).dropna()
    if len(lab) < 4:
        return {}
    d = d[d.c.isin(lab.index)]
    m = d.groupby("c").s.mean().reindex(lab.index)
    rho = spearmanr(m, lab).statistic
    groups = {c: g.s.values for c, g in d.groupby("c")}
    bs = []
    for _ in range(n_boot):
        mm = pd.Series({c: rng.choice(v, len(v)).mean() for c, v in groups.items()}).reindex(lab.index)
        bs.append(spearmanr(mm, lab).statistic)
    if len(lab) <= 8:
        null = [spearmanr(m, lab.values[list(p)]).statistic for p in itertools.permutations(range(len(lab)))]
    else:
        null = [spearmanr(m, rng.permutation(lab.values)).statistic for _ in range(n_perm)]
    return dict(n_casas=len(lab), spearman=rho, pearson=pearsonr(m, lab)[0], ic_lo=np.nanpercentile(bs, 2.5),
                ic_hi=np.nanpercentile(bs, 97.5), p_perm=float((np.sum(np.abs(null) >= abs(rho) - 1e-12)) / len(null)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", default=str(BASE_DIR / "data/noticias/casas_puntaje.csv"))
    ap.add_argument("--variantes", nargs="+", default=["base"])
    ap.add_argument("--n-capas", type=int, default=3, help="capas (top por CV con diputados) que se promedian")
    ap.add_argument("--capas-paso", type=int, default=2, help="evalua una de cada n capas")
    ap.add_argument("--k-cpca", type=int, default=8)
    ap.add_argument("--alpha-cpca", type=float, default=2.0)
    ap.add_argument("--k-knn", type=int, default=50)
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--min-art", type=int, default=20, help="descarta casas con menos articulos")
    ap.add_argument("--neutral-acts", default=None, help="por defecto results/noticias/<m>_neutral_acts.pkl")
    ap.add_argument("--news-acts", default=None, help="por defecto results/noticias/<m>_noticias_acts.pkl")
    args = ap.parse_args()
    safe = args.model.replace("/", "_")
    out_dir = OUT_DIR / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    d = pickle.load(open(BASE_DIR / f"results/caa/{safe}_statement_acts.pkl", "rb"))
    S, mp_idx = d["acts"], d["mp_idx"]
    y = d["p6"][mp_idx].astype(float)
    ok = ~np.isnan(y)
    S, y, g = S[ok], y[ok], mp_idx[ok]
    nd = pickle.load(open(args.news_acts or OUT_DIR / f"{safe}_noticias_acts.pkl", "rb"))
    N, meta = nd["acts"], nd["meta"]
    cnt = meta.casa.value_counts()
    keep = meta.casa.isin(cnt[cnt >= args.min_art].index).values
    N, meta = N[keep], meta[keep].reset_index(drop=True)
    casas = meta.casa.values
    nb = Path(args.neutral_acts) if args.neutral_acts else OUT_DIR / f"{safe}_neutral_acts.pkl"
    Bg = pickle.load(open(nb, "rb"))["acts"] if nb.exists() else None
    labels = pd.read_csv(args.labels).set_index("casa")["puntaje"] if Path(args.labels).exists() else pd.Series(dtype=float)
    print(f"{args.model}: {len(S)} declaraciones ({len(np.unique(g))} diputados), {len(N)} articulos, {meta.casa.nunique()} casas; "
          f"etiquetas: {len(labels)}; fondo cPCA: {None if Bg is None else len(Bg)}", flush=True)

    L = S.shape[1]
    layers = list(range(L // 4, L, args.capas_paso))
    mp_texts = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")["intervenciones"]
    s_texts = [str(mp_texts.iloc[m][k]) for m, k in zip(mp_idx[ok], d["stmt_k"][ok])]  # alineacion aproximada (como probes_control_tema)

    rows, scores_all = [], {}
    for var in args.variantes:
        if var == "resid_tema":
            tf = TfidfVectorizer(min_df=3, max_df=0.5, sublinear_tf=True, strip_accents="unicode")
            Z = TruncatedSVD(50, random_state=0).fit(tf.fit_transform(s_texts + meta.texto.astype(str).tolist()))
            M = tf.transform(s_texts + meta.texto.astype(str).tolist())
            Zall = Z.transform(M)
            Sv, Nv = residualize(S, Zall[:len(S)]), residualize(N, Zall[len(S):])
        else:
            Sv, Nv = S, N

        def top_layers(cv):
            return sorted(sorted(cv, key=cv.get, reverse=True)[:args.n_capas])

        def record(name, score, cv_info=""):
            scores_all[(var, name)] = score
            r = dict(variante=var, metodo=name, capas=cv_info, eta2=eta2(score, casas), split_half=split_half(score, casas))
            r.update(evaluate(score, casas, labels))
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "resumen.csv", index=False)

        # ---- ridge y cPCA (por capa)
        cv_r, pr_r, cv_c, pr_c, pr_k = {}, {}, {}, {}, {}
        for l in layers:
            Xs, Xn, Xb = prep(Sv, Nv, Bg, l)
            mk = lambda: RidgeCV(alphas=ALPHAS)
            cv_r[l] = dep_spearman(cv_pred(Xs, y, g, mk), y, g)
            pr_r[l] = mk().fit(Xs, y).predict(Xn)
            if Xb is not None:
                V = cpca_basis(Xs, Xb, args.alpha_cpca, args.k_cpca)
                Ps, Pn = Xs @ V, Xn @ V
                cv_c[l] = dep_spearman(cv_pred(Ps, y, g, mk), y, g)
                pr_c[l] = mk().fit(Ps, y).predict(Pn)
                pr_k[l] = knn_pred(Pn - Pn.mean(0) + Ps.mean(0), Ps, y, args.k_knn)
            print(f"[{var}] capa {l}: ridge cv={cv_r[l]:.2f}" + (f" cpca cv={cv_c[l]:.2f}" if l in cv_c else ""), flush=True)
        tl = top_layers(cv_r)
        record("ridge", np.mean([pr_r[l] for l in tl], 0), str(tl))
        # sensibilidad (no usar para elegir): cada capa por separado
        for l in layers:
            scores_all[(var, f"ridge_capa{l}")] = pr_r[l]
        if pr_c:
            tl = top_layers(cv_c)
            record("cpca_ridge", np.mean([pr_c[l] for l in tl], 0), str(tl))
            record("cpca_knn", np.mean([pr_k[l] for l in tl], 0), str(tl))

        # ---- CAA y CAA con control de tema
        for fname in ("caa", "caatopic"):
            p = BASE_DIR / f"results/caa/{safe}_{fname}_directions.pkl"
            if not p.exists():
                continue
            dd = pickle.load(open(p, "rb"))["layers"]
            tl = sorted(sorted(dd, key=lambda l: dd[l]["auc_cv"], reverse=True)[:args.n_capas])
            sc = []
            for l in tl:
                mu = Sv[:, l].astype(np.float32).mean(0)
                sc.append((Nv[:, l].astype(np.float32) - mu) @ dd[l]["vector"] / dd[l]["diff_norm"])
            record(fname, np.mean(sc, 0), str(tl))

        # ---- TF-IDF (solo variante base)
        if var == "base":
            tf = TfidfVectorizer(min_df=3, max_df=0.5, sublinear_tf=True, strip_accents="unicode")
            Xt = tf.fit_transform(s_texts)
            rg = RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(Xt, y)
            record("tfidf", rg.predict(tf.transform(meta.texto.astype(str))))

        # ---- nulo: direcciones aleatorias en las capas ridge elegidas
        if len(labels):
            tl = top_layers(cv_r)
            rng = np.random.RandomState(0)
            nul = []
            for _ in range(args.n_random):
                sc = []
                for l in tl:
                    v = rng.randn(Nv.shape[2])
                    v /= np.linalg.norm(v)
                    sc.append(Nv[:, l].astype(np.float32) @ v)
                r = evaluate(np.mean(sc, 0) / np.std(sc), casas, labels, n_boot=2, n_perm=2)
                if r:
                    nul.append(abs(r["spearman"]))
            if nul:
                rows.append(dict(variante=var, metodo="aleatoria", spearman=float(np.mean(nul)), ic_hi=float(np.percentile(nul, 95)),
                                 capas=f"|rho| medio y p95 de {len(nul)} direcciones"))
                print("aleatoria |rho| medio/p95:", np.mean(nul), np.percentile(nul, 95), flush=True)

    pd.DataFrame(rows).to_csv(out_dir / "resumen.csv", index=False)
    art = meta.drop(columns=["texto"]).copy()
    for (v, m), sc in scores_all.items():
        art[f"{v}|{m}"] = sc
    art.to_parquet(out_dir / "scores_articulos.parquet", index=False)
    tab = pd.concat({f"{v}|{m}": outlet_table(s, casas)["mean"] for (v, m), s in scores_all.items() if "_capa" not in m}, axis=1)
    tab["n_articulos"] = outlet_table(next(iter(scores_all.values())), casas)["count"]
    tab.to_csv(out_dir / "puntaje_por_casa.csv")
    print(tab.round(2).to_string())


if __name__ == "__main__":
    main()
