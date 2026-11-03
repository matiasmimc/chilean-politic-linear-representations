"""s(c): ¿el signo/peso de la ideologia depende del proyecto, y se puede predecir SIN usar votos?

Modelo por proyecto c:  logit P(En Contra) = a_c + s_c * z_i      (z_i = P6 estandarizado del diputado)

Comparaciones (todas leave-bill-out salvo que se indique; metricas pareadas diputado x proyecto):
  global_ideologia : un unico (a, s) ajustado en los demas proyectos
  media_partido    : tasa de En Contra del partido en los demas proyectos (sin ideologia)
  oraculo_ideologia: (a_c, s_c) ajustados con los votos del propio proyecto (CV 2-fold sobre diputados) -> techo de
                     "ideologia condicionada al proyecto" (usa votos del proyecto: NO es una prediccion)
  sc_metadata      : (a_c, s_c) predichos solo con metadatos pre-votacion (iniciativa, origen, urgencia, anio)
  sc_texto         : idem con TF-IDF de titulo+resumen
  sc_meta_texto    : ambos
  techo_partido_proyecto: voto mayoritario de los demas diputados del mismo partido en el mismo proyecto
                     (usa votos del proyecto: referencia de techo, NO es una prediccion)
No se usan a_favor_total/en_contra_total, resultado_final ni estado (post-votacion).

Validacion: --validacion bill (leave-bill-out, por defecto) o tema (leave-cluster-out: se excluyen del entrenamiento todos los
proyectos del mismo grupo tematico que el proyecto evaluado; grupos = KMeans sobre TF-IDF de titulo+resumen, --n-temas).

Uso: uv run python scripts/analisis/sc_condicionada.py [--validacion tema --n-temas 10]
"""
import argparse
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts/steering"))
import predecir_votos_steered as pv  # noqa: E402  (get_mp_uid_by_name, get_contested_bills)

OUT = BASE_DIR / "results/sc_condicionada"
OUT.mkdir(parents=True, exist_ok=True)


def load_pairs():
    df = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    df["mp_uid"] = [pv.get_mp_uid_by_name(n) for n in df["Name"]]
    df = df.dropna(subset=["mp_uid"]).copy()
    df["mp_uid"] = df["mp_uid"].astype(int)
    df["z"] = (df["dim_ideology_P6"] - df["dim_ideology_P6"].mean()) / df["dim_ideology_P6"].std()
    mp = df[["mp_uid", "z", "party_1_P6"]].drop_duplicates("mp_uid").rename(columns={"party_1_P6": "party"})
    q = """SELECT b.bill_id, v.mp_uid, v.voto FROM bills b
           JOIN sesiones_votacion sv ON b.bill_id = sv.bill_id
           JOIN votos_parlamentario v ON sv.sesion_votacion_id = v.sesion_votacion_id
           WHERE v.voto IN ('A Favor','En Contra') AND b.resumen IS NOT NULL AND length(b.resumen) > 50"""
    with sqlite3.connect(pv.DB_PATH) as conn:
        votes = pd.read_sql(q, conn)
        bills = pd.read_sql("SELECT bill_id, titulo, resumen, iniciativa, origen, urgencia, fecha_ingreso FROM bills "
                            "WHERE resumen IS NOT NULL AND length(resumen) > 50", conn)
    votes["ec"] = (votes.voto == "En Contra").astype(int)
    g = votes.groupby(["bill_id", "mp_uid"]).ec.agg(["mean", "size"]).reset_index()
    g = g[g["mean"] != 0.5]
    g["ec"] = (g["mean"] > 0.5).astype(int)
    pairs = g[["bill_id", "mp_uid", "ec"]].merge(mp, on="mp_uid")
    n = pairs.groupby("bill_id").size()
    keep = n[n >= 60].index
    return pairs[pairs.bill_id.isin(keep)].reset_index(drop=True), bills[bills.bill_id.isin(keep)].reset_index(drop=True)


def fit_bill(z, y, C=1.0):
    if len(np.unique(y)) < 2:
        return (-5.0 if y[0] == 0 else 5.0), 0.0
    m = LogisticRegression(C=C).fit(z.reshape(-1, 1), y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def bill_features(bills):
    meta = pd.DataFrame({
        "mensaje": (bills.iniciativa == "Mensaje").astype(float),
        "senado": (bills.origen == "Senado").astype(float),
        "urg_inmediata": (bills.urgencia == "Discusión inmediata").astype(float),
        "urg_suma": (bills.urgencia == "Suma").astype(float),
        "anio": pd.to_datetime(bills.fecha_ingreso).dt.year.astype(float) - 2022,
    })
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=3000, sublinear_tf=True)
    text = tf.fit_transform((bills.titulo.fillna("") + ". " + bills.resumen.fillna("")))
    return meta.values, text


def pair_metrics(df, score_col):
    y, s = df.ec.values, df[score_col].values
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    within = [roc_auc_score(g.ec, g[score_col]) for _, g in df.groupby("bill_id")
              if g.ec.nunique() == 2 and g[score_col].nunique() > 1]
    thr = np.quantile(s, 1 - y.mean())
    pred = (s > thr).astype(int)
    rec = [(pred[y == c] == c).mean() for c in (0, 1) if (y == c).any()]
    return {"auc_pooled": auc, "auc_within_bill": float(np.mean(within)) if within else np.nan,
            "bal_acc": float(np.mean(rec)) * 100, "n_pares": len(df), "n_proyectos": df.bill_id.nunique()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validacion", choices=["bill", "tema"], default="bill")
    ap.add_argument("--n-temas", type=int, default=10)
    args = ap.parse_args()
    suffix = "" if args.validacion == "bill" else f"_tema{args.n_temas}"
    pairs, bills = load_pairs()
    print(f"{len(pairs)} pares diputado x proyecto, {pairs.bill_id.nunique()} proyectos, {pairs.mp_uid.nunique()} diputados, "
          f"En Contra={pairs.ec.mean():.3f}")
    bills = bills.set_index("bill_id").loc[sorted(pairs.bill_id.unique())].reset_index()
    ids = bills.bill_id.tolist()
    meta, text = bill_features(bills)
    if args.validacion == "tema":
        Z = normalize(TruncatedSVD(50, random_state=0).fit_transform(text))
        group = KMeans(args.n_temas, n_init=10, random_state=0).fit_predict(Z)
        print("Tamano de los grupos tematicos:", np.bincount(group).tolist())
    else:
        group = np.arange(len(ids))
    group_of = dict(zip(ids, group))

    # parametros por proyecto con sus propios votos (objetivo de s(c))
    params = {}
    for b, g in pairs.groupby("bill_id"):
        params[b] = fit_bill(g.z.values, g.ec.values)
    A = np.array([params[b][0] for b in ids]); S = np.array([params[b][1] for b in ids])
    print(f"s_c real: media {S.mean():.2f}, %positivo {(S > 0.05).mean():.2f}, %negativo {(S < -0.05).mean():.2f}")

    feats = {"sc_metadata": sparse.csr_matrix(meta), "sc_texto": text,
             "sc_meta_texto": sparse.hstack([sparse.csr_matrix(meta), text]).tocsr()}
    pred_params = {k: np.zeros((len(ids), 2)) for k in feats}
    for i in range(len(ids)):
        tr = np.array([j for j in range(len(ids)) if group[j] != group[i]])
        for k, X in feats.items():
            for t, target in enumerate((A, S)):
                r = RidgeCV(alphas=np.logspace(-2, 3, 8)).fit(X[tr], target[tr])
                pred_params[k][i, t] = r.predict(X[i])[0]
    print(f"\nPredictibilidad de s_c (validacion {args.validacion}):")
    rows_sc = []
    for k in feats:
        rho = spearmanr(S, pred_params[k][:, 1]).statistic
        sign = np.mean(np.sign(pred_params[k][:, 1]) == np.sign(S))
        rows_sc.append({"features": k, "spearman_s": rho, "acierto_signo": sign})
        print(f"  {k:14s} Spearman(s_c, s_pred)={rho:.2f}  acierto de signo={sign:.2f}")
    pd.DataFrame(rows_sc).to_csv(OUT / f"predictibilidad_sc{suffix}.csv", index=False)

    # puntajes pareados
    out = pairs.copy()
    for name in ["global_ideologia", "media_partido", "oraculo_ideologia", "sc_metadata", "sc_texto", "sc_meta_texto",
                 "techo_partido_proyecto"]:
        out[name] = np.nan
    idx_of = {b: i for i, b in enumerate(ids)}
    for b, g in pairs.groupby("bill_id"):
        rest = pairs[pairs.bill_id.map(group_of) != group_of[b]]
        a_g, s_g = fit_bill(rest.z.values, rest.ec.values)
        out.loc[g.index, "global_ideologia"] = a_g + s_g * g.z.values
        pm = rest.groupby("party").ec.mean()
        out.loc[g.index, "media_partido"] = g.party.map(pm).fillna(rest.ec.mean()).values
        # oraculo: 2-fold sobre diputados dentro del proyecto
        perm = np.random.RandomState(0).permutation(len(g))
        folds = np.array_split(perm, 2)
        sc = np.zeros(len(g))
        for f in range(2):
            te, tr_ = folds[f], folds[1 - f]
            a_o, s_o = fit_bill(g.z.values[tr_], g.ec.values[tr_])
            sc[te] = a_o + s_o * g.z.values[te]
        out.loc[g.index, "oraculo_ideologia"] = sc
        for k in feats:
            a_p, s_p = pred_params[k][idx_of[b]]
            out.loc[g.index, k] = a_p + s_p * g.z.values
        # techo partido x proyecto (leave-one-out)
        pg = g.groupby("party").ec
        tot, cnt = pg.transform("sum"), pg.transform("size")
        loo = np.where(cnt > 1, (tot - g.ec) / (cnt - 1).clip(lower=1), g.ec.mean())
        out.loc[g.index, "techo_partido_proyecto"] = loo

    divided = pv.get_contested_bills(0.25)
    summary = []
    for label, sub in (("todos", out), ("divididos(>=25% en contra)", out[out.bill_id.isin(divided)])):
        for name in ["global_ideologia", "media_partido", "oraculo_ideologia", "sc_metadata", "sc_texto",
                     "sc_meta_texto", "techo_partido_proyecto"]:
            summary.append({"proyectos": label, "metodo": name, **pair_metrics(sub, name)})
    res = pd.DataFrame(summary).round(3)
    res.to_csv(OUT / f"comparacion{suffix}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n", res.to_string(index=False))
    pd.DataFrame({"bill_id": ids, "a_real": A, "s_real": S, **{f"s_{k}": v[:, 1] for k, v in pred_params.items()},
                  **{f"a_{k}": v[:, 0] for k, v in pred_params.items()}}).to_csv(OUT / f"parametros_por_proyecto{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
