"""Direcciones CAA con control de tema: contraste derecha-izquierda DENTRO de cada tema.

1. Cada intervencion se asigna a un tema con TF-IDF + SVD + KMeans (K temas, no supervisado).
2. Por capa: v = sum_t w_t * (media_derecha_t - media_izquierda_t) / sum_t w_t, con w_t = media armonica de los
   conteos de ambos lados (solo temas con >= 3 intervenciones por lado en el entrenamiento).
3. Validacion: CV 5-fold sobre diputados extremos (la direccion se construye sin los retenidos); proyeccion de cada
   intervencion RESTANDO la media de su tema; AUC por diputado. Spearman con P6 en el 50% central (nunca usado).
   Se calcula tambien la version sin control (contraste global, sin centrar por tema) para comparar.
Salida: results/caa/<modelo>_caatopic_directions.pkl (mismo formato que caa_directions.pkl).

Uso: uv run python scripts/caa/construir_direcciones_caa_tema.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "results/caa"
MODEL_CACHE = BASE_DIR / "model"


def stmt_texts(df, tok, min_tokens=20, max_tokens=350):
    """Textos en el mismo orden y con el mismo filtro que extraer_intervenciones_caa.py."""
    out = []
    for mp_i, row in enumerate(df.itertuples()):
        for t in row.intervenciones:
            if len(tok(str(t), add_special_tokens=False)["input_ids"][:max_tokens]) >= min_tokens:
                out.append((mp_i, str(t)))
    return out


def topic_contrast(acts, topic, side, train_mask, n_topics, min_per_side=3):
    """acts [n,L,d] (float32). Devuelve (v [L,d] sin normalizar, dif global [L,d])."""
    num, den = 0.0, 0.0
    for t in range(n_topics):
        r = train_mask & (topic == t) & (side == 1)
        l = train_mask & (topic == t) & (side == 0)
        if r.sum() < min_per_side or l.sum() < min_per_side:
            continue
        w = 2 * r.sum() * l.sum() / (r.sum() + l.sum())
        num = num + w * (acts[r].mean(0) - acts[l].mean(0))
        den += w
    glob = acts[train_mask & (side == 1)].mean(0) - acts[train_mask & (side == 0)].mean(0)
    return num / den, glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--q", type=float, default=0.25)
    ap.add_argument("--topics", type=int, default=20)
    args = ap.parse_args()
    safe = args.model.replace("/", "_")

    df = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    d = pickle.load(open(OUT_DIR / f"{safe}_statement_acts.pkl", "rb"))
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=str(MODEL_CACHE))
    st = stmt_texts(df, tok)
    assert len(st) == len(d["acts"]) and np.array_equal(np.array([m for m, _ in st]), d["mp_idx"]), \
        f"desalineado: {len(st)} textos vs {len(d['acts'])} activaciones"
    texts, mp_idx = [t for _, t in st], d["mp_idx"]
    p6 = d["p6"]

    # temas
    tf = TfidfVectorizer(max_df=0.4, min_df=5, sublinear_tf=True, strip_accents="unicode", max_features=20000)
    X = normalize(TruncatedSVD(100, random_state=0).fit_transform(tf.fit_transform(texts)))
    topic = KMeans(args.topics, n_init=5, random_state=0).fit_predict(X)

    lo, hi = np.quantile(p6, [args.q, 1 - args.q])
    mp_side = np.where(p6 >= hi, 1, np.where(p6 <= lo, 0, -1))   # -1 = central
    side = mp_side[mp_idx]
    ext_mp = np.where(mp_side >= 0)[0]
    mid_mp = np.where(mp_side < 0)[0]
    tab = pd.crosstab(topic[side >= 0], side[side >= 0])
    print(f"{safe}: {len(texts)} intervenciones; temas con ambos lados >=3: {int(((tab >= 3).all(axis=1)).sum())}/{args.topics}")

    acts = d["acts"].astype(np.float32)            # [n, L, dim]
    L = acts.shape[1]
    tmean = np.stack([acts[topic == t].mean(0) if (topic == t).any() else np.zeros(acts.shape[1:]) for t in range(args.topics)])
    centered = acts - tmean[topic]                 # restar la media del tema

    def eval_dir(U, X_eval_acts, mp_ids):
        """Proyeccion media por diputado para cada capa: U [L,d]."""
        proj = np.einsum("nld,ld->nl", X_eval_acts, U)
        return np.stack([proj[mp_idx == m].mean(0) for m in mp_ids])      # [n_mp, L]

    y_ext = mp_side[ext_mp]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(ext_mp, y_ext))
    sc_topic, sc_glob = np.zeros((len(ext_mp), L)), np.zeros((len(ext_mp), L))
    for tr, te in folds:
        train = np.isin(mp_idx, ext_mp[tr])
        v, g = topic_contrast(acts, topic, side, train, args.topics)
        u = v / np.linalg.norm(v, axis=1, keepdims=True); ug = g / np.linalg.norm(g, axis=1, keepdims=True)
        sc_topic[te] = eval_dir(u, centered, ext_mp[te])
        sc_glob[te] = eval_dir(ug, acts, ext_mp[te])
    auc_t = np.array([roc_auc_score(y_ext, sc_topic[:, l]) for l in range(L)])
    auc_g = np.array([roc_auc_score(y_ext, sc_glob[:, l]) for l in range(L)])

    train_all = np.isin(mp_idx, ext_mp)
    v, g = topic_contrast(acts, topic, side, train_all, args.topics)
    u = v / np.linalg.norm(v, axis=1, keepdims=True); ug = g / np.linalg.norm(g, axis=1, keepdims=True)
    mid_t, mid_g = eval_dir(u, centered, mid_mp), eval_dir(ug, acts, mid_mp)
    rho_t = np.array([spearmanr(mid_t[:, l], p6[mid_mp]).statistic for l in range(L)])
    rho_g = np.array([spearmanr(mid_g[:, l], p6[mid_mp]).statistic for l in range(L)])

    res = pd.DataFrame({"layer": range(L), "auc_tema": auc_t, "auc_global": auc_g, "rho_medio_tema": rho_t,
                        "rho_medio_global": rho_g, "diff_norm_tema": np.linalg.norm(v, axis=1)})
    res.to_csv(OUT_DIR / f"{safe}_caatopic_validacion.csv", index=False)
    pickle.dump({"layers": {l: {"vector": u[l].astype(np.float32), "diff_norm": float(np.linalg.norm(v[l])),
                                "auc_cv": float(auc_t[l]), "rho_medio": float(rho_t[l])} for l in range(L)},
                 "q": args.q, "topics": args.topics}, open(OUT_DIR / f"{safe}_caatopic_directions.pkl", "wb"))
    pd.set_option("display.width", 200)
    print(res.sort_values("auc_tema", ascending=False).head(8).round(3).to_string(index=False))
    print(f"AUC max: con control {auc_t.max():.3f} | sin control {auc_g.max():.3f}")
    print(f"rho_medio: con control mediana {np.median(rho_t):.2f} max {rho_t.max():.2f} | sin control mediana {np.median(rho_g):.2f} max {rho_g.max():.2f}")
    terms = np.array(tf.get_feature_names_out())
    print("Temas (tamano, lado derecho %):", [(int((topic == t).sum()), round(float(side[topic == t].mean()), 2)) for t in range(args.topics)][:10], "...")


if __name__ == "__main__":
    main()
