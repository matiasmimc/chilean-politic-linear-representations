"""Prediccion de votos a nivel partido x proyecto con leave-bill-out anidado (CPU).

Pregunta: ¿se puede estimar mejor como vota cada partido en un proyecto NO visto, solo con rasgos del proyecto conocidos
antes de votar (iniciativa Mensaje/Mocion, texto) y la historia de votos del partido en otros proyectos? Es el cuello de botella
de la prediccion global (techo partido x proyecto vs. partido solo).

Para cada proyecto retenido b y cada proyecto de entrenamiento b' != b, los rasgos de (partido, b') se calculan sin b ni b'
(sin fuga). Segunda etapa: regresion logistica sobre celdas partido x proyecto (peso = n diputados, clases balanceadas).
Rasgos: tasa EN CONTRA del partido (a), en proyectos de la misma iniciativa (b), ponderada por similitud de texto (kNN sobre
embeddings de bills, c), y EN CONTRA media de los vecinos (dificultad del proyecto, d). Metricas a nivel diputado, en todos
los proyectos retenidos: exactitud balanceada y AUC; referencias: partido solo, techo partido x proyecto (leave-MP-out).

Uso: uv run python scripts/analisis/votos_partido_proyecto.py
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent.parent
OUT = BASE / "results/votos_partido_proyecto"
SM = 2.0  # suavizado hacia la tasa global


def bal_acc(y, p):
    return 100 * np.mean([(p[y == c] == c).mean() for c in (0, 1)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(BASE / "results/voting_predictions/residual/divided/voting_tasks.parquet")
    t["y"] = (t.voto_real == "EN CONTRA").astype(int)
    conn = sqlite3.connect(BASE / "data/parlamento.db")
    bills = pd.read_sql("select bill_id, titulo, iniciativa from bills", conn)
    bills = bills[bills.titulo.isin(t.ley.unique())].drop_duplicates("titulo").set_index("titulo")
    emb = {r[0]: np.array(json.loads(r[1])) for r in conn.execute("select bill_id, embedding from bill_embeddings")}
    t["bill"] = t.ley.map(bills.bill_id)
    B = sorted(t.bill.unique())
    P = sorted(t.partido.unique())
    bi, pi = {b: i for i, b in enumerate(B)}, {p: i for i, p in enumerate(P)}
    nB, nP = len(B), len(P)
    EC = np.zeros((nP, nB))
    N = np.zeros((nP, nB))
    for (p, b), g in t.groupby(["partido", "bill"]):
        EC[pi[p], bi[b]], N[pi[p], bi[b]] = g.y.sum(), len(g)
    mens = np.array([float(bills.set_index("bill_id").loc[b, "iniciativa"] == "Mensaje") for b in B])
    E = np.stack([emb[b] / np.linalg.norm(emb[b]) for b in B])
    S = E @ E.T
    print(f"{len(t)} votos, {nB} proyectos ({int(mens.sum())} Mensaje), {nP} partidos", flush=True)

    def feats(b, train):
        """Rasgos [nP, 4] del proyecto b usando solo los proyectos `train` (indices)."""
        tr = np.array(train)
        g = EC[:, tr].sum() / N[:, tr].sum()
        ra = (EC[:, tr].sum(1) + SM * g) / (N[:, tr].sum(1) + SM)
        same = tr[mens[tr] == mens[b]]
        rb = (EC[:, same].sum(1) + SM * g) / (N[:, same].sum(1) + SM) if len(same) else ra
        k = tr[np.argsort(-S[b, tr])[:6]]
        w = np.exp(S[b, k] * 10)
        w = w / w.sum()
        rc = ((EC[:, k] * w).sum(1) + SM * ra) / ((N[:, k] * w).sum(1) + SM)
        d = np.full(nP, (EC[:, k].sum() / N[:, k].sum()))
        lg = lambda x: np.log(np.clip(x, 1e-3, 1 - 1e-3) / (1 - np.clip(x, 1e-3, 1 - 1e-3)))
        return np.column_stack([lg(ra), lg(rb), lg(rc), lg(d)])

    specs = {"partido (a)": [0], "partido + iniciativa (a,b)": [0, 1], "+ vecinos de texto (a,b,c)": [0, 1, 2],
             "+ dificultad del proyecto (a,b,c,d)": [0, 1, 2, 3]}
    rows, preds = [], {k: np.zeros(len(t)) for k in specs}
    for b in range(nB):
        others = [x for x in range(nB) if x != b]
        Xtr, ytr, wtr = [], [], []
        for b2 in others:
            f = feats(b2, [x for x in others if x != b2])
            for p in range(nP):
                if N[p, b2] == 0:
                    continue
                Xtr += [f[p], f[p]]
                ytr += [1, 0]
                wtr += [EC[p, b2], N[p, b2] - EC[p, b2]]
        Xtr, ytr, wtr = np.array(Xtr), np.array(ytr), np.array(wtr)
        keep = wtr > 0
        fb = feats(b, others)
        m = (t.bill == B[b]).values
        for name, cols in specs.items():
            sc = StandardScaler().fit(Xtr[keep][:, cols])
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
            clf.fit(sc.transform(Xtr[keep][:, cols]), ytr[keep], sample_weight=wtr[keep])
            pcell = clf.predict_proba(sc.transform(fb[:, cols]))[:, 1]
            preds[name][m] = pcell[t.partido[m].map(pi).values]
    y = t.y.values
    res = []
    for name, p in preds.items():
        res.append(dict(modelo=name, bal_acc=bal_acc(y, (p > 0.5).astype(int)), auc=100 * roc_auc_score(y, p),
                        auc_medio_por_proyecto=100 * np.mean([roc_auc_score(y[t.bill == b], p[(t.bill == b).values])
                                                              for b in B if len(np.unique(y[t.bill == b])) == 2])))
    # techo: mayoria partido x proyecto dejando fuera al propio diputado
    s = t.groupby(["partido", "bill"]).y.transform("sum")
    n = t.groupby(["partido", "bill"]).y.transform("count")
    loo = (s - t.y) / (n - 1).replace(0, np.nan)
    ok = loo.notna()
    res.append(dict(modelo="techo: mayoria partido x proyecto (leave-MP-out)",
                    bal_acc=bal_acc(y[ok], (loo[ok] > 0.5).astype(int).values), auc=100 * roc_auc_score(y[ok], loo[ok]),
                    auc_medio_por_proyecto=np.nan))
    r = pd.DataFrame(res)
    r.to_csv(OUT / "resumen.csv", index=False)
    pd.DataFrame({k: v for k, v in preds.items()} | {"mp_uid": t.mp_uid, "bill": t.bill, "partido": t.partido, "y": y}).to_parquet(
        OUT / "predicciones.parquet")
    print(r.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
