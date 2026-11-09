"""Steering oraculo (GPU): ¿el modelo usa una direccion de ideologia para votar distinto DENTRO de un partido?

A cada tarea de voto (prompt con nombre y partido) se le suma, en las capas medias, alpha * z_mp * sigma_l * v_l, donde
z_mp es el P6 VERDADERO del diputado (el mejor caso posible para una representacion individual), v_l la direccion y
sigma_l la desviacion entre diputados de la proyeccion media de sus declaraciones sobre v_l (unidades de variacion natural).
  z_full   : P6 estandarizado (incluye diferencias entre partidos)
  z_within : P6 centrado por partido (solo diferencias dentro de partido)
Direcciones: inlp (ridge a P6 sobre declaraciones, rango 1), persona (derecha - izquierda), random (2 semillas).

Metricas por condicion (score = logit EN CONTRA - logit A FAVOR):
  beta_dosis      : pendiente de (score - score_base) sobre z (logits por unidad de z); respuesta causal
  auc_celda_*     : AUC medio en celdas partido x proyecto con ambas clases y >= 4 diputados:
                    base (score sin steer), steer (score con steer), delta (score - base, orientado), z (referencia sin modelo).
                    La orientacion de un proyecto es el signo de corr(z_within, voto) en los OTROS partidos del proyecto.
  auc_proyecto    : AUC medio dentro de proyecto (como en la ablacion)

Uso: CUDA_VISIBLE_DEVICES=1 uv run python scripts/steering/oraculo_steering.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analisis"))
from ablacion_direcciones import chat, first_id, layer_modules, orthonormal, persona_directions  # noqa: E402
from ablacion_v2 import inlp_directions  # noqa: E402
from cpca_features_votos import map_mps  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "results/oraculo_steering"


class Steerer:
    def __init__(self, layers):
        self.layers, self.handles, self.coef = layers, [], None

    def set(self, shifts):
        """shifts: {capa: vector [d] ya escalado por sigma}. Vacio = sin steering. El coeficiente por fila va en self.coef."""
        for h in self.handles:
            h.remove()
        self.handles = []
        for l, v in shifts.items():
            vt = torch.tensor(v, dtype=torch.bfloat16, device=next(self.layers[l].parameters()).device)

            def hook(_m, _i, out, vt=vt):
                h = out[0] if isinstance(out, tuple) else out
                h = h + self.coef.to(h.device, h.dtype)[:, None, None] * vt
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self.handles.append(self.layers[l].register_forward_hook(hook))


@torch.no_grad()
def scores(model, tok, st, prompts, coef, ids, bs):
    order = np.argsort([len(p) for p in prompts])
    out = np.zeros(len(prompts), dtype=np.float32)
    for i in range(0, len(prompts), bs):
        idx = order[i:i + bs]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
        st.coef = torch.tensor(coef[idx], dtype=torch.float32)
        lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), -1)[:, ids].cpu().numpy()
        out[idx] = lp[:, 0] - lp[:, 1]
    return out


def cell_auc(s, y, cells, orient=None, min_n=4):
    """AUC medio sobre celdas; orient: signo por celda (si no None, AUC -> 1-AUC cuando el signo es negativo)."""
    a = []
    for c in np.unique(cells):
        m = cells == c
        if m.sum() >= min_n and len(np.unique(y[m])) == 2:
            v = roc_auc_score(y[m], s[m])
            a.append(1 - v if (orient is not None and orient[c] < 0) else v)
    return float(np.mean(a)) if a else np.nan, len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tasks", type=int, default=0, help="0 = todas")
    ap.add_argument("--alphas", type=float, nargs="+", default=[3.0, 8.0])
    ap.add_argument("--n-rand", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_")

    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=str(BASE_DIR / "model"))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=str(BASE_DIR / "model"), low_cpu_mem_usage=True,
                                                 torch_dtype=torch.bfloat16, device_map="auto").eval()
    layers = layer_modules(model)
    L = len(layers)
    mid = sorted(set(round(L * f) for f in (0.375, 0.4, 0.44, 0.47, 0.5)))

    # --- tareas y P6 verdadero por diputado
    mp = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    uid = map_mps(mp["Name"].tolist())
    p6 = dict(zip(uid, mp["dim_ideology_P6"].astype(float)))
    t = pd.read_parquet(BASE_DIR / "results/voting_predictions/residual/divided/voting_tasks.parquet")
    t["p6"] = t.mp_uid.map(p6)
    t = t[t.p6.notna()].reset_index(drop=True)
    if args.n_tasks:
        t = t.sample(n=min(args.n_tasks, len(t)), random_state=0).reset_index(drop=True)
    pm = mp.assign(uid=uid, p6=mp["dim_ideology_P6"].astype(float)).dropna(subset=["p6"])
    zf = ((t.p6 - pm.p6.mean()) / pm.p6.std()).values
    party_mean = t.groupby("partido").p6.transform("mean")
    zw = ((t.p6 - party_mean) / (t.p6 - party_mean).std()).values
    y = (t.voto_real == "EN CONTRA").astype(int).values
    cells = (t.partido + "|" + t.ley).values
    bill = t.ley.values
    print(f"{len(t)} tareas, {t.mp_uid.nunique()} diputados, {len(np.unique(cells))} celdas", flush=True)

    # orientacion por celda: signo de corr(z_within, voto) en los otros partidos del proyecto
    orient = {}
    for c in np.unique(cells):
        b, p = c.split("|")[1], c.split("|")[0]
        o = (bill == b) & (t.partido.values != p)
        orient[c] = np.sign(np.corrcoef(zw[o], y[o])[0, 1]) if o.sum() > 5 and zw[o].std() > 0 and y[o].std() > 0 else 1.0

    prompts = [chat(tok, p, prefix="VOTO:") for p in t.prompt]
    ids = [first_id(tok, " EN"), first_id(tok, " A")]
    assert ids[0] != ids[1]

    # --- direcciones y escala
    d_stmt = pickle.load(open(BASE_DIR / f"results/caa/{safe}_statement_acts.pkl", "rb"))
    y_stmt = mp["dim_ideology_P6"].astype(float).values[d_stmt["mp_idx"]]
    pers = persona_directions(model, tok, list(range(L)), args.bs)
    dirs = {"inlp": {}, "persona": {}}
    for sd in range(args.n_rand):
        dirs[f"random_s{sd}"] = {}
    for l in mid:
        A = d_stmt["acts"][:, l].astype(np.float32)
        dirs["inlp"][l] = inlp_directions(A, y_stmt, 1)[0]
        dirs["persona"][l] = orthonormal([pers[l]])[0]
        for sd in range(args.n_rand):
            dirs[f"random_s{sd}"][l] = orthonormal(np.random.RandomState(1000 + sd).randn(1, A.shape[1]))[0]
    sigma = {}
    for name, dd in dirs.items():
        sigma[name] = {}
        for l in mid:
            A = d_stmt["acts"][:, l].astype(np.float32) @ dd[l]
            mean_mp = pd.Series(A).groupby(d_stmt["mp_idx"]).mean()
            sigma[name][l] = float(mean_mp.std())
    print("sigma inlp:", {l: round(s, 3) for l, s in sigma["inlp"].items()}, flush=True)

    st = Steerer(layers)
    st.set({})
    s0 = scores(model, tok, st, prompts, np.zeros(len(t), np.float32), ids, args.bs)
    ref = dict(auc_celda_z=cell_auc(zw, y, cells, orient)[0], auc_celda_base=cell_auc(s0, y, cells)[0],
               n_celdas=cell_auc(s0, y, cells)[1],
               auc_proyecto_base=float(np.mean([roc_auc_score(y[bill == b], s0[bill == b]) for b in np.unique(bill)
                                                if len(np.unique(y[bill == b])) == 2])))
    print("base", ref, flush=True)

    rows, raw = [dict(direccion="base", z="-", alpha=0, **ref)], {"base": s0}
    conds = [(name, zt, a) for name in dirs for zt in ("full", "within") for a in args.alphas]
    for name, zt, a in tqdm(conds, desc="condiciones"):
        z = zf if zt == "full" else zw
        st.set({l: sigma[name][l] * dirs[name][l] for l in mid})
        s = scores(model, tok, st, prompts, (a * z).astype(np.float32), ids, args.bs)
        d = s - s0
        r = dict(direccion=name, z=zt, alpha=a, beta_dosis=float((d * z).mean() / (z * z).mean()) / a,
                 corr_delta_z=float(np.corrcoef(d, z)[0, 1]), media_delta=float(d.mean()),
                 auc_celda_steer=cell_auc(s, y, cells)[0], auc_celda_delta=cell_auc(d, y, cells, orient)[0],
                 auc_celda_base=ref["auc_celda_base"], auc_celda_z=ref["auc_celda_z"],
                 auc_proyecto=float(np.mean([roc_auc_score(y[bill == b], s[bill == b]) for b in np.unique(bill)
                                             if len(np.unique(y[bill == b])) == 2])))
        rows.append(r)
        raw[(name, zt, a)] = s
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / f"resultados_{safe}.csv", index=False)
    st.set({})
    pickle.dump({"raw": raw, "y": y, "cells": cells, "zf": zf, "zw": zw}, open(OUT_DIR / f"raw_{safe}.pkl", "wb"))


if __name__ == "__main__":
    main()
