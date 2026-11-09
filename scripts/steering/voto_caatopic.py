"""Test de voto con la direccion que SI mueve el slant (CAA con control de tema), GPU.

Pregunta: la direccion caatopic (derecha - izquierda, capas top-4 por AUC de CAA) que cambia la postura de los textos (slant v3),
¿cambia tambien los votos en el sentido politico esperado? Se suma alpha * diff_norm_l * v_l (v_l unitaria) en esas capas, en todos
los tokens del prompt de voto (mismo hook que oraculo_steering), a TODOS los diputados con el mismo alpha (sin P6 individual).
Direcciones: caatopic, persona (derecha - izquierda por prompts), 3 aleatorias en las mismas capas y escala.

score = logit EN CONTRA - logit A FAVOR. delta = score(alpha) - score(0).
Orientacion de cada celda partido x proyecto: r_c = corr(P6 estandarizado, voto EN CONTRA) entre diputados de OTROS partidos en el
mismo proyecto (>0: la derecha vota mas EN CONTRA). Metricas por condicion (IC95% por bootstrap de proyectos):
  beta_or : pendiente de delta/alpha sobre r_c (ponderada por tareas): si caatopic empuja "a la derecha", delta sube donde r_c > 0
  media_or: media de delta/alpha * signo(r_c) (logits por unidad de alpha en el sentido esperado)
  media_delta: desplazamiento medio de EN CONTRA (efecto global)
  auc_celda_steer: AUC medio dentro de celdas partido x proyecto (junto con el base) para ver si el steering rompe la prediccion.

Uso: CUDA_VISIBLE_DEVICES=1 uv run python scripts/steering/voto_caatopic.py --model meta-llama/Llama-3.1-8B-Instruct --alphas 2 6
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analisis"))
from ablacion_direcciones import chat, first_id, layer_modules, orthonormal, persona_directions  # noqa: E402
from cpca_features_votos import map_mps  # noqa: E402
from oraculo_steering import Steerer, cell_auc, scores  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "results/voto_caatopic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[2.0, 6.0])
    ap.add_argument("--n-rand", type=int, default=3)
    ap.add_argument("--n-tasks", type=int, default=0, help="0 = todas (piloto: 300)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--tag", default="")
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

    mp = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    uid = map_mps(mp["Name"].tolist())
    p6 = dict(zip(uid, mp["dim_ideology_P6"].astype(float)))
    t = pd.read_parquet(BASE_DIR / "results/voting_predictions/residual/divided/voting_tasks.parquet")
    t["p6"] = t.mp_uid.map(p6)
    t = t[t.p6.notna()].reset_index(drop=True)
    pm = mp.assign(p6=mp["dim_ideology_P6"].astype(float)).dropna(subset=["p6"])
    zf = ((t.p6 - pm.p6.mean()) / pm.p6.std()).values
    y = (t.voto_real == "EN CONTRA").astype(int).values
    cells = (t.partido + "|" + t.ley).values
    bill = t.ley.values
    r_cell = {}
    for c in np.unique(cells):
        p, b = c.split("|")[0], c.split("|")[1]
        o = (bill == b) & (t.partido.values != p)
        r_cell[c] = float(np.corrcoef(zf[o], y[o])[0, 1]) if o.sum() > 5 and zf[o].std() > 0 and y[o].std() > 0 else 0.0
    r = np.array([r_cell[c] for c in cells])
    keep = np.arange(len(t))
    if args.n_tasks:
        keep = np.random.RandomState(0).choice(len(t), min(args.n_tasks, len(t)), replace=False)
    prompts = [chat(tok, p, prefix="VOTO:") for p in t.prompt.values[keep]]
    ids = [first_id(tok, " EN"), first_id(tok, " A")]
    assert ids[0] != ids[1]
    yk, ck, bk, rk = y[keep], cells[keep], bill[keep], r[keep]
    print(f"{len(keep)} tareas, {len(np.unique(ck))} celdas, {len(np.unique(bk))} proyectos", flush=True)

    caa = pickle.load(open(BASE_DIR / f"results/caa/{safe}_caatopic_directions.pkl", "rb"))["layers"]
    top = sorted(sorted(caa, key=lambda l: caa[l]["auc_cv"], reverse=True)[:4])
    scale = {l: float(caa[l]["diff_norm"]) for l in top}
    pers = persona_directions(model, tok, list(range(L)), args.bs)
    dirs = {"caatopic": {l: caa[l]["vector"] / np.linalg.norm(caa[l]["vector"]) for l in top},
            "persona": {l: orthonormal([pers[l]])[0] for l in top}}
    for sd in range(args.n_rand):
        dirs[f"random_s{sd}"] = {l: orthonormal(np.random.RandomState(2000 + sd).randn(1, dirs["caatopic"][l].shape[0]))[0]
                                  for l in top}
    print("capas", top, "diff_norm", {l: round(s, 3) for l, s in scale.items()}, flush=True)

    st = Steerer(layers)
    st.set({})
    s0 = scores(model, tok, st, prompts, np.zeros(len(prompts), np.float32), ids, args.bs)
    base_auc = cell_auc(s0, yk, ck)[0]
    print(f"base: AUC celda {base_auc:.3f}; media score {s0.mean():.2f}", flush=True)

    def boot_ci(d, a, n=300):
        ub = np.unique(bk)
        by = {b: np.where(bk == b)[0] for b in ub}
        rng = np.random.RandomState(0)
        out = []
        for _ in range(n):
            idx = np.concatenate([by[b] for b in rng.choice(ub, len(ub))])
            dd, rr = d[idx] / a, rk[idx]
            out.append(((dd * rr).sum() / (rr ** 2).sum(), (dd * np.sign(rr)).mean()))
        return np.percentile(out, [2.5, 97.5], axis=0)

    rows = [dict(direccion="base", alpha=0.0, auc_celda_steer=base_auc)]
    conds = [(name, s * a) for name in dirs for a in args.alphas for s in (+1, -1)]
    for name, a in tqdm(conds, desc="condiciones"):
        st.set({l: scale[l] * dirs[name][l] for l in top})
        sc = scores(model, tok, st, prompts, np.full(len(prompts), a, np.float32), ids, args.bs)
        d = sc - s0
        dd = d / a
        ci = boot_ci(d, a)
        row = dict(direccion=name, alpha=a, media_delta=float(d.mean()), media_abs_delta=float(np.abs(d).mean()),
                   beta_or=float((dd * rk).sum() / (rk ** 2).sum()), beta_lo=ci[0][0], beta_hi=ci[1][0],
                   media_or=float((dd * np.sign(rk)).mean()), or_lo=ci[0][1], or_hi=ci[1][1],
                   auc_celda_steer=cell_auc(sc, yk, ck)[0], flips_pct=100 * float(((sc > 0) != (s0 > 0)).mean()))
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / f"resultados_{safe}{args.tag}.csv", index=False)
    st.set({})
    print("VOTO_CAATOPIC TERMINADO", flush=True)


if __name__ == "__main__":
    main()
