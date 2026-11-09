"""Ablacion v2 (GPU): control positivo (persona) y subespacios de mayor rango.

Anade a `ablacion_direcciones.py`:
  * conducta "persona": el mismo prompt de voto SIN partido, con "Eres un diputado chileno de izquierda/derecha";
    brecha = media de [puntaje EN CONTRA | derecha] - [puntaje EN CONTRA | izquierda] (pareado por tarea).
    Control positivo: ablar la direccion de persona deberia reducir la brecha. Prueba funcional: ¿ablar las direcciones
    de ideologia (INLP sobre P6, cPCA) tambien la reduce?
  * subespacios de rango k por capa: inlp (proyecciones nulas iterativas de un ridge a P6 sobre activaciones por
    declaracion, orto-normales), cpca (k autovectores top de la capa) y random (k vectores ortonormales).
  * conjuntos de capas: mid (12-16), late (26-30), wide (6-30; solo rango bajo).
Conductas: votos con partido (AUC dentro de proyecto), texto izquierda/derecha (AUC), brecha de persona.

Uso: CUDA_VISIBLE_DEVICES=1 uv run python scripts/steering/ablacion_v2.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablacion_direcciones import (Ablator, PROMPT_TEXTO, build_directions, chat, first_id,  # noqa: E402
                                  layer_modules, metrics_votes, orthonormal)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_CACHE = BASE_DIR / "model"
OUT_DIR = BASE_DIR / "results/ablacion"
HEADER = re.compile(r"^Eres el diputado chileno .*? del partido .*?\.", re.M)


@torch.no_grad()
def first_token_logits(model, tok, prompts, token_ids, bs):
    order = np.argsort([len(p) for p in prompts])
    out = np.zeros((len(prompts), len(token_ids)), dtype=np.float32)
    for i in range(0, len(prompts), bs):
        idx = order[i:i + bs]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        out[idx] = torch.log_softmax(logits, -1)[:, token_ids].cpu().numpy()
    return out


def inlp_directions(acts_l, y, k, lam_rel=0.3, device="cuda"):
    """k direcciones ortonormales: ridge a y, proyectar fuera, repetir. acts_l [S, d], y [S] (NaN = descartar)."""
    ok = ~np.isnan(y)
    X = torch.tensor(acts_l[ok], dtype=torch.float32, device=device)
    X = X - X.mean(0)
    yy = torch.tensor((y[ok] - y[ok].mean()) / (y[ok].std() + 1e-8), dtype=torch.float32, device=device)
    d = X.shape[1]
    B = []
    for _ in range(k):
        G = X.T @ X
        lam = lam_rel * torch.trace(G) / d
        w = torch.linalg.solve(G + lam * torch.eye(d, device=device), X.T @ yy)
        w = w / (w.norm() + 1e-12)
        B.append(w.cpu().numpy())
        X = X - (X @ w[:, None]) @ w[None, :]
    return orthonormal(np.stack(B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-votes", type=int, default=800)
    ap.add_argument("--n-persona", type=int, default=400)
    ap.add_argument("--n-texts", type=int, default=400)
    ap.add_argument("--n-rand", type=int, default=2)
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 64])
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_")

    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=str(MODEL_CACHE))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=str(MODEL_CACHE), low_cpu_mem_usage=True,
                                                 torch_dtype=torch.bfloat16, device_map="auto").eval()
    layers = layer_modules(model)
    L = len(layers)
    abl = Ablator(layers)
    args.n_rand = max(args.n_rand, 0)
    base_dirs = build_directions(argparse.Namespace(model=args.model, n_rand=0, bs=args.bs), model, tok,
                                 list(range(L)), L)  # k=1: cpca, cpca_k4, ridge, caa, persona
    sets = {"mid": sorted(base_dirs["mid"]["cpca"]), "late": sorted(base_dirs["late"]["cpca"]),
            "wide": list(range(round(L * 0.19), round(L * 0.94)))}

    # --- subespacios de rango k
    d_stmt = pickle.load(open(BASE_DIR / f"results/caa/{safe}_statement_acts.pkl", "rb"))
    mp = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    y_stmt = mp["dim_ideology_P6"].astype(float).values[d_stmt["mp_idx"]]
    cp = pickle.load(open(BASE_DIR / f"results/cpca_analysis/residual/{safe}_cpca_analysis_global.pkl", "rb"))["all_components"]
    rng = np.random.RandomState(1)
    d_model = d_stmt["acts"].shape[2]
    inlp_cache = {}

    def inlp(l, k):
        if (l, max(args.ranks)) not in inlp_cache:
            inlp_cache[(l, max(args.ranks))] = inlp_directions(d_stmt["acts"][:, l].astype(np.float32), y_stmt,
                                                               max(args.ranks))
        return inlp_cache[(l, max(args.ranks))][:k]

    def cpca_k(l, k):
        comps = sorted([c for c in cp if c["layer"] == l], key=lambda c: -c["eigenvalue"])[:k]
        return orthonormal([c["vector"] for c in comps])

    def rand_k(k, seed):
        r = np.random.RandomState(1000 + seed)
        return orthonormal(r.randn(k, d_model))

    conds = []  # (nombre, conjunto de capas, rango, dirs)
    pers = {s: base_dirs[s]["persona"] for s in ("mid", "late")}
    for s in ("mid", "late"):
        conds.append(("persona", s, 1, pers[s]))
        conds.append(("inlp", s, 1, {l: inlp(l, 1) for l in sets[s]}))
        for sd in range(args.n_rand):
            conds.append((f"random_s{sd}", s, 1, {l: rand_k(1, sd) for l in sets[s]}))
        for k in args.ranks:
            conds.append(("inlp", s, k, {l: inlp(l, k) for l in sets[s]}))
            conds.append(("cpca", s, k, {l: cpca_k(l, k) for l in sets[s]}))
            for sd in range(args.n_rand):
                conds.append((f"random_s{sd}", s, k, {l: rand_k(k, sd) for l in sets[s]}))
    for k in (1, 8):
        conds.append(("inlp", "wide", k, {l: inlp(l, k) for l in sets["wide"]}))
        for sd in range(args.n_rand):
            conds.append((f"random_s{sd}", "wide", k, {l: rand_k(k, sd) for l in sets["wide"]}))
    print(f"{len(conds)} condiciones", flush=True)

    # --- datos de las conductas
    t = pd.read_parquet(BASE_DIR / "results/voting_predictions/residual/divided/voting_tasks.parquet")
    tv = t.sample(n=min(args.n_votes, len(t)), random_state=0).reset_index(drop=True)
    y_v, bill_v = (tv.voto_real == "EN CONTRA").astype(int).values, tv.ley.values
    prompts_v = [chat(tok, p, prefix="VOTO:") for p in tv.prompt]
    ids_v = [first_id(tok, " EN"), first_id(tok, " A")]
    tp = t.sample(n=min(args.n_persona, len(t)), random_state=1).reset_index(drop=True)
    prompts_pl = [chat(tok, HEADER.sub("Eres un diputado chileno de izquierda.", p), prefix="VOTO:") for p in tp.prompt]
    prompts_pr = [chat(tok, HEADER.sub("Eres un diputado chileno de derecha.", p), prefix="VOTO:") for p in tp.prompt]
    assert prompts_pl[0] != chat(tok, tp.prompt[0], prefix="VOTO:"), "no se reemplazo el encabezado"
    p6 = mp["dim_ideology_P6"].astype(float).values
    lo, hi = np.nanpercentile(p6, [25, 75])
    pool = [(str(tx)[:1500], int(p6[i] >= hi)) for i, row in enumerate(mp.itertuples())
            if not np.isnan(p6[i]) and (p6[i] <= lo or p6[i] >= hi) for tx in row.intervenciones if len(str(tx)) > 200]
    sel = np.random.RandomState(0).choice(len(pool), size=min(args.n_texts, len(pool)), replace=False)
    texts = [chat(tok, PROMPT_TEXTO.format(texto=pool[i][0])) for i in sel]
    y_t = np.array([pool[i][1] for i in sel])
    ids_t = [first_id(tok, "izquierda"), first_id(tok, "derecha"), first_id(tok, "Izquierda"), first_id(tok, "Derecha")]

    def behaviours():
        lv = first_token_logits(model, tok, prompts_v, ids_v, args.bs)
        s_v = lv[:, 0] - lv[:, 1]
        lt = first_token_logits(model, tok, texts, ids_t, args.bs)
        s_t = np.logaddexp(lt[:, 1], lt[:, 3]) - np.logaddexp(lt[:, 0], lt[:, 2])
        ll = first_token_logits(model, tok, prompts_pl, ids_v, args.bs)
        lr = first_token_logits(model, tok, prompts_pr, ids_v, args.bs)
        gap = (lr[:, 0] - lr[:, 1]) - (ll[:, 0] - ll[:, 1])
        return s_v, s_t, gap

    rows, raw = [], {}
    abl.set({})
    s_v0, s_t0, gap0 = behaviours()
    thr0 = float(np.median(s_v0))
    base = dict(auc_voto=metrics_votes(s_v0, y_v, bill_v)["auc_within_bill"], auc_texto=float(roc_auc_score(y_t, s_t0)),
                brecha_persona=float(gap0.mean()))
    rows.append(dict(modelo=args.model, direccion="base", capas="-", k=0, **base))
    raw["base"] = (s_v0, s_t0, gap0)
    print("base", base, flush=True)
    for name, sname, k, dd in tqdm(conds, desc="condiciones"):
        abl.set(dd)
        s_v, s_t, gap = behaviours()
        r = dict(modelo=args.model, direccion=name, capas=sname, k=k,
                 auc_voto=metrics_votes(s_v, y_v, bill_v, ref=thr0)["auc_within_bill"],
                 auc_texto=float(roc_auc_score(y_t, s_t)), brecha_persona=float(gap.mean()),
                 brecha_rel=float(gap.mean() / gap0.mean()), frac_brecha_pos=float((gap > 0).mean()),
                 corr_voto_base=float(np.corrcoef(s_v, s_v0)[0, 1]))
        rows.append(r)
        raw[(name, sname, k)] = (s_v, s_t, gap)
        print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items() if a != "modelo"}, flush=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / f"resultados_v2_{safe}.csv", index=False)
    abl.set({})
    pickle.dump({"raw": raw, "y_v": y_v, "bill_v": bill_v, "y_t": y_t}, open(OUT_DIR / f"raw_v2_{safe}.pkl", "wb"))


if __name__ == "__main__":
    main()
