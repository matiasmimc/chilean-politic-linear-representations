"""Ablacion de direcciones (GPU): ¿el modelo USA la direccion para tareas ligadas a ideologia?

Para cada direccion se proyecta fuera de la salida del bloque residual (todas las posiciones) en un conjunto de capas
y se mide el cambio de dos conductas, SIN generar texto (logits del primer token):
  votos : P(A FAVOR) vs P(EN CONTRA) con el prompt de predecir_votos_steered (tareas divididas, con partido).
          Metricas: AUC dentro de proyecto del puntaje "EN CONTRA", exactitud balanceada con umbral = mediana base.
  texto : clasificacion zero-shot izquierda/derecha de una intervencion (sin nombre ni partido).
          Metrica: AUC contra los cuartiles extremos de P6 (diputados de izquierda vs derecha).
Direcciones (k=1 por capa salvo `cpca_k4`): cpca (autovectores top de all_components), ridge (P6 sobre residual por
diputado, prompts sin partido), caa (caatopic), persona (media "diputado de izquierda" - "de derecha" en el ultimo
token) y random (vectores unitarios; N semillas). Conjuntos de capas: mid=12-16, late=27-31 (Llama; se escala por
profundidad para otros modelos).

Uso: CUDA_VISIBLE_DEVICES=1 uv run python scripts/steering/ablacion_direcciones.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_CACHE = BASE_DIR / "model"
OUT_DIR = BASE_DIR / "results/ablacion"
TOPICS = ["aborto", "inmigración", "pensiones", "educación gratuita", "cambio climático", "porte de armas",
          "matrimonio igualitario", "impuestos a las grandes fortunas", "reforma de salud", "seguridad ciudadana",
          "pueblos originarios", "privatización de empresas públicas", "salario mínimo", "libertad de prensa",
          "uso de la fuerza policial", "reforma constitucional", "royalty minero", "TPP-11 y libre comercio",
          "educación sexual", "eutanasia"]
PROMPT_TEXTO = ("Lee la siguiente intervención parlamentaria y responde con una sola palabra si quien habla es "
                "de izquierda o de derecha (responde \"izquierda\" o \"derecha\").\n\nIntervención:\n{texto}")


def layer_modules(model):
    mods = {}
    for name, m in model.named_modules():
        r = re.search(r"layers\.(\d+)$", name)
        if r and "vision" not in name and "audio" not in name:
            mods[int(r.group(1))] = m
    return [mods[i] for i in sorted(mods)]


class Ablator:
    def __init__(self, layers):
        self.layers = layers
        self.handles = []

    def set(self, dirs):
        """dirs: {capa: array [k, d] ortonormal}. Vacio = sin ablacion."""
        for h in self.handles:
            h.remove()
        self.handles = []
        for l, B in dirs.items():
            Bt = torch.tensor(B, dtype=torch.bfloat16, device=self.layers[l].parameters().__next__().device)

            def hook(_m, _i, out, Bt=Bt):
                h = out[0] if isinstance(out, tuple) else out
                h = h - (h @ Bt.T) @ Bt
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self.handles.append(self.layers[l].register_forward_hook(hook))


def orthonormal(M):
    q, _ = np.linalg.qr(np.asarray(M, dtype=np.float64).T)
    return q.T.astype(np.float32)


@torch.no_grad()
def first_token_logits(model, tok, prompts, token_ids, bs):
    out = []
    for i in tqdm(range(0, len(prompts), bs), leave=False):
        batch = prompts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        out.append(torch.log_softmax(logits, -1)[:, token_ids].cpu().numpy())
    return np.concatenate(out)


def chat(tok, content, prefix=""):
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True) + prefix


def first_id(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"][0]


def persona_directions(model, tok, layers_idx, bs):
    acts = {}
    for side, txt in (("izq", "izquierda"), ("der", "derecha")):
        prompts = [chat(tok, f"Eres un diputado chileno de {txt}. Escribe una declaración sobre {t}.") for t in TOPICS]
        vs = []
        for i in range(0, len(prompts), bs):
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states[1:]
            vs.append(torch.stack([h[:, -1, :].float() for h in hs], 1).cpu().numpy())
        acts[side] = np.concatenate(vs).mean(0)  # [L, d]
    return acts["der"] - acts["izq"]


def build_directions(args, model, tok, layers_idx, L):
    safe = args.model.replace("/", "_")
    sets = {"mid": [round(L * f) for f in (0.375, 0.4, 0.44, 0.47, 0.5)],
            "late": [round(L * f) - 1 for f in (0.84, 0.88, 0.91, 0.94, 0.97)]}
    sets = {k: sorted(set(min(max(l, 0), L - 1) for l in v)) for k, v in sets.items()}
    cp = pickle.load(open(BASE_DIR / f"results/cpca_analysis/residual/{safe}_cpca_analysis_global.pkl", "rb"))["all_components"]
    caa = pickle.load(open(BASE_DIR / f"results/caa/{safe}_caatopic_directions.pkl", "rb"))["layers"]
    feats, _ = pickle.load(open(BASE_DIR / f"results/replication/v3_noparty/residual/{safe}_politician_features.pkl", "rb"))
    mp = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    p6 = mp["dim_ideology_P6"].astype(float).values
    ok = ~np.isnan(p6)
    persona = persona_directions(model, tok, layers_idx, args.bs)
    rng = np.random.RandomState(0)
    dirs = {}
    for sname, ls in sets.items():
        by_layer = {l: sorted([c for c in cp if c["layer"] == l], key=lambda c: -c["eigenvalue"]) for l in ls}
        d = {"cpca": {}, "cpca_k4": {}, "ridge": {}, "caa": {}, "persona": {}}
        for l in ls:
            d["cpca"][l] = orthonormal([by_layer[l][0]["vector"]])
            d["cpca_k4"][l] = orthonormal([c["vector"] for c in by_layer[l][:4]])
            X = np.nan_to_num(feats[ok, l, :]).astype(np.float64)
            w = RidgeCV(alphas=np.logspace(0, 5, 11)).fit((X - X.mean(0)) / (X.std() + 1e-8), p6[ok]).coef_
            d["ridge"][l] = orthonormal([w])
            d["caa"][l] = orthonormal([caa[l]["vector"]])
            d["persona"][l] = orthonormal([persona[l]])
        for s in range(args.n_rand):
            d[f"random_s{s}"] = {l: orthonormal([rng.randn(persona.shape[1])]) for l in ls}
        dirs[sname] = d
        # cosenos entre metodos (diagnostico)
        for a in ("cpca", "ridge", "caa", "persona"):
            for b in ("cpca", "ridge", "caa", "persona"):
                if a < b:
                    cs = [abs(float(d[a][l][0] @ d[b][l][0])) for l in ls]
                    print(f"cos {sname} {a}-{b}: {np.round(cs, 2)}", flush=True)
    return dirs


def metrics_votes(score, y, bill, ref=None):
    """score: logit(EN CONTRA) - logit(A FAVOR). y=1 EN CONTRA."""
    aucs = [roc_auc_score(y[bill == b], score[bill == b]) for b in np.unique(bill) if len(np.unique(y[bill == b])) == 2]
    thr = np.median(score) if ref is None else ref
    pred = score > thr
    ba = 100 * np.mean([np.mean(pred[y == c] == c) for c in (0, 1)])
    return dict(auc_within_bill=float(np.mean(aucs)), bal_acc_umbral_mediana=float(ba), media_score=float(score.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-votes", type=int, default=1500)
    ap.add_argument("--n-texts", type=int, default=600)
    ap.add_argument("--n-rand", type=int, default=5)
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
    dirs = build_directions(args, model, tok, list(range(L)), L)

    # --- tareas de voto
    t = pd.read_parquet(BASE_DIR / "results/voting_predictions/residual/divided/voting_tasks.parquet")
    t = t.sample(n=min(args.n_votes, len(t)), random_state=0).reset_index(drop=True)
    y_v = (t.voto_real == "EN CONTRA").astype(int).values
    bill_v = t.ley.values
    # la respuesta del asistente empieza con "VOTO:"; el siguiente token es " A" (A FAVOR) o " EN" (EN CONTRA)
    prompts_v = [chat(tok, p, prefix="VOTO:") for p in t.prompt]
    ids_v = [first_id(tok, " EN"), first_id(tok, " A")]

    # --- textos para clasificacion izquierda/derecha
    mp = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    p6 = mp["dim_ideology_P6"].astype(float).values
    lo, hi = np.nanpercentile(p6, [25, 75])
    texts, y_t = [], []
    rng = np.random.RandomState(0)
    pool = []
    for i, row in enumerate(mp.itertuples()):
        if np.isnan(p6[i]) or not (p6[i] <= lo or p6[i] >= hi):
            continue
        for tx in row.intervenciones:
            tx = str(tx)
            if len(tx) > 200:
                pool.append((tx[:1500], int(p6[i] >= hi)))
    sel = rng.choice(len(pool), size=min(args.n_texts, len(pool)), replace=False)
    texts = [chat(tok, PROMPT_TEXTO.format(texto=pool[i][0])) for i in sel]
    y_t = np.array([pool[i][1] for i in sel])
    ids_t = [first_id(tok, "izquierda"), first_id(tok, "derecha")]
    assert ids_t[0] != ids_t[1] and ids_v[0] != ids_v[1], "primeros tokens ambiguos"
    ids_t_alt = [first_id(tok, "Izquierda"), first_id(tok, "Derecha")]

    def run_conditions(name, dset, layer_set):
        abl.set(dset)
        lv = first_token_logits(model, tok, prompts_v, ids_v, args.bs)
        s_v = lv[:, 0] - lv[:, 1]
        lt = first_token_logits(model, tok, texts, ids_t + ids_t_alt, args.bs)
        s_t = np.logaddexp(lt[:, 1], lt[:, 3]) - np.logaddexp(lt[:, 0], lt[:, 2])  # derecha - izquierda
        return s_v, s_t

    rows = []
    s_v0, s_t0 = run_conditions("base", {}, None)
    thr0 = float(np.median(s_v0))
    base_v = metrics_votes(s_v0, y_v, bill_v)
    base_t = float(roc_auc_score(y_t, s_t0))
    rows.append(dict(modelo=args.model, capas="-", direccion="base", **base_v, auc_texto=base_t,
                     media_score_texto=float(s_t0.mean())))
    print("base", base_v, base_t, flush=True)
    for sname, dd in dirs.items():
        for method, dset in dd.items():
            s_v, s_t = run_conditions(method, dset, sname)
            mv = metrics_votes(s_v, y_v, bill_v, ref=thr0)
            auc_t = float(roc_auc_score(y_t, s_t))
            rows.append(dict(modelo=args.model, capas=sname, direccion=method, **mv, auc_texto=auc_t,
                             media_score_texto=float(s_t.mean()),
                             corr_voto_base=float(np.corrcoef(s_v, s_v0)[0, 1])))
            print(sname, method, {k: round(v, 3) for k, v in mv.items()}, round(auc_t, 3), flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / f"resultados_{safe}.csv", index=False)
    abl.set({})


if __name__ == "__main__":
    main()
