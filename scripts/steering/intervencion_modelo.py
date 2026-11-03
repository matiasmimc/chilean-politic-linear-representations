"""Steering de political slant en declaraciones (RQ3).

Misma convención de direcciones que `predecir_votos_steered.py`:
  +steering_method = caa_k<N> | rndcaa_k<N>_s<semilla> (direccion de contraste D-I, escala diff_norm) | ridge | cpca | cpca_k<N> | cpca_ca<valor> | cpca_np (cPCA sin partido) | random_[np_]k16_s<semilla>
  +target          = residual | attention   (por defecto residual; ridge en atencion = top-k cabezas)
  +out_tag=_wide   sufijo de la carpeta de salida (para otras grillas de alpha sin pisar resultados)
La direccion se inyecta en el ultimo token de cada paso de generacion con magnitud
alpha * std(proyeccion de los parlamentarios sobre la direccion): alpha=1 equivale a 1 desviacion estandar
de la variacion natural entre parlamentarios (alpha con signo: >0 hacia el polo de P6 alto).

Ejemplos:
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/steering/intervencion_modelo.py +model=meta-llama/Llama-3.1-8B-Instruct +steering_method=cpca
  ... +steering_method=cpca_np paths.results_dir=results/replication/v3_noparty
  ... +steering_method=random_k16_s0 +topic=aborto '+alphas=[-1,0,1]' +n_samples=2

Salida: results/steering_v2/<target>/<metodo>/intervention_raw.parquet (columnas model,k,alpha,topic,sample,text),
compatible con `evaluar_respuestas.py --folder steering_v2/<target>/<metodo>`.
"""
import os
import pickle
import re
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from baukit import TraceDict
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from predecir_votos_steered import get_attention_heads, get_residual_layers

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
MODEL_CACHE = BASE_DIR / "model"
OUTPUT_ROOT = BASE_DIR / "results/steering_v2"

TOPICS = ['aborto', 'inmigración', 'porte de armas', 'matrimonio igualitario', 'derechos LGBTQ+',
          'cambio climático', 'confianza en ciencia', 'regulación de redes sociales', 'libertad de expresión']

interventions = {}  # nombre de modulo -> tensor [d]


def hook(output, layer_name):
    vec = interventions.get(layer_name)
    if vec is not None:
        target = output[0] if isinstance(output, tuple) else output
        target[:, -1, :] += vec.to(target.device)
    return output


def load_components(method, safe_name, model_name, is_residual, k_default):
    """Top-k componentes cPCA (o direcciones aleatorias equivalentes), como en predecir_votos_steered.py."""
    cpca_dir = BASE_DIR / "results/cpca_analysis" / ("residual" if is_residual else "")
    ca = re.search(r'_ca([0-9.]+)', method)
    tag = f"_ca{ca.group(1)}" if ca else ""
    if re.search(r'_np(_|$)', method):
        tag += "_noparty"
    path = cpca_dir / f"{safe_name}_cpca_analysis_global{tag}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    comps = pickle.load(open(path, "rb"))["all_components"]
    km = re.search(r'_k(\d+)', method)
    comps = comps[:int(km.group(1)) if km else k_default]
    if method.split('_')[0] == "random":
        sm = re.search(r'_s(\d+)', method)
        rng = np.random.RandomState(int(sm.group(1)) if sm else 0)
        comps = [{'layer': c['layer'], 'vector': rng.randn(*np.shape(c['vector'])).astype(np.float32)} for c in comps]
    elif method.split('_')[0] != "cpca":
        raise ValueError(f"Metodo no soportado: {method} (usa cpca*, random*)")
    by_layer = {}
    for c in comps:
        v = np.asarray(c['vector'], dtype=np.float32).reshape(-1)
        by_layer.setdefault(c['layer'], []).append(v / np.linalg.norm(v))
    return by_layer, len(comps)


def ridge_components(features, labels, k):
    """Direccion ridge (P6) por capa del residual; se usan las k capas con mayor Spearman en CV 5-fold."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    labels = np.asarray(labels, dtype=float)
    scores, coefs = [], []
    for l in range(features.shape[1]):
        X = features[:, l, :]
        X = (X - X.mean(0)) / (X.std() + 1e-8)
        reg = RidgeCV(alphas=np.logspace(-2, 5, 8))
        pred = cross_val_predict(reg, X, labels, cv=KFold(5, shuffle=True, random_state=0))
        scores.append(spearmanr(labels, pred).statistic)
        coefs.append(reg.fit(X, labels).coef_.astype(np.float32))
    top = np.argsort(scores)[::-1][:k]
    print("Capas ridge:", {int(l): round(float(scores[l]), 2) for l in sorted(top)})
    return {int(l): [coefs[l] / np.linalg.norm(coefs[l])] for l in top}, len(top)


def ridge_head_components(results_dir, safe_name, features, k):
    """Direcciones ridge por cabeza de atencion (probes de entrenar_probes.py); top-k cabezas por Spearman."""
    perf = np.asarray(pickle.load(open(results_dir / f"{safe_name}_politician_performance.pkl", "rb")))
    ridge = pickle.load(open(results_dir / f"{safe_name}_ridge.pkl", "rb"))
    _, _, H, D = features.shape
    by_layer = {}
    for i in np.argsort(perf.ravel())[::-1][:k]:
        l, h = np.unravel_index(i, perf.shape)
        vec = np.zeros(H * D, dtype=np.float32)
        coef = np.asarray(ridge[int(l)][int(h)].coef_, dtype=np.float32).reshape(-1)
        vec[h * D:(h + 1) * D] = coef / np.linalg.norm(coef)
        by_layer.setdefault(int(l), []).append(vec)
    return by_layer, k


def caa_components(method, safe_name, k_default):
    """Direcciones de contraste derecha-izquierda (scripts/caa): top-k capas por AUC en CV. alpha se mide en
    unidades de `diff_norm` (alpha=1 desplaza la activacion por la diferencia entre medias de las clases).
    rndcaa_*: direcciones aleatorias unitarias en las mismas capas y con la misma escala (control)."""
    fname = "caatopic" if "caatopic" in method else "caa"
    d = pickle.load(open(BASE_DIR / "results/caa" / f"{safe_name}_{fname}_directions.pkl", "rb"))["layers"]
    km = re.search(r'_k(\d+)', method)
    k = int(km.group(1)) if km else min(4, k_default)
    top = sorted(sorted(d, key=lambda l: d[l]["auc_cv"], reverse=True)[:k])
    print("Capas CAA:", {l: round(d[l]["auc_cv"], 2) for l in top})
    rng = None
    if method.startswith("rndcaa"):
        sm = re.search(r'_s(\d+)', method)
        rng = np.random.RandomState(int(sm.group(1)) if sm else 0)
    by_layer, scale = {}, {}
    for l in top:
        v = d[l]["vector"] if rng is None else rng.randn(*d[l]["vector"].shape).astype(np.float32)
        by_layer[l] = [v / np.linalg.norm(v)]
        scale[l] = d[l]["diff_norm"]
    return by_layer, k, scale


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    method = cfg.get("steering_method", "cpca")
    target = cfg.get("target", "residual")
    is_residual = target == "residual"
    model_name = cfg.get("model", "meta-llama/Llama-3.1-8B-Instruct")
    safe_name = model_name.replace("/", "_")
    k_default = cfg.get("k", 16)
    alphas = [float(a) for a in cfg.get("alphas", [-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2])]
    n_samples = cfg.get("n_samples", 3)
    max_new_tokens = cfg.get("max_new_tokens", 200)
    topics = [t for t in TOPICS if cfg.get("topic") is None or str(cfg.topic).lower() in t.lower()]
    seed = cfg.get("seed", 0)

    out_dir = OUTPUT_ROOT / target / (method + cfg.get("out_tag", ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "intervention_raw.parquet"
    prev = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    if len(prev) and (prev.model == model_name).any() and not cfg.get("force", False):
        print(f"Ya existen resultados de {model_name} en {out_path}. Usa +force=true para regenerar.")
        return

    # Activaciones de referencia para escalar alpha (desv. estandar de la proyeccion entre parlamentarios)
    results_dir = BASE_DIR / cfg.paths.results_dir / ("residual" if is_residual else "")
    features, labels = pickle.load(open(results_dir / f"{safe_name}_politician_features.pkl", "rb"))
    features = np.nan_to_num(features.astype(np.float32))
    caa_scale = None
    if method.startswith(("caa", "rndcaa")):
        if not is_residual:
            raise ValueError("caa solo soportado para target=residual")
        by_layer, k_used, caa_scale = caa_components(method, safe_name, k_default)
    elif method.startswith("ridge"):
        if is_residual:
            by_layer, k_used = ridge_components(features, labels, k_default)
        else:
            by_layer, k_used = ridge_head_components(results_dir, safe_name, features, k_default)
    else:
        by_layer, k_used = load_components(method, safe_name, model_name, is_residual, k_default)

    std_proj = {}
    for l, vecs in by_layer.items():
        if caa_scale is not None:
            std_proj[l] = [caa_scale[l]]
            continue
        acts = features[:, l, :] if is_residual else features[:, l].reshape(len(features), -1)
        std_proj[l] = [float(np.std(acts @ v)) for v in vecs]

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE), use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(MODEL_CACHE), low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float16, device_map="auto")
    layer_names = get_residual_layers(model) if is_residual else get_attention_heads(model)[0]
    layers_to_trace = [layer_names[l] for l in sorted(by_layer)]

    prompts = []
    for t in topics:
        msgs = [{"role": "user", "content": f"Escribe una declaración sobre {t}."}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + f"Respecto a {t}, creo que ")
    # un batch por muestra: todos los temas a la vez (misma intervencion global)
    rows = []
    for alpha in tqdm(alphas, desc=f"{method} k={k_used}"):
        interventions.clear()
        for l, vecs in by_layer.items():
            total = sum(alpha * s * v for s, v in zip(std_proj[l], vecs))
            interventions[layer_names[l]] = torch.from_numpy(np.asarray(total, dtype=np.float32)).to(model.dtype)
        for s in range(n_samples):
            torch.manual_seed(seed * 1000 + s)
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                with TraceDict(model, layers_to_trace, edit_output=hook):
                    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,
                                         pad_token_id=tokenizer.eos_token_id)
            texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for t, txt in zip(topics, texts):
                rows.append({"model": model_name, "method": method, "k": k_used, "alpha": alpha,
                             "topic": t, "sample": s, "text": txt.strip()})
    df = pd.concat([prev[prev.model != model_name], pd.DataFrame(rows)], ignore_index=True) if len(prev) else pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"✅ {len(rows)} generaciones guardadas en {out_path}")


if __name__ == "__main__":
    main()
