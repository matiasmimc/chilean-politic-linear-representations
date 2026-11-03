"""Validacion causal de la interpretacion por unembedding.

Para una direccion v en la capa l (cPC1, ridge) se toman los tokens de cada polo segun el logit lens
(`results/unembedding/<modelo>/tokens_por_capa.csv`) y se mide si inyectar alpha * |h| * v sube la
log-prob del polo "+" respecto al polo "-" en prompts neutros (ultimo token).
Controles: direccion aleatoria de igual norma y cPC1 ortogonalizada respecto del probe ridge.

Uso:
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/interpretabilidad/validacion_causal_logit_lens.py model=meta-llama/Llama-3.1-8B-Instruct
"""
import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from logit_lens_cpca import ridge_pca_directions  # mismo directorio

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_CACHE = BASE_DIR / "model"

NEUTRAL_PROMPTS = [
    "Escribe una breve declaración sobre el tema más importante de este mes en Chile.",
    "Comenta la situación actual del país en un párrafo.",
    "¿Qué debería priorizar el Congreso este año?",
    "Redacta una opinión sobre los desafíos de la próxima década.",
    "Resume en una frase tu visión del futuro de Chile.",
    "Describe un problema que afecta a las familias chilenas.",
    "Da tu opinión sobre cómo mejorar la vida en las ciudades.",
    "Escribe un mensaje para los ciudadanos al comenzar el año.",
]


def get_layers(model):
    for attr in ("model.layers", "model.language_model.layers", "transformer.encoder.layers", "transformer.h"):
        obj = model
        try:
            for a in attr.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise ValueError("No se encontraron las capas del modelo")


def token_ids(tokenizer, tokens):
    ids = []
    for t in tokens:
        for variant in (" " + t, t, " " + t.capitalize(), t.capitalize()):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
    return sorted(set(ids))


@torch.no_grad()
def pole_score(model, tokenizer, prompts, layer_module, vec, pos_ids, neg_ids):
    """mean logprob(polo +) - mean logprob(polo -) en el siguiente token, con vec sumado al ultimo token."""
    handle = None
    if vec is not None:
        def hook(_, __, out):
            h = out[0] if isinstance(out, tuple) else out
            h[:, -1, :] += vec.to(h.dtype)
            return out
        handle = layer_module.register_forward_hook(hook)
    scores = []
    for p in prompts:
        text = tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        logp = torch.log_softmax(model(**ids).logits[0, -1].float(), dim=-1)
        scores.append((logp[pos_ids].mean() - logp[neg_ids].mean()).item())
    if handle:
        handle.remove()
    return float(np.mean(scores))


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    model_name = cfg.get("model", "meta-llama/Llama-3.1-8B-Instruct")
    safe = model_name.replace("/", "_")
    alphas = cfg.get("alphas_causal", [0.0, 0.25, 0.5, 1.0, 2.0])
    top_n = cfg.get("top_n_tokens", 30)
    rng = np.random.RandomState(0)

    tok_df = pd.read_csv(BASE_DIR / "results/unembedding" / safe / "tokens_por_capa.csv")
    sel = pd.read_json(BASE_DIR / "results/unembedding" / safe / "selection.json", typ="series")["layers"]

    with open(BASE_DIR / cfg.paths.results_dir / "residual" / f"{safe}_politician_features.pkl", "rb") as f:
        features, labels = pickle.load(f)
    cpca_path = BASE_DIR / "results/cpca_analysis/residual" / f"{safe}_cpca_analysis_global.pkl"
    if not cpca_path.exists():
        cpca_path = BASE_DIR / "results/cpca_analysis/cpca_analysis/residual" / f"{safe}_cpca_analysis_global.pkl"
    with open(cpca_path, "rb") as f:
        cpca = pickle.load(f)
    ridge, _ = ridge_pca_directions(features, labels)
    cpc1 = {c["layer"]: c["vector"] for c in cpca["all_components"] if c["component_idx"] == 0}

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE))
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(MODEL_CACHE),
                                                 torch_dtype=torch.bfloat16, device_map="auto").eval()
    layers = get_layers(model)

    rows = []
    for l in sel:
        mean_norm = float(np.linalg.norm(features[:, l, :], axis=1).mean())
        w = ridge[l] / np.linalg.norm(ridge[l])
        c = cpc1[l] / np.linalg.norm(cpc1[l])
        c_orth = c - np.dot(c, w) * w
        c_orth /= np.linalg.norm(c_orth)
        rnd = rng.randn(len(c)); rnd /= np.linalg.norm(rnd)
        dirs = {"cPC1": c, "ridge": w, "cPC1_ortho_ridge": c_orth, "random": rnd}

        for dname in ("cPC1", "ridge"):
            sub = tok_df[(tok_df.layer == l) & (tok_df.direction == dname)]
            pos_ids = token_ids(tokenizer, sub[(sub.pole == "+") & (sub["rank"] < top_n)].token.tolist())
            neg_ids = token_ids(tokenizer, sub[(sub.pole == "-") & (sub["rank"] < top_n)].token.tolist())
            if not pos_ids or not neg_ids:
                continue
            base = pole_score(model, tokenizer, NEUTRAL_PROMPTS, layers[l], None, pos_ids, neg_ids)
            for steer_name, v in dirs.items():
                for a in alphas:
                    vec = torch.tensor(a * mean_norm * v, dtype=torch.float32, device=model.device)
                    sc = pole_score(model, tokenizer, NEUTRAL_PROMPTS, layers[l], vec, pos_ids, neg_ids)
                    rows.append({"layer": l, "token_set_from": dname, "steer_dir": steer_name,
                                 "alpha": a, "score": sc, "delta_vs_base": sc - base,
                                 "n_pos": len(pos_ids), "n_neg": len(neg_ids)})
            print(f"capa {l} / tokens de {dname} listo")

    out = BASE_DIR / "results/unembedding" / safe / "validacion_causal.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
