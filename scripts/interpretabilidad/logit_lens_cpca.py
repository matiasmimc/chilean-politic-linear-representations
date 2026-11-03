"""Logit lens de direcciones (cPCA, PCA, Ridge) del residual stream via matriz de unembedding.

Para cada direccion v de la capa l se calcula W_U @ (gamma * v) (gamma = pesos del RMSNorm final)
y se listan los tokens de ambos polos (+v / -v).

Notas metodologicas:
  * hidden_states[-1] de HF ya incluye la norma final: en la ultima capa NO se aplica gamma otra vez.
  * La norma final se aproxima por gamma (RMSNorm es invariante a escala para una direccion).
    Gemma usa (1 + gamma).
  * Se resta la media de logits sobre el vocabulario para quitar el desplazamiento uniforme.
  * Se omite el softcapping final de Gemma (monotono; no cambia el ranking).

Criterio de seleccion de capas/componentes (fijado antes de mirar tokens, ver `select_layers`):
  las `n_layers` capas con mayor |cos(cPC1, ridge Nominate)| entre las capas >= 40% de la profundidad; en ellas se reportan cPC1 y cPC2.
  El analisis por capa completo (todas las capas) se guarda aparte para el heatmap.

Uso:
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/interpretabilidad/logit_lens_cpca.py model=meta-llama/Llama-3.1-8B-Instruct
"""
import json
import pickle
import re
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.linear_model import RidgeCV
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
MODEL_CACHE = BASE_DIR / "model"

# Raices de un lexicon politico (es/en); se compara con startswith sobre el token normalizado.
POLITICAL_STEMS = [
    "gobiern", "estado", "reforma", "derech", "izquierd", "derech", "partid", "ley", "leyes",
    "president", "congres", "diput", "senad", "democra", "liberal", "conservad", "social",
    "comunis", "socialis", "republican", "capital", "trabaj", "sindic", "impuest", "tribut",
    "pension", "salud", "educaci", "seguridad", "delincuen", "inmigra", "aborto", "constituc",
    "pinochet", "allende", "dictadur", "pueblo", "ciudadan", "elecci", "voto", "poder",
    "polític", "politic", "justicia", "igualdad", "libertad", "mercado", "privat", "públic",
    "public", "regulaci", "propiedad", "obrer", "empresari", "fascis", "marxis", "neoliberal",
    "progres", "reaccion", "revoluci", "autorit", "feminis", "familia", "orden", "patria",
    "nacion", "immigra", "tax", "welfare", "government", "election", "party", "right", "left",
    "approv", "aprob", "legislat", "vote", "voting", "votaci", "reform", "crisis", "situa", "support", "apoy",
    "oppos", "opon", "demand", "exig", "require", "requier", "law", "bill", "proyect", "minist", "policy",
    "pol\u00edtic", "reject", "rechaz", "protest", "labor", "union", "econom", "tax", "state", "citizen",
]


def find_final_norm(model, d_model):
    mods = dict(model.named_modules())
    cands = [n for n, m in mods.items()
             if re.search(r'(^|\.)norm$', n) and '.layers.' not in n and 'vision' not in n
             and getattr(getattr(m, "weight", None), "shape", (None,))[-1] == d_model]
    if not cands:
        raise ValueError("No se encontro la norma final del modelo")
    return mods[cands[-1]]


def load_unembedding(model_name, device):
    """Devuelve (W_U [V, d], gamma [d]) en float32 sin conservar el modelo completo."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=str(MODEL_CACHE), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    W_U = model.get_output_embeddings().weight.detach().float().to(device)
    norm = find_final_norm(model, W_U.shape[1])
    gamma = norm.weight.detach().float().to(device)
    if "gemma" in type(norm).__name__.lower():
        gamma = 1.0 + gamma
    del model
    return W_U, gamma


def build_vocab_mask(tokenizer, vocab_size, min_len=3):
    """Tokens alfabeticos (latinos con tildes) de largo >= min_len; excluye especiales."""
    keep = np.zeros(vocab_size, dtype=bool)
    special = set(tokenizer.all_special_ids)
    for i in range(min(vocab_size, len(tokenizer))):
        if i in special:
            continue
        s = tokenizer.decode([i]).strip()
        if len(s) >= min_len and re.fullmatch(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", s):
            keep[i] = True
    return keep


def is_political(token):
    t = token.strip().lower()
    return any(t.startswith(stem) for stem in POLITICAL_STEMS)


def lens_direction(v, W_U, gamma, apply_gamma, vocab_mask):
    """Logits de tokens para +v (cuyo reverso es -v)."""
    v = torch.as_tensor(v, dtype=torch.float32, device=W_U.device)
    v = v / v.norm()
    if apply_gamma:
        v = v * gamma
    logits = W_U @ v
    logits = logits - logits.mean()
    logits[~vocab_mask] = float("nan")
    return logits


def top_tokens(logits, tokenizer, k):
    l = torch.nan_to_num(logits, nan=-1e9)
    pos = torch.topk(l, k).indices.cpu().tolist()
    l2 = torch.nan_to_num(-logits, nan=-1e9)
    neg = torch.topk(l2, k).indices.cpu().tolist()
    dec = lambda ids: [tokenizer.decode([i]).strip() for i in ids]
    return (pos, dec(pos), logits[pos].cpu().tolist()), (neg, dec(neg), logits[neg].cpu().tolist())


def lexicon_pvalue(tokens, vocab_tokens, rng, n_perm=1000):
    """Fraccion politica observada y p-valor de permutacion vs. k tokens al azar del vocabulario filtrado."""
    obs = np.mean([is_political(t) for t in tokens])
    k = len(tokens)
    null = np.array([np.mean([is_political(t) for t in rng.choice(vocab_tokens, k, replace=False)])
                     for _ in range(n_perm)])
    return float(obs), float((np.sum(null >= obs) + 1) / (n_perm + 1))


def ridge_pca_directions(features, labels):
    """Direcciones por capa: ridge (orientada con las etiquetas) y PC1 (orientada con ridge)."""
    n, L, d = features.shape
    ridge, pc1 = [], []
    y = np.asarray(labels, dtype=float)
    for l in range(L):
        X = np.nan_to_num(features[:, l, :])
        w = RidgeCV(alphas=np.logspace(-3, 4, 20)).fit(X, y).coef_
        Xc = X - X.mean(0)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        p = vt[0]
        if np.dot(p, w) < 0:
            p = -p
        ridge.append(w)
        pc1.append(p)
    return np.stack(ridge), np.stack(pc1)


def select_layers(cpca, ridge, n_layers, min_layer_frac=0.4):
    """Criterio previo: capas con mayor |cos(cPC1, ridge)| entre las capas >= min_layer_frac * L
    (el logit lens crudo no es fiable en capas tempranas; ver Fase 2 paso 5)."""
    cos = {}
    min_layer = int(min_layer_frac * len(ridge))
    for c in cpca["all_components"]:
        if c["component_idx"] == 0 and c["layer"] >= min_layer:
            r = ridge[c["layer"]]
            cos[c["layer"]] = abs(float(np.dot(c["vector"], r) / (np.linalg.norm(c["vector"]) * np.linalg.norm(r))))
    return sorted(sorted(cos, key=cos.get, reverse=True)[:n_layers]), cos


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    model_name = cfg.get("model", "meta-llama/Llama-3.1-8B-Instruct")
    safe = model_name.replace("/", "_")
    top_k = cfg.get("top_k", 30)
    n_layers_sel = cfg.get("n_layers_sel", 4)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(42)

    feat_path = BASE_DIR / cfg.paths.results_dir / "residual" / f"{safe}_politician_features.pkl"
    cpca_path = BASE_DIR / "results/cpca_analysis/residual" / f"{safe}_cpca_analysis_global.pkl"
    if not cpca_path.exists():  # estructura antigua con carpeta duplicada
        cpca_path = BASE_DIR / "results/cpca_analysis/cpca_analysis/residual" / f"{safe}_cpca_analysis_global.pkl"
    out_dir = BASE_DIR / "results/unembedding" / safe
    fig_dir = BASE_DIR / "figures/unembedding"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(feat_path, "rb") as f:
        features, labels = pickle.load(f)
    with open(cpca_path, "rb") as f:
        cpca = pickle.load(f)
    n_samples, L, d = features.shape
    ridge, pc1 = ridge_pca_directions(features, labels)

    # 1) Criterio de seleccion ANTES de ver tokens
    sel_layers, cos_by_layer = select_layers(cpca, ridge, n_layers_sel)
    with open(out_dir / "selection.json", "w") as f:
        json.dump({"criterio": "top capas por |cos(cPC1, ridge)|", "layers": sel_layers,
                   "cos_cpc1_ridge": cos_by_layer, "contrast_alpha": cpca["alpha"]}, f, indent=2)
    print(f"Capas seleccionadas: {sel_layers}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE))
    W_U, gamma = load_unembedding(model_name, device)
    vocab_mask = torch.from_numpy(build_vocab_mask(tokenizer, W_U.shape[0])).to(device)
    vocab_tokens = np.array([tokenizer.decode([i]).strip() for i in torch.nonzero(vocab_mask).flatten().cpu().tolist()])
    print(f"Vocabulario filtrado: {len(vocab_tokens)} tokens")

    cpc = {(c["layer"], c["component_idx"]): c["vector"] for c in cpca["all_components"] if c["component_idx"] < 2}

    rows, summary = [], []
    for l in range(L):
        apply_gamma = l < L - 1  # la ultima capa de hidden_states ya esta normalizada
        directions = {"cPC1": cpc.get((l, 0)), "cPC2": cpc.get((l, 1)), "PC1": pc1[l], "ridge": ridge[l]}
        for name, v in directions.items():
            if v is None:
                continue
            logits = lens_direction(v, W_U, gamma, apply_gamma, vocab_mask)
            (_, ptoks, pvals), (_, ntoks, nvals) = top_tokens(logits, tokenizer, top_k)
            for pole, toks, vals in (("+", ptoks, pvals), ("-", ntoks, nvals)):
                for r, (t, val) in enumerate(zip(toks, vals)):
                    rows.append({"model": safe, "direction": name, "layer": l, "pole": pole,
                                 "rank": r, "token": t, "logit": val})
            frac, p = lexicon_pvalue(ptoks + ntoks, vocab_tokens, rng)
            summary.append({"direction": name, "layer": l, "frac_political": frac, "p_perm": p,
                            "selected": l in sel_layers})

    df = pd.DataFrame(rows)
    df_sum = pd.DataFrame(summary)
    df.to_csv(out_dir / "tokens_por_capa.csv", index=False)
    df_sum.to_csv(out_dir / "lexicon_fraccion.csv", index=False)

    # Tabla legible para las capas seleccionadas
    with open(out_dir / "tokens_capas_seleccionadas.md", "w") as f:
        for l in sel_layers:
            f.write(f"## Capa {l}\n")
            for name in ["cPC1", "cPC2", "PC1", "ridge"]:
                for pole in ["+", "-"]:
                    sub = df[(df.layer == l) & (df.direction == name) & (df.pole == pole)].sort_values("rank")
                    if len(sub):
                        f.write(f"- **{name} {pole}**: {', '.join(sub.token.head(15))}\n")
            f.write("\n")

    # Figura: fraccion politica por capa y direccion
    plt.figure(figsize=(10, 5))
    for name, g in df_sum.groupby("direction"):
        plt.plot(g.layer, g.frac_political, marker="o", label=name)
    plt.xlabel("Capa")
    plt.ylabel(f"Fraccion de tokens politicos (top-{top_k} por polo)")
    plt.title(f"Logit lens de direcciones | {safe}")
    plt.grid(alpha=0.4, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{safe}_lexicon_fraccion.png", dpi=200)
    plt.close()
    print(f"✅ Resultados en {out_dir}")


if __name__ == "__main__":
    main()
