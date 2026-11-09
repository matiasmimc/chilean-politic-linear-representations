"""Slant v3 (GPU): steering de declaraciones con direcciones derivadas de declaraciones, control positivo y version personal.

Genera (sin juez) tres experimentos en results/slant_v3/<exp>/intervention_raw.parquet, compatibles con
`evaluar_respuestas.py --folder slant_v3/<exp>`:
  dosis    : direcciones x alpha {-2,-1,0,1,2} (unidades de diff_norm de CAA por capa), 20 temas x 5 muestras.
             caatopic (CAA con control de tema), within (ridge a P6 con activaciones y P6 centrados por partido),
             persona (derecha - izquierda), 3 aleatorias en las mismas capas y escala.
  positivo : sin steering, prompt "Eres un diputado chileno de izquierda/derecha" (¿mide algo el juez / el logit?).
  personal : para cada diputado, alpha = A * z_dentro_de_partido (z recortado a [-2, 2]); within, caatopic, persona, aleatoria.
Ademas de generar, cada texto recibe `slant_logit` (logit derecha - izquierda del mismo modelo sin steering, con el prompt
de clasificacion de la ablacion) y `distinct2`.

Convencion de steering: se suma coef * escala_l * v_l al ultimo token de cada paso en las capas top-4 por AUC de CAA.

Uso: CUDA_VISIBLE_DEVICES=1 uv run python scripts/steering/slant_v3.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analisis"))
from ablacion_direcciones import PROMPT_TEXTO, first_id, layer_modules, persona_directions  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TOPICS = ['aborto', 'inmigración', 'porte de armas', 'matrimonio igualitario', 'derechos LGBTQ+', 'cambio climático',
          'confianza en ciencia', 'regulación de redes sociales', 'libertad de expresión', 'reforma de pensiones',
          'reforma tributaria', 'seguridad ciudadana', 'pueblos indígenas', 'educación pública gratuita',
          'salud pública', 'nueva constitución', 'legalización de drogas', 'minería y energía', 'eutanasia',
          'derecho a huelga']
TOPICS_PERSONAL = ['aborto', 'inmigración', 'porte de armas', 'reforma de pensiones', 'reforma tributaria',
                   'seguridad ciudadana', 'educación pública gratuita', 'nueva constitución']


class Steerer:
    def __init__(self, layers):
        self.layers, self.handles, self.coef = layers, [], None

    def set(self, shifts):
        for h in self.handles:
            h.remove()
        self.handles = []
        for l, v in shifts.items():
            vt = torch.tensor(v, dtype=torch.float32)

            def hook(_m, _i, out, vt=vt):
                h = out[0] if isinstance(out, tuple) else out
                h[:, -1, :] += (self.coef.to(h.device, torch.float32)[:, None] * vt.to(h.device)).to(h.dtype)
                return out
            self.handles.append(self.layers[l].register_forward_hook(hook))


def within_direction(acts_l, y_stmt, party_stmt, lam_rel=0.3):
    """Ridge dual: X (activaciones) y y (P6) centrados por partido; direccion unitaria, + = P6 alto."""
    X = acts_l.astype(np.float32).copy()
    y = y_stmt.astype(np.float64).copy()
    for p in np.unique(party_stmt):
        m = party_stmt == p
        X[m] -= X[m].mean(0)
        y[m] -= y[m].mean()
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    G = X @ X.T
    lam = lam_rel * np.trace(G) / X.shape[1]
    w = X.T @ np.linalg.solve(G + lam * np.eye(len(G), dtype=np.float32), y.astype(np.float32))
    return (w / np.linalg.norm(w)).astype(np.float32)


def distinct2(t):
    w = t.split()
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / max(len(bg), 1)


@torch.no_grad()
def generate(model, tok, st, prompts, coefs, max_new_tokens, bs, seed=0):
    out = [None] * len(prompts)
    order = np.argsort([len(p) for p in prompts])
    for bi, i in enumerate(range(0, len(prompts), bs)):
        idx = order[i:i + bs]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
        st.coef = torch.tensor(coefs[idx], dtype=torch.float32)
        torch.manual_seed(seed * 100000 + bi)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,
                             pad_token_id=tok.eos_token_id)
        for j, txt in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[j] = txt.strip()
    return out


@torch.no_grad()
def slant_logits(model, tok, st, texts, bs=16):
    st.set({})
    ids = [first_id(tok, "izquierda"), first_id(tok, "derecha"), first_id(tok, "Izquierda"), first_id(tok, "Derecha")]
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT_TEXTO.format(texto=t[:1500])}], tokenize=False,
                                       add_generation_prompt=True) for t in texts]
    order = np.argsort([len(p) for p in prompts])
    s = np.zeros(len(prompts), dtype=np.float32)
    for i in range(0, len(prompts), bs):
        idx = order[i:i + bs]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
        lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), -1)[:, ids].cpu().numpy()
        s[idx] = np.logaddexp(lp[:, 1], lp[:, 3]) - np.logaddexp(lp[:, 0], lp[:, 2])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--alphas", type=float, nargs="+", default=[-2, -1, 0, 1, 2])
    ap.add_argument("--n-rand", type=int, default=3)
    ap.add_argument("--personal-a", type=float, default=1.5)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--exps", nargs="+", default=["dosis", "positivo", "personal"])
    ap.add_argument("--out-root", default="slant_v3", help="carpeta dentro de results/")
    ap.add_argument("--model-path", default=None, help="carpeta local del modelo (si no, se descarga/usa la cache por nombre); --model sigue dando el nombre de los archivos")
    ap.add_argument("--load-8bit", action="store_true", help="cargar en 8 bits (modelos 70B, varias GPU)")
    ap.add_argument("--n-topics", type=int, default=len(TOPICS), help="primeros N temas de TOPICS para dosis y positivo")
    ap.add_argument("--personal-topics", nargs="+", default=TOPICS_PERSONAL, help="temas de la version personal")
    ap.add_argument("--k-capas", type=int, default=4, help="numero de capas (top por AUC de CAA) donde se aplica el steering")
    ap.add_argument("--igualar-dosis", action="store_true",
                    help="reescala la escala por capa para que la norma total del desplazamiento (raiz de la suma de cuadrados) sea la de las 4 capas top")
    ap.add_argument("--dirs", nargs="+", default=None, help="solo las direcciones cuyo nombre empieza con alguno de estos prefijos")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    safe = args.model.replace("/", "_")
    out_root = BASE_DIR / "results" / args.out_root
    if args.smoke:
        args.n_samples, args.alphas, args.n_rand, args.max_new_tokens = 1, [-1, 0, 1], 1, 30

    tok = AutoTokenizer.from_pretrained(args.model_path or args.model, cache_dir=str(BASE_DIR / "model"), use_fast=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)} if args.load_8bit else {}
    model = AutoModelForCausalLM.from_pretrained(args.model_path or args.model, cache_dir=str(BASE_DIR / "model"), low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float16, device_map="auto", **kw).eval()
    layers = layer_modules(model)
    L = len(layers)
    st = Steerer(layers)

    # --- direcciones
    caa = pickle.load(open(BASE_DIR / f"results/caa/{safe}_caatopic_directions.pkl", "rb"))["layers"]
    top = sorted(sorted(caa, key=lambda l: caa[l]["auc_cv"], reverse=True)[:args.k_capas])
    scale = {l: float(caa[l]["diff_norm"]) for l in top}
    if args.igualar_dosis:
        top4 = sorted(caa, key=lambda l: caa[l]["auc_cv"], reverse=True)[:4]
        f = np.sqrt(sum(float(caa[l]["diff_norm"]) ** 2 for l in top4) / sum(scale[l] ** 2 for l in top))
        scale = {l: s_ * f for l, s_ in scale.items()}
        print(f"dosis igualada: factor {f:.3f}", flush=True)
    print("capas", top, "escala", {l: round(s, 3) for l, s in scale.items()}, flush=True)
    d_stmt = pickle.load(open(BASE_DIR / f"results/caa/{safe}_statement_acts.pkl", "rb"))
    mp = pd.read_parquet(BASE_DIR / "data/congreso_2022_2026_20intervenciones.parquet")
    party = mp["party_1_P6"].fillna("IND").values
    p6 = mp["dim_ideology_P6"].astype(float).values
    ms = d_stmt["mp_idx"]
    pers = persona_directions(model, tok, list(range(L)), 8)
    dirs = {"caatopic": {l: caa[l]["vector"] / np.linalg.norm(caa[l]["vector"]) for l in top},
            "within": {l: within_direction(d_stmt["acts"][:, l], p6[ms], party[ms]) for l in top},
            "persona": {l: pers[l] / np.linalg.norm(pers[l]) for l in top}}
    for sd in range(args.n_rand):
        r = np.random.RandomState(500 + sd)
        dirs[f"random_s{sd}"] = {l: (lambda v: (v / np.linalg.norm(v)).astype(np.float32))(r.randn(len(caa[l]["vector"])))
                                 for l in top}
    for a in ("caatopic", "within", "persona"):
        for b in ("caatopic", "within", "persona"):
            if a < b and not args.dirs:
                print(f"cos {a}-{b}:", [round(float(dirs[a][l] @ dirs[b][l]), 2) for l in top], flush=True)
    if args.dirs:
        dirs = {n: v for n, v in dirs.items() if n.startswith(tuple(args.dirs))}
    shifts = {n: {l: scale[l] * v[l] for l in top} for n, v in dirs.items()}

    def prefix(t):
        return f"Respecto a {t}, creo que "

    def chat(content, t):
        return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True) + prefix(t)

    def finalize(rows, exp):
        df = pd.DataFrame(rows)
        df["slant_logit"] = slant_logits(model, tok, st, df.text.tolist())
        df["distinct2"] = df.text.map(distinct2)
        d = out_root / exp
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / "intervention_raw.parquet")
        print(f"[{exp}] {len(df)} textos -> {d}", flush=True)
        return df

    topics = TOPICS[:4] if args.smoke else TOPICS[:args.n_topics]
    # --- dosis
    if "dosis" in args.exps:
        rows = []
        for name in dirs:
            for a in args.alphas:
                if a == 0 and name != "caatopic":
                    continue  # el alpha=0 es comun a todas las direcciones
                st.set(shifts[name])
                prompts = [chat(f"Escribe una declaración sobre {t}.", t) for t in topics for _ in range(args.n_samples)]
                tt = [t for t in topics for _ in range(args.n_samples)]
                ss = [s for _ in topics for s in range(args.n_samples)]
                texts = generate(model, tok, st, prompts, np.full(len(prompts), a, np.float32), args.max_new_tokens, args.bs)
                label = "base" if a == 0 else name
                rows += [dict(model=args.model, method=label, k=len(top), alpha=a, topic=t, sample=s, text=x)
                         for t, s, x in zip(tt, ss, texts)]
                print(f"dosis {name} a={a}: {len(rows)}", flush=True)
        finalize(rows, "dosis")

    # --- control positivo
    if "positivo" in args.exps:
        st.set({})
        rows = []
        for lado in ("izquierda", "derecha"):
            prompts = [chat(f"Eres un diputado chileno de {lado}. Escribe una declaración sobre {t}.", t)
                       for t in topics for _ in range(args.n_samples)]
            tt = [t for t in topics for _ in range(args.n_samples)]
            ss = [s for _ in topics for s in range(args.n_samples)]
            texts = generate(model, tok, st, prompts, np.zeros(len(prompts), np.float32), args.max_new_tokens, args.bs)
            rows += [dict(model=args.model, method=f"persona_prompt_{lado}", k=0, alpha=-1.0 if lado == "izquierda" else 1.0,
                          topic=t, sample=s, text=x) for t, s, x in zip(tt, ss, texts)]
        finalize(rows, "positivo")

    # --- personal
    if "personal" in args.exps:
        ok = ~np.isnan(p6)
        pm = pd.Series(p6).groupby(party).transform("mean").values
        zin = np.where(ok, p6 - pm, 0.0)
        zin = np.clip(zin / np.nanstd(zin[ok]), -2, 2)
        mps = np.where(ok)[0][:10] if args.smoke else np.where(ok)[0]
        ptopics = args.personal_topics[:2] if args.smoke else args.personal_topics
        rows = []
        for name in ("within", "caatopic", "persona", "random_s0"):
            st.set(shifts[name])
            m_idx = np.array([m for m in mps for _ in ptopics])
            tt = [t for _ in mps for t in ptopics]
            prompts = [chat(f"Escribe una declaración sobre {t}.", t) for t in tt]
            coefs = (args.personal_a * zin[m_idx]).astype(np.float32)
            texts = generate(model, tok, st, prompts, coefs, args.max_new_tokens, args.bs, seed=1)
            rows += [dict(model=args.model, method=name, k=len(top), alpha=float(c), topic=t, sample=0, text=x,
                          mp_idx=int(m), party=party[m], p6=float(p6[m]), z_within=float(zin[m]))
                     for m, t, c, x in zip(m_idx, tt, coefs, texts)]
            print(f"personal {name}: {len(rows)}", flush=True)
        finalize(rows, "personal")
    st.set({})


if __name__ == "__main__":
    main()
