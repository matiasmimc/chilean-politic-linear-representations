"""Juez rapido (GPU): mismos prompts y escala que `evaluar_respuestas.py`, pero el juez responde solo el numero
({"coherencia": n} / {"postura": n}, ~14 tokens) en vez de JSON con justificacion. Salida compatible:
results/<folder>/intervention_evaluated_<modelo>.parquet con score_coherence y score_postura.

Uso: CUDA_VISIBLE_DEVICES=1 uv run python scripts/steering/juez_rapido.py --folders slant_v3/positivo slant_v3/dosis --model Llama
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluar_respuestas as E  # noqa: E402

BASE_DIR = E.BASE_DIR
CUT = "RESPONDE ÚNICAMENTE CON UN JSON"


def short(prompt, key, spec):
    return prompt[:prompt.index(CUT)] + f'{CUT} (sin explicaciones): {{"{key}": {spec}}}'


def first_number(resp):
    m = re.search(r'-?\d+(?:\.\d+)?', resp.split(":", 1)[-1] if ":" in resp else resp)
    return float(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", nargs="+", required=True)
    ap.add_argument("--model", default="Llama")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--judge-model", default=E.JUDGE_MODEL_NAME, help="otro juez (p. ej. Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--tag", default="", help="sufijo del archivo de salida cuando se usa otro juez")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.judge_model, cache_dir=str(E.MODEL_CACHE), token=E.HUGGINGFACE_TOKEN)
    tok.padding_side = "left"
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")]
    eos = sorted({t for t in eos if isinstance(t, int) and t > 0})
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    kw = {}
    if "27b" in args.judge_model.lower():
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                       bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.judge_model, cache_dir=str(E.MODEL_CACHE), token=E.HUGGINGFACE_TOKEN,
                                                 device_map="auto", **kw)

    def judge(prompts):
        convs = [[{"role": "system", "content": "Responde siempre en formato JSON."}, {"role": "user", "content": p}]
                 for p in prompts]
        enc = tok.apply_chat_template(convs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                      padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=14, do_sample=False, eos_token_id=eos, pad_token_id=pad)
        return [tok.decode(o[enc["input_ids"].shape[-1]:], skip_special_tokens=True) for o in out]

    for folder in args.folders:
        df = pd.read_parquet(BASE_DIR / f"results/{folder}/intervention_raw.parquet")
        df = df[df.model.str.contains(args.model, case=False, na=False)].reset_index(drop=True)
        texts = [str(t)[:2000] + ("... [TRUNCADO POR EXCESO DE LONGITUD]" if len(str(t)) > 2000 else "") for t in df.text]
        order = sorted(range(len(df)), key=lambda i: len(texts[i]))
        coh, pos = [None] * len(df), [None] * len(df)
        for s in tqdm(range(0, len(order), args.batch_size), desc=folder):
            idx = order[s:s + args.batch_size]
            rc = judge([short(E.get_judge_prompt_coherence(texts[i]), "coherencia", "[número 1-5]") for i in idx])
            rp = judge([short(E.get_judge_prompt_posture(df.topic[i], texts[i]), "postura", "[número entre -2 y 2]") for i in idx])
            for i, a, b in zip(idx, rc, rp):
                coh[i], pos[i] = first_number(a), first_number(b)
        df["score_coherence"], df["score_postura"] = coh, pos
        out = BASE_DIR / f"results/{folder}/intervention_evaluated_{args.model}{args.tag}.parquet"
        df.to_parquet(out)
        print(f"{folder}: {len(df)} -> {out}; sin parsear: coh {df.score_coherence.isna().sum()} pos {df.score_postura.isna().sum()}",
              flush=True)


if __name__ == "__main__":
    main()
