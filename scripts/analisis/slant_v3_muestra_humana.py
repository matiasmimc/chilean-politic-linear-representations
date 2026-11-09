"""Hoja de anotacion humana ciega para validar al juez de slant v3 (CPU).

Muestrea textos de los tres modelos (caatopic alpha -2/+2, aleatoria s0 alpha -2/+2 y prompts de persona izquierda/derecha),
los mezcla y escribe results/slant_v3_validacion/anotacion_humana.csv (id, tema, texto, columnas vacias postura_humana
[-2..2] y coherencia_humana [1..5]) y clave.csv (id, modelo, metodo, alpha, puntaje de cada juez disponible).
Uso: uv run python scripts/analisis/slant_v3_muestra_humana.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent.parent / "results"
OUT = R / "slant_v3_validacion"
MODELOS = {"llama": ("slant_v3", "Llama"), "gemma": ("slant_v3_gemma", "gemma"), "glm": ("slant_v3_glm", "glm")}
PLAN = [("dosis", "caatopic", -2.0, 10), ("dosis", "caatopic", 2.0, 10), ("dosis", "random_s0", -2.0, 5),
        ("dosis", "random_s0", 2.0, 5), ("positivo", "persona_prompt_izquierda", None, 5),
        ("positivo", "persona_prompt_derecha", None, 5)]


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.RandomState(0)
    rows = []
    for mod, (root, jtag) in MODELOS.items():
        for exp, metodo, alpha, n in PLAN:
            f = R / root / exp / f"intervention_evaluated_{jtag}.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(f)
            d = d[(d.method == metodo) & (pd.to_numeric(d.score_coherence, errors="coerce") >= 4)]
            if alpha is not None:
                d = d[d.alpha == alpha]
            if len(d):
                rows.append(d.sample(n=min(n, len(d)), random_state=int(rng.randint(1e6))).assign(modelo=mod, exp=exp))
    df = pd.concat(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    df["id"] = np.arange(len(df))
    cols = [c for c in ("score_postura", "score_coherence", "slant_logit") if c in df]
    df[["id", "topic", "text"]].rename(columns={"topic": "tema", "text": "texto"}).assign(
        postura_humana="", coherencia_humana="").to_csv(OUT / "anotacion_humana.csv", index=False)
    df[["id", "modelo", "exp", "method", "alpha", "topic"] + cols].to_csv(OUT / "clave.csv", index=False)
    print(f"{len(df)} textos -> {OUT}")


if __name__ == "__main__":
    main()
