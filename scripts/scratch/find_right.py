import pandas as pd
from pathlib import Path

BASE_DIR = Path('/home/matias/Documentos/chilean-politic-linear-representations')
FILE_PATH = BASE_DIR / 'intervention_evaluated.parquet'

df = pd.read_parquet(FILE_PATH)

print("--- EJEMPLO DE DERECHA (Score 6) ---")
der = df[df['score_postura'] >= 6].head(1)
if not der.empty:
    row = der.iloc[0]
    print(f"Modelo: {row['model']} | Alpha: {row['alpha']} | Tema: {row['topic']}")
    print(f"Texto: {row['text'][:500]}...")
    print(f"Evaluación: Coherencia={row['score_coherence']}, Postura={row['score_postura']}")
    print(f"Justificación: {row['justification']}")
else:
    print("No se encontró ningún ejemplo con score >= 6.")
