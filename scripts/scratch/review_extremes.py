import pandas as pd
from pathlib import Path

BASE_DIR = Path('/home/matias/Documentos/chilean-politic-linear-representations')
FILE_PATH = BASE_DIR / 'intervention_evaluated.parquet'

df = pd.read_parquet(FILE_PATH)

# Filtrar por coherencia aceptable (>3)
df_valid = df[df['score_coherence'] >= 4]

print("--- EXTREMOS DE IZQUIERDA (Postura 1-2) ---")
izq = df_valid[df_valid['score_postura'] <= 2].head(2)
for _, row in izq.iterrows():
    print(f"Mod: {row['model']} | Alpha: {row['alpha']} | Tema: {row['topic']}")
    print(f"Txt: {row['text'][:200]}...")
    print(f"Juez: {row['justification']}\n")

print("--- EXTREMOS DE DERECHA (Postura 6-7) ---")
der = df_valid[df_valid['score_postura'] >= 6].head(2)
for _, row in der.iterrows():
    print(f"Mod: {row['model']} | Alpha: {row['alpha']} | Tema: {row['topic']}")
    print(f"Txt: {row['text'][:200]}...")
    print(f"Juez: {row['justification']}\n")
