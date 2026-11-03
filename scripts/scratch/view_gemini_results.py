import pandas as pd
from pathlib import Path

BASE_DIR = Path('/home/matias/Documentos/chilean-politic-linear-representations')
FILE_PATH = BASE_DIR / 'intervention_evaluated_gemini.parquet'

df = pd.read_parquet(FILE_PATH)

print("--- DISTRIBUCIÓN DE POSTURAS (Gemini 3 Flash) ---")
dist = df['score_postura'].value_counts().sort_index()
print(dist)

print("\n--- ESTADÍSTICAS POR ALPHA ---")
stats = df.groupby('alpha')['score_postura'].agg(['mean', 'std', 'min', 'max'])
print(stats)

print("\n--- EJEMPLOS EXTREMOS ---")
# Izquierda (-3)
izq = df[df['score_postura'] <= -2.5].head(1)
if not izq.empty:
    row = izq.iloc[0]
    print(f"IZQUIERDA (-3) | Alpha: {row['alpha']} | Tema: {row['topic']}")
    print(f"Texto: {row['text'][:250]}...")
    print(f"Juez: {row['justification']}\n")

# Derecha (+3)
der = df[df['score_postura'] >= 2.5].head(1)
if not der.empty:
    row = der.iloc[0]
    print(f"DERECHA (+3) | Alpha: {row['alpha']} | Tema: {row['topic']}")
    print(f"Texto: {row['text'][:250]}...")
    print(f"Juez: {row['justification']}\n")
