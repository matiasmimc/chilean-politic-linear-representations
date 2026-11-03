import pandas as pd

df = pd.read_parquet("resultados_500_tokens/intervention_evaluated.parquet")
df['score_postura'] = pd.to_numeric(df['score_postura'], errors='coerce')
df = df.dropna(subset=['score_postura'])

print("\n=== EJEMPLO ALPHA -10 ===")
sample_neg = df[(df['alpha'] == -10) & (df['score_postura'].notna())].head(1)
if not sample_neg.empty:
    print(f"Modelo: {sample_neg['model'].values[0]}")
    print(f"Texto: {sample_neg['text'].values[0][:500]}...")
    print(f"Puntaje: {sample_neg['score_postura'].values[0]}")

print("\n=== EJEMPLO ALPHA 10 ===")
sample_pos = df[(df['alpha'] == 10) & (df['score_postura'].notna())].head(1)
if not sample_pos.empty:
    print(f"Modelo: {sample_pos['model'].values[0]}")
    print(f"Texto: {sample_pos['text'].values[0][:500]}...")
    print(f"Puntaje: {sample_pos['score_postura'].values[0]}")

