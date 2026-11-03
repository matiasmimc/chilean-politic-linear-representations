import pandas as pd
from pathlib import Path

BASE_DIR = Path('/home/matias/Documentos/chilean-politic-linear-representations')
FILE_PATH = BASE_DIR / 'intervention_evaluated.parquet'

df = pd.read_parquet(FILE_PATH)

# Filtrar por un modelo y tema para ver la progresión
example_topic = 'aborto'
example_model = 'meta-llama/Meta-Llama-3-8B-Instruct'

subset = df[(df['topic'] == example_topic) & (df['model'] == example_model)].sort_values('alpha')

print(f"Pivoteando sobre tema: {example_topic} | Modelo: {example_model}\n")

alphas_to_show = [-20, 0, 20]

for alpha in alphas_to_show:
    row = subset[subset['alpha'] == alpha].iloc[0]
    print(f"ALPHA: {alpha}")
    print(f"TEXTO: {row['text'][:300]}...")
    print(f"SCORES: Coherencia={row['score_coherence']} | Postura={row['score_postura']}")
    print(f"JUEZ: {row['justification']}")
    print("-" * 50)
