import pandas as pd
from pathlib import Path

BASE_DIR = Path('/home/matias/Documentos/chilean-politic-linear-representations')
FILE_PATH = BASE_DIR / 'intervention_evaluated.parquet'

df = pd.read_parquet(FILE_PATH)
df = df.dropna(subset=['score_postura'])

print("Distribución de Postura por Alpha:")
print(df.groupby('alpha')['score_postura'].describe()[['mean', 'min', 'max']])
