import pandas as pd
from pathlib import Path

BASE_DIR = Path('/home/matias/Documentos/chilean-politic-linear-representations')
FILE_PATH = BASE_DIR / 'intervention_evaluated.parquet'

if not FILE_PATH.exists():
    print(f"Error: {FILE_PATH} no existe.")
else:
    df = pd.read_parquet(FILE_PATH)
    
    # Seleccionar algunos ejemplos interesantes
    # 1. Baseline (alpha=0)
    baseline = df[df['alpha'] == 0].head(1)
    
    # 2. Steering positivo (Derecha)
    right = df[df['alpha'] == 20].head(1)
    
    # 3. Steering negativo (Izquierda)
    left = df[df['alpha'] == -20].head(1)
    
    # 4. Incoherente (Extremo)
    incoherent = df[df['alpha'].abs() == 50].head(1)
    
    examples = pd.concat([baseline, right, left, incoherent])
    
    for idx, row in examples.iterrows():
        print(f"--- Ejemplo {idx} ---")
        print(f"Modelo: {row['model']}")
        print(f"Tema: {row['topic']}")
        print(f"Alpha: {row['alpha']}")
        print(f"Texto: {row['text'][:200]}...")
        print(f"Evaluación:")
        print(f"  - Coherencia: {row['score_coherence']}")
        print(f"  - Postura: {row['score_postura']}")
        print(f"  - Justificación: {row['justification']}")
        print("-" * 30)
