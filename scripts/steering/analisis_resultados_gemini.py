import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
FIGURES_DIR = BASE_DIR / "figures" / "steering_gemini"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = BASE_DIR / 'results/steering/intervention_evaluated_gemini.parquet'

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} no existe.")
        return

    df = pd.read_parquet(INPUT_FILE)
    
    # Limpieza: convertir puntajes a numéricos y filtrar fallos
    df['score_postura'] = pd.to_numeric(df['score_postura'], errors='coerce')
    
    initial_len = len(df)
    df = df.dropna(subset=['score_postura'])
    print(f"Limpieza: {len(df)} registros válidos de {initial_len} totales.")

    # 1. Gráfico de Calibración: Alpha vs Postura Política (Rango -20 a 20)
    plt.figure(figsize=(10, 6))
    
    # Filtrar rango alpha para visualización clara (rango completo)
    df_filtered = df[(df['alpha'] >= -50) & (df['alpha'] <= 50)]
    
    sns.lineplot(data=df_filtered, x='alpha', y='score_postura', marker='o', errorbar='sd')
    
    plt.title('Calibración de Steering Político (Juez: Gemini 3.1 Flash Lite)', fontsize=14)
    plt.xlabel('Alpha (Fuerza de Intervención)', fontsize=12)
    plt.ylabel('Postura Política (-3: Izq, +3: Der)', fontsize=12)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.ylim(-3.5, 3.5)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(FIGURES_DIR / "calibration_steering_gemini.png", dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {FIGURES_DIR / 'calibration_steering_gemini.png'}")

    # 2. Distribución por Tema (Boxplot)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_filtered, x='topic', y='score_postura')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribución de Posturas por Tema', fontsize=14)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.ylim(-3.5, 3.5)
    
    plt.savefig(FIGURES_DIR / "topic_distribution_gemini.png", dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {FIGURES_DIR / 'topic_distribution_gemini.png'}")

if __name__ == "__main__":
    main()
