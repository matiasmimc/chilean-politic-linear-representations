import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, help="Tema específico a analizar (ej. aborto)")
    parser.add_argument("--folder", type=str, default="steering_100", help="Carpeta de resultados (ej. steering_100, steering_200)")
    args = parser.parse_args()

    # Configuración de Rutas
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE_DIR = SCRIPT_DIR.parent.parent
    
    if args.topic:
        safe_topic = args.topic.replace(' ', '_')
        INPUT_FILE = BASE_DIR / f'results/{args.folder}/intervention_evaluated_{safe_topic}.parquet'
        FIGURES_DIR = BASE_DIR / "figures" / args.folder / "steering_por_tema" / safe_topic
    else:
        INPUT_FILE = BASE_DIR / f'results/{args.folder}/intervention_evaluated.parquet'
        FIGURES_DIR = BASE_DIR / "figures" / args.folder
        
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} no existe. Ejecuta evaluar_respuestas.py primero.")
        return

    df = pd.read_parquet(INPUT_FILE)
    
    # Limpieza básica: convertir puntajes a numéricos y filtrar fallos
    df['score_coherence'] = pd.to_numeric(df['score_coherence'], errors='coerce')
    df['score_postura'] = pd.to_numeric(df['score_postura'], errors='coerce')
    
    initial_len = len(df)
    df = df.dropna(subset=['score_coherence', 'score_postura'])
    print(f"Limpieza: {len(df)} registros válidos de {initial_len} totales.")

    df_slant = df[(df['alpha'] >= -30) & (df['alpha'] <= 30)]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_slant, x='alpha', y='score_postura', hue='model', style='model', markers=True, dashes=False, errorbar='sd', err_style='bars', err_kws={'capsize': 3})
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Centro')
    plt.title(f'Calibración: Alpha vs Postura Política{" - " + args.topic if args.topic else ""}', pad=20)
    plt.xlabel('Alpha (Fuerza del Steering)')
    plt.ylabel('Puntaje Ideológico (Juez LLM)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'calibration_steering.png', dpi=300)
    print(f"Gráfico de calibración guardado en {FIGURES_DIR / 'calibration_steering.png'}")

    # 2. Gráfico de Coherencia: Alpha vs Coherencia
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='alpha', y='score_coherence', hue='model', marker='o')
    plt.title(f'Degradación del Lenguaje vs Alpha{" - " + args.topic if args.topic else ""}', pad=20)
    plt.xlabel('Alpha')
    plt.ylabel('Puntaje de Coherencia (1-5)')
    plt.ylim(1, 5.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'coherence_degradation.png', dpi=300)
    print(f"Gráfico de coherencia guardado en {FIGURES_DIR / 'coherence_degradation.png'}")

    # 3. Gráfico de Trade-off: Desviación Ideológica vs Coherencia
    # Calculamos la desviación del centro (4)
    df['ideological_shift'] = (df['score_postura'] - 4).abs()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='ideological_shift', y='score_coherence', hue='model', alpha=0.5)
    plt.title(f'Trade-off: Desviación vs Coherencia{" - " + args.topic if args.topic else ""}', pad=20)
    plt.xlabel('Desviación Absoluta del Centro (Shift)')
    plt.ylabel('Coherencia')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'steering_tradeoff.png', dpi=300)
    print(f"Gráfico de trade-off guardado en {FIGURES_DIR / 'steering_tradeoff.png'}")

    # Estadísticas resumidas por modelo
    summary = df.groupby(['model', 'alpha']).agg({
        'score_postura': ['mean', 'std'],
        'score_coherence': 'mean'
    }).reset_index()
    
    if args.topic:
        safe_topic = args.topic.replace(' ', '_')
        csv_path = BASE_DIR / f'results/steering/steering_summary_stats_{safe_topic}.csv'
    else:
        csv_path = BASE_DIR / 'results/steering/steering_summary_stats.csv'
        
    summary.to_csv(csv_path, index=False)
    print(f"Estadísticas resumidas guardadas en {csv_path}")

if __name__ == "__main__":
    main()
