import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results_replication"
FIGURES_DIR = BASE_DIR / "figures"

# Asegurar que el directorio de figuras existe
FIGURES_DIR.mkdir(exist_ok=True)

# Configuración de estilo
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

def generar_figura_2(models):
    """Genera Heatmaps de Accuracy por Capa y Cabeza."""
    print("\n--- Generando Figura 2: Heatmaps de Accuracy ---")
    
    for model_name in models:
        safe_name = model_name.replace("/", "_")
        path = RESULTS_DIR / f"{safe_name}_politician_performance.pkl"
        
        if not path.exists():
            print(f"Archivo no encontrado: {path}")
            continue
            
        with open(path, 'rb') as f:
            performance = pickle.load(f)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(performance, cmap='viridis', cbar_kws={'label': 'Spearman Correlation'})
        plt.xlabel('Attention Heads')
        plt.ylabel('Layers')
        plt.title(f'Layer-Head Accuracy - {model_name}')
        plt.tight_layout()
        
        save_path = FIGURES_DIR / f"{safe_name}_heatmap.pdf"
        plt.savefig(save_path)
        plt.savefig(save_path.with_suffix('.png')) # También guardamos en PNG para previsualización rápida
        plt.close()
        
        # Reporte de las Top 20 Heads
        top_indices = np.dstack(np.unravel_index(np.argsort(performance.ravel()), performance.shape))[0][-20:, :][::-1]
        print(f"Top 20 Heads for {model_name} guardadas.")
        
        with open(FIGURES_DIR / f"{safe_name}_top_heads.txt", "w") as f_out:
            f_out.write(f"Top 20 Heads for {model_name}\n")
            f_out.write("Layer, Head\n")
            for l, h in top_indices:
                f_out.write(f"{l}, {h}\n")

def generar_figura_3(models):
    """Genera resultados de intervención."""
    print("\n--- Generando Figura 3: Resultados de Intervención ---")
    
    for model_name in models:
        safe_name = model_name.replace("/", "_")
        results_path = RESULTS_DIR / f"{safe_name}_intervention_results.pkl"
        
        if not results_path.exists():
            print(f"Archivo no encontrado: {results_path}")
            continue
            
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
            
        # results format: [k, alpha, topic, model_gen_str]
        df_res = pd.DataFrame(results, columns=['k', 'alpha', 'topic', 'text'])
        
        if df_res.empty:
            continue

        plt.figure(figsize=(12, 6))
        sns.countplot(data=df_res, x='alpha', hue='k')
        plt.title(f'Distribución de muestras de intervención ({model_name})')
        plt.xlabel('Alpha (Intensidad Intervención)')
        plt.ylabel('Cantidad de muestras')
        plt.legend(title='Top-K Heads', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = FIGURES_DIR / f"{safe_name}_intervention_dist.pdf"
        plt.savefig(save_path)
        plt.savefig(save_path.with_suffix('.png'))
        plt.close()
        print(f"Figura de intervención para {model_name} guardada.")

def main():
    models = ['meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', 'lmsys/vicuna-7b-v1.5']
    
    generar_figura_2(models)
    generar_figura_3(models)
    
    print(f"\nProceso finalizado. Las figuras se encuentran en: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
