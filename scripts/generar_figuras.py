import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# Configuración de Rutas Base
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2_names_only"
DEFAULT_FIGURES_DIR = BASE_DIR / "figures/v2_names_only"

# Configuración de estilo
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12


def generar_figura_2_para_archivo(perf_path, figures_dir):
    """Genera Heatmaps de Accuracy por Capa y Cabeza y lista Top 20."""
    safe_name = perf_path.name.replace("_politician_performance.pkl", "")
    print(f"\n--- Generando Heatmap y Top Heads para: {safe_name} ---")
    
    with open(perf_path, 'rb') as f:
        performance = pickle.load(f)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(performance, cmap='viridis', cbar_kws={'label': 'Spearman Correlation'})
    plt.xlabel('Attention Heads')
    plt.ylabel('Layers')
    plt.title(f'Layer-Head Accuracy (Spearman) - {safe_name}')
    plt.tight_layout()
    
    save_path = figures_dir / f"{safe_name}_heatmap.pdf"
    plt.savefig(save_path)
    plt.savefig(save_path.with_suffix('.png'))
    plt.close()
    
    # Reporte de las Top 20 Heads
    top_indices = np.dstack(np.unravel_index(np.argsort(performance.ravel()), performance.shape))[0][-20:, :][::-1]
    
    top_heads_path = figures_dir / f"{safe_name}_top_heads.txt"
    with open(top_heads_path, "w") as f_out:
        f_out.write(f"Top 20 Heads for {safe_name}\n")
        f_out.write("Rank, Layer, Head, Spearman_Correlation\n")
        for rank, (l, h) in enumerate(top_indices, 1):
            score = performance[l, h]
            f_out.write(f"{rank}, {l}, {h}, {score:.4f}\n")
            
    print(f"Heatmap guardado en: {save_path.with_suffix('.png')}")
    print(f"Top 20 Heads guardadas en: {top_heads_path}")
    print(f"Máximo Spearman Correlation: {np.max(performance):.4f}")


def generar_figura_3(results_dir, figures_dir):
    """Genera resultados de intervención si existen."""
    intervention_files = list(results_dir.glob("*_intervention_results.pkl"))
    
    # Filtrar solo modelos state-of-the-art
    sota_models = [
        'google_gemma-4-E4B-it', 
        'meta-llama_Llama-3.1-8B-Instruct', 
        'Qwen_Qwen3.5-9B'
    ]
    intervention_files = [f for f in intervention_files if any(sota in f.name for sota in sota_models)]
    
    if not intervention_files:
        print("No se encontraron archivos de intervención para los modelos seleccionados.")
        return

    print("\n--- Generando Figuras de Intervención ---")
    for res_path in intervention_files:
        safe_name = res_path.name.replace("_intervention_results.pkl", "")
        with open(res_path, 'rb') as f:
            results = pickle.load(f)
            
        df_res = pd.DataFrame(results, columns=['k', 'alpha', 'topic', 'text'])
        if df_res.empty:
            continue

        plt.figure(figsize=(12, 6))
        sns.countplot(data=df_res, x='alpha', hue='k')
        plt.title(f'Distribución de muestras de intervención ({safe_name})')
        plt.xlabel('Alpha (Intensidad Intervención)')
        plt.ylabel('Cantidad de muestras')
        plt.legend(title='Top-K Heads', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = figures_dir / f"{safe_name}_intervention_dist.pdf"
        plt.savefig(save_path)
        plt.savefig(save_path.with_suffix('.png'))
        plt.close()
        print(f"Figura de intervención guardada: {save_path.with_suffix('.png')}")


def main():
    parser = argparse.ArgumentParser(description="Generación de figuras y reportes de probing a partir de los resultados.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Directorio donde residen los archivos de resultados.")
    parser.add_argument("--figures-dir", type=str, default=str(DEFAULT_FIGURES_DIR), help="Directorio donde guardar las figuras y reportes.")
    parser.add_argument("--model", type=str, default=None, help="Filtrar por modelo específico (opcional).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Error: No existe el directorio de resultados {results_dir}")
        return

    if args.model:
        perf_files = [f for f in results_dir.glob("*_politician_performance.pkl") if args.model.lower().replace("/", "_") in f.name.lower()]
    else:
        perf_files = sorted(list(results_dir.glob("*_politician_performance.pkl")))
        
    # Filtrar solo modelos state-of-the-art
    sota_models = [
        'google_gemma-4-E4B-it', 
        'meta-llama_Llama-3.1-8B-Instruct', 
        'Qwen_Qwen3.5-9B'
    ]
    perf_files = [f for f in perf_files if any(sota in f.name for sota in sota_models)]

    if not perf_files:
        print(f"No se encontraron archivos '*_politician_performance.pkl' en {results_dir}")
        print("Saltando la generación de Heatmaps de precisión (Figura 2)...")
    else:
        print(f"Encontrados {len(perf_files)} archivo(s) de rendimiento en {results_dir}:")
        for f in perf_files:
            print(f" - {f.name}")

        for perf_file in perf_files:
            generar_figura_2_para_archivo(perf_file, figures_dir)

    generar_figura_3(results_dir, figures_dir)
    
    print(f"\n¡Figuras y reportes generados con éxito en: {figures_dir}!")


if __name__ == "__main__":
    main()
