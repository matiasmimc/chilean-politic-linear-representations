import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_FULL_DIR = BASE_DIR / "results/replication/v2"
DEFAULT_NAMES_DIR = BASE_DIR / "results/replication/v2_names_only"
DEFAULT_FIGURES_DIR = BASE_DIR / "figures/comparacion_baselines"

plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12


def load_model_performance(perf_file):
    with open(perf_file, 'rb') as f:
        perf = pickle.load(f)
    return perf


def format_model_label(filename):
    name = filename.replace("_politician_performance.pkl", "")
    # Simplificar nombres para gráficos
    name = name.replace("meta-llama_", "").replace("google_", "").replace("Qwen_", "").replace("mistralai_", "")
    return name


def main():
    parser = argparse.ArgumentParser(description="Comparación de rendimiento de Probing: Baseline (Names-Only) vs Full Context (Intervenciones).")
    parser.add_argument("--full-dir", type=str, default=str(DEFAULT_FULL_DIR), help="Ruta a resultados con intervenciones.")
    parser.add_argument("--names-dir", type=str, default=str(DEFAULT_NAMES_DIR), help="Ruta a resultados solo con nombres.")
    parser.add_argument("--figures-dir", type=str, default=str(DEFAULT_FIGURES_DIR), help="Ruta donde guardar gráficos y tabla resumen.")
    args = parser.parse_args()

    full_dir = Path(args.full_dir)
    names_dir = Path(args.names_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    full_files = {p.name: p for p in full_dir.glob("*_politician_performance.pkl")}
    names_files = {p.name: p for p in names_dir.glob("*_politician_performance.pkl")}

    all_models = sorted(list(set(list(full_files.keys()) + list(names_files.keys()))))
    
    records = []
    
    for fname in all_models:
        model_label = format_model_label(fname)
        
        has_full = fname in full_files
        has_names = fname in names_files
        
        full_max, full_mean = np.nan, np.nan
        names_max, names_mean = np.nan, np.nan
        
        if has_full:
            perf_full = load_model_performance(full_files[fname])
            full_max = np.max(perf_full)
            full_mean = np.mean(perf_full)
            
        if has_names:
            perf_names = load_model_performance(names_files[fname])
            names_max = np.max(perf_names)
            names_mean = np.mean(perf_names)
            
        records.append({
            "filename": fname,
            "model": model_label,
            "full_context_max_spearman": full_max,
            "full_context_mean_spearman": full_mean,
            "names_only_max_spearman": names_max,
            "names_only_mean_spearman": names_mean,
            "delta_max": full_max - names_max if pd.notna(full_max) and pd.notna(names_max) else np.nan
        })

    df_summary = pd.DataFrame(records)
    csv_path = figures_dir / "comparacion_rendimiento_probing.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\n📊 Tabla resumen guardada en: {csv_path}\n")
    print(df_summary.to_string())

    # 1. Gráfico de Barras: Comparación Max Spearman
    plot_data = []
    for _, row in df_summary.iterrows():
        if pd.notna(row['names_only_max_spearman']):
            plot_data.append({"Model": row['model'], "Type": "Names Only (Baseline)", "Spearman": row['names_only_max_spearman']})
        if pd.notna(row['full_context_max_spearman']):
            plot_data.append({"Model": row['model'], "Type": "Full Context (20 Intervenciones)", "Spearman": row['full_context_max_spearman']})

    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        plt.figure(figsize=(12, 6))
        palette = {'Names Only (Baseline)': '#e74c3c', 'Full Context (20 Intervenciones)': '#2ecc71'}
        sns.barplot(data=df_plot, x='Model', y='Spearman', hue='Type', palette=palette)
        plt.title('Impacto del Contexto Parlamentario en la Representación Política (Linear Probing)', fontsize=14, fontweight='bold')
        plt.ylabel('Máximo Spearman Correlation (Ideología vs Probe)')
        plt.xlabel('Modelo')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=25, ha='right')
        plt.legend(title='Condición de Entrada')
        plt.tight_layout()
        
        save_bar = figures_dir / "comparacion_max_spearman.png"
        plt.savefig(save_bar, dpi=300)
        plt.savefig(save_bar.with_suffix(".pdf"))
        plt.close()
        print(f"📈 Gráfico de barras guardado en: {save_bar}")

    # 2. Curvas de Rendimiento por Capa (para modelos disponibles en ambas condiciones)
    paired_models = [fname for fname in all_models if fname in full_files and fname in names_files]
    if paired_models:
        for fname in paired_models:
            model_label = format_model_label(fname)
            perf_full = load_model_performance(full_files[fname])
            perf_names = load_model_performance(names_files[fname])
            
            # Max por capa
            layer_max_full = np.max(perf_full, axis=1)
            layer_max_names = np.max(perf_names, axis=1)
            
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(layer_max_full)), layer_max_full, label='Full Context (Intervenciones)', marker='o', color='#2ecc71', linewidth=2)
            plt.plot(range(len(layer_max_names)), layer_max_names, label='Names Only (Baseline)', marker='s', color='#e74c3c', linestyle='--', linewidth=2)
            plt.title(f'Evolución del Rendimiento por Capa: {model_label}', fontsize=13, fontweight='bold')
            plt.xlabel('Capa del Modelo')
            plt.ylabel('Máximo Spearman Correlation')
            plt.ylim(0, 1.0)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            
            save_curve = figures_dir / f"{model_label}_layerwise_comparison.png"
            plt.savefig(save_curve, dpi=300)
            plt.savefig(save_curve.with_suffix(".pdf"))
            plt.close()
            print(f"📈 Curva de capas guardada para {model_label} en: {save_curve}")


if __name__ == "__main__":
    main()
