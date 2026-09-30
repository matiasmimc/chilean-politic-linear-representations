import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
results_dir = BASE_DIR / 'results/voting_predictions'
figures_dir = BASE_DIR / 'figures/steering_200'
figures_dir.mkdir(parents=True, exist_ok=True)

def load_data():
    files = list(results_dir.glob("*.parquet"))
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        parts = f.name.replace("voting_predictions_", "").replace(".parquet", "").split("_")
        
        # Parse model name (it can have multiple underscores)
        # e.g., google_gemma-4-E4B-it_pca.parquet -> parts = ['google', 'gemma-4-E4B-it', 'pca']
        # Wait, bcall_d1 has an underscore: 'bcall', 'd1'
        
        if parts[-2] == 'bcall':
            method = parts[-2] + "_" + parts[-1]
            model = "_".join(parts[:-2])
        else:
            method = parts[-1]
            model = "_".join(parts[:-1])
        
        # Mapeo nombres
        method_map = {'pca': 'PCA', 'cpca': 'cPCA', 'nmf': 'NMF', 'cnmf': 'cNMF', 'ridge': 'Ridge'}
        method = method_map.get(method, method)
        
        # Simplificar nombre del modelo
        model = model.replace("google_", "").replace("meta-llama_", "").replace("Qwen_", "")
        
        df['model'] = model
        df['method'] = method
        dfs.append(df)
        
    return pd.concat(dfs, ignore_index=True)

def main():
    all_results = load_data()
    
    # Descartar Qwen 3.5
    all_results = all_results[~all_results['model'].str.contains('Qwen', case=False)]
    
    # Calcular accuracy por modelo, método y alpha
    accuracy_df = all_results.groupby(['model', 'method', 'alpha'])['acierto'].mean().reset_index()
    
    # Baseline: accuracy para alpha=0
    baseline_df = accuracy_df[accuracy_df['alpha'] == 0].groupby('model')['acierto'].mean().reset_index()
    baseline_df['method'] = 'Baseline (α=0)'
    baseline_df['alpha'] = 0
    
    # Encontrar el mejor alpha para cada modelo y método
    best_idx = accuracy_df.groupby(['model', 'method'])['acierto'].idxmax()
    best_results = accuracy_df.loc[best_idx].copy()
    
    # Combinar con baseline
    best_results = pd.concat([best_results, baseline_df], ignore_index=True)
    best_results.rename(columns={'acierto': 'best_accuracy'}, inplace=True)
    
    # Generar tabla pivot con accuracies (y alphas)
    def format_accuracy_alpha(row):
        acc = row['best_accuracy'] * 100
        alpha = row['alpha']
        if row['method'] == 'Baseline (α=0)':
            return f"{acc:.1f}%"
        return f"{acc:.1f}% (α={alpha})"
        
    def format_accuracy_latex(row):
        acc = row['best_accuracy'] * 100
        alpha = row['alpha']
        if row['method'] == 'Baseline (α=0)':
            return f"{acc:.1f}\\%"
        return f"{acc:.1f}\\% \\scriptsize{{(α={alpha})}}"
        
    best_results['Accuracy_MD'] = best_results.apply(format_accuracy_alpha, axis=1)
    best_results['Accuracy_TEX'] = best_results.apply(format_accuracy_latex, axis=1)
    
    # Orden de columnas para la tabla
    order = ['Baseline (α=0)', 'Ridge', 'bcall_d1', 'bcall_d2', 'PCA', 'NMF', 'cPCA', 'cNMF']
    methods_present = [m for m in order if m in best_results['method'].unique()]
    
    # Pivot MD
    pivot_md = best_results.pivot(index='model', columns='method', values='Accuracy_MD')
    pivot_md = pivot_md[methods_present] # reordenar
    
    with open(figures_dir / "resultados_steering.md", "w") as f:
        f.write("### Resultados de Precisión en Predicción de Votos (con mejor α)\n\n")
        f.write(pivot_md.to_markdown())
        
    # Pivot TEX
    pivot_tex = best_results.pivot(index='model', columns='method', values='Accuracy_TEX')
    pivot_tex = pivot_tex[methods_present]
    
    with open(figures_dir / "resultados_steering.tex", "w") as f:
        # A simple latex table export
        tex_str = pivot_tex.to_latex(escape=False, column_format='l' + 'c'*len(methods_present))
        f.write(tex_str)

    # GRAFICO
    # Para el gráfico es mejor mostrar solo la accuracy (sin el alpha anotado) para no saturar.
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['font.size'] = 12
    sns.set_theme(style="whitegrid")
    
    plt.figure()
    
    # Colores bonitos, destacando el baseline
    palette = sns.color_palette("husl", len(methods_present) - 1)
    color_dict = {m: c for m, c in zip([m for m in methods_present if m != 'Baseline (α=0)'], palette)}
    color_dict['Baseline (α=0)'] = '#34495e' # Gris oscuro
    
    ax = sns.barplot(
        data=best_results, 
        x='model', 
        y='best_accuracy', 
        hue='method', 
        hue_order=methods_present,
        palette=color_dict,
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.title('Impacto del Steering en la Predicción de Votos (Mejor α)', fontsize=16, pad=15)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Modelo', fontsize=14)
    plt.ylim(0.4, 1.0) # Zoom para ver diferencias
    plt.xticks(rotation=15)
    
    # Mejorar la leyenda
    plt.legend(title='Método de Steering', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.tight_layout()
    
    plt.savefig(figures_dir / 'prediccion_votos_steering.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'prediccion_votos_steering.pdf', bbox_inches='tight')
    print("Figuras y tablas guardadas en", figures_dir)
    print("\nTabla Markdown:")
    print(pivot_md.to_markdown())

if __name__ == '__main__':
    main()
