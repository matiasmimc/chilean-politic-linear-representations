import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from pathlib import Path
import os
import glob
import argparse

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2"
DEFAULT_PCA_DIR = BASE_DIR / "results/pca_analysis"
DEFAULT_FIGURES_DIR = BASE_DIR / "figures/pca_analysis"


def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.abs(np.dot(u, v)) / (norm_u * norm_v))


def main():
    parser = argparse.ArgumentParser(description="Descubrimiento No Supervisado (PCA) y Alineamiento con Eje Ideológico.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Directorio con archivos *_politician_features.pkl.")
    parser.add_argument("--pca-dir", type=str, default=str(DEFAULT_PCA_DIR), help="Directorio para guardar resultados de PCA.")
    parser.add_argument("--figures-dir", type=str, default=str(DEFAULT_FIGURES_DIR), help="Directorio para guardar gráficos.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pca_dir = Path(args.pca_dir)
    figures_dir = Path(args.figures_dir)
    
    pca_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_path = DATA_DIR / "congreso_2022_2026_20intervenciones.parquet"
    if not df_path.exists():
        print(f"Error: No se encuentra {df_path}")
        return
    df = pd.read_parquet(df_path)
    
    feature_files = glob.glob(str(results_dir / "*_politician_features.pkl"))
    
    # Filtrar solo modelos state-of-the-art
    sota_models = [
        'google_gemma-4-E4B-it', 
        'meta-llama_Llama-3.1-8B-Instruct', 
        'Qwen_Qwen3.5-9B'
    ]
    feature_files = [f for f in feature_files if any(sota in os.path.basename(f) for sota in sota_models)]
    
    if not feature_files:
        print(f"No se encontraron archivos *_politician_features.pkl en {results_dir}")
        return

    for file_path in feature_files:
        model_name = os.path.basename(file_path).replace("_politician_features.pkl", "")
        print(f"\n=======================================================")
        print(f"Procesando PCA y Alineamiento para: {model_name}")
        print(f"=======================================================")
        
        with open(file_path, 'rb') as f:
            features, labels = pickle.load(f)
            
        num_statements, num_layers, num_heads, d_head = features.shape
        hidden_dim = num_heads * d_head
        features_flat = features.reshape(num_statements, num_layers, hidden_dim)
        
        all_components = []
        pc1_variances = []
        pc1_ideology_alignment = []
        pc2_ideology_alignment = []
        
        # Procesar capa por capa
        for layer_idx in range(num_layers):
            X_layer = features_flat[:, layer_idx, :] # [num_statements, hidden_dim]
            y = np.array(labels, dtype=float)
            
            if np.isnan(X_layer).any() or np.isinf(X_layer).any():
                X_layer = np.nan_to_num(X_layer, nan=0.0, posinf=0.0, neginf=0.0)
                
            # 1. PCA no supervisado
            pca = PCA()
            projections = pca.fit_transform(X_layer)
            ratios = pca.explained_variance_ratio_
            components = pca.components_
            
            pc1_variances.append(ratios[0])
            
            # 2. Probe lineal supervisado (Ridge) para obtener el vector ideológico de esta capa
            ridge = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge.fit(X_layer, y)
            w_ideology = ridge.coef_
            
            # 3. Alineamiento (Cosine Similarity) entre PCA no supervisado y vector ideológico
            pc1_align = cosine_similarity(components[0], w_ideology)
            pc2_align = cosine_similarity(components[1], w_ideology) if len(components) > 1 else 0.0
            
            pc1_ideology_alignment.append(pc1_align)
            pc2_ideology_alignment.append(pc2_align)
            
            for comp_idx, (ratio, vec) in enumerate(zip(ratios, components)):
                all_components.append({
                    'layer': layer_idx,
                    'pca_component_idx': comp_idx,
                    'explained_variance_ratio': ratio,
                    'vector': vec,
                    'projection': projections[:, comp_idx],
                    'ideology_cosine_sim': cosine_similarity(vec, w_ideology)
                })
        
        all_components_sorted = sorted(all_components, key=lambda x: x['explained_variance_ratio'], reverse=True)
        top_10 = all_components_sorted[:10]
        
        print(f"Top 5 componentes principales globales para {model_name}:")
        for i, comp in enumerate(top_10[:5]):
            print(f"  {i+1}. Capa {comp['layer']}, PC{comp['pca_component_idx']+1} | Varianza: {comp['explained_variance_ratio']:.4f} | CosSim con Ideología: {comp['ideology_cosine_sim']:.4f}")
            
        output_file = pca_dir / f"{model_name}_pca_analysis.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump({
                "all_components": all_components_sorted,
                "pc1_variances": pc1_variances,
                "pc1_ideology_alignment": pc1_ideology_alignment,
                "pc2_ideology_alignment": pc2_ideology_alignment
            }, f)
            
        # Gráfico 1: Varianza explicada por PC1 a través de las capas
        plt.figure(figsize=(10, 5))
        plt.plot(range(num_layers), pc1_variances, marker='o', color='#3498db', linewidth=2)
        plt.title(f'Varianza Explicada por PC1 vs Capa\nModelo: {model_name}', fontsize=12, fontweight='bold')
        plt.xlabel('Capa del Modelo')
        plt.ylabel('Ratio de Varianza Explicada')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{model_name}_pc1_variance_by_layer.png", dpi=300)
        plt.close()
        
        # Gráfico 2: Cosine Similarity entre Eje No Supervisado (PC1) y Vector Ideológico Supervisado
        plt.figure(figsize=(10, 5))
        plt.plot(range(num_layers), pc1_ideology_alignment, marker='o', label='PC1 vs Vector Ideología', color='#e67e22', linewidth=2)
        plt.plot(range(num_layers), pc2_ideology_alignment, marker='s', label='PC2 vs Vector Ideología', color='#95a5a6', linestyle='--', linewidth=2)
        plt.title(f'Emergencia Espontánea de Ideología en Componentes No Supervisados\nModelo: {model_name}', fontsize=12, fontweight='bold')
        plt.xlabel('Capa del Modelo')
        plt.ylabel('Similitud Coseno (|cos θ|)')
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"{model_name}_pca_vs_ideology_alignment.png", dpi=300)
        plt.savefig(figures_dir / f"{model_name}_pca_vs_ideology_alignment.pdf")
        plt.close()
        
        # Gráfico 3: Scatter Plot 2D de Diputados coloreado por Dimensión Ideológica Real
        if len(top_10) >= 2:
            top1 = top_10[0]
            top2 = top_10[1]
            
            plot_df = pd.DataFrame({
                'Nombre': df['Name'],
                'Partido': df['party_1_P6'],
                'Ideologia': df['dim_ideology_P6'],
                'Top1': top1['projection'],
                'Top2': top2['projection']
            })
            
            plt.figure(figsize=(11, 7))
            scatter = plt.scatter(plot_df['Top1'], plot_df['Top2'], c=plot_df['Ideologia'], cmap='coolwarm', s=80, alpha=0.85, edgecolors='k', linewidth=0.5)
            cbar = plt.colorbar(scatter)
            cbar.set_label('Ideología Oficial (P6: Izquierda < 0 < Derecha)')
            plt.title(f'Proyección No Supervisada (PCA) Coloreada por Ideología Oficial\nModelo: {model_name}', fontsize=12, fontweight='bold')
            plt.xlabel(f"Global Top 1 (Capa {top1['layer']}, PC{top1['pca_component_idx']+1})")
            plt.ylabel(f"Global Top 2 (Capa {top2['layer']}, PC{top2['pca_component_idx']+1})")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(figures_dir / f"{model_name}_scatter_pca_ideology.png", dpi=300)
            plt.savefig(figures_dir / f"{model_name}_scatter_pca_ideology.pdf")
            plt.close()

    print("\n✅ Análisis de PCA y alineamiento no supervisado completado con éxito.")


if __name__ == "__main__":
    main()
