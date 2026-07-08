import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
import os
import glob

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results/replication/v2"
PCA_RESULTS_DIR = BASE_DIR / "results/pca_analysis"
FIGURES_DIR = BASE_DIR / "figures/pca_analysis"

PCA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1. Cargar el DataFrame original para obtener metadatos (partidos, nombres)
    df_path = DATA_DIR / "congreso_2022_2026_20intervenciones.parquet"
    if not df_path.exists():
        print(f"Error: No se encuentra {df_path}")
        return
    df = pd.read_parquet(df_path)
    
    # Buscar todos los archivos de características generados por extraer_activaciones.py
    feature_files = glob.glob(str(RESULTS_DIR / "*_politician_features.pkl"))
    
    if not feature_files:
        print(f"No se encontraron archivos *_politician_features.pkl en {RESULTS_DIR}")
        return

    for file_path in feature_files:
        model_name = os.path.basename(file_path).replace("_politician_features.pkl", "")
        print(f"\n--- Procesando PCA para el modelo: {model_name} ---")
        
        # Cargar características
        with open(file_path, 'rb') as f:
            features, labels = pickle.load(f)
            
        # features tiene forma [num_statements, layers, heads, d_head]
        # Aplanar cabezas de atención: [N, L, H, D] -> [N, L, H*D]
        num_statements, num_layers, num_heads, d_head = features.shape
        hidden_dim = num_heads * d_head
        features_flat = features.reshape(num_statements, num_layers, hidden_dim)
        
        print(f"Forma de activaciones aplanadas: {features_flat.shape}")
        
        all_components = []
        pc1_variances = []
        
        # 2. Aplicar PCA por capa
        for layer_idx in range(num_layers):
            X_layer = features_flat[:, layer_idx, :] # [num_statements, hidden_dim]
            
            # Limpiar NaNs o Infs si existen debido a precisión fp16
            if np.isnan(X_layer).any() or np.isinf(X_layer).any():
                X_layer = np.nan_to_num(X_layer, nan=0.0, posinf=0.0, neginf=0.0)
                
            # PCA normal
            pca = PCA()
            # Ojo: si hay más dimensiones que muestras (ej. 155 muestras y 4096 dims),
            # sklearn devolverá min(n_samples, n_features) componentes.
            projections = pca.fit_transform(X_layer)
            
            ratios = pca.explained_variance_ratio_
            components = pca.components_
            
            pc1_variances.append(ratios[0])
            
            # Recopilar información de los componentes de esta capa
            # Para eficiencia en memoria y tiempo de sorting, podemos guardar sólo 
            # aquellos que expliquen más del 1% (0.01) o simplemente los top 20 por capa,
            # ya que el Top 10 global seguramente saldrá de ahí. Pero guardaremos todos por completitud.
            for comp_idx, (ratio, vec) in enumerate(zip(ratios, components)):
                all_components.append({
                    'layer': layer_idx,
                    'pca_component_idx': comp_idx, # 0 para PC1, 1 para PC2, etc.
                    'explained_variance_ratio': ratio,
                    'vector': vec,
                    'projection': projections[:, comp_idx] # La coordenada de los diputados en este eje
                })
        
        # 3. Ordenar globalmente por ratio de varianza
        all_components_sorted = sorted(all_components, key=lambda x: x['explained_variance_ratio'], reverse=True)
        
        # 4. Seleccionar Top 10 dimensiones a nivel global
        top_10 = all_components_sorted[:10]
        
        print(f"Top 10 componentes encontrados para {model_name}:")
        for i, comp in enumerate(top_10):
            print(f"  {i+1}. Capa {comp['layer']}, PC{comp['pca_component_idx']+1} - Varianza Explicada: {comp['explained_variance_ratio']:.4f}")
            
        # 5. Guardar resultados
        output_file = PCA_RESULTS_DIR / f"{model_name}_top10_pca.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(top_10, f)
        print(f"Resultados guardados en {output_file}")
        
        # 6. Gráfico 1: Evolución del PC1 por capas
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_layers), pc1_variances, marker='o', linestyle='-', color='b')
        plt.title(f'Varianza Explicada por el Componente Principal 1 (PC1) vs Capa\nModelo: {model_name}')
        plt.xlabel('Capa del Modelo')
        plt.ylabel('Ratio de Varianza Explicada (PC1)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{model_name}_pc1_variance_by_layer.png", dpi=300)
        plt.close()
        
        # 7. Gráfico 2: Scatter Plot de Diputados en Global Top 1 vs Global Top 2
        if len(top_10) >= 2:
            top1 = top_10[0]
            top2 = top_10[1]
            
            plot_df = pd.DataFrame({
                'Nombre': df['Name'],
                'Partido': df['party_1_P6'],
                'Ideologia': df['dim_ideology_P6'],
                f"Global Top 1 (L{top1['layer']}_PC{top1['pca_component_idx']+1})": top1['projection'],
                f"Global Top 2 (L{top2['layer']}_PC{top2['pca_component_idx']+1})": top2['projection']
            })
            
            plt.figure(figsize=(12, 8))
            # Pintar por partido
            sns.scatterplot(
                data=plot_df, 
                x=f"Global Top 1 (L{top1['layer']}_PC{top1['pca_component_idx']+1})", 
                y=f"Global Top 2 (L{top2['layer']}_PC{top2['pca_component_idx']+1})",
                hue='Partido',
                palette='tab20',
                s=100, alpha=0.8
            )
            
            plt.title(f'Proyección de Diputados: Top 1 vs Top 2 Componentes Globales\nModelo: {model_name}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{model_name}_scatter_top1_vs_top2_party.png", dpi=300)
            plt.close()

if __name__ == "__main__":
    main()
