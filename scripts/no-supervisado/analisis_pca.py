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
import difflib

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


def load_bcall_labels(names_list, bcall_path):
    """
    Intenta hacer un fuzzy match entre los nombres de los parlamentarios
    y el formato del b-call ('Apellido1_Apellido2_Nombre').
    """
    if not os.path.exists(bcall_path):
        print(f"⚠️ No se encontró el archivo b-call en {bcall_path}. Se usará Nominate para b-call (fallback).")
        return None
        
    bcall_df = pd.read_csv(bcall_path)
    bcall_legislators = bcall_df['legislator'].tolist()
    
    # Crear un diccionario para mapeo rápido
    bcall_map = dict(zip(bcall_df['legislator'], bcall_df['d1']))
    
    # Limpiar un poco los nombres de bcall para el matcher
    # Ej: "Acevedo_Sáez_María Candelaria" -> "Acevedo Sáez María Candelaria"
    bcall_clean_names = {leg.replace('_', ' '): leg for leg in bcall_legislators}
    clean_choices = list(bcall_clean_names.keys())
    
    matched_bcall = []
    
    for name in names_list:
        # Formato usual Name_BCN es "Apellido1 Apellido2 Nombre"
        # Hacemos fuzzy match
        matches = difflib.get_close_matches(name, clean_choices, n=1, cutoff=0.6)
        if matches:
            best_match = matches[0]
            original_leg = bcall_clean_names[best_match]
            matched_bcall.append(bcall_map[original_leg])
        else:
            # Si no hay match, usamos NaN temporalmente
            matched_bcall.append(np.nan)
            
    # Llenar NaNs con la media o 0 para no romper el modelo (o se podrían descartar)
    matched_bcall = np.array(matched_bcall)
    nan_mask = np.isnan(matched_bcall)
    if nan_mask.any():
        print(f"⚠️ No se encontró b-call para {nan_mask.sum()} instancias. Se rellenarán con la media.")
        matched_bcall[nan_mask] = np.nanmean(matched_bcall)
        
    return matched_bcall


def main():
    parser = argparse.ArgumentParser(description="Descubrimiento No Supervisado (PCA) con Varianza Global.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Directorio con archivos *_politician_features.pkl.")
    parser.add_argument("--pca-dir", type=str, default=str(DEFAULT_PCA_DIR), help="Directorio para guardar resultados de PCA.")
    parser.add_argument("--figures-dir", type=str, default=str(DEFAULT_FIGURES_DIR), help="Directorio para guardar gráficos.")
    parser.add_argument("--bcall-path", type=str, default=str(DATA_DIR / "bcall_data_chile" / "CHL-bcall-scores-2025.csv"), help="Ruta al archivo CSV de b-call.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pca_dir = Path(args.pca_dir)
    figures_dir = Path(args.figures_dir)
    
    pca_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"Procesando PCA y Alineamiento (Varianza Global) para: {model_name}")
        print(f"=======================================================")
        
        with open(file_path, 'rb') as f:
            features, labels_nominate = pickle.load(f)
            
        # Intentar cargar metadata para b-call si existe
        metadata_path = results_dir / f"{model_name}_politicians_metadata.pkl"
        labels_bcall = None
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                meta_dict = pickle.load(f)
            names_list = [m['name'] for m in meta_dict['metadata']]
            labels_bcall = load_bcall_labels(names_list, args.bcall_path)
            
        if labels_bcall is None:
            # Fallback a usar nominate
            labels_bcall = labels_nominate
            
        is_residual = features.ndim == 3
        if is_residual:
            num_statements, num_layers, hidden_dim = features.shape
            features_flat = features
        else:
            num_statements, num_layers, num_heads, d_head = features.shape
            hidden_dim = num_heads * d_head
            features_flat = features.reshape(num_statements, num_layers, hidden_dim)
        
        all_components = []
        total_model_variance = 0.0
        
        pc1_variances_by_layer = []
        pc1_nominate_alignment = []
        pc1_bcall_alignment = []
        
        # Procesar capa por capa
        for layer_idx in range(num_layers):
            X_layer = features_flat[:, layer_idx, :] # [num_statements, hidden_dim]
            y_nom = np.array(labels_nominate, dtype=float)
            y_bcall = np.array(labels_bcall, dtype=float)
            
            if np.isnan(X_layer).any() or np.isinf(X_layer).any():
                X_layer = np.nan_to_num(X_layer, nan=0.0, posinf=0.0, neginf=0.0)
                
            # 1. PCA no supervisado
            pca = PCA()
            projections = pca.fit_transform(X_layer)
            
            # Autovalores absolutos
            eigenvalues = pca.explained_variance_ 
            components = pca.components_
            
            layer_total_variance = np.sum(eigenvalues)
            total_model_variance += layer_total_variance
            
            pc1_variances_by_layer.append(pca.explained_variance_ratio_[0])
            
            # 2. Probe lineal supervisado (Ridge) para Nominate
            ridge_nom = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge_nom.fit(X_layer, y_nom)
            w_nominate = ridge_nom.coef_
            
            # 3. Probe lineal supervisado (Ridge) para b-call
            ridge_bcall = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge_bcall.fit(X_layer, y_bcall)
            w_bcall = ridge_bcall.coef_
            
            # Alineamiento del PC1 de esta capa
            pc1_nominate_alignment.append(cosine_similarity(components[0], w_nominate))
            pc1_bcall_alignment.append(cosine_similarity(components[0], w_bcall))
            
            for comp_idx, (eval_abs, vec) in enumerate(zip(eigenvalues, components)):
                all_components.append({
                    'layer': layer_idx,
                    'pca_component_idx': comp_idx,
                    'absolute_variance': eval_abs,
                    'layer_variance_ratio': eval_abs / layer_total_variance if layer_total_variance > 0 else 0,
                    'vector': vec,
                    'projection': projections[:, comp_idx],
                    'nominate_cosine_sim': cosine_similarity(vec, w_nominate),
                    'bcall_cosine_sim': cosine_similarity(vec, w_bcall)
                })
        
        # Ordenar globalmente por varianza absoluta
        all_components.sort(key=lambda x: x['absolute_variance'], reverse=True)
        
        # Calcular varianza global acumulada y seleccionar componentes hasta el 90%
        cumulative_variance = 0.0
        top_90_components = []
        for comp in all_components:
            cumulative_variance += comp['absolute_variance']
            comp['global_variance_ratio'] = comp['absolute_variance'] / total_model_variance
            comp['cumulative_global_variance'] = cumulative_variance / total_model_variance
            top_90_components.append(comp)
            if (cumulative_variance / total_model_variance) >= 0.90:
                break
                
        print(f"Para el modelo {model_name}, se necesitan {len(top_90_components)} componentes de un total de {len(all_components)} para explicar el 90% de la varianza global.")
        
        top_5 = all_components[:5]
        print(f"Top 5 componentes principales globales absolutas:")
        for i, comp in enumerate(top_5):
            print(f"  {i+1}. Capa {comp['layer']}, PC{comp['pca_component_idx']+1} | Var Global Ratio: {comp['global_variance_ratio']:.4f} | CosSim(Nominate): {comp['nominate_cosine_sim']:.4f} | CosSim(b-call): {comp['bcall_cosine_sim']:.4f}")
            
        curr_pca_dir = pca_dir / "residual" if is_residual else pca_dir
        curr_figures_dir = figures_dir / "residual" if is_residual else figures_dir
        curr_pca_dir.mkdir(parents=True, exist_ok=True)
        curr_figures_dir.mkdir(parents=True, exist_ok=True)

        output_file = curr_pca_dir / f"{model_name}_pca_analysis_global.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump({
                "all_components": all_components,
                "top_90_components": top_90_components,
                "total_model_variance": total_model_variance,
                "pc1_variances_by_layer": pc1_variances_by_layer,
                "pc1_nominate_alignment": pc1_nominate_alignment,
                "pc1_bcall_alignment": pc1_bcall_alignment
            }, f)
            
        # Graficar alineamiento Nominate vs B-call
        plt.figure(figsize=(10, 5))
        plt.plot(range(num_layers), pc1_nominate_alignment, marker='o', label='PC1 vs Nominate (P6)', color='#3498db', linewidth=2)
        plt.plot(range(num_layers), pc1_bcall_alignment, marker='s', label='PC1 vs b-call', color='#e74c3c', linestyle='--', linewidth=2)
        plt.title(f'Alineamiento Ideológico de PC1 por Capa\nModelo: {model_name}', fontsize=12, fontweight='bold')
        plt.xlabel('Capa del Modelo')
        plt.ylabel('Similitud Coseno (|cos θ|)')
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(curr_figures_dir / f"{model_name}_pca_alignment_nominate_vs_bcall.png", dpi=300)
        plt.close()

    print("\n✅ Análisis PCA con varianza global y b-call completado con éxito.")

if __name__ == "__main__":
    main()
