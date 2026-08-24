import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import NMF
from sklearn.linear_model import RidgeCV

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2"
DEFAULT_NMF_DIR = BASE_DIR / "results/nmf_analysis"

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.abs(np.dot(u, v)) / (norm_u * norm_v))

def main():
    parser = argparse.ArgumentParser(description="Extracción de Componentes con NMF Puro")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--nmf-dir", type=str, default=str(DEFAULT_NMF_DIR))
    parser.add_argument("--nmf-components", type=int, default=10, help="Número de macro-perfiles a extraer")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    nmf_dir = Path(args.nmf_dir)
    
    nmf_dir.mkdir(parents=True, exist_ok=True)
    
    df_path = DATA_DIR / "congreso_2022_2026_20intervenciones.parquet"
    if not df_path.exists():
        print(f"Error: No se encuentra {df_path}")
        return
    df = pd.read_parquet(df_path)
    
    sota_models = ['google_gemma-4-E4B-it', 'meta-llama_Llama-3.1-8B-Instruct', 'Qwen_Qwen3.5-9B']
    feature_files = [f for f in results_dir.glob("*_politician_features.pkl") if any(sota in f.name for sota in sota_models)]
    
    if not feature_files:
        print(f"No se encontraron archivos SOTA en {results_dir}")
        return

    for file_path in feature_files:
        model_name = file_path.name.replace("_politician_features.pkl", "")
        print(f"\n{'='*60}")
        print(f"Procesando NMF Puro para: {model_name}")
        print(f"{'='*60}")
        
        with open(file_path, 'rb') as f:
            features, labels = pickle.load(f)
            
        num_statements, num_layers, num_heads, d_head = features.shape
        hidden_dim = num_heads * d_head
        features_flat = features.reshape(num_statements, num_layers, hidden_dim)
        
        all_nmf_components = []
        
        for layer_idx in range(num_layers):
            print(f"\nProcesando Capa {layer_idx}/{num_layers-1}...")
            X_layer = features_flat[:, layer_idx, :]
            y = np.array(labels, dtype=float)
            
            if np.isnan(X_layer).any() or np.isinf(X_layer).any():
                X_layer = np.nan_to_num(X_layer, nan=0.0, posinf=0.0, neginf=0.0)
                
            # 1. Probe lineal supervisado para obtener el vector ideológico original
            ridge = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge.fit(X_layer, y)
            w_ideology = ridge.coef_ # Vector en el espacio original [hidden_dim]
            
            # --- MÉTODO: NMF PURO ---
            print(f"  [NMF Puro] Aplicando NMF directo sobre activaciones pasadas por ReLU...")
            X_relu = np.maximum(X_layer, 0) # Forzar no-negatividad
            nmf_pure = NMF(n_components=args.nmf_components, init='random', random_state=42, max_iter=500)
            W_nmf_pure = nmf_pure.fit_transform(X_relu)
            decoded_components_pure = nmf_pure.components_ # [k, hidden_dim]
            
            for comp_idx in range(args.nmf_components):
                vec_original_space = decoded_components_pure[comp_idx]
                sim = cosine_similarity(vec_original_space, w_ideology)
                pseudo_variance = np.var(W_nmf_pure[:, comp_idx])
                
                all_nmf_components.append({
                    'layer': layer_idx,
                    'component_idx': comp_idx,
                    'vector': vec_original_space,
                    'ideology_cosine_sim': sim,
                    'explained_variance_ratio': pseudo_variance
                })
                
        # --- GUARDAR RESULTADOS NMF PURO ---
        all_nmf_sorted = sorted(all_nmf_components, key=lambda x: x['explained_variance_ratio'], reverse=True)
        print(f"\nTop 5 Macro-Perfiles globales [NMF PURO] para {model_name}:")
        for i, comp in enumerate(all_nmf_sorted[:5]):
            print(f"  {i+1}. Capa {comp['layer']}, Comp{comp['component_idx']+1} | Pseudo-Var: {comp['explained_variance_ratio']:.4f} | CosSim Ideología: {comp['ideology_cosine_sim']:.4f}")
            
        with open(nmf_dir / f"{model_name}_nmf_analysis.pkl", 'wb') as f:
            pickle.dump({"all_components": all_nmf_sorted}, f)
            
    print("\n✅ Análisis NMF completado con éxito.")

if __name__ == "__main__":
    main()
