import os
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2"
DEFAULT_CPCA_DIR = BASE_DIR / "results/cpca_analysis"

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.abs(np.dot(u, v)) / (norm_u * norm_v))

def compute_covariance(X):
    """Calcula matriz de covarianza empírica (asume datos ya centrados o centra internamente)."""
    X_centered = X - np.mean(X, axis=0)
    return np.cov(X_centered, rowvar=False)

def main():
    parser = argparse.ArgumentParser(description="Contrastive PCA (cPCA) sobre activaciones de capas.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--cpca-dir", type=str, default=str(DEFAULT_CPCA_DIR))
    parser.add_argument("--alpha", type=float, default=2.0, help="Fuerza de penalización del background")
    parser.add_argument("--k", type=int, default=10, help="Número de componentes a extraer por capa")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cpca_dir = Path(args.cpca_dir)
    cpca_dir.mkdir(parents=True, exist_ok=True)
    
    sota_models = ['google_gemma-4-E4B-it', 'meta-llama_Llama-3.1-8B-Instruct', 'Qwen_Qwen3.5-9B']
    
    # Buscar archivos de activaciones politicas (Target)
    target_files = [f for f in results_dir.glob("*_politician_features.pkl") if any(sota in f.name for sota in sota_models)]
    
    for t_file in target_files:
        model_name = t_file.name.replace("_politician_features.pkl", "")
        b_file = results_dir / f"{model_name}_neutral_features.pkl"
        
        if not b_file.exists():
            print(f"⚠️ No se encontró dataset de fondo (neutral) para {model_name}. Saltando.")
            continue
            
        print(f"\n{'='*60}")
        print(f"Procesando Contrastive PCA (cPCA) | Modelo: {model_name} | Alpha: {args.alpha}")
        print(f"{'='*60}")
        
        # Cargar Target
        with open(t_file, 'rb') as f:
            t_features, labels = pickle.load(f)
            
        # Cargar Background
        with open(b_file, 'rb') as f:
            b_features, _ = pickle.load(f)
            
        # features shape: [num_statements, num_layers, num_heads, d_head]
        num_statements_t, num_layers_t, num_heads_t, d_head_t = t_features.shape
        num_statements_b, num_layers_b, num_heads_b, d_head_b = b_features.shape
        
        if num_layers_t != num_layers_b:
            print(f"⚠️ Advertencia: Discrepancia en número de capas (Target: {num_layers_t}, Background: {num_layers_b}). Truncando al mínimo disponible.")
        
        num_layers = min(num_layers_t, num_layers_b)
        hidden_dim = num_heads_t * d_head_t
        
        t_flat = t_features[:, :num_layers, :, :].reshape(num_statements_t, num_layers, hidden_dim)
        b_flat = b_features[:, :num_layers, :, :].reshape(num_statements_b, num_layers, hidden_dim)
        
        all_cpca_components = []
        
        for layer_idx in range(num_layers):
            print(f"Procesando Capa {layer_idx}/{num_layers-1}...", end=" ")
            
            X_T = t_flat[:, layer_idx, :]
            X_B = b_flat[:, layer_idx, :]
            y = np.array(labels, dtype=float)
            
            if np.isnan(X_T).any(): X_T = np.nan_to_num(X_T)
            if np.isnan(X_B).any(): X_B = np.nan_to_num(X_B)
            
            # Entrenar Probe Lineal como ground truth de referencia
            ridge = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge.fit(X_T, y)
            w_ideology = ridge.coef_
            
            # --- Contrastive PCA ---
            C_T = compute_covariance(X_T)
            C_B = compute_covariance(X_B)
            
            # Matriz de diferencia
            C_contrastive = C_T - args.alpha * C_B
            
            # Encontrar autovectores (Como C es simétrica real, usamos eigh)
            eigenvalues, eigenvectors = np.linalg.eigh(C_contrastive)
            
            # Ordenar de mayor a menor varianza (autovalores)
            sorted_idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_idx]
            eigenvectors = eigenvectors[:, sorted_idx]
            
            # Tomar los top-k componentes cPCA
            for i in range(args.k):
                vec = eigenvectors[:, i]
                sim = cosine_similarity(vec, w_ideology)
                
                # Para facilitar interpretabilidad, alinear dirección
                if np.dot(vec, w_ideology) < 0:
                    vec = -vec
                    
                all_cpca_components.append({
                    'layer': layer_idx,
                    'component_idx': i,
                    'vector': vec,
                    'eigenvalue': float(eigenvalues[i]),
                    'sim_ideology': sim
                })
            print("OK")
            
        # Ordenar todos los componentes encontrados por eigenvalue global
        all_cpca_components.sort(key=lambda x: x['eigenvalue'], reverse=True)
        
        out_dict = {
            "all_components": all_cpca_components,
            "alpha": args.alpha
        }
        
        out_path = cpca_dir / f"{model_name}_cpca_analysis.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump(out_dict, f)
            
        print(f"✅ Guardado cPCA en {out_path}")

if __name__ == "__main__":
    main()
