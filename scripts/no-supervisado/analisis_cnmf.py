import os
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import NMF
from sklearn.linear_model import RidgeCV

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2"
DEFAULT_CNMF_DIR = BASE_DIR / "results/cnmf_analysis"

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0: return 0.0
    return float(np.abs(np.dot(u, v)) / (norm_u * norm_v))

def main():
    parser = argparse.ArgumentParser(description="Contrastive NMF (cNMF) sobre activaciones de capas.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--cnmf-dir", type=str, default=str(DEFAULT_CNMF_DIR))
    parser.add_argument("--k", type=int, default=15, help="Número de componentes NMF a extraer")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cnmf_dir = Path(args.cnmf_dir)
    cnmf_dir.mkdir(parents=True, exist_ok=True)
    
    sota_models = ['google_gemma-4-E4B-it', 'meta-llama_Llama-3.1-8B-Instruct', 'Qwen_Qwen3.5-9B']
    target_files = [f for f in results_dir.glob("*_politician_features.pkl") if any(sota in f.name for sota in sota_models)]
    
    for t_file in target_files:
        model_name = t_file.name.replace("_politician_features.pkl", "")
        b_file = results_dir / f"{model_name}_neutral_features.pkl"
        
        if not b_file.exists():
            print(f"⚠️ No se encontró dataset neutral para {model_name}. Saltando.")
            continue
            
        print(f"\n{'='*60}")
        print(f"Procesando Contrastive NMF (cNMF) | Modelo: {model_name}")
        print(f"{'='*60}")
        
        with open(t_file, 'rb') as f:
            t_features, labels = pickle.load(f)
        with open(b_file, 'rb') as f:
            b_features, _ = pickle.load(f)
            
        num_statements_t, num_layers_t, num_heads_t, d_head_t = t_features.shape
        num_statements_b, num_layers_b, num_heads_b, d_head_b = b_features.shape
        
        if num_layers_t != num_layers_b:
            print(f"⚠️ Advertencia: Discrepancia en número de capas (Target: {num_layers_t}, Background: {num_layers_b}). Truncando al mínimo disponible.")
            
        num_layers = min(num_layers_t, num_layers_b)
        hidden_dim = num_heads_t * d_head_t
        
        t_flat = t_features[:, :num_layers, :, :].reshape(num_statements_t, num_layers, hidden_dim)
        b_flat = b_features[:, :num_layers, :, :].reshape(num_statements_b, num_layers, hidden_dim)
        
        all_cnmf_components = []
        
        for layer_idx in range(num_layers):
            print(f"Procesando Capa {layer_idx}/{num_layers-1}...", end=" ")
            
            X_T = np.maximum(t_flat[:, layer_idx, :], 0) # ReLU
            X_B = np.maximum(b_flat[:, layer_idx, :], 0) # ReLU
            
            # Concatenar para descubrir diccionario conjunto
            X_concat = np.vstack([X_T, X_B])
            
            if np.isnan(X_concat).any(): X_concat = np.nan_to_num(X_concat)
            
            y_t = np.array(labels, dtype=float)
            ridge = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge.fit(X_T, y_t)
            w_ideology = ridge.coef_
            
            # Entrenar NMF en el dataset concatenado
            nmf = NMF(n_components=args.k, init='nndsvda', max_iter=100000, random_state=42)
            W_concat = nmf.fit_transform(X_concat) # Shape: [N_T + N_B, k]
            H = nmf.components_                  # Shape: [k, hidden_dim]
            
            # Separar las activaciones (W) de vuelta en Target y Background
            W_T = W_concat[:num_statements_t, :]
            W_B = W_concat[num_statements_t:, :]
            
            # Evaluar el "Contrastive Score" de cada componente
            # Queremos componentes que se activen mucho en Target y poco en Background
            mean_act_T = np.mean(W_T, axis=0)
            mean_act_B = np.mean(W_B, axis=0)
            
            # Evitar division por cero
            contrastive_scores = mean_act_T / (mean_act_B + 1e-6)
            
            # Ordenar los índices de componentes por su score contrastivo
            sorted_idx = np.argsort(contrastive_scores)[::-1]
            
            # Extraer el Top 5 más contrastivo de la capa
            for rank, i in enumerate(sorted_idx[:5]):
                vec = H[i, :]
                sim = cosine_similarity(vec, w_ideology)
                
                if np.dot(vec, w_ideology) < 0:
                    vec = -vec
                    
                all_cnmf_components.append({
                    'layer': layer_idx,
                    'component_idx': i,
                    'vector': vec,
                    'contrastive_score': float(contrastive_scores[i]),
                    'sim_ideology': sim
                })
            print("OK")
            
        # Ordenar componentes globales por Contrastive Score
        all_cnmf_components.sort(key=lambda x: x['contrastive_score'], reverse=True)
        
        out_dict = {
            "all_components": all_cnmf_components
        }
        
        out_path = cnmf_dir / f"{model_name}_cnmf_analysis.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump(out_dict, f)
            
        print(f"✅ Guardado cNMF en {out_path}")

if __name__ == "__main__":
    main()
