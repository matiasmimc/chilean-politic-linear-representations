import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import NMF
from sklearn.linear_model import RidgeCV
import difflib

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2_por_declaracion"
DEFAULT_NMF_DIR = BASE_DIR / "results/nmf_analysis"
DEFAULT_FIGURES_DIR = BASE_DIR / "figures/nmf_analysis"

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.abs(np.dot(u, v)) / (norm_u * norm_v))

def load_bcall_labels(names_list, bcall_path):
    if not os.path.exists(bcall_path):
        print(f"⚠️ No se encontró el archivo b-call en {bcall_path}. Se usará Nominate (fallback).")
        return None
        
    bcall_df = pd.read_csv(bcall_path)
    bcall_legislators = bcall_df['legislator'].tolist()
    bcall_map = dict(zip(bcall_df['legislator'], bcall_df['d1']))
    bcall_clean_names = {leg.replace('_', ' '): leg for leg in bcall_legislators}
    clean_choices = list(bcall_clean_names.keys())
    
    matched_bcall = []
    for name in names_list:
        matches = difflib.get_close_matches(name, clean_choices, n=1, cutoff=0.6)
        if matches:
            best_match = matches[0]
            original_leg = bcall_clean_names[best_match]
            matched_bcall.append(bcall_map[original_leg])
        else:
            matched_bcall.append(np.nan)
            
    matched_bcall = np.array(matched_bcall)
    nan_mask = np.isnan(matched_bcall)
    if nan_mask.any():
        print(f"⚠️ No se encontró b-call para {nan_mask.sum()} instancias. Se rellenarán con la media.")
        matched_bcall[nan_mask] = np.nanmean(matched_bcall)
        
    return matched_bcall

def main():
    parser = argparse.ArgumentParser(description="Extracción de Componentes con NMF Puro y Pseudo-Varianza Global")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--nmf-dir", type=str, default=str(DEFAULT_NMF_DIR))
    parser.add_argument("--nmf-components", type=int, default=10, help="Número de macro-perfiles a extraer por capa")
    parser.add_argument("--bcall-path", type=str, default=str(DATA_DIR / "bcall_data_chile" / "CHL-bcall-scores-2025.csv"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    nmf_dir = Path(args.nmf_dir)
    nmf_dir.mkdir(parents=True, exist_ok=True)
    
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
            features, labels_nominate = pickle.load(f)
            
        metadata_path = results_dir / f"{model_name}_politicians_metadata.pkl"
        labels_bcall = None
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                meta_dict = pickle.load(f)
            names_list = [m['name'] for m in meta_dict['metadata']]
            labels_bcall = load_bcall_labels(names_list, args.bcall_path)
            
        if labels_bcall is None:
            labels_bcall = labels_nominate
            
        num_statements, num_layers, num_heads, d_head = features.shape
        hidden_dim = num_heads * d_head
        features_flat = features.reshape(num_statements, num_layers, hidden_dim)
        
        all_nmf_components = []
        total_pseudo_variance = 0.0
        
        for layer_idx in range(num_layers):
            print(f"Procesando Capa {layer_idx}/{num_layers-1}...", end=" ")
            X_layer = features_flat[:, layer_idx, :]
            y_nom = np.array(labels_nominate, dtype=float)
            y_bcall = np.array(labels_bcall, dtype=float)
            
            if np.isnan(X_layer).any() or np.isinf(X_layer).any():
                X_layer = np.nan_to_num(X_layer, nan=0.0, posinf=0.0, neginf=0.0)
                
            ridge_nom = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge_nom.fit(X_layer, y_nom)
            w_nominate = ridge_nom.coef_
            
            ridge_bcall = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge_bcall.fit(X_layer, y_bcall)
            w_bcall = ridge_bcall.coef_
            
            # --- MÉTODO: NMF PURO ---
            X_relu = np.maximum(X_layer, 0)
            nmf_pure = NMF(n_components=args.nmf_components, init='random', random_state=42, max_iter=500)
            W_nmf_pure = nmf_pure.fit_transform(X_relu)
            decoded_components_pure = nmf_pure.components_
            
            for comp_idx in range(args.nmf_components):
                vec = decoded_components_pure[comp_idx]
                pseudo_variance = float(np.var(W_nmf_pure[:, comp_idx]))
                total_pseudo_variance += pseudo_variance
                
                # Alinear vector para consistencia visual
                if np.dot(vec, w_nominate) < 0:
                    vec = -vec
                
                all_nmf_components.append({
                    'layer': layer_idx,
                    'component_idx': comp_idx,
                    'vector': vec,
                    'nominate_cosine_sim': cosine_similarity(vec, w_nominate),
                    'bcall_cosine_sim': cosine_similarity(vec, w_bcall),
                    'pseudo_variance': pseudo_variance
                })
            print("OK")
                
        all_nmf_components.sort(key=lambda x: x['pseudo_variance'], reverse=True)
        
        cumulative_pvar = 0.0
        top_90_components = []
        for comp in all_nmf_components:
            cumulative_pvar += comp['pseudo_variance']
            comp['global_pvar_ratio'] = comp['pseudo_variance'] / total_pseudo_variance if total_pseudo_variance > 0 else 0
            top_90_components.append(comp)
            if (cumulative_pvar / total_pseudo_variance) >= 0.90:
                break
                
        print(f"\nPara {model_name}, se necesitan {len(top_90_components)} macro-perfiles NMF de un total de {len(all_nmf_components)} para explicar el 90% de la pseudo-varianza global.")
        
        print(f"Top 5 Macro-Perfiles globales [NMF PURO]:")
        for i, comp in enumerate(all_nmf_components[:5]):
            print(f"  {i+1}. Capa {comp['layer']}, Comp{comp['component_idx']+1} | Pseudo-Var Ratio: {comp['global_pvar_ratio']:.4f} | CosSim(Nominate): {comp['nominate_cosine_sim']:.4f} | CosSim(b-call): {comp['bcall_cosine_sim']:.4f}")
            
        out_dict = {
            "all_components": all_nmf_components,
            "top_90_components": top_90_components,
            "total_pseudo_variance": total_pseudo_variance
        }
        
        out_path = nmf_dir / f"{model_name}_nmf_analysis_global.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump(out_dict, f)
            
    print("\n✅ Análisis NMF completado con éxito.")

if __name__ == "__main__":
    main()
