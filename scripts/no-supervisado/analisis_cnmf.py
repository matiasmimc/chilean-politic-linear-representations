import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import NMF
from sklearn.linear_model import RidgeCV
import difflib
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0: return 0.0
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

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # Mostrar la configuración actual
    print("="*60)
    print("Configuración del Experimento:")
    print(OmegaConf.to_yaml(cfg))
    print("="*60)

    # Rutas de entrada relativas a la raíz del proyecto original (BASE_DIR)
    results_dir = BASE_DIR / cfg.paths.results_dir
    bcall_path = BASE_DIR / cfg.paths.bcall_path
    mp_path = BASE_DIR / cfg.paths.mp_path
    
    # Rutas de salida absolutas en BASE_DIR
    cnmf_dir = BASE_DIR / "results" / "cnmf_analysis"
    figures_dir = BASE_DIR / "figures" / "cnmf_analysis"
    cnmf_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    sota_models = [
        'google/gemma-4-E4B-it',
        'meta-llama/Llama-3.1-8B-Instruct',
        # 'Qwen/Qwen3.5-9B',
        'zai-org/GLM-4-9B-0414',
        # 'mistralai/Ministral-3-8B-Instruct-2512',
    ]
    target_files = [f for f in results_dir.glob("*_politician_features.pkl") if any(sota.replace("/", "_") in f.name for sota in sota_models)]
    
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
            t_features, labels_nominate = pickle.load(f)
            
        metadata_path = results_dir / f"{model_name}_politicians_metadata.pkl"
        labels_bcall = None
        names_list = None
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                meta_dict = pickle.load(f)
            names_list = [m['name'] for m in meta_dict['metadata']]
        else:
            df_path = Path(mp_path)
            if df_path.exists():
                df_mps = pd.read_parquet(df_path)
                names_list = df_mps['Name'].tolist()
                
        if names_list:
            labels_bcall = load_bcall_labels(names_list, bcall_path)
            
        if labels_bcall is None:
            labels_bcall = labels_nominate
            
        with open(b_file, 'rb') as f:
            b_features, _ = pickle.load(f)
            
        num_statements_t, num_layers_t, num_heads_t, d_head_t = t_features.shape
        num_statements_b, num_layers_b, num_heads_b, d_head_b = b_features.shape
        
        if num_layers_t != num_layers_b:
            print(f"⚠️ Advertencia: Discrepancia en número de capas (Target: {num_layers_t}, Background: {num_layers_b}).")
            
        num_layers = min(num_layers_t, num_layers_b)
        hidden_dim = num_heads_t * d_head_t
        
        t_flat = t_features[:, :num_layers, :, :].reshape(num_statements_t, num_layers, hidden_dim)
        b_flat = b_features[:, :num_layers, :, :].reshape(num_statements_b, num_layers, hidden_dim)
        
        all_cnmf_components = []
        total_target_pseudo_variance = 0.0
        
        for layer_idx in range(num_layers):
            print(f"Procesando Capa {layer_idx}/{num_layers-1}...", end=" ")
            
            X_T = np.maximum(t_flat[:, layer_idx, :], 0) # ReLU
            X_B = np.maximum(b_flat[:, layer_idx, :], 0) # ReLU
            
            y_nom = np.array(labels_nominate, dtype=float)
            y_bcall = np.array(labels_bcall, dtype=float)
            
            X_concat = np.vstack([X_T, X_B])
            if np.isnan(X_concat).any(): X_concat = np.nan_to_num(X_concat)
            
            ridge_nom = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge_nom.fit(X_T, y_nom)
            w_nominate = ridge_nom.coef_
            
            ridge_bcall = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge_bcall.fit(X_T, y_bcall)
            w_bcall = ridge_bcall.coef_
            
            nmf = NMF(n_components=cfg.params.get('k', 16), init='nndsvda', max_iter=100000, random_state=42)
            W_concat = nmf.fit_transform(X_concat)
            H = nmf.components_
            
            W_T = W_concat[:num_statements_t, :]
            W_B = W_concat[num_statements_t:, :]
            
            mean_act_T = np.mean(W_T, axis=0)
            mean_act_B = np.mean(W_B, axis=0)
            
            contrastive_scores = mean_act_T / (mean_act_B + 1e-6)
            
            for i in range(cfg.params.get('k', 16)):
                vec = H[i, :]
                pvar_T = float(np.var(W_T[:, i]))
                total_target_pseudo_variance += pvar_T
                
                if np.dot(vec, w_nominate) < 0:
                    vec = -vec
                    
                all_cnmf_components.append({
                    'layer': layer_idx,
                    'component_idx': i,
                    'vector': vec,
                    'contrastive_score': float(contrastive_scores[i]),
                    'target_pseudo_variance': pvar_T,
                    'nominate_cosine_sim': cosine_similarity(vec, w_nominate),
                    'bcall_cosine_sim': cosine_similarity(vec, w_bcall)
                })
            print("OK")
            
        all_cnmf_components.sort(key=lambda x: x['contrastive_score'], reverse=True)
        
        cumulative_pvar = 0.0
        top_90_components = []
        for comp in all_cnmf_components:
            cumulative_pvar += comp['target_pseudo_variance']
            comp['global_pvar_ratio'] = comp['target_pseudo_variance'] / total_target_pseudo_variance if total_target_pseudo_variance > 0 else 0
            top_90_components.append(comp)
            if (cumulative_pvar / total_target_pseudo_variance) >= 0.90:
                break
                
        print(f"\nPara {model_name}, se necesitan {len(top_90_components)} componentes de {len(all_cnmf_components)} para explicar el 90% de la pseudo-varianza objetivo en orden de mayor score contrastivo.")
        
        out_dict = {
            "all_components": all_cnmf_components,
            "top_90_components": top_90_components,
            "total_target_pseudo_variance": total_target_pseudo_variance
        }
        
        out_path = cnmf_dir / f"{model_name}_cnmf_analysis_global.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump(out_dict, f)
            
        print(f"✅ Guardado cNMF en {out_path}")
        
if __name__ == "__main__":
    main()
