import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import RidgeCV
import difflib
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2_por_declaracion"
DEFAULT_CPCA_DIR = BASE_DIR / "results/cpca_analysis"
DEFAULT_FIGURES_DIR = BASE_DIR / "figures/cpca_analysis"

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
    
    # Rutas de salida relativas a la carpeta generada por Hydra
    cpca_dir = Path("cpca_results")
    figures_dir = Path("figures")
    
    cpca_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    sota_models = ['google_gemma-4-E4B-it', 'meta-llama_Llama-3.1-8B-Instruct', 'Qwen_Qwen3.5-9B']
    
    target_files = [f for f in results_dir.glob("*_politician_features.pkl") if any(sota in f.name for sota in sota_models)]
    
    if not target_files:
        print(f"No se encontraron archivos *_politician_features.pkl en {results_dir}")
        return
        
    for t_file in target_files:
        model_name = t_file.name.replace("_politician_features.pkl", "")
        b_file = results_dir / f"{model_name}_neutral_features.pkl"
        
        if not b_file.exists():
            print(f"⚠️ No se encontró dataset de fondo (neutral) para {model_name}. Saltando.")
            continue
            
        print(f"\n{'='*60}")
        print(f"Procesando Contrastive PCA (cPCA) | Modelo: {model_name} | Alpha: {cfg.params.alpha}")
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
            print(f"⚠️ Advertencia: Discrepancia en número de capas (Target: {num_layers_t}, Background: {num_layers_b}). Truncando al mínimo disponible.")
        
        num_layers = min(num_layers_t, num_layers_b)
        hidden_dim = num_heads_t * d_head_t
        
        t_flat = t_features[:, :num_layers, :, :].reshape(num_statements_t, num_layers, hidden_dim)
        b_flat = b_features[:, :num_layers, :, :].reshape(num_statements_b, num_layers, hidden_dim)
        
        all_cpca_components = []
        total_model_positive_variance = 0.0
        
        pc1_nominate_alignment = []
        pc1_bcall_alignment = []
        pc2_nominate_alignment = []
        pc2_bcall_alignment = []
        
        for layer_idx in range(num_layers):
            X_T = t_flat[:, layer_idx, :]
            X_B = b_flat[:, layer_idx, :]
            y_nom = np.array(labels_nominate, dtype=float)
            y_bcall = np.array(labels_bcall, dtype=float)
            
            if np.isnan(X_T).any(): X_T = np.nan_to_num(X_T)
            if np.isnan(X_B).any(): X_B = np.nan_to_num(X_B)
            
            ridge_nom = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge_nom.fit(X_T, y_nom)
            w_nominate = ridge_nom.coef_
            
            ridge_bcall = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge_bcall.fit(X_T, y_bcall)
            w_bcall = ridge_bcall.coef_
            
            C_T = compute_covariance(X_T)
            C_B = compute_covariance(X_B)
            C_contrastive = C_T - cfg.params.alpha * C_B
            
            eigenvalues, eigenvectors = np.linalg.eigh(C_contrastive)
            
            sorted_idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_idx]
            eigenvectors = eigenvectors[:, sorted_idx]
            
            layer_positive_variance = np.sum(eigenvalues[eigenvalues > 0])
            total_model_positive_variance += layer_positive_variance
            
            # Evaluar top 1 y top 2 para gráficos por capa
            vec_top1 = eigenvectors[:, 0]
            if np.dot(vec_top1, w_nominate) < 0:
                vec_top1 = -vec_top1
            pc1_nominate_alignment.append(cosine_similarity(vec_top1, w_nominate))
            pc1_bcall_alignment.append(cosine_similarity(vec_top1, w_bcall))
            
            vec_top2 = eigenvectors[:, 1]
            if np.dot(vec_top2, w_nominate) < 0:
                vec_top2 = -vec_top2
            pc2_nominate_alignment.append(cosine_similarity(vec_top2, w_nominate))
            pc2_bcall_alignment.append(cosine_similarity(vec_top2, w_bcall))
            
            # Guardamos todos los autovectores con autovalor positivo
            for i in range(len(eigenvalues)):
                eval_val = eigenvalues[i]
                if eval_val <= 0:
                    break # Como están ordenados descendentemente, los demás también serán <= 0
                
                vec = eigenvectors[:, i]
                
                if np.dot(vec, w_nominate) < 0:
                    vec = -vec
                    
                all_cpca_components.append({
                    'layer': layer_idx,
                    'component_idx': i,
                    'eigenvalue': float(eval_val),
                    'vector': vec,
                    'nominate_cosine_sim': cosine_similarity(vec, w_nominate),
                    'bcall_cosine_sim': cosine_similarity(vec, w_bcall)
                })
            
        all_cpca_components.sort(key=lambda x: x['eigenvalue'], reverse=True)
        
        cumulative_variance = 0.0
        top_90_components = []
        for comp in all_cpca_components:
            cumulative_variance += comp['eigenvalue']
            comp['global_variance_ratio'] = comp['eigenvalue'] / total_model_positive_variance if total_model_positive_variance > 0 else 0
            top_90_components.append(comp)
            if (cumulative_variance / total_model_positive_variance) >= 0.90:
                break
        
        print(f"Para el modelo {model_name} (cPCA), se necesitan {len(top_90_components)} componentes de un total de {len(all_cpca_components)} para explicar el 90% de la varianza positiva global contrastiva.")
        
        out_dict = {
            "all_components": all_cpca_components,
            "top_90_components": top_90_components,
            "alpha": cfg.params.alpha,
            "total_model_positive_variance": total_model_positive_variance,
            "pc1_nominate_alignment": pc1_nominate_alignment,
            "pc1_bcall_alignment": pc1_bcall_alignment,
            "pc2_nominate_alignment": pc2_nominate_alignment,
            "pc2_bcall_alignment": pc2_bcall_alignment
        }
        
        out_path = cpca_dir / f"{model_name}_cpca_analysis_global.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump(out_dict, f)
            
        print(f"✅ Guardado cPCA en {out_path}")
        
        plt.figure(figsize=(10, 5))
        
        if cfg.params.compare_target == "nominate":
            plt.plot(range(num_layers), pc1_nominate_alignment, marker='o', label='cPC1 vs Nominate', color='#3498db', linewidth=2)
            plt.plot(range(num_layers), pc2_nominate_alignment, marker='^', label='cPC2 vs Nominate', color='#9b59b6', linewidth=2)
            target_title = "Nominate"
        else:
            plt.plot(range(num_layers), pc1_bcall_alignment, marker='o', label='cPC1 vs b-call', color='#e74c3c', linewidth=2)
            plt.plot(range(num_layers), pc2_bcall_alignment, marker='^', label='cPC2 vs b-call', color='#f39c12', linewidth=2)
            target_title = "b-call"
            
        plt.title(f'Alineamiento Ideológico de cPC1 y cPC2 vs {target_title} por Capa\nModelo: {model_name} | Alpha: {cfg.params.alpha}', fontsize=12, fontweight='bold')
        plt.xlabel('Capa del Modelo')
        plt.ylabel('Similitud Coseno (|cos θ|)')
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"{model_name}_cpca_alignment_{cfg.params.compare_target}.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
