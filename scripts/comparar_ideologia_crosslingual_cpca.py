import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RidgeCV
import hydra
from omegaconf import DictConfig, OmegaConf

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
US_DIR = BASE_DIR / "results/us_analysis"
FIGURES_DIR = BASE_DIR / "figures/crosslingual_analysis_cpca"


FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.abs(np.dot(u, v)) / (norm_u * norm_v))

def compute_covariance(X):
    X_centered = X - np.mean(X, axis=0)
    return np.cov(X_centered, rowvar=False)

def get_cpc1_aligned(X_T, X_B, labels, contrast_alpha=2.0):
    """
    Calcula el cPCA (C_T - contrast_alpha * C_B) y retorna el autovector principal (cPC1),
    alineado con la dirección ideológica de un clasificador supervisado (Ridge).
    """
    if np.isnan(X_T).any(): X_T = np.nan_to_num(X_T)
    if np.isnan(X_B).any(): X_B = np.nan_to_num(X_B)
    
    y = np.array(labels, dtype=float)
    
    # 1. Obtener la dirección ideológica supervisada como referencia (prioridad de signo)
    ridge = RidgeCV(alphas=np.logspace(-3, 4, 20))
    ridge.fit(X_T, y)
    w_ideology = ridge.coef_
    
    # 2. Calcular cPCA
    C_T = compute_covariance(X_T)
    C_B = compute_covariance(X_B)
    C_contrastive = C_T - contrast_alpha * C_B
    
    eigenvalues, eigenvectors = np.linalg.eigh(C_contrastive)
    
    # Ordenar por autovalor de mayor a menor
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    cPC1 = eigenvectors[:, 0]
    
    # 3. Alinear el cPC1 con la ideología (si el dot product es negativo, invertir)
    if np.dot(cPC1, w_ideology) < 0:
        cPC1 = -cPC1
        
    return cPC1

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Mostrar la configuración actual
    print("="*60)
    print("Configuración del Experimento Crosslingual:")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))

    target = cfg.get("target", "attention")
    if target == "residual":
        CHILE_DIR = (BASE_DIR / cfg.paths.results_dir) / "residual" if not str(cfg.paths.results_dir).endswith("residual") else (BASE_DIR / cfg.paths.results_dir)
        US_DIR = (BASE_DIR / "results/us_analysis") / "residual"
        FIGURES_DIR = BASE_DIR / "figures/crosslingual_analysis_cpca/residual"
    else:
        CHILE_DIR = BASE_DIR / cfg.paths.results_dir
        US_DIR = BASE_DIR / "results/us_analysis"
        FIGURES_DIR = BASE_DIR / "figures/crosslingual_analysis_cpca"
        
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    contrast_alpha = cfg.params.contrast_alpha
    
    sota_models = [
        'google_gemma-4-E4B-it', 
        'meta-llama_Llama-3.1-8B-Instruct', 
        'zai-org_GLM-4-9B-0414',
        'meta-llama_Llama-3.1-70B-Instruct',
        'latam-gpt_Llama-3.1-70B-LatamGPT-SFT-1.0'
    ]

    selected_model = cfg.get("model", None)
    if selected_model:
        model_pattern = selected_model.replace("/", "_")
        target_files = [f for f in CHILE_DIR.glob("*_politician_features.pkl") if model_pattern in f.name]
    else:
        target_files = [f for f in CHILE_DIR.glob("*_politician_features.pkl") if any(sota in f.name for sota in sota_models)]
    
    for t_file in target_files:
        model_name = t_file.name.replace("_politician_features.pkl", "")
        safe_name = model_name.replace("/", "_")
        
        print(f"\n{'='*60}")
        print(f"Análisis Cross-Lingual cPCA | Modelo: {model_name} | Alpha constrastivo: {contrast_alpha}")
        print(f"{'='*60}")
        
        # Rutas Chile
        cl_t_path = t_file
        cl_b_path = CHILE_DIR / f"{safe_name}_neutral_features.pkl"
        
        # Rutas US
        us_t_path = US_DIR / f"{safe_name}_politician_features.pkl"
        us_b_path = US_DIR / f"{safe_name}_neutral_features.pkl"
        
        paths_to_check = [cl_t_path, cl_b_path, us_t_path, us_b_path]
        skip_model = False
        for p in paths_to_check:
            if not p.exists():
                print(f"⚠️ Falta el archivo {p}. Saltando modelo.")
                skip_model = True
                break
        if skip_model:
            continue
                
        # 1. Cargar Activaciones
        print("Cargando activaciones...")
        with open(cl_t_path, 'rb') as f:
            cl_T, cl_labels = pickle.load(f)
        with open(cl_b_path, 'rb') as f:
            cl_B, _ = pickle.load(f)
            
        with open(us_t_path, 'rb') as f:
            us_T, us_labels = pickle.load(f)
        with open(us_b_path, 'rb') as f:
            us_B, _ = pickle.load(f)

        # Si alguna de las activaciones neutrales fue guardada como lista (por scripts anteriores), las apilamos
        if isinstance(cl_T, list): cl_T = np.stack(cl_T, axis=1)
        if isinstance(cl_B, list): cl_B = np.stack(cl_B, axis=1)
        if isinstance(us_T, list): us_T = np.stack(us_T, axis=1)
        if isinstance(us_B, list): us_B = np.stack(us_B, axis=1)

        # Aplanar representaciones según si son 3D (residual) o 4D (attention heads)
        if cl_T.ndim == 3:
            num_stmts_cl, num_layers_cl, hidden_dim = cl_T.shape
            num_layers_us = us_T.shape[1]
            num_layers = min(num_layers_cl, num_layers_us)
            cl_T_flat = cl_T[:, :num_layers, :]
            cl_B_flat = cl_B[:, :num_layers, :]
            us_T_flat = us_T[:, :num_layers, :]
            us_B_flat = us_B[:, :num_layers, :]
        else:
            num_stmts_cl, num_layers_cl, num_heads, d_head = cl_T.shape
            num_layers_us = us_T.shape[1]
            num_layers = min(num_layers_cl, num_layers_us)
            hidden_dim = num_heads * d_head
            cl_T_flat = cl_T[:, :num_layers, :, :].reshape(-1, num_layers, hidden_dim)
            cl_B_flat = cl_B[:, :num_layers, :, :].reshape(-1, num_layers, hidden_dim)
            us_T_flat = us_T[:, :num_layers, :, :].reshape(-1, num_layers, hidden_dim)
            us_B_flat = us_B[:, :num_layers, :, :].reshape(-1, num_layers, hidden_dim)
        
        similarities_cpca = []
        
        print("\nCalculando cPC1 capa por capa para ambos contextos...")
        for layer in range(num_layers):
            # Obtener datos de la capa
            X_T_cl = cl_T_flat[:, layer, :]
            X_B_cl = cl_B_flat[:, layer, :]
            
            X_T_us = us_T_flat[:, layer, :]
            X_B_us = us_B_flat[:, layer, :]
            
            # Calcular cPC1 alineado
            cPC1_cl = get_cpc1_aligned(X_T_cl, X_B_cl, cl_labels, contrast_alpha=contrast_alpha)
            cPC1_us = get_cpc1_aligned(X_T_us, X_B_us, us_labels, contrast_alpha=contrast_alpha)
            
            # Similitud coseno entre los vectores de máxima varianza ideológica
            sim_cos = cosine_similarity(cPC1_cl, cPC1_us)
            similarities_cpca.append(sim_cos)
            
        promedio = np.mean(similarities_cpca)
        print(f"\n✅ Análisis completado. Similitud promedio: {promedio:.4f}")
        
        # Graficar
        plt.figure(figsize=(8, 5))
        plt.plot(range(num_layers), similarities_cpca, marker='o', color='#8e44ad', linewidth=2, label='cPC1 Cosine Sim')
        plt.axhline(y=promedio, color='r', linestyle='--', label=f'Promedio: {promedio:.2f}')
        
        plt.title(f'Similitud Cross-Lingual del Eje Ideológico (cPC1)\nChile vs EEUU | Modelo: {model_name}', fontsize=12, fontweight='bold')
        plt.xlabel('Capa del Modelo')
        plt.ylabel('Similitud Coseno (|cos θ|)')
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        fig_path = FIGURES_DIR / f"{safe_name}_crosslingual_cpca_similarity.png"
        plt.savefig(fig_path, dpi=300)
        print(f"✅ Gráfico guardado en {fig_path}")
        plt.close()
    
        # Guardar en JSON
        import json
        metrics_data = {
            "model": model_name,
            "method": "cPCA (Cosine Similarity)",
            "contrastive_alpha": contrast_alpha,
            "cosine_similarity": similarities_cpca,
            "average": float(promedio)
        }
        json_path = FIGURES_DIR / f"{safe_name}_crosslingual_cpca_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        print(f"✅ Métricas guardadas en {json_path}")

if __name__ == "__main__":
    main()
