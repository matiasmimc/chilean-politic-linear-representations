import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
CHILE_DIR = BASE_DIR / "results/replication/v2"
US_DIR = BASE_DIR / "results/us_analysis"
FIGURES_DIR = BASE_DIR / "figures/crosslingual_analysis"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.abs(np.dot(u, v)) / (norm_u * norm_v))

def linear_cka(X, Y):
    """Linear CKA (Centered Kernel Alignment) entre dos matrices (N, d)."""
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)
    
    dot_XX = np.dot(X_c, X_c.T)
    dot_YY = np.dot(Y_c, Y_c.T)
    dot_XY = np.dot(X_c, Y_c.T)
    
    numerator = np.sum(dot_XY ** 2)
    denominator = np.sqrt(np.sum(dot_XX ** 2) * np.sum(dot_YY ** 2))
    return float(numerator / denominator) if denominator > 0 else 0.0

def compute_pwcca(X, Y, n_components=10):
    """Aproximación de PWCCA mediante reducción por PCA previa (SVCCA)."""
    # Prevenir que n_components sea mayor que N o d
    n_comp = min(n_components, X.shape[0], X.shape[1])
    
    pca_x = PCA(n_components=n_comp).fit_transform(X)
    pca_y = PCA(n_components=n_comp).fit_transform(Y)
    
    cca = CCA(n_components=n_comp)
    cca.fit(pca_x, pca_y)
    X_c, Y_c = cca.transform(pca_x, pca_y)
    
    corrs = []
    for i in range(n_comp):
        std_x = np.std(X_c[:, i])
        std_y = np.std(Y_c[:, i])
        if std_x == 0 or std_y == 0:
            corrs.append(0.0)
        else:
            corrs.append(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1])
            
    # Pesos basados en la proyección absoluta de X sobre las direcciones canónicas
    weights = np.sum(np.abs(np.dot(pca_x.T, X_c)), axis=0)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(n_comp) / n_comp
        
    pwcca = np.sum(weights * np.abs(corrs))
    return float(pwcca)

def get_pca_first_component(features_flat):
    num_statements, num_layers, hidden_dim = features_flat.shape
    components = []
    
    for layer_idx in range(num_layers):
        X_layer = features_flat[:, layer_idx, :]
        if np.isnan(X_layer).any() or np.isinf(X_layer).any():
            X_layer = np.nan_to_num(X_layer, nan=0.0, posinf=0.0, neginf=0.0)
            
        pca = PCA(n_components=1)
        pca.fit(X_layer)
        components.append(pca.components_[0])
        
    return np.array(components) # [num_layers, hidden_dim]

def main():
    parser = argparse.ArgumentParser(description="Compara la similitud de la Dimensión Ideológica entre Chile y EEUU.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    model_name = args.model
    safe_name = model_name.replace("/", "_")
    
    print(f"\n{'='*60}")
    print(f"Análisis Cross-Lingual de Ideología | Modelo: {model_name}")
    print(f"{'='*60}")
    
    chile_path = CHILE_DIR / f"{safe_name}_politician_features.pkl"
    us_path = US_DIR / f"{safe_name}_politician_features.pkl"
    
    if not chile_path.exists():
        print(f"Error: Faltan activaciones de Chile en {chile_path}")
        return
    if not us_path.exists():
        print(f"Error: Faltan activaciones de EEUU en {us_path}. Corre extraer_activaciones_us.py primero.")
        return
        
    # 1. Cargar Activaciones de Chile
    with open(chile_path, 'rb') as f:
        chile_features, chile_labels = pickle.load(f)
        
    num_stmts_cl, num_layers, num_heads, d_head = chile_features.shape
    hidden_dim = num_heads * d_head
    chile_flat = chile_features.reshape(num_stmts_cl, num_layers, hidden_dim)
    
    # 2. Cargar Activaciones de EEUU
    with open(us_path, 'rb') as f:
        us_features, us_labels = pickle.load(f)
        
    num_stmts_us = us_features.shape[0]
    us_flat = us_features.reshape(num_stmts_us, num_layers, hidden_dim)
    
    # Igualar N y ALINEAR SEMÁNTICAMENTE por rango ideológico
    N_samples = min(num_stmts_cl, num_stmts_us)
    print(f"Alineando matrices por percentil ideológico: N = {N_samples}")
    
    # Ordenar por ideología (de izquierda a derecha)
    idx_cl_sorted = np.argsort(chile_labels)
    idx_us_sorted = np.argsort(us_labels)
    
    # Tomar una muestra equiespaciada si un dataset es más grande que otro
    if num_stmts_cl > N_samples:
        idx_cl_sorted = idx_cl_sorted[np.linspace(0, num_stmts_cl-1, N_samples, dtype=int)]
    if num_stmts_us > N_samples:
        idx_us_sorted = idx_us_sorted[np.linspace(0, num_stmts_us-1, N_samples, dtype=int)]
    
    chile_trunc = chile_flat[idx_cl_sorted]
    us_trunc = us_flat[idx_us_sorted]
    
    print("Calculando Eje PCA Ideológico...")
    chile_pca_components = get_pca_first_component(chile_flat) # Se mantiene todo para el PCA global
    us_pca_components = get_pca_first_component(us_flat)
    
    # 3. Calcular métricas representacionales por capa
    similarities_cos = []
    similarities_cka = []
    similarities_pwcca = []
    
    print("\nCalculando métricas capa por capa...")
    for layer in range(num_layers):
        # Cosine Similarity (sobre el vector principal de PCA global)
        sim_cos = cosine_similarity(chile_pca_components[layer], us_pca_components[layer])
        similarities_cos.append(sim_cos)
        
        # CKA y PWCCA (sobre los espacios de representación truncados, N=20)
        X_layer = chile_trunc[:, layer, :]
        Y_layer = us_trunc[:, layer, :]
        
        sim_cka = linear_cka(X_layer, Y_layer)
        similarities_cka.append(sim_cka)
        
        sim_pwcca = compute_pwcca(X_layer, Y_layer, n_components=10)
        similarities_pwcca.append(sim_pwcca)
        
    # 4. Graficar
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Similitud Representacional Cross-Lingual (Chile vs EEUU)\nModelo: {model_name}', fontsize=14, fontweight='bold')
    
    metrics = [
        (similarities_cos, 'Similitud Coseno (PCA 1)', '#8e44ad'),
        (similarities_cka, 'Linear CKA', '#2980b9'),
        (similarities_pwcca, 'PWCCA', '#27ae60')
    ]
    
    for ax, (data, title, color) in zip(axes, metrics):
        ax.plot(range(num_layers), data, marker='o', color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Capa del Modelo')
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(y=np.mean(data), color='r', linestyle='--', label=f'Promedio: {np.mean(data):.2f}')
        ax.legend()
        
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / f"{safe_name}_crosslingual_similarity.png"
    plt.savefig(fig_path, dpi=300)
    print(f"\n✅ Gráfico guardado en {fig_path}")

    # Guardar métricas en JSON
    import json
    metrics_data = {
        "model": model_name,
        "cosine_similarity": similarities_cos,
        "linear_cka": similarities_cka,
        "pwcca": similarities_pwcca
    }
    json_path = FIGURES_DIR / f"{safe_name}_crosslingual_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"✅ Métricas guardadas en {json_path}")

if __name__ == "__main__":
    main()
