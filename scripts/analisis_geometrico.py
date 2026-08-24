import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2"
GEOMETRY_RESULTS_DIR = BASE_DIR / "results/geometric_similarity"
FIGURES_DIR = BASE_DIR / "figures/geometric_similarity"

GEOMETRY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_gram_matrices(layer_acts):
    """
    Precomputa las matrices Gram centradas K = X_c @ X_c.T para cada capa.
    Dado que N (155) << D (4096), operar en espacio de muestras NxN es órdenes
    de magnitud más rápido.
    """
    gram_list = []
    frob_norms = []
    for X in layer_acts:
        X = X.astype(np.float64)
        X_c = X - np.mean(X, axis=0, keepdims=True)
        K = X_c @ X_c.T  # [N, N]
        norm_k = np.sqrt(np.sum(K ** 2))
        gram_list.append(K)
        frob_norms.append(norm_k)
    return gram_list, frob_norms


def compute_fast_cka_matrix(grams1, norms1, grams2, norms2):
    """
    Calcula la matriz CKA completa [L1, L2] de forma vectorizada y rápida.
    """
    L1 = len(grams1)
    L2 = len(grams2)
    sim_matrix = np.zeros((L1, L2))
    
    for i in range(L1):
        K = grams1[i]
        norm_k = norms1[i]
        for j in range(L2):
            L = grams2[j]
            norm_l = norms2[j]
            hsic = np.sum(K * L)
            denom = norm_k * norm_l
            sim_matrix[i, j] = float(hsic / denom) if denom > 0 else 0.0
            
    return sim_matrix


def precompute_pwcca_bases(layer_acts, top_k=20):
    """
    Precomputa las bases ortonormales Q_x (top_k componentes de SVD) y las matrices centradas X_c
    para cada capa una sola vez.
    """
    bases = []
    centered_acts = []
    for X in layer_acts:
        X = X.astype(np.float64)
        X_c = X - np.mean(X, axis=0, keepdims=True)
        k = min(top_k, X_c.shape[0] - 1, X_c.shape[1])
        Ux, _, _ = np.linalg.svd(X_c, full_matrices=False)
        Q_x = Ux[:, :k]
        bases.append(Q_x)
        centered_acts.append(X_c)
    return bases, centered_acts


def compute_fast_pwcca_matrix(bases1, X_cents1, bases2, epsilon=1e-8):
    """
    Calcula la matriz PWCCA entre dos conjuntos de capas usando las bases precomputadas.
    """
    L1 = len(bases1)
    L2 = len(bases2)
    sim_matrix = np.zeros((L1, L2))
    
    for i in range(L1):
        Q_x = bases1[i]
        X_c = X_cents1[i]
        for j in range(L2):
            Q_y = bases2[j]
            # SVD del producto de subespacios ortogonales
            U, s, _ = np.linalg.svd(Q_x.T @ Q_y, full_matrices=False)
            s = np.clip(s, 0.0, 1.0)
            
            # Pesos de proyección sobre X_c
            weights = []
            for k_idx in range(len(s)):
                u_k = Q_x @ U[:, k_idx]
                weight = np.sum(np.abs(X_c.T @ u_k))
                weights.append(weight)
                
            weights = np.array(weights)
            total_weight = np.sum(weights)
            
            if total_weight < epsilon:
                sim_matrix[i, j] = float(np.mean(s))
            else:
                sim_matrix[i, j] = float(np.sum((weights / total_weight) * s))
                
    return sim_matrix


def load_model_activations(file_path):
    with open(file_path, 'rb') as f:
        features, labels = pickle.load(f)
        
    n_samples, n_layers, n_heads, d_head = features.shape
    hidden_dim = n_heads * d_head
    
    if np.isnan(features).any() or np.isinf(features).any():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
    layer_acts = []
    for l in range(n_layers):
        layer_acts.append(features[:, l, :, :].reshape(n_samples, hidden_dim))
        
    return layer_acts, labels


def plot_similarity_heatmap(sim_matrix, model1_name, model2_name, metric, output_path):
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(
        sim_matrix,
        cmap='magma' if metric == 'cka' else 'viridis',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': f'{metric.upper()} Similarity'}
    )
    plt.title(f'Alineamiento Geométrico ({metric.upper()}):\n{model1_name} vs {model2_name}', fontsize=12, fontweight='bold')
    plt.xlabel(f'Capas de {model2_name}')
    plt.ylabel(f'Capas de {model1_name}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.close()
    print(f"🗺️ Mapa de calor guardado en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Análisis Geométrico de Espacios Latentes (Linear CKA y PWCCA).")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Directorio con archivos de características.")
    parser.add_argument("--metric", type=str, default="cka", choices=["cka", "pwcca", "both"], help="Métrica de similitud geométrica a calcular.")
    parser.add_argument("--model-pair", type=str, default=None, help="Par de modelos a comparar separados por coma.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    feature_files = sorted(list(results_dir.glob("*_politician_features.pkl")))
    
    if not feature_files:
        print(f"No se encontraron archivos en {results_dir}")
        return

    model_names = [f.name.replace("_politician_features.pkl", "") for f in feature_files]
    print(f"Modelos encontrados ({len(model_names)}): {model_names}")

    metrics_to_run = ["cka", "pwcca"] if args.metric == "both" else [args.metric]

    if args.model_pair:
        parts = [p.strip() for p in args.model_pair.split(",")]
        pairs = [(parts[0], parts[1])]
    else:
        pairs = []
        for i in range(len(model_names)):
            for j in range(i, len(model_names)):
                pairs.append((model_names[i], model_names[j]))

    for m1, m2 in pairs:
        file1 = results_dir / f"{m1}_politician_features.pkl"
        file2 = results_dir / f"{m2}_politician_features.pkl"
        
        if not file1.exists() or not file2.exists():
            print(f"Saltando par {m1} - {m2} (uno o ambos archivos no existen).")
            continue
            
        print(f"\n=======================================================")
        print(f"Comparando geometría entre: {m1} y {m2}")
        print(f"=======================================================")
        
        acts1, _ = load_model_activations(file1)
        acts2, _ = load_model_activations(file2)
        
        if "cka" in metrics_to_run:
            print("Calculando Linear CKA optimizado...")
            grams1, norms1 = compute_gram_matrices(acts1)
            grams2, norms2 = compute_gram_matrices(acts2)
            sim_matrix_cka = compute_fast_cka_matrix(grams1, norms1, grams2, norms2)
            
            out_pkl = GEOMETRY_RESULTS_DIR / f"{m1}_vs_{m2}_cka.pkl"
            with open(out_pkl, 'wb') as f:
                pickle.dump({"model_1": m1, "model_2": m2, "metric": "cka", "similarity_matrix": sim_matrix_cka}, f)
                
            out_plot = FIGURES_DIR / f"{m1}_vs_{m2}_cka_heatmap.png"
            plot_similarity_heatmap(sim_matrix_cka, m1, m2, "cka", out_plot)
            
        if "pwcca" in metrics_to_run:
            print("Calculando PWCCA optimizado...")
            bases1, X_cents1 = precompute_pwcca_bases(acts1)
            bases2, _ = precompute_pwcca_bases(acts2)
            sim_matrix_pwcca = compute_fast_pwcca_matrix(bases1, X_cents1, bases2)
            
            out_pkl = GEOMETRY_RESULTS_DIR / f"{m1}_vs_{m2}_pwcca.pkl"
            with open(out_pkl, 'wb') as f:
                pickle.dump({"model_1": m1, "model_2": m2, "metric": "pwcca", "similarity_matrix": sim_matrix_pwcca}, f)
                
            out_plot = FIGURES_DIR / f"{m1}_vs_{m2}_pwcca_heatmap.png"
            plot_similarity_heatmap(sim_matrix_pwcca, m1, m2, "pwcca", out_plot)

    print("\n✅ Análisis geométrico completado con éxito.")


if __name__ == "__main__":
    main()
