import numpy as np
import pickle
import argparse
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from pathlib import Path

# Configuración de Rutas Base
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2_names_only"


def train_probes_for_file(features_path, results_dir, force=False):
    safe_name = features_path.name.replace("_politician_features.pkl", "")
    perf_path = results_dir / f"{safe_name}_politician_performance.pkl"
    model_path = results_dir / f"{safe_name}_ridge.pkl"
    
    if perf_path.exists() and model_path.exists() and not force:
        print(f"Probes ya entrenadas para {safe_name}. Saltando (usa --force para reentrenar)...")
        return
        
    print(f"\n--- Entrenando Probes para: {safe_name} ---")
    
    with open(features_path, 'rb') as f:
        features, labels = pickle.load(f)

    # 1. Filter out samples with NaN labels
    valid_idx = ~np.isnan(labels)
    features = features[valid_idx]
    labels = labels[valid_idx]

    # 2. Replace NaNs in features with 0.0 (happens with float16 overflow in some models)
    if np.isnan(features).any():
        features = np.nan_to_num(features, nan=0.0)

    # features: [num_statements, layers, heads, d_head]
    num_statements, num_layers, num_heads, d_head = features.shape
    print(f"Dimensiones de activaciones: {num_statements} parlamentarios, {num_layers} capas, {num_heads} cabezas, d_head={d_head}")
    
    performance = np.zeros((num_layers, num_heads))
    ridge_dict = {}

    for i in tqdm(range(num_layers), desc="Processing Layers"):
        ridge_dict[i] = {}
        for j in range(num_heads):
            kf = KFold(n_splits=2, shuffle=True, random_state=42)
            
            # features[:, i, j, :] -> [num_statements, d_head]
            X = features[:, i, j, :]
            y = labels

            for train_indices, test_indices in kf.split(X):
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
                
                ridge_model = Ridge(alpha=1, fit_intercept=False)
                ridge_model.fit(X_train, y_train)
                
                # Store the model
                ridge_dict[i][j] = ridge_model
                
                y_pred = ridge_model.predict(X_test)
                sc = spearmanr(y_test, y_pred).statistic
                if np.isnan(sc): sc = 0
                performance[i, j] += sc
                
    performance /= 2 # n_splits
    
    with open(perf_path, 'wb') as f:
        pickle.dump(performance, f)
    with open(model_path, 'wb') as f:
        pickle.dump(ridge_dict, f)
        
    print(f"Resultados guardados en {results_dir}")
    print(f"Máximo Spearman Correlation: {np.max(performance):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de Linear Probes (Ridge Regression) por cabeza de atención.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Directorio donde residen los archivos de activaciones.")
    parser.add_argument("--model", type=str, default=None, help="Nombre o filtro de modelo a procesar (opcional).")
    parser.add_argument("--force", action="store_true", help="Forzar reentrenamiento aunque ya existan los resultados.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: No existe el directorio {results_dir}")
        return

    if args.model:
        feature_files = [f for f in results_dir.glob("*_politician_features.pkl") if args.model.lower().replace("/", "_") in f.name.lower()]
        if not feature_files:
            # Fallback if raw string matches filename directly
            feature_files = [f for f in results_dir.glob("*_politician_features.pkl") if args.model.lower() in f.name.lower()]
    else:
        feature_files = sorted(list(results_dir.glob("*_politician_features.pkl")))

    if not feature_files:
        print(f"No se encontraron archivos '*_politician_features.pkl' en {results_dir}")
        return

    print(f"Encontrados {len(feature_files)} archivo(s) de activaciones para evaluar en {results_dir}:")
    for f in feature_files:
        print(f" - {f.name}")

    for feat_file in feature_files:
        train_probes_for_file(feat_file, results_dir, force=args.force)

    print("\n¡Probing completado para todos los modelos!")


if __name__ == "__main__":
    main()
