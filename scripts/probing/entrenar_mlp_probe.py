import numpy as np
import pickle
import argparse
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from pathlib import Path

# Configuración de Rutas Base
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2_names_only"


def train_mlp_probe_for_model(safe_name, results_dir, k_heads=32, force=False):
    output_path = results_dir / f"{safe_name}_mlp_probing_results.pkl"
    if output_path.exists() and not force:
        print(f"Resultados MLP ya existentes para {safe_name}. Saltando (usa --force para reentrenar)...")
        return

    features_path = results_dir / f"{safe_name}_politician_features.pkl"
    perf_path = results_dir / f"{safe_name}_politician_performance.pkl"
    
    if not features_path.exists() or not perf_path.exists():
        print(f"Error: Faltan archivos de activaciones o rendimiento para {safe_name}. Asegúrate de haber ejecutado entrenar_probes.py.")
        return

    print(f"\n--- Evaluando Predictibilidad No Lineal (MLP) para: {safe_name} ---")

    # Cargar activaciones y etiquetas
    with open(features_path, 'rb') as f:
        features, labels = pickle.load(f)
    
    # Cargar rendimiento individual de las cabezas
    with open(perf_path, 'rb') as f:
        performance = pickle.load(f)

    # 1. Filtrar muestras con etiquetas NaN
    valid_idx = ~np.isnan(labels)
    features = features[valid_idx]
    labels = labels[valid_idx]

    # 2. Reemplazar NaNs en features por 0.0
    if np.isnan(features).any():
        features = np.nan_to_num(features, nan=0.0)

    num_statements, num_layers, num_heads, d_head = features.shape

    # 3. Identificar las top K cabezas más predictivas
    total_heads = num_layers * num_heads
    actual_k = min(k_heads, total_heads)
    
    flat_perf = performance.flatten()
    top_indices = np.argsort(flat_perf)[-actual_k:][::-1]
    
    # Convertir índices planos a (capa, cabeza)
    top_heads = []
    for idx in top_indices:
        layer = idx // num_heads
        head = idx % num_heads
        top_heads.append((layer, head))
        
    print(f"Top {actual_k} heads seleccionadas. Mejor correlación individual: {flat_perf[top_indices[0]]:.4f}")

    # 4. Preparar el dataset ensemble (concatenar activaciones de las top K cabezas)
    X_ensemble = np.zeros((num_statements, actual_k * d_head))
    for i, (layer, head) in enumerate(top_heads):
        X_ensemble[:, i*d_head : (i+1)*d_head] = features[:, layer, head, :]

    # 5. Validación Cruzada
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    mlp_correlations = []
    ridge_correlations = []

    for train_idx, test_idx in tqdm(kf.split(X_ensemble), total=n_splits, desc="Cross-Validation"):
        X_train, X_test = X_ensemble[train_idx], X_ensemble[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # --- Escalamiento de Datos (Vital para MLP) ---
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # --- MLP Probe Ultra-Robusto (Optimizado para N pequeño) ---
        mlp = MLPRegressor(
            hidden_layer_sizes=(16,),
            activation='relu', 
            solver='lbfgs',
            alpha=10.0,
            max_iter=2000, 
            random_state=42
        )
        mlp.fit(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test)
        
        sc_mlp = spearmanr(y_test, y_pred_mlp).statistic
        if not np.isnan(sc_mlp):
            mlp_correlations.append(sc_mlp)
        
        # --- Ridge Baseline (Ensemble Lineal) ---
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        
        sc_ridge = spearmanr(y_test, y_pred_ridge).statistic
        if not np.isnan(sc_ridge):
            ridge_correlations.append(sc_ridge)

    mean_mlp = np.mean(mlp_correlations) if mlp_correlations else 0.0
    mean_ridge = np.mean(ridge_correlations) if ridge_correlations else 0.0
    
    print(f"\nResultados para {safe_name}:")
    print(f"  - Máximo Individual (referencia): {np.max(performance):.4f}")
    print(f"  - Ensemble Lineal (Ridge, K={actual_k}): {mean_ridge:.4f}")
    print(f"  - Ensemble No Lineal (MLP, K={actual_k}): {mean_mlp:.4f}")
    
    # Guardar resultados
    output_results = {
        'top_heads': top_heads,
        'mean_spearman_mlp': mean_mlp,
        'mean_spearman_ridge': mean_ridge,
        'mlp_folds': mlp_correlations,
        'ridge_folds': ridge_correlations
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_results, f)
        
    print(f"Resultados guardados en {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluación de predictibilidad no lineal (MLP vs Ridge Ensemble) para top-k heads.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Directorio donde residen los archivos de resultados.")
    parser.add_argument("--model", type=str, default=None, help="Filtrar por modelo específico (opcional).")
    parser.add_argument("--k-heads", type=int, default=32, help="Cantidad de cabezas top a concatenar en el ensemble.")
    parser.add_argument("--force", action="store_true", help="Forzar reevaluación aunque ya existan los resultados.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: No existe el directorio {results_dir}")
        return

    if args.model:
        perf_files = [f for f in results_dir.glob("*_politician_performance.pkl") if args.model.lower().replace("/", "_") in f.name.lower()]
    else:
        perf_files = sorted(list(results_dir.glob("*_politician_performance.pkl")))

    if not perf_files:
        print(f"No se encontraron archivos '*_politician_performance.pkl' en {results_dir}")
        print("Ejecuta 'scripts/entrenar_probes.py' primero.")
        return

    for perf_file in perf_files:
        safe_name = perf_file.name.replace("_politician_performance.pkl", "")
        train_mlp_probe_for_model(safe_name, results_dir, k_heads=args.k_heads, force=args.force)

    print("\n¡Evaluación MLP completada!")


if __name__ == "__main__":
    main()
