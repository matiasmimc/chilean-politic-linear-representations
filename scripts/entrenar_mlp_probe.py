import numpy as np
import pickle
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from pathlib import Path

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results/replication/v2"

def main():
    models = [
        'meta-llama/Meta-Llama-3-8B-Instruct', 
        'mistralai/Mistral-7B-Instruct-v0.3', 
        'Qwen/Qwen2.5-7B-Instruct',
        'failspy/Meta-Llama-3-8B-Instruct-abliterated-v3'
    ]
    K_HEADS = 32
    
    for model_name in models:
        print(f"\n--- Evaluando Predictibilidad No Lineal (MLP) para: {model_name} ---")
        safe_name = model_name.replace("/", "_")
        output_path = RESULTS_DIR / f"{safe_name}_mlp_probing_results.pkl"
        
        if output_path.exists():
            print(f"Resultados MLP ya existentes para {model_name}. Saltando...")
            continue
            
        features_path = RESULTS_DIR / f"{safe_name}_politician_features.pkl"
        perf_path = RESULTS_DIR / f"{safe_name}_politician_performance.pkl"
        
        if not features_path.exists() or not perf_path.exists():
            print(f"Error: Faltan archivos para {model_name}. Asegúrate de haber ejecutado entrenar_probes.py.")
            continue

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
        # Aplanamos el rendimiento para obtener los índices globales
        flat_perf = performance.flatten()
        top_indices = np.argsort(flat_perf)[-K_HEADS:][::-1]
        
        # Convertir índices planos a (capa, cabeza)
        top_heads = []
        for idx in top_indices:
            layer = idx // num_heads
            head = idx % num_heads
            top_heads.append((layer, head))
            
        print(f"Top {K_HEADS} heads seleccionadas. Mejor correlación individual: {flat_perf[top_indices[0]]:.4f}")

        # 4. Preparar el dataset ensemble (concatenar activaciones de las top K cabezas)
        # X_ensemble shape: [num_statements, K_HEADS * d_head]
        X_ensemble = np.zeros((num_statements, K_HEADS * d_head))
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
                hidden_layer_sizes=(16,), # Muy pequeño para evitar memorización
                activation='relu', 
                solver='lbfgs',    # Mucho mejor para datasets pequeños
                alpha=10.0,        # Regularización fuerte para forzar simplicidad
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

        mean_mlp = np.mean(mlp_correlations)
        mean_ridge = np.mean(ridge_correlations)
        
        print(f"\nResultados para {model_name}:")
        print(f"  - Máximo Individual (referencia): {np.max(performance):.4f}")
        print(f"  - Ensemble Lineal (Ridge, K={K_HEADS}): {mean_ridge:.4f}")
        print(f"  - Ensemble No Lineal (MLP, K={K_HEADS}): {mean_mlp:.4f}")
        
        # Guardar resultados
        output_results = {
            'top_heads': top_heads,
            'mean_spearman_mlp': mean_mlp,
            'mean_spearman_ridge': mean_ridge,
            'mlp_folds': mlp_correlations,
            'ridge_folds': ridge_correlations
        }
        
        output_path = RESULTS_DIR / f"{safe_name}_mlp_probing_results.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(output_results, f)
            
        print(f"Resultados guardados en {output_path}")

if __name__ == "__main__":
    main()
