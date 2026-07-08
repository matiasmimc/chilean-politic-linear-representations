import numpy as np
import pickle
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
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
    for model_name in models:
        print(f"\n--- Entrenando Probes para: {model_name} ---")
        safe_name = model_name.replace("/", "_")
        features_path = RESULTS_DIR / f"{safe_name}_politician_features.pkl"
        perf_path = RESULTS_DIR / f"{safe_name}_politician_performance.pkl"
        model_path = RESULTS_DIR / f"{safe_name}_ridge.pkl"
        
        if perf_path.exists() and model_path.exists():
            print(f"Probes ya entrenadas para {model_name}. Saltando...")
            continue
        
        if not features_path.exists():
            print(f"Error: No se encuentra {features_path}. Ejecuta extraer_activaciones.py primero.")
            continue

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
                    
                    # Store the model (overwrites for simplicity, or we could average)
                    ridge_dict[i][j] = ridge_model
                    
                    y_pred = ridge_model.predict(X_test)
                    sc = spearmanr(y_test, y_pred).statistic
                    if np.isnan(sc): sc = 0
                    performance[i, j] += sc
                    
        performance /= 2 # n_splits
        
        perf_path = RESULTS_DIR / f"{safe_name}_politician_performance.pkl"
        model_path = RESULTS_DIR / f"{safe_name}_ridge.pkl"
        
        with open(perf_path, 'wb') as f:
            pickle.dump(performance, f)
        with open(model_path, 'wb') as f:
            pickle.dump(ridge_dict, f)
            
        print(f"Resultados guardados en {RESULTS_DIR}")
        print(f"Máximo Spearman Correlation: {np.max(performance):.4f}")

if __name__ == "__main__":
    main()
