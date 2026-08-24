import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from pathlib import Path
import difflib

# Configuración de Rutas Base
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2_names_only"
BCALL_RESULTS_DIR = BASE_DIR / "results/probing/bcall"
BCALL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def match_names(name, bcall_names):
    """
    Empareja los nombres del parquet con los nombres en formato del bcall_scores usando difflib.
    """
    clean_name = name.lower().replace(',', '').replace(' ', '')
    b_names_clean = [n.lower().replace('_', '') for n in bcall_names]
    
    matches = difflib.get_close_matches(clean_name, b_names_clean, n=1, cutoff=0.5)
    if matches:
        idx = b_names_clean.index(matches[0])
        return bcall_names[idx]
    return None

def train_bcall_probes_for_file(features_path, df_parq, df_bcall, output_dir, force=False):
    safe_name = features_path.name.replace("_politician_features.pkl", "")
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    perf_path = out_path / f"{safe_name}_bcall_performance.pkl"
    model_path = out_path / f"{safe_name}_bcall_ridge.pkl"
    
    if perf_path.exists() and model_path.exists() and not force:
        print(f"Probes B-Call ya entrenadas para {safe_name}. Saltando (usa --force para reentrenar)...")
        return
        
    print(f"\n--- Entrenando Probes B-Call para: {safe_name} ---")
    
    with open(features_path, 'rb') as f:
        features, _ = pickle.load(f)

    d1_labels = []
    d2_labels = []
    valid_mask = []
    
    bcall_dict = df_bcall.set_index('legislator')[['d1', 'd2']].to_dict('index')
    bcall_names = list(bcall_dict.keys())
    
    matched_count = 0
    for name in df_parq['Name']:
        match = match_names(name, bcall_names)
        if match:
            d1_labels.append(bcall_dict[match]['d1'])
            d2_labels.append(bcall_dict[match]['d2'])
            valid_mask.append(True)
            matched_count += 1
        else:
            d1_labels.append(np.nan)
            d2_labels.append(np.nan)
            valid_mask.append(False)
            
    print(f"Emparejados {matched_count}/{len(df_parq)} legisladores con B-Call exitosamente.")
            
    d1_labels = np.array(d1_labels)
    d2_labels = np.array(d2_labels)
    valid_mask = np.array(valid_mask)
    
    features = features[valid_mask]
    d1_labels = d1_labels[valid_mask]
    d2_labels = d2_labels[valid_mask]

    if np.isnan(features).any():
        features = np.nan_to_num(features, nan=0.0)

    num_statements, num_layers, num_heads, d_head = features.shape
    
    performance_d1 = np.zeros((num_layers, num_heads))
    performance_d2 = np.zeros((num_layers, num_heads))
    ridge_dict = {'d1': {}, 'd2': {}}

    for i in tqdm(range(num_layers), desc="Procesando Capas"):
        ridge_dict['d1'][i] = {}
        ridge_dict['d2'][i] = {}
        for j in range(num_heads):
            kf = KFold(n_splits=2, shuffle=True, random_state=42)
            X = features[:, i, j, :]
            
            # Entrenar D1 (Ideología)
            sc_d1 = 0
            model_d1 = Ridge(alpha=1, fit_intercept=True)
            for train_idx, test_idx in kf.split(X):
                model_d1.fit(X[train_idx], d1_labels[train_idx])
                preds = model_d1.predict(X[test_idx])
                s = spearmanr(d1_labels[test_idx], preds).statistic
                if not np.isnan(s): sc_d1 += s
            performance_d1[i, j] = sc_d1 / 2
            model_d1.fit(X, d1_labels) # Fit final en todos los datos
            ridge_dict['d1'][i][j] = model_d1
            
            # Entrenar D2 (Volatilidad / Cohesión)
            sc_d2 = 0
            model_d2 = Ridge(alpha=1, fit_intercept=True)
            for train_idx, test_idx in kf.split(X):
                model_d2.fit(X[train_idx], d2_labels[train_idx])
                preds = model_d2.predict(X[test_idx])
                s = spearmanr(d2_labels[test_idx], preds).statistic
                if not np.isnan(s): sc_d2 += s
            performance_d2[i, j] = sc_d2 / 2
            model_d2.fit(X, d2_labels)
            ridge_dict['d2'][i][j] = model_d2
                
    perf_dict = {'d1': performance_d1, 'd2': performance_d2}
    
    with open(perf_path, 'wb') as f:
        pickle.dump(perf_dict, f)
    with open(model_path, 'wb') as f:
        pickle.dump(ridge_dict, f)
        
    print(f"Resultados guardados en {output_dir}")
    print(f"Máximo Spearman D1 (Ideología): {np.max(performance_d1):.4f}")
    print(f"Máximo Spearman D2 (Volatilidad): {np.max(performance_d2):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Probing para las dos dimensiones de B-Call (d1 y d2)")
    parser.add_argument("--features-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(BCALL_RESULTS_DIR / "names_only"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df_parq = pd.read_parquet(DATA_DIR / "congreso_2022_2026_20intervenciones.parquet")
    bcall_path = DATA_DIR / "bcall_data" / "CHL-bcall-scores-2025.csv"
    if not bcall_path.exists():
        print(f"Error: No se encuentra {bcall_path}. Ejecuta el script R de B-Call primero.")
        return
    df_bcall = pd.read_csv(bcall_path)

    feat_dir = Path(args.features_dir)
    if args.model:
        feature_files = [f for f in feat_dir.glob("*_politician_features.pkl") if args.model.lower().replace("/", "_") in f.name.lower()]
    else:
        feature_files = sorted(list(feat_dir.glob("*_politician_features.pkl")))

    for feat_file in feature_files:
        train_bcall_probes_for_file(feat_file, df_parq, df_bcall, args.output_dir, force=args.force)

if __name__ == "__main__":
    main()
