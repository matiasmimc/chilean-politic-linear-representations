import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import NMF
from sklearn.linear_model import RidgeCV

try:
    from sae_lens import SAE
    SAELENS_AVAILABLE = True
except ImportError:
    SAELENS_AVAILABLE = False
    print("Advertencia: sae_lens no está instalado. Instálalo con `pip install saelens`.")

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2"
DEFAULT_SAE_DIR = BASE_DIR / "results/sae_analysis"

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.abs(np.dot(u, v)) / (norm_u * norm_v))

def get_saelens_ids(model_name, layer_idx):
    """Mapeo heurístico de modelos a releases de SAELens. Modificar según disponibilidad."""
    name_lower = model_name.lower()
    if 'gemma' in name_lower:
        return "gemma-2-9b-res-tc21", f"layer_{layer_idx}/width_16k/average_l0_71"
    elif 'llama-3.1' in name_lower:
        return "llama_3_1_8b_instruct_sae", f"blocks.{layer_idx}.hook_resid_post"
    return None, None

def load_and_finetune_saelens(model_name, layer_idx, X_layer, device='cuda'):
    if not SAELENS_AVAILABLE:
        return None, None
        
    release, sae_id = get_saelens_ids(model_name, layer_idx)
    if not release:
        print(f"  [SAELens] No hay mapeo configurado para {model_name}.")
        return None, None
        
    try:
        print(f"  [SAELens] Intentando descargar: release='{release}', id='{sae_id}'...")
        sae, _, _ = SAE.from_pretrained(release, sae_id, device=device)
        
        # Fine-Tuning
        print(f"  [SAELens] Fine-Tuning SAE preentrenado con datos locales...")
        X_tensor = torch.tensor(X_layer, dtype=torch.float32, device=device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)
        sae.train()
        for epoch in range(10): # Pocas épocas porque ya está preentrenado
            for batch in dataloader:
                x = batch[0]
                optimizer.zero_grad()
                f = sae.encode(x)
                x_hat = sae.decode(f)
                mse_loss = nn.functional.mse_loss(x_hat, x)
                l1_loss = 1e-3 * torch.norm(f, p=1, dim=1).mean()
                loss = mse_loss + l1_loss
                loss.backward()
                optimizer.step()
                
        sae.eval()
        with torch.no_grad():
            f_all = sae.encode(X_tensor)
            
        # SAELens usa sae.W_dec de tamaño [d_sae, d_model]
        # Transponemos para que sea compatible con [d_model, d_sae] si es necesario
        return sae.W_dec.T, f_all.cpu().numpy()
        
    except Exception as e:
        print(f"  [SAELens] Falló la descarga o fine-tuning: {str(e)[:100]}...")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Extracción de Componentes Experimentales con SAE + NMF")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--sae-dir", type=str, default=str(DEFAULT_SAE_DIR))
    parser.add_argument("--nmf-components", type=int, default=10, help="Número de macro-perfiles a extraer")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    sae_dir = Path(args.sae_dir)
    
    sae_dir.mkdir(parents=True, exist_ok=True)
    
    df_path = DATA_DIR / "congreso_2022_2026_20intervenciones.parquet"
    if not df_path.exists():
        print(f"Error: No se encuentra {df_path}")
        return
    df = pd.read_parquet(df_path)
    
    # Este script experimental es ideal para modelos con SAE SOTA
    sota_models = ['meta-llama_Llama-3.1-8B-Instruct']
    feature_files = [f for f in results_dir.glob("*_politician_features.pkl") if any(sota in f.name for sota in sota_models)]
    
    if not feature_files:
        print(f"No se encontraron archivos para modelos con SAE soportado en {results_dir}")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for file_path in feature_files:
        model_name = file_path.name.replace("_politician_features.pkl", "")
        print(f"\n{'='*60}")
        print(f"Procesando Experimental SAE + NMF para: {model_name}")
        print(f"{'='*60}")
        
        with open(file_path, 'rb') as f:
            features, labels = pickle.load(f)
            
        num_statements, num_layers, num_heads, d_head = features.shape
        hidden_dim = num_heads * d_head
        features_flat = features.reshape(num_statements, num_layers, hidden_dim)
        
        all_sae_components = []
        
        for layer_idx in range(num_layers):
            print(f"\nProcesando Capa {layer_idx}/{num_layers-1}...")
            X_layer = features_flat[:, layer_idx, :]
            y = np.array(labels, dtype=float)
            
            if np.isnan(X_layer).any() or np.isinf(X_layer).any():
                X_layer = np.nan_to_num(X_layer, nan=0.0, posinf=0.0, neginf=0.0)
                
            ridge = RidgeCV(alphas=np.logspace(-3, 4, 20))
            ridge.fit(X_layer, y)
            w_ideology = ridge.coef_ # Vector en el espacio original [hidden_dim]
            
            # --- MÉTODO EXPERIMENTAL: SAE + NMF ---
            W_dec_tensor, H_sparse = load_and_finetune_saelens(model_name, layer_idx, X_layer, device=device)
            
            if H_sparse is not None:
                print(f"  [SAE+NMF] Aplicando NMF (k={args.nmf_components}) sobre activaciones dispersas latentes...")
                nmf_sae = NMF(n_components=args.nmf_components, init='random', random_state=42, max_iter=500)
                W_nmf_sae = nmf_sae.fit_transform(H_sparse) # [num_statements, k]
                H_nmf_sae = nmf_sae.components_ # [k, d_sae]
                
                # Proyectar de vuelta al espacio original usando el Decoder del SAE
                H_nmf_tensor = torch.tensor(H_nmf_sae, dtype=torch.float32, device=device)
                with torch.no_grad():
                    if W_dec_tensor.shape[0] == hidden_dim:
                        decoder_matrix = W_dec_tensor.T
                    else:
                        decoder_matrix = W_dec_tensor
                    decoded_components_sae = torch.matmul(H_nmf_tensor, decoder_matrix).cpu().numpy()
            
                for comp_idx in range(args.nmf_components):
                    vec_original_space = decoded_components_sae[comp_idx]
                    sim = cosine_similarity(vec_original_space, w_ideology)
                    pseudo_variance = np.var(W_nmf_sae[:, comp_idx])
                    
                    all_sae_components.append({
                        'layer': layer_idx,
                        'component_idx': comp_idx,
                        'vector': vec_original_space,
                        'ideology_cosine_sim': sim,
                        'explained_variance_ratio': pseudo_variance
                    })
            else:
                print(f"  [SAE+NMF] Falló la carga del SAE para la capa {layer_idx}. Omitiendo.")
                
        # --- GUARDAR RESULTADOS SAE+NMF ---
        if all_sae_components:
            all_sae_sorted = sorted(all_sae_components, key=lambda x: x['explained_variance_ratio'], reverse=True)
            print(f"\nTop 5 Macro-Perfiles globales [SAE+NMF] para {model_name}:")
            for i, comp in enumerate(all_sae_sorted[:5]):
                print(f"  {i+1}. Capa {comp['layer']}, Comp{comp['component_idx']+1} | Pseudo-Var: {comp['explained_variance_ratio']:.4f} | CosSim Ideología: {comp['ideology_cosine_sim']:.4f}")
                
            with open(sae_dir / f"{model_name}_sae_analysis.pkl", 'wb') as f:
                pickle.dump({"all_components": all_sae_sorted}, f)
            
    print("\n✅ Análisis SAE+NMF completado con éxito.")

if __name__ == "__main__":
    main()
