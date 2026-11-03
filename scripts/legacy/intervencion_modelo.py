import torch
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from tqdm import tqdm
from baukit import TraceDict
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results/replication/v2"
MODEL_CACHE = BASE_DIR / "model"

RESULTS_DIR.mkdir(exist_ok=True)

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# Variable global para el hook
current_alpha = 0
current_focal_ridge_dict = {}
current_head_dict = {}
current_features = None
current_model_config = None

def lt_modulated_vector_add(head_output, layer_name):
    global current_alpha, current_focal_ridge_dict, current_head_dict, current_features, current_model_config
    
    # Extract layer index from name (e.g., 'model.layers.15.self_attn.o_proj')
    layer_index = int(layer_name.split('.')[2])
    
    # head_output shape: [batch, seq, hidden_dim]
    head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=current_model_config.num_attention_heads)
    
    for head_index in current_head_dict[layer_index]:
        # Intervention vector: alpha * ridge_coef * std(features)
        # We target the last token's activation
        ridge_coef = current_focal_ridge_dict[(layer_index, head_index)]
        feat_std = np.std(current_features[:, layer_index, head_index, :], axis=0) # features: [num_statements, layers, heads, d_head]
        
        # Add to the last token
        intervention_vec = torch.from_numpy(current_alpha * ridge_coef * feat_std).to(head_output.device).to(head_output.dtype)
        head_output[:, -1, head_index, :] += intervention_vec
        
    head_output = rearrange(head_output, 'b s h d -> b s (h d)')
    return head_output

def main():
    global current_alpha, current_focal_ridge_dict, current_head_dict, current_features, current_model_config
    
    models = [
        'meta-llama/Meta-Llama-3-8B-Instruct', 
        'mistralai/Mistral-7B-Instruct-v0.3', 
        'Qwen/Qwen2.5-7B-Instruct'
    ]
    topics = ['aborto', 'inmigración', 'porte de armas', 'matrimonio igualitario', 'derechos LGBTQ+', 
              'cambio climático', 'confianza en ciencia', 'regulación de redes sociales', 'libertad de expresión']

    for model_name in models:
        print(f"\n--- Intervención en modelo: {model_name} ---")
        safe_name = model_name.replace("/", "_")
        output_results_path = RESULTS_DIR / f"{safe_name}_intervention_results.pkl"
        
        # Omitir generación si ya existen los resultados
        if output_results_path.exists():
            print(f"Resultados para {model_name} ya existen. Saltando generación...")
            continue
            
        print(f"\n--- Intervención en modelo: {model_name} ---")
        
        perf_path = RESULTS_DIR / f"{safe_name}_politician_performance.pkl"
        feat_path = RESULTS_DIR / f"{safe_name}_politician_features.pkl"
        ridge_path = RESULTS_DIR / f"{safe_name}_ridge.pkl"
        
        if not (perf_path.exists() and feat_path.exists() and ridge_path.exists()):
            print(f"Error: Faltan archivos de resultados para {model_name}. Ejecuta los scripts anteriores primero.")
            continue

        performance = pickle.load(open(perf_path, 'rb'))
        with open(feat_path, 'rb') as f:
            current_features, _ = pickle.load(f)
        trained_ridge_dict = pickle.load(open(ridge_path, 'rb'))

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE), token=HUGGINGFACE_TOKEN, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=str(MODEL_CACHE), 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16, 
            token=HUGGINGFACE_TOKEN,
            device_map="auto"
        )
        current_model_config = model.config

        results = []
        # Parámetros de intervención tal cual el notebook
        k_values = reversed([16, 32, 48, 64, 80, 96])
        alpha_values = reversed([-30, -20, -10, 0, 10, 20, 30])

        for k in k_values:
            # Seleccionar las top-k cabezas según performance
            top_indices = np.dstack(np.unravel_index(np.argsort(performance.ravel()), performance.shape))[0][-k:, :][::-1]
            
            # Preparar diccionarios para el hook
            current_focal_ridge_dict = {}
            current_head_dict = {}
            for i in top_indices:
                l_idx, h_idx = i
                current_focal_ridge_dict[(l_idx, h_idx)] = trained_ridge_dict[l_idx][h_idx].coef_
                if l_idx not in current_head_dict:
                    current_head_dict[l_idx] = [h_idx]
                else:
                    current_head_dict[l_idx].append(h_idx)
            
            # Capas a intervenir
            layers_to_trace = [f'model.layers.{i}.self_attn.o_proj' for i in sorted(current_head_dict.keys())]

            for alpha in tqdm(alpha_values, desc=f"k={k}"):
                current_alpha = alpha
                for topic in topics:
                    original_prompt = f"Escribe una declaración sobre {topic}."
                    prompt_formatted = f"USER: {original_prompt}\nASSISTANT: Respecto a {topic}, creo que "
                    input_ids = tokenizer(prompt_formatted, return_tensors="pt")['input_ids'].to(model.device)
                    
                    with torch.no_grad():
                        with TraceDict(model, layers_to_trace, edit_output=lt_modulated_vector_add):
                            # generate often appends to the input, we start from input_ids-1 as in notebook? 
                            # the notebook does: input_ids.to('cuda')[0][:-1].unsqueeze(0)
                            # let's follow that logic if it's meant to bypass the last space/token
                            model_gen_tokens = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.7)
                    
                    model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True)
                    results.append([k, alpha, topic, model_gen_str.strip()])

        output_results_path = RESULTS_DIR / f"{safe_name}_intervention_results.pkl"

        with open(output_results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Intervención completada para {model_name}. Resultados guardados.")
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Consolidación final de resultados
    print("\nConsolidando todos los resultados en intervention_raw.parquet...")
    all_results = []
    for model_name in models:
        safe_name = model_name.replace("/", "_")
        results_path = RESULTS_DIR / f"{safe_name}_intervention_results.pkl"
        if results_path.exists():
            with open(results_path, 'rb') as f:
                all_results += pickle.load(f)
    
    if all_results:
        df_final = pd.DataFrame(all_results, columns=['k', 'alpha', 'topic', 'text'])
        df_final.to_parquet(BASE_DIR / 'results/steering/intervention_raw.parquet')
        print(f"Archivo consolidado guardado en {BASE_DIR / 'results/steering/intervention_raw.parquet'}")

if __name__ == "__main__":
    main()
