import torch
import numpy as np
import pandas as pd
import pickle
import os
import sys
import argparse
import subprocess
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm
from baukit import TraceDict
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = BASE_DIR / "results/replication/v2"
# OUTPUT_DIR = BASE_DIR / "results/steering"
OUTPUT_DIR = BASE_DIR / "results/steering_200"
MODEL_CACHE = BASE_DIR / "model"

MAX_NEW_TOKENS = 200

RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('HUGGINGFACE_TOKEN='):
                    HUGGINGFACE_TOKEN = line.strip().split('=', 1)[1]

# Variable global para el hook
current_alpha = 0
current_focal_ridge_dict = {}
current_head_dict = {}
current_features = None
current_model_config = None
accumulated_activations = {}

def lt_modulated_vector_add(head_output, layer_name):
    global current_alpha, current_focal_ridge_dict, current_head_dict, current_features, current_model_config, accumulated_activations
    
    # Extract layer index from name (e.g., 'model.layers.15.self_attn.o_proj')
    layer_index = int(layer_name.split('.')[2])
    
    # head_output shape: [batch, seq, hidden_dim]
    head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=current_model_config.num_attention_heads)
    
    for head_index in current_head_dict[layer_index]:
        # Intervention vector: alpha * (ridge_coef / norm(ridge_coef)) * std(projection)
        # We target the last token's activation
        ridge_coef = current_focal_ridge_dict[(layer_index, head_index)]
        
        # Calcular la desviación estándar de la proyección de las activaciones sobre la dirección de la sonda (ridge_coef)
        # features shape: [num_statements, layers, heads, d_head]
        acts = current_features[:, layer_index, head_index, :] # [num_statements, d_head]
        proj = acts @ ridge_coef # [num_statements]
        std_proj = np.std(proj)
        
        # Normalizar ridge_coef a norma 1 para no depender de la escala de los coeficientes originales
        norm_ridge = np.linalg.norm(ridge_coef)
        if norm_ridge > 0:
            direction = ridge_coef / norm_ridge
        else:
            direction = ridge_coef
            
        # El vector de intervención es escalado por alpha y la desviación estándar de la proyección
        intervention_vec = torch.from_numpy(current_alpha * std_proj * direction).to(head_output.device).to(head_output.dtype)
        
        # Save activations before intervention
        act = head_output[:, :, head_index, :].detach().cpu()
        key = (layer_index, head_index)
        if key not in accumulated_activations:
            accumulated_activations[key] = []
        accumulated_activations[key].append(act)
        
        # Add to the last token
        head_output[:, -1, head_index, :] += intervention_vec
        
    head_output = rearrange(head_output, 'b s h d -> b s (h d)')
    return head_output

def main():
    global current_alpha, current_focal_ridge_dict, current_head_dict, current_features, current_model_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--total-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--topic", type=str, help="Filtrar un tema específico para evaluar")
    parser.add_argument("--model", type=str, help="Filtrar un modelo específico para evaluar")
    args = parser.parse_args()

    models = [
        'meta-llama/Meta-Llama-3-8B-Instruct', 
        'mistralai/Mistral-7B-Instruct-v0.3', 
        'Qwen/Qwen2.5-7B-Instruct',
        'failspy/Meta-Llama-3-8B-Instruct-abliterated-v3'
    ]
    
    if args.model:
        models = [m for m in models if args.model.lower() in m.lower()]
        if not models:
            print(f"Error: El modelo '{args.model}' no coincide con los disponibles.")
            return

    # Lógica del orquestador principal
    if not args.worker:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Detectadas {num_gpus} GPUs. Repartiendo modelos en {num_gpus} procesos paralelos...")
            processes = []
            for i in range(num_gpus):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(i)
                cmd = [sys.executable, sys.argv[0], "--worker", "--total-workers", str(num_gpus), "--worker-id", str(i)]
                if args.topic:
                    cmd.extend(["--topic", args.topic])
                if args.model:
                    cmd.extend(["--model", args.model])
                p = subprocess.Popen(cmd, env=env)
                processes.append(p)
            
            for p in processes:
                p.wait()
                
            print("\nTodos los workers terminaron. Consolidando resultados...")
            all_results = []
            for model_name in models:
                safe_name = model_name.replace("/", "_")
                results_path = OUTPUT_DIR / f"{safe_name}_intervention_results.pkl"
                if results_path.exists():
                    with open(results_path, 'rb') as f:
                        all_results += pickle.load(f)
            
            if all_results:
                df_final = pd.DataFrame(all_results, columns=['model', 'k', 'alpha', 'topic', 'text'])
                df_final.to_parquet(OUTPUT_DIR / 'intervention_raw.parquet')
                print(f"Archivo consolidado exitosamente en {OUTPUT_DIR / 'intervention_raw.parquet'}")
            return
            
    # --- Lógica del Worker o Single GPU ---
    if args.total_workers > 1:
        # Repartir modelos según el worker id usando módulo
        models = [m for i, m in enumerate(models) if i % args.total_workers == args.worker_id]
        if not models:
            print(f"Worker {args.worker_id}: No hay modelos asignados para este worker. Terminando.")
            return
        print(f"Worker {args.worker_id}: Procesando modelos {models} en GPU {args.worker_id}")

    topics = ['aborto', 'inmigración', 'porte de armas', 'matrimonio igualitario', 'derechos LGBTQ+', 
              'cambio climático', 'confianza en ciencia', 'regulación de redes sociales', 'libertad de expresión']
              
    if args.topic:
        topics = [t for t in topics if args.topic.lower() in t.lower()]
        if not topics:
            print(f"Error: El tema '{args.topic}' no coincide con los disponibles.")
            return

    for model_name in models:
        print(f"\n--- Intervención en modelo: {model_name} ---")
        safe_name = model_name.replace("/", "_")
        output_results_path = OUTPUT_DIR / f"{safe_name}_intervention_results.pkl"
        
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
            dtype=torch.float16, 
            token=HUGGINGFACE_TOKEN,
            device_map="auto"
        )
        current_model_config = model.config

        results = []
        # Parámetros de intervención tal cual el notebook
        k_values = reversed([16, 32, 48, 64, 80, 96])
        alpha_values = sorted([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])

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
                    messages = [{"role": "user", "content": f"Escribe una declaración sobre {topic}."}]
                    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompt_formatted += f"Respecto a {topic}, creo que "
                    input_ids = tokenizer(prompt_formatted, return_tensors="pt")['input_ids'].to(model.device)
                    
                    accumulated_activations.clear()
                    with torch.no_grad():
                        with TraceDict(model, layers_to_trace, edit_output=lt_modulated_vector_add):
                            model_gen_tokens = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
                    
                    # Process and save activations
                    token_activations = {}
                    for key, act_list in accumulated_activations.items():
                        full_act = torch.cat(act_list, dim=1).squeeze(0).numpy() # [total_seq_len, d_head]
                        token_activations[key] = full_act
                    
                    token_ids = model_gen_tokens[0].cpu().numpy()
                    try:
                        token_strings = tokenizer.convert_ids_to_tokens(model_gen_tokens[0].tolist())
                    except OverflowError:
                        token_strings = ["[ERROR_DECODING]"] * len(token_ids)
                    
                    save_data = {
                        'token_ids': token_ids,
                        'token_strings': token_strings,
                        'activations': token_activations
                    }
                    
                    act_dir = OUTPUT_DIR / "token_activations" / safe_name / f"k_{k}" / f"alpha_{alpha}"
                    act_dir.mkdir(parents=True, exist_ok=True)
                    act_file = act_dir / f"{topic}.pkl"
                    
                    with open(act_file, 'wb') as f:
                        pickle.dump(save_data, f)
                        
                    try:
                        model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True)
                    except OverflowError:
                        model_gen_str = "[ERROR_DECODING: El modelo generó tokens inválidos debido a intervención extrema]"
                    results.append([model_name, k, alpha, topic, model_gen_str.strip()])

        output_results_path = OUTPUT_DIR / f"{safe_name}_intervention_results.pkl"

        with open(output_results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Intervención completada para {model_name}. Resultados guardados.")
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Consolidación final de resultados (Solo si es Single GPU)
    if not args.worker:
        print("\nConsolidando todos los resultados en intervention_raw.parquet...")
        all_results = []
        for model_name in models:
            safe_name = model_name.replace("/", "_")
            results_path = OUTPUT_DIR / f"{safe_name}_intervention_results.pkl"
            if results_path.exists():
                with open(results_path, 'rb') as f:
                    all_results += pickle.load(f)
        
        if all_results:
            df_final = pd.DataFrame(all_results, columns=['model', 'k', 'alpha', 'topic', 'text'])
            df_final.to_parquet(OUTPUT_DIR / 'intervention_raw.parquet')
            print(f"Archivo consolidado guardado en {OUTPUT_DIR / 'intervention_raw.parquet'}")

if __name__ == "__main__":
    main()
