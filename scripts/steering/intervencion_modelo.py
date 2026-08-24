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
import re
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
current_method = "ridge"
current_focal_ridge_dict = {}
current_head_dict = {}
current_features = None
current_model_config = None
current_num_heads = None
accumulated_activations = {}
precomputed_interventions = {}

def get_attention_heads(model):
    # Buscar dinámicamente el prefijo exacto de las capas usando named_modules
    prefix = ""
    for name, _ in model.named_modules():
        # Buscar la primera capa de atención
        match = re.search(r'^(.*)\.0\.self_attn\.o_proj$', name)
        if match:
            prefix = match.group(1)
            break
            
    if not prefix:
        # Intento de fallback para arquitecturas sin o_proj o con otra convención
        for name, _ in model.named_modules():
            match = re.search(r'^(.*)\.0\.attn\.c_proj$', name)
            if match:
                prefix = match.group(1)
                break
                
    if not prefix:
        for name, _ in model.named_modules():
            match = re.search(r'^(.*)\.0\.linear_attn\.out_proj$', name)
            if match:
                prefix = match.group(1)
                break
                
    if not prefix:
        raise ValueError(f"No se pudo localizar el prefijo del bloque de atención para {type(model).__name__}")
        
    # Navegar por el modelo para obtener la lista de capas usando el prefijo encontrado
    parts = prefix.split('.')
    layers = model
    for p in parts:
        layers = getattr(layers, p)
        
    heads = []
    for i, layer in enumerate(layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            heads.append(f"{prefix}.{i}.self_attn.o_proj")
        elif hasattr(layer, "attn") and hasattr(layer.attn, "c_proj"):
            heads.append(f"{prefix}.{i}.attn.c_proj")
        elif hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
            heads.append(f"{prefix}.{i}.linear_attn.out_proj")
            
    if not heads:
        raise ValueError(f"No se encontraron proyecciones de atención en {type(model).__name__} bajo el prefijo {prefix}")
        
    return heads, prefix

def lt_modulated_vector_add(head_output, layer_name):
    global current_alpha, current_focal_ridge_dict, current_head_dict, current_features, current_num_heads, accumulated_activations, current_method
    
    # Extraer el índice de la capa de manera robusta
    match = re.search(r'\.(\d+)\.', layer_name) or re.search(r'^(\d+)\.', layer_name)
    if not match:
        raise ValueError(f"No se pudo extraer el índice de la capa de {layer_name}")
    layer_index = int(match.group(1))
    
    if current_method == "ridge":
        # head_output shape: [batch, seq, hidden_dim] -> [batch, seq, heads, d_head]
        head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=current_num_heads)
        
        for head_index in current_head_dict.get(layer_index, []):
            intervention_vec = precomputed_interventions[(layer_index, head_index)]
            if intervention_vec.device != head_output.device:
                intervention_vec = intervention_vec.to(head_output.device)
                precomputed_interventions[(layer_index, head_index)] = intervention_vec
            
            # Save activations before intervention
            act = head_output[:, :, head_index, :].detach().cpu()
            key = (layer_index, head_index)
            if key not in accumulated_activations:
                accumulated_activations[key] = []
            accumulated_activations[key].append(act)
            
            # Add to the last token
            head_output[:, -1, head_index, :] += intervention_vec
            
        head_output = rearrange(head_output, 'b s h d -> b s (h d)')
    else:
        # Para PCA, SAE y NMF operamos sobre el vector completo de tamaño hidden_dim
        if layer_index in precomputed_interventions:
            intervention_vec = precomputed_interventions[layer_index]
            if intervention_vec.device != head_output.device:
                intervention_vec = intervention_vec.to(head_output.device)
                precomputed_interventions[layer_index] = intervention_vec
                
            # Save activations before intervention
            act = head_output[:, :, :].detach().cpu()
            key = layer_index
            if key not in accumulated_activations:
                accumulated_activations[key] = []
            accumulated_activations[key].append(act)
            
            # Add to the last token (intervention_vec tiene tamaño hidden_dim)
            head_output[:, -1, :] += intervention_vec
            
    return head_output

def main():
    global current_alpha, current_focal_ridge_dict, current_head_dict, current_features, current_num_heads, precomputed_interventions
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--total-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--topic", type=str, help="Filtrar un tema específico para evaluar")
    parser.add_argument("--model", type=str, help="Filtrar un modelo específico para evaluar")
    parser.add_argument("--steering-method", type=str, choices=["ridge", "pca", "sae", "nmf"], default="ridge", help="Método de extracción del vector de steering")
    args = parser.parse_args()
    
    global current_method
    current_method = args.steering_method

    models = [
        'google/gemma-4-E4B-it',
        'meta-llama/Llama-3.1-8B-Instruct',
        'Qwen/Qwen3.5-9B',
        # 'huihui-ai/Huihui-Qwen3.5-9B-abliterated',
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
                out_pq = OUTPUT_DIR / f'intervention_{args.steering_method}.parquet'
                df_final.to_parquet(out_pq)
                print(f"Archivo consolidado exitosamente en {out_pq}")
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
        print(f"\n--- Intervención en modelo: {model_name} (Método: {args.steering_method}) ---")
        safe_name = model_name.replace("/", "_")
        output_results_path = OUTPUT_DIR / f"{safe_name}_intervention_results_{args.steering_method}.pkl"
        
        # Omitir generación si ya existen los resultados
        if output_results_path.exists():
            print(f"Resultados para {model_name} ya existen. Saltando generación...")
            continue
            
        print(f"\n--- Intervención en modelo: {model_name} ---")
        
        perf_path = RESULTS_DIR / f"{safe_name}_politician_performance.pkl"
        feat_path = RESULTS_DIR / f"{safe_name}_politician_features.pkl"
        
        if not feat_path.exists():
            print(f"Error: Faltan archivos de características para {model_name}.")
            continue
            
        with open(feat_path, 'rb') as f:
            current_features, _ = pickle.load(f)
            
        if args.steering_method == "ridge":
            ridge_path = RESULTS_DIR / f"{safe_name}_ridge.pkl"
            if not (perf_path.exists() and ridge_path.exists()):
                print(f"Error: Faltan archivos de resultados ridge para {model_name}. Ejecuta entrenar_probes.py")
                continue
            performance = pickle.load(open(perf_path, 'rb'))
            trained_model_dict = pickle.load(open(ridge_path, 'rb'))
        elif args.steering_method == "pca":
            pca_path = BASE_DIR / f"results/pca_analysis/{safe_name}_pca_analysis.pkl"
            if not pca_path.exists():
                print(f"Error: Falta archivo PCA para {model_name}. Ejecuta analisis_pca_activaciones.py")
                continue
            trained_model_dict = pickle.load(open(pca_path, 'rb'))
        elif args.steering_method in ["sae", "nmf"]:
            # Marcador de posición para cuando se implemente SAE/NMF
            method_path = BASE_DIR / f"results/{args.steering_method}_analysis/{safe_name}_{args.steering_method}_analysis.pkl"
            if not method_path.exists():
                print(f"Error: Falta archivo {args.steering_method} para {model_name}.")
                continue
            trained_model_dict = pickle.load(open(method_path, 'rb'))

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE), token=HUGGINGFACE_TOKEN, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=str(MODEL_CACHE), 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16, 
            token=HUGGINGFACE_TOKEN,
            device_map="auto"
        )
        
        input_device = getattr(model, "device", None) or next(model.parameters()).device
        
        # Obtener num_attention_heads de manera robusta
        current_num_heads = getattr(model.config, "num_attention_heads", None)
        if current_num_heads is None and hasattr(model.config, "text_config"):
            current_num_heads = getattr(model.config.text_config, "num_attention_heads", None)
        if current_num_heads is None:
            current_num_heads = getattr(model.config, "n_head", None)
        if current_num_heads is None:
            raise ValueError(f"No se pudo determinar el número de cabezas (num_attention_heads) para {model_name}.")
            
        heads_list, model_prefix = get_attention_heads(model)

        results = []
        # Parámetros de intervención tal cual el notebook
        k_values = reversed([16, 32, 48, 64, 80, 96])
        alpha_values = sorted([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])

        for k in k_values:
            layers_to_trace = []
            
            if args.steering_method == "ridge":
                # Seleccionar las top-k cabezas según performance
                top_indices = np.dstack(np.unravel_index(np.argsort(performance.ravel()), performance.shape))[0][-k:, :][::-1]
                
                current_focal_ridge_dict = {}
                current_head_dict = {}
                for i in top_indices:
                    l_idx, h_idx = i
                    current_focal_ridge_dict[(l_idx, h_idx)] = trained_model_dict[l_idx][h_idx].coef_
                    if l_idx not in current_head_dict:
                        current_head_dict[l_idx] = [h_idx]
                    else:
                        current_head_dict[l_idx].append(h_idx)
                
                layers_to_trace = [heads_list[i] for i in sorted(current_head_dict.keys())]
            else:
                # Para PCA, SAE, NMF
                # Extraemos los top K componentes globales
                top_components = trained_model_dict["all_components"][:k]
                current_head_dict = {} # Guardaremos los vectores agrupados por capa
                
                for comp in top_components:
                    l_idx = comp['layer']
                    vec = comp['vector'] # [hidden_dim]
                    if l_idx not in current_head_dict:
                        current_head_dict[l_idx] = [vec]
                    else:
                        current_head_dict[l_idx].append(vec)
                        
                layers_to_trace = [heads_list[i] for i in sorted(current_head_dict.keys())]

            for alpha in tqdm(alpha_values, desc=f"k={k}"):
                current_alpha = alpha
                
                precomputed_interventions.clear()
                
                if args.steering_method == "ridge":
                    for l_idx, h_idx in current_focal_ridge_dict.keys():
                        ridge_coef = current_focal_ridge_dict[(l_idx, h_idx)]
                        acts = current_features[:, l_idx, h_idx, :]
                        proj = acts @ ridge_coef
                        std_proj = np.std(proj)
                        norm_coef = np.linalg.norm(ridge_coef)
                        direction = ridge_coef / (norm_coef if norm_coef > 0 else 1)
                        
                        vec = torch.from_numpy(current_alpha * std_proj * direction).to(model.dtype).to(input_device)
                        precomputed_interventions[(l_idx, h_idx)] = vec
                else:
                    # Precomputar intervenciones globales por capa para PCA/SAE/NMF
                    for l_idx, vecs in current_head_dict.items():
                        layer_total_vec = np.zeros_like(vecs[0])
                        # acts_flat shape: [num_statements, hidden_dim]
                        num_stmts = current_features.shape[0]
                        acts_flat = current_features[:, l_idx, :, :].reshape(num_stmts, -1)
                        
                        for vec in vecs:
                            proj = acts_flat @ vec
                            std_proj = np.std(proj)
                            norm_coef = np.linalg.norm(vec)
                            direction = vec / (norm_coef if norm_coef > 0 else 1)
                            layer_total_vec += (current_alpha * std_proj * direction)
                            
                        t_vec = torch.from_numpy(layer_total_vec).to(model.dtype).to(input_device)
                        precomputed_interventions[l_idx] = t_vec
                
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
                        full_act = torch.cat(act_list, dim=1).squeeze(0).float().numpy() # [total_seq_len, d_head]
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

        output_results_path = OUTPUT_DIR / f"{safe_name}_intervention_results_{args.steering_method}.pkl"

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
            results_path = OUTPUT_DIR / f"{safe_name}_intervention_results_{args.steering_method}.pkl"
            if results_path.exists():
                with open(results_path, 'rb') as f:
                    all_results += pickle.load(f)
        
        if all_results:
            df_final = pd.DataFrame(all_results, columns=['model', 'k', 'alpha', 'topic', 'text'])
            out_pq = OUTPUT_DIR / f'intervention_{args.steering_method}.parquet'
            df_final.to_parquet(out_pq)
            print(f"Archivo consolidado guardado en {out_pq}")

if __name__ == "__main__":
    main()
