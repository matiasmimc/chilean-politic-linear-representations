import torch
import numpy as np
import pandas as pd
import pickle
import os
import gc
import argparse
from tqdm import tqdm
from baukit import TraceDict
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results/replication/v2_names_only"
MODEL_CACHE = BASE_DIR / "model"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('HUGGINGFACE_TOKEN='):
                    HUGGINGFACE_TOKEN = line.strip().split('=', 1)[1]


def get_attention_heads(model):
    layers = None
    prefix = ""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        prefix = "model.layers"
    elif hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        layers = model.model.language_model.layers
        prefix = "model.language_model.layers"
    elif hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        layers = model.language_model.model.layers
        prefix = "language_model.model.layers"
    elif hasattr(model, "layers"):
        layers = model.layers
        prefix = "layers"
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        prefix = "transformer.h"
        
    if layers is None:
        raise ValueError(f"No se pudo localizar el bloque de capas del modelo de lenguaje para {type(model).__name__}")
        
    heads = []
    for i, layer in enumerate(layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            heads.append(f"{prefix}.{i}.self_attn.o_proj")
            
    if not heads:
        raise ValueError(f"No se encontraron capas con 'self_attn.o_proj' en {type(model).__name__}")
        
    return heads


def extract_activations(model, statements, target="attention"):
    if target == "residual":
        features = []
        for prompt in tqdm(statements, desc="Extracting residual stream", total=len(statements)):
            with torch.no_grad():
                outputs = model(prompt.to(model.device), output_hidden_states=True)
                hidden_states = torch.stack(outputs.hidden_states[1:]).squeeze() # [layers, seq, d_model]
                last_token_states = hidden_states[:, -1, :].cpu().float().numpy()
                features.append(last_token_states)
        features = np.stack(features, axis=0) # [num_statements, layers, d_model]
        return features

    HEADS = get_attention_heads(model)
    head_wise_hidden_states_list = []
    
    device = next(model.parameters()).device
    
    for prompt in tqdm(statements, desc="Extracting activations", total=len(statements)):
        # prompt is a tensor [1, seq_len]
        with torch.no_grad():
            with TraceDict(model, HEADS) as ret:
                _ = model(prompt.to(device))
                # output from o_proj is [batch, seq, hidden_dim]
                head_wise_hidden_states = [ret[head].output.squeeze().detach().float().cpu() for head in HEADS]
                head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).numpy() # [layers, seq, hidden]
                head_wise_hidden_states_list.append(head_wise_hidden_states)
    
    features = []
    num_heads = getattr(model.config, "num_attention_heads", None)
    if num_heads is None and hasattr(model.config, "text_config"):
        num_heads = getattr(model.config.text_config, "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(model.config, "n_head", None)
    if num_heads is None:
        raise ValueError("No se pudo determinar el número de cabezas (num_attention_heads) del modelo.")
        
    for head_wise_hidden_states in head_wise_hidden_states_list:
        # We take the state of the last token in the sequence
        last_token_states = head_wise_hidden_states[:, -1, :] # [layers, hidden]
        # Reshape to [layers, heads, d_head]
        reshaped = rearrange(last_token_states, 'l (h d) -> l h d', h=num_heads)
        features.append(reshaped)
        
    features = np.stack(features, axis=0) # [num_statements, layers, heads, d_head]
    return features


def main():
    parser = argparse.ArgumentParser(description="Extracción de activaciones usando solo nombres de parlamentarios.")
    parser.add_argument("--model", type=str, default=None, help="Nombre o filtro del modelo a procesar (opcional).")
    parser.add_argument("--data-path", type=str, default=str(DATA_DIR / "congreso_2022_2026_20intervenciones.parquet"), help="Ruta al archivo de datos.")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), help="Directorio de salida para los resultados.")
    parser.add_argument("--cache-dir", type=str, default=str(MODEL_CACHE), help="Directorio del caché de los modelos descargados.")
    parser.add_argument("--target", type=str, default="attention", choices=["attention", "residual"], help="Objetivo de extracción (cabezas de atención o residual stream).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if args.target == "residual":
        out_dir = out_dir / "residual"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_path = Path(args.data_path)
    if not df_path.exists():
        print(f"Error: No se encuentra {df_path}")
        return

    if df_path.suffix == ".parquet":
        df = pd.read_parquet(df_path)
    else:
        df = pd.read_csv(df_path)
    
    # Modelos evaluados para 1x GPU en fp16
    models = [
        'google/gemma-4-E4B-it',
        'meta-llama/Llama-3.1-8B-Instruct',
        # 'Qwen/Qwen3.5-9B',
        'zai-org/GLM-4-9B-0414',
        # 'mistralai/Ministral-3-8B-Instruct-2512',
    ]

    if args.model:
        models = [m for m in models if args.model.lower() in m.lower()]
        if not models:
            # Si no coincide con la lista por defecto, permitir ejecutar el modelo exacto pasado por argumento
            models = [args.model]

    for model_name in models:
        print(f"\n--- Procesando modelo: {model_name} (Solo Nombres) ---")
        safe_name = model_name.replace("/", "_")
        
        output_path = out_dir / f"{safe_name}_politician_features.pkl"
        if output_path.exists():
            print(f"Activaciones ya extraídas para {model_name} en {output_path}. Saltando...")
            continue
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir, token=HUGGINGFACE_TOKEN, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=args.cache_dir, 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16, 
            token=HUGGINGFACE_TOKEN,
            device_map="auto"
        )

        # Preparar prompts usando solo los nombres de los parlamentarios
        print("Preparando prompts (solo nombres)...")
        statements = []
        for _, row in df.iterrows():
            name = row['Name']
            messages = [{"role": "user", "content": f"Genera una declaración de {name}, un/a parlamentario/a en Chile."}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_text += f"En 2025, {name} dijo que"

            statements.append(tokenizer(prompt_text, return_tensors="pt")['input_ids'])
        
        # Guardar prompts
        prompts_path = out_dir / f"{safe_name}_politicians.pkl"
        with open(prompts_path, 'wb') as f:
            pickle.dump(statements, f)

        # Extraer activaciones
        labels = np.array(df['dim_ideology_P6'].astype(float))
        features = extract_activations(model, statements, target=args.target)
        
        with open(output_path, 'wb') as f:
            pickle.dump((features, labels), f)
        
        print(f"Activaciones guardadas exitosamente en {output_path}")
        
        # Liberar memoria
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
