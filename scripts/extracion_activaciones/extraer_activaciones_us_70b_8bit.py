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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results/us_analysis"
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

def extract_activations_us_70b(model, statements, target="attention"):
    try:
        input_device = model.get_input_embeddings().weight.device
    except Exception:
        input_device = next(model.parameters()).device

    if target == "residual":
        features = []
        for prompt in tqdm(statements, desc="Extrayendo residual stream (US 70B 8-bit)", total=len(statements)):
            with torch.no_grad():
                outputs = model(prompt.to(input_device), output_hidden_states=True)
                hidden_states = torch.stack(outputs.hidden_states[1:]).squeeze() # [layers, seq, d_model]
                last_token_states = hidden_states[-1, :].cpu().float().numpy() if hidden_states.ndim == 2 else hidden_states[:, -1, :].cpu().float().numpy()
                features.append(last_token_states.copy())
        features = np.stack(features, axis=0) # [num_statements, layers, d_model]
        return features

    HEADS = get_attention_heads(model)
    
    num_heads = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", None))
    if num_heads is None and hasattr(model.config, "text_config"):
        num_heads = getattr(model.config.text_config, "num_attention_heads", None)
        
    features = []
    
    for prompt in tqdm(statements, desc="Extrayendo activaciones (US 70B 8-bit)", total=len(statements)):
        with torch.no_grad():
            with TraceDict(model, HEADS) as ret:
                _ = model(prompt.to(input_device))
                head_wise_hidden_states = [ret[head].output.squeeze().detach().float().cpu() for head in HEADS]
                head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0)
                
                # Extraemos y guardamos SOLO el último token para no explotar la memoria RAM
                last_token_states = head_wise_hidden_states[:, -1, :] 
                
                # CRÍTICO: Usar .copy() para romper la referencia de memoria compartida
                reshaped = rearrange(last_token_states.numpy().copy(), 'l (h d) -> l h d', h=num_heads)
                features.append(reshaped)
        
    features = np.stack(features, axis=0)
    return features

def clean_bioname(bioname):
    """Limpia el formato 'APELLIDO, Nombre (Apodo)' a 'Nombre Apellido'."""
    if pd.isna(bioname):
        return ""
    import re
    bioname = re.sub(r'\(.*?\)', '', bioname)
    
    parts = bioname.split(',')
    if len(parts) == 2:
        last = parts[0].strip().title()
        first = parts[1].strip().title()
        return f"{first} {last}"
    return bioname.strip().title()

def main():
    parser = argparse.ArgumentParser(description="Extracción de activaciones para US Politicians con modelos 70B (8-bits y multi-GPU).")
    parser.add_argument("--model", type=str, default=None, help="Nombre o filtro del modelo a procesar (opcional).")
    parser.add_argument("--target", type=str, default="attention", choices=["attention", "residual"], help="Objetivo de extracción (cabezas de atención o residual stream).")
    parser.add_argument("--cache-dir", type=str, default=str(MODEL_CACHE), help="Directorio del caché de los modelos descargados.")
    args = parser.parse_args()

    results_dir = RESULTS_DIR
    if args.target == "residual":
        results_dir = results_dir / "residual"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Cargar CSV de Nominate
    csv_path = DATA_DIR / "EEUU-nominate.csv"
    if not csv_path.exists():
        print(f"Error: No se encuentra {csv_path}")
        return
        
    df_raw = pd.read_csv(csv_path)
    
    df_dems = df_raw[df_raw['party_code'] == 100].copy()
    df_reps = df_raw[df_raw['party_code'] == 200].copy()
    
    np.random.seed(42)
    sample_dems = df_dems.sample(78)
    sample_reps = df_reps.sample(78)
    
    df = pd.concat([sample_dems, sample_reps]).reset_index(drop=True)
    
    df['Name'] = df['bioname'].apply(clean_bioname)
    df['Ideology'] = df['nominate_dim1'].astype(float)
    df['Party'] = df['party_code'].map({100: 'Democrat', 200: 'Republican'})
    
    df.to_parquet(results_dir / "us_politicians.parquet")

    # Modelos 70B a ejecutar en paralelo cuantizados en 8-bits
    models = [
        'meta-llama/Llama-3.1-70B-Instruct',
        'latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0',
    ]

    if args.model:
        models = [m for m in models if args.model.lower() in m.lower()]
        if not models:
            models = [args.model]

    # Configuración de cuantización en 8-bits
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    for model_name in models:
        print(f"\n--- Procesando modelo 70B: {model_name} (US Politicians - 8-bits / Multi-GPU) ---")
        safe_name = model_name.replace("/", "_")
        
        output_path = results_dir / f"{safe_name}_politician_features.pkl"
        if output_path.exists():
            print(f"Activaciones ya extraídas para {model_name} en {output_path}. Saltando...")
            continue
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir, token=HUGGINGFACE_TOKEN, use_fast=False)
        
        print("Cargando modelo en 8-bits con device_map='auto'...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=args.cache_dir, 
            quantization_config=quantization_config,
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16, 
            token=HUGGINGFACE_TOKEN,
            device_map="auto"
        )

        print("Preparando prompts en inglés...")
        statements = []
        for _, row in df.iterrows():
            name = row['Name']
            messages = [{"role": "user", "content": f"Generate a political statement by {name}, a US politician."}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_text += f"In 2025, {name} said that"

            statements.append(tokenizer(prompt_text, return_tensors="pt")['input_ids'])
        
        labels = np.array(df['Ideology'].astype(float))
        features = extract_activations_us_70b(model, statements, target=args.target)
        
        with open(output_path, 'wb') as f:
            pickle.dump((features, labels), f)
        
        print(f"Activaciones 70B de EEUU guardadas exitosamente en {output_path}")
        
        # Liberar memoria
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
