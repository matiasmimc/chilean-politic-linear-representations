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
RESULTS_DIR = BASE_DIR / "results/us_analysis"
MODEL_CACHE = BASE_DIR / "model"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
if not HUGGINGFACE_TOKEN:
    print("Warning: HUGGINGFACE_TOKEN no encontrado en el archivo .env")

sota_models = ['google/gemma-4-E4B-it', 'meta-llama/Llama-3.1-8B-Instruct', 'zai-org/GLM-4-9B-0414']

def get_attention_heads(model):
    import re
    heads = []
    # Buscar usando regex sobre todos los modulos nombrados, para evitar
    # que wrappers de accelerate (offload a CPU/disco) oculten los atributos.
    for name, _ in model.named_modules():
        if "vision_tower" in name:
            continue
        if re.search(r'\.\d+\.self_attn\.o_proj$', name):
            heads.append(name)
        elif re.search(r'\.\d+\.attn\.c_proj$', name):
            heads.append(name)
        elif re.search(r'\.\d+\.linear_attn\.out_proj$', name):
            heads.append(name)
            
    # Ordenar por el número de capa para asegurar el orden correcto
    def extract_layer_num(name):
        match = re.search(r'\.(\d+)\.', name)
        return int(match.group(1)) if match else 0
        
    heads = sorted(heads, key=extract_layer_num)
    
    if not heads:
        raise ValueError(f"No se encontraron proyecciones de atención en {type(model).__name__}")
        
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
    
    for prompt in tqdm(statements, desc="Extrayendo activaciones", total=len(statements)):
        with torch.no_grad():
            with TraceDict(model, HEADS) as ret:
                _ = model(prompt.to(device))
                head_wise_hidden_states = [ret[head].output.squeeze().detach().float().cpu() for head in HEADS]
                head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).numpy()
                head_wise_hidden_states_list.append(head_wise_hidden_states)
    
    features = []
    num_heads = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", None))
    if num_heads is None and hasattr(model.config, "text_config"):
        num_heads = getattr(model.config.text_config, "num_attention_heads", getattr(model.config.text_config, "n_head", None))
    
    if num_heads is None:
        raise ValueError("No se pudo determinar 'num_heads' desde la configuración del modelo.")
        
    for head_wise_hidden_states in head_wise_hidden_states_list:
        last_token_states = head_wise_hidden_states[:, -1, :] 
        reshaped = rearrange(last_token_states, 'l (h d) -> l h d', h=num_heads)
        features.append(reshaped)
        
    features = np.stack(features, axis=0)
    return features

def clean_bioname(bioname):
    """Limpia el formato 'APELLIDO, Nombre (Apodo)' a 'Nombre Apellido'."""
    if pd.isna(bioname):
        return ""
    # Quitar lo que esté entre paréntesis
    import re
    bioname = re.sub(r'\(.*?\)', '', bioname)
    
    parts = bioname.split(',')
    if len(parts) == 2:
        last = parts[0].strip().title()
        first = parts[1].strip().title()
        return f"{first} {last}"
    return bioname.strip().title()

def main():
    parser = argparse.ArgumentParser(description="Extracción de activaciones para Políticos de EEUU desde Nominate.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--target", type=str, default="attention", choices=["attention", "residual"], help="Objetivo de extracción (cabezas de atención o residual stream).")
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
    
    # Filtrar Democratas (100) y Republicanos (200)
    df_dems = df_raw[df_raw['party_code'] == 100].copy()
    df_reps = df_raw[df_raw['party_code'] == 200].copy()
    
    # Muestra estratificada para igualar los 156 de Chile (78 y 78)
    np.random.seed(42)
    sample_dems = df_dems.sample(78)
    sample_reps = df_reps.sample(78)
    
    df = pd.concat([sample_dems, sample_reps]).reset_index(drop=True)
    
    # Limpiar nombres y mapear ideología
    df['Name'] = df['bioname'].apply(clean_bioname)
    df['Ideology'] = df['nominate_dim1'].astype(float)
    df['Party'] = df['party_code'].map({100: 'Democrat', 200: 'Republican'})
    
    df.to_parquet(results_dir / "us_politicians.parquet")

    for model_name in sota_models:
        print(f"\n--- Procesando modelo: {model_name} (US Politicians) ---")
        safe_name = model_name.replace("/", "_")
        
        output_path = results_dir / f"{safe_name}_politician_features.pkl"
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE), token=HUGGINGFACE_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=str(MODEL_CACHE), 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16, 
            token=HUGGINGFACE_TOKEN,
            device_map="auto"
        )
    
        print("Preparando prompts en inglés...")
        statements = []
        for _, row in df.iterrows():
            name = row['Name']
            # El paper original demuestra que los nombres bastan si el prompt es en cero-shot
            messages = [{"role": "user", "content": f"Generate a political statement by {name}, a US politician."}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_text += f"In 2025, {name} said that"
    
            statements.append(tokenizer(prompt_text, return_tensors="pt")['input_ids'])
        
        labels = np.array(df['Ideology'].astype(float))
        features = extract_activations(model, statements, target=args.target)
        
        with open(output_path, 'wb') as f:
            pickle.dump((features, labels), f)
        
        print(f"Activaciones de EEUU guardadas exitosamente en {output_path}")
        
        # Limpiar memoria GPU
        del model
        del tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
