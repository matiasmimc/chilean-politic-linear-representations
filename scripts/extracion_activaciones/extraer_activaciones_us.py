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
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results/us_analysis"
MODEL_CACHE = BASE_DIR / "model"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")

def get_attention_heads(model):
    prefix = ""
    for name, _ in model.named_modules():
        import re
        match = re.search(r'^(.*)\.0\.self_attn\.o_proj$', name)
        if match:
            prefix = match.group(1)
            break
            
    if not prefix:
        for name, _ in model.named_modules():
            import re
            match = re.search(r'^(.*)\.0\.attn\.c_proj$', name)
            if match:
                prefix = match.group(1)
                break
                
    if not prefix:
        for name, _ in model.named_modules():
            import re
            match = re.search(r'^(.*)\.0\.linear_attn\.out_proj$', name)
            if match:
                prefix = match.group(1)
                break
                
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
            
    return heads

def extract_attention_head_activations(model, statements):
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
    args = parser.parse_args()

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
    
    df.to_parquet(RESULTS_DIR / "us_politicians.parquet")

    model_name = args.model
    print(f"\n--- Procesando modelo: {model_name} (US Politicians) ---")
    safe_name = model_name.replace("/", "_")
    
    output_path = RESULTS_DIR / f"{safe_name}_politician_features.pkl"
        
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
    features = extract_attention_head_activations(model, statements)
    
    with open(output_path, 'wb') as f:
        pickle.dump((features, labels), f)
    
    print(f"Activaciones de EEUU guardadas exitosamente en {output_path}")

if __name__ == "__main__":
    main()
