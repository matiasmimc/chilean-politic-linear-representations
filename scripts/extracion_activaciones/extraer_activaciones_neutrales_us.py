import pickle
import os
import gc
import argparse
import random
from tqdm import tqdm
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import pandas as pd
import torch

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
        
    return heads, ""

def build_neutral_prompts(df_neutral, df_mps, tokenizer):
    statements = []
    labels = []
    
    mps_list = df_mps.to_dict('records')
    
    for idx, row in df_neutral.iterrows():
        # Elegir un político de EEUU aleatorio
        random_mp = random.choice(mps_list)
        name = random_mp['Name']
        ideology = random_mp['Ideology']
        
        texto = row['intervenciones']
        
        # Formato de prompt para contexto neutral (inglés)
        messages = [{"role": "user", "content": f"The following is a non-political statement by {name}, a US politician:\n\n\"{texto}\""}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        statements.append(tokenizer(prompt_text, return_tensors="pt")['input_ids'])
        labels.append(float(ideology))
        
    import numpy as np
    return statements, np.array(labels)

def extract_activations(model, inputs, target="attention"):
    if target == "residual":
        features = []
        for inp in tqdm(inputs, desc="Extrayendo residual stream"):
            inp = inp.to(model.device)
            with torch.no_grad():
                outputs = model(inp, output_hidden_states=True)
                hidden_states = torch.stack(outputs.hidden_states[1:]).squeeze() # [layers, seq, d_model]
                last_token_states = hidden_states[-1, :].cpu().float().numpy() if hidden_states.ndim == 2 else hidden_states[:, -1, :].cpu().float().numpy()
                features.append(last_token_states)
        # Transponer para emparejar la dimensionalidad [num_layers, num_statements, d_model]
        features = np.stack(features) # [num_statements, layers, d_model]
        # Wait, the current logic returns a list of arrays (num_layers), each shape [num_statements, ...]
        # Let's match the old return shape convention for this script:
        num_layers = features.shape[1]
        final_features = []
        for l in range(num_layers):
            final_features.append(features[:, l, :])
        return final_features

    heads_list, _ = get_attention_heads(model)
    
    # Asegurar que el pad token esté definido de forma segura (y manejar listas en eos_token_id)
    if getattr(model.config, "pad_token_id", None) is None:
        eos = getattr(model.config, "eos_token_id", None)
        if isinstance(eos, list) and len(eos) > 0:
            eos = eos[0]
        model.config.pad_token_id = eos

    # Encontrar número de cabezales para el rearreglo (soporta LLMs y multimodales)
    num_heads = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", None))
    if num_heads is None and hasattr(model.config, "text_config"):
        num_heads = getattr(model.config.text_config, "num_attention_heads", getattr(model.config.text_config, "n_head", None))
    
    if num_heads is None:
        raise ValueError("No se pudo determinar 'num_heads' desde la configuración del modelo.")
        
    # Hook hook hook
    from baukit import TraceDict
    activations = []
    
    # Procesar batch por batch (de tamaño 1 para no tener problemas de padding)
    for inp in tqdm(inputs, desc="Extrayendo activaciones"):
        inp = inp.to(model.device)
        with torch.no_grad():
            with TraceDict(model, heads_list) as ret:
                model(inp)
                
        # Extraer las representaciones de cada cabeza en la última posición (last token)
        head_wise_hidden_states_list = [ret[head].output[0].detach() for head in heads_list]
        features = []
        for head_wise_hidden_states in head_wise_hidden_states_list:
            last_token_states = head_wise_hidden_states[-1:, :] 
            reshaped = rearrange(last_token_states, 'b (h d) -> b h d', h=num_heads)
            features.append(reshaped.cpu())
            
        activations.append(features)
        
    # Acomodar lista de listas a (num_capas, num_muestras, num_cabezales, dim_cabezal)
    num_layers = len(heads_list)
    final_features = []
    for l in range(num_layers):
        layer_features = torch.cat([act[l] for act in activations], dim=0)
        final_features.append(layer_features.numpy())
        
    return final_features

def main():
    parser = argparse.ArgumentParser(description="Extracción de activaciones sobre texto neutral para US.")
    parser.add_argument("--neutral-path", type=str, default=str(DATA_DIR / "us_intervenciones_neutrales.parquet"))
    parser.add_argument("--mp-path", type=str, default=str(RESULTS_DIR / "us_politicians.parquet"))
    parser.add_argument("--target", type=str, default="attention", choices=["attention", "residual"], help="Objetivo de extracción (cabezas de atención o residual stream).")
    args = parser.parse_args()

    results_dir = RESULTS_DIR
    if args.target == "residual":
        results_dir = results_dir / "residual"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_neutral_path = Path(args.neutral_path)
    if not df_neutral_path.exists():
        print(f"Error: No se encuentra el dataset neutral en {df_neutral_path}")
        print("Corre primero el script: python scripts/generar_texto_neutral_us.py")
        return

    df_neutral = pd.read_parquet(df_neutral_path)
    df_mps = pd.read_parquet(args.mp_path)
    print(f"Cargado dataset neutral de EEUU de {len(df_neutral)} textos.")
    
    for model_name in sota_models:
        print(f"\n=======================================================")
        print(f"Extrayendo fondo neutral para EE.UU.: {model_name}")
        print(f"=======================================================")
        safe_name = model_name.replace("/", "_")
        
        output_path = results_dir / f"{safe_name}_neutral_features.pkl"
        
        print(f"Cargando modelo y tokenizer {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE), token=HUGGINGFACE_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=str(MODEL_CACHE), 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16, 
            token=HUGGINGFACE_TOKEN,
            device_map="auto"
        )
        
        print("Preparando prompts neutrales en inglés...")
        statements, labels = build_neutral_prompts(df_neutral, df_mps, tokenizer)
        
        print("Extrayendo características...")
        features = extract_activations(model, statements, target=args.target)
        
        with open(output_path, 'wb') as f:
            import pickle
            pickle.dump((features, labels), f)
            
        print(f"✅ Activaciones neutrales de EEUU guardadas en {output_path}")
        
        # Limpiar GPU
        del model
        del tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
