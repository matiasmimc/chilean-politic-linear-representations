import torch
import numpy as np
import pandas as pd
import pickle
import os
import re
import argparse
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from einops import rearrange

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_RESULTS_DIR = BASE_DIR / "results/replication/v2"
MODEL_CACHE = BASE_DIR / "model"

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('HUGGINGFACE_TOKEN='):
                    HUGGINGFACE_TOKEN = line.strip().split('=', 1)[1]

DEFAULT_MODELS = [
    'google/gemma-4-E4B-it',
    'meta-llama/Llama-3.1-8B-Instruct',
    # 'Qwen/Qwen3.5-9B',
    'zai-org/GLM-4-9B-0414',
    # 'mistralai/Ministral-3-8B-Instruct-2512',
]

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
        # Elegir un diputado real completamente al azar para neutralizar el sesgo del nombre
        random_mp = random.choice(mps_list)
        name = random_mp['Name']
        party = random_mp['party_1_P6']
        
        # Ojo: la etiqueta ideológica NO DEBE ser la del diputado, porque el texto es neutral.
        # Pero guardaremos la etiqueta del diputado por si cPCA necesita alinear el DF (aunque no la usaremos para cPCA).
        # Lo mejor es guardar label = 0.0 o la del MP al azar
        label = random_mp.get('dim_ideology_P6', 0.0)
        
        texto_neutral = row.get('intervenciones', "")
        prompt_intervenciones = f"\nA continuación, algunas intervenciones parlamentarias suyas:\n{texto_neutral}\n"
            
        messages = [{"role": "user", "content": f"Genera una declaración de {name}, parlamentario/a en Chile del partido {party}.{prompt_intervenciones}"}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_text += f"En 2025, {name} dijo que"

        statements.append(tokenizer(prompt_text, return_tensors="pt")['input_ids'])
        labels.append(label)
        
    return statements, np.array(labels, dtype=float)

def main():
    parser = argparse.ArgumentParser(description="Extracción de activaciones sobre texto neutral.")
    parser.add_argument("--model", type=str, default=None, help="Nombre de modelo.")
    parser.add_argument("--neutral-path", type=str, default=str(DATA_DIR / "intervenciones_neutrales.parquet"))
    parser.add_argument("--mp-path", type=str, default=str(DATA_DIR / "congreso_2022_2026_20intervenciones.parquet"))
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--target", type=str, default="attention", choices=["attention", "residual"])
    parser.add_argument("--force", action="store_true", help="Sobrescribir si el archivo ya existe.")
    args = parser.parse_args()

    results_dir = DEFAULT_RESULTS_DIR
    if args.target == "residual":
        results_dir = results_dir / "residual"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_neutral_path = Path(args.neutral_path)
    if not df_neutral_path.exists():
        print(f"Error: No se encuentra el dataset neutral en {df_neutral_path}")
        return

    df_neutral = pd.read_parquet(df_neutral_path)
    df_mps = pd.read_parquet(args.mp_path)
    print(f"Cargado dataset neutral de {len(df_neutral)} textos.")
    
    models = [m.strip() for m in args.model.split(",")] if args.model else DEFAULT_MODELS
    torch_dtype = getattr(torch, args.dtype)

    for model_name in models:
        print(f"\n=======================================================")
        print(f"Extrayendo fondo (Background) para: {model_name}")
        print(f"=======================================================")
        safe_name = model_name.replace("/", "_")
        
        output_path = results_dir / f"{safe_name}_neutral_features.pkl"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE), token=HUGGINGFACE_TOKEN, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=str(MODEL_CACHE), 
                low_cpu_mem_usage=True, 
                torch_dtype=torch_dtype, 
                token=HUGGINGFACE_TOKEN,
                device_map="auto"
            )
        except Exception as e:
            print(f"⚠️ Error cargando modelo {model_name}: {e}")
            continue

        print("Preparando prompts neutrales...")
        statements, labels = build_neutral_prompts(df_neutral, df_mps, tokenizer)
        
        all_features = []
        model.eval()

        if args.target == "residual":
            with torch.no_grad():
                for input_ids in tqdm(statements, desc="Extrayendo residual stream (neutral)"):
                    input_ids = input_ids.to(model.device)
                    outputs = model(input_ids, output_hidden_states=True)
                    hidden_states = torch.stack(outputs.hidden_states[1:]).squeeze() # [layers, seq, d_model]
                    last_token_states = hidden_states[-1, :].cpu().float().numpy() if hidden_states.ndim == 2 else hidden_states[:, -1, :].cpu().float().numpy()
                    all_features.append(last_token_states)
            all_features = np.stack(all_features) # [num_statements, layers, d_model]
        else:
            heads_list, _ = get_attention_heads(model)
            num_layers = len(heads_list)
            
            num_heads = getattr(model.config, "num_attention_heads", None)
            if num_heads is None and hasattr(model.config, "text_config"):
                num_heads = getattr(model.config.text_config, "num_attention_heads", None)
            if num_heads is None:
                num_heads = getattr(model.config, "n_head", None)
            if num_heads is None:
                raise ValueError("No se pudo determinar el número de cabezas (num_attention_heads) del modelo.")

            with torch.no_grad():
                for input_ids in tqdm(statements, desc="Extrayendo activaciones neutrales"):
                    input_ids = input_ids.to(model.device)
                    
                    layer_features = []
                    def hook_fn(output, layer_name):
                        if isinstance(output, tuple):
                            act = output[0]
                        else:
                            act = output
                        # Obtener la activación del ÚLTIMO token
                        last_token_act = act[0, -1, :].detach().float().cpu().numpy()
                        last_token_act = rearrange(last_token_act, '(h d) -> h d', h=num_heads)
                        layer_features.append(last_token_act)
                        return output

                    hooks = {layer: hook_fn for layer in heads_list}
                    
                    try:
                        from baukit import TraceDict
                        with TraceDict(model, layers=heads_list, edit_output=hook_fn):
                            model(input_ids)
                    except Exception as e:
                        print(f"Error en baukit, omitiendo: {e}")
                        raise e
                        
                    all_features.append(np.stack(layer_features))

            all_features = np.stack(all_features) # [num_statements, num_layers, num_heads, d_head]
        
        with open(output_path, 'wb') as f:
            pickle.dump((all_features, labels), f)
            
        print(f"✅ Guardado background en {output_path}")
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
