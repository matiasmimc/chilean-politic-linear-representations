import torch
import numpy as np
import pandas as pd
import pickle
import os
import re
import argparse
from tqdm import tqdm
from baukit import TraceDict
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

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
    prefix = ""
    for name, _ in model.named_modules():
        match = re.search(r'^(.*)\.0\.self_attn\.o_proj$', name)
        if match:
            prefix = match.group(1)
            break
            
    if not prefix:
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
        
    return heads


def extract_attention_head_activations(model, statements):
    HEADS = get_attention_heads(model)
    head_wise_hidden_states_list = []
    
    for prompt in tqdm(statements, desc="Extracting activations", total=len(statements)):
        # prompt: [1, seq_len]
        with torch.no_grad():
            with TraceDict(model, HEADS) as ret:
                _ = model(prompt.to(model.device))
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


def build_prompts(df, tokenizer, names_only=False):
    statements = []
    for _, row in df.iterrows():
        name = row['Name']
        party = row['party_1_P6']
        
        if names_only:
            prompt_intervenciones = ""
        else:
            intervenciones = row.get('intervenciones')
            if isinstance(intervenciones, (list, np.ndarray)) and len(intervenciones) > 0:
                if isinstance(intervenciones, list):
                    texto_intervenciones = "\n".join(intervenciones)
                else:
                    texto_intervenciones = "\n".join(intervenciones.astype(str))
            elif pd.notna(intervenciones) and not isinstance(intervenciones, (list, np.ndarray)) and str(intervenciones).strip():
                texto_intervenciones = str(intervenciones)
            else:
                texto_intervenciones = ""
                
            if texto_intervenciones:
                texto_intervenciones = texto_intervenciones[:3000]
                prompt_intervenciones = f"\nA continuación, algunas intervenciones parlamentarias suyas:\n{texto_intervenciones}\n"
            else:
                prompt_intervenciones = "\nNo hay intervenciones registradas.\n"
            
        messages = [{"role": "user", "content": f"Genera una declaración de {name}, parlamentario/a en Chile del partido {party}.{prompt_intervenciones}"}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_text += f"En 2025, {name} dijo que"

        statements.append(tokenizer(prompt_text, return_tensors="pt")['input_ids'])
    return statements


def main():
    parser = argparse.ArgumentParser(description="Extracción de activaciones de atención para parlamentarios.")
    parser.add_argument("--model", type=str, default=None, help="Nombre de modelo de HuggingFace o lista separada por comas.")
    parser.add_argument("--data-path", type=str, default=str(DATA_DIR / "congreso_2022_2026_20intervenciones.parquet"), help="Ruta al archivo parquet con parlamentarios e intervenciones.")
    parser.add_argument("--results-dir", type=str, default=None, help="Directorio donde guardar los resultados extraídos.")
    parser.add_argument("--names-only", action="store_true", help="Extraer solo usando nombres (sin declaraciones/intervenciones).")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Tipo de dato para cargar el modelo.")
    parser.add_argument("--force", action="store_true", help="Sobrescribir si el archivo ya existe.")
    args = parser.parse_args()

    # Definir directorio de salida
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.names_only:
        results_dir = BASE_DIR / "results/replication/v2_names_only"
    else:
        results_dir = DEFAULT_RESULTS_DIR

    results_dir.mkdir(parents=True, exist_ok=True)

    df_path = Path(args.data_path)
    if not df_path.exists():
        print(f"Error: No se encuentra el dataset en {df_path}")
        return

    df = pd.read_parquet(df_path)
    print(f"Cargado dataset de {len(df)} parlamentarios desde {df_path}")
    
    if args.model:
        models = [m.strip() for m in args.model.split(",")]
    else:
        models = DEFAULT_MODELS

    torch_dtype = getattr(torch, args.dtype)

    for model_name in models:
        print(f"\n=======================================================")
        print(f"Procesando modelo: {model_name} (names_only={args.names_only}, dtype={args.dtype})")
        print(f"=======================================================")
        safe_name = model_name.replace("/", "_")
        
        output_path = results_dir / f"{safe_name}_politician_features.pkl"
        if output_path.exists() and not args.force:
            print(f"Activaciones ya extraídas en {output_path}. Saltando (usa --force para sobrescribir)...")
            continue
            
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

        # Preparar prompts
        print("Preparando prompts...")
        statements = build_prompts(df, tokenizer, names_only=args.names_only)
        
        # Guardar prompts
        with open(results_dir / f"{safe_name}_politicians.pkl", 'wb') as f:
            pickle.dump(statements, f)

        # Extraer activaciones
        labels = np.array(df['dim_ideology_P6'].astype(float))
        features = extract_attention_head_activations(model, statements)
        
        with open(output_path, 'wb') as f:
            pickle.dump((features, labels), f)
        
        print(f"✅ Activaciones guardadas con éxito en: {output_path}")
        
        # Liberar memoria VRAM
        del model
        del tokenizer
        torch.cuda.empty_cache()

    print("\n🎉 Extracción finalizada para todos los modelos seleccionados.")


if __name__ == "__main__":
    main()
