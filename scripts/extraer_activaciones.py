import torch
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from baukit import TraceDict
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results/replication/v2"
MODEL_CACHE = BASE_DIR / "model"

RESULTS_DIR.mkdir(exist_ok=True)

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

def extract_attention_head_activations(model, statements):
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    head_wise_hidden_states_list = []
    
    for prompt in tqdm(statements, desc="Extracting activations", total=len(statements)):
        # prompt is likely a tensor [1, seq_len]
        with torch.no_grad():
            with TraceDict(model, HEADS) as ret:
                _ = model(prompt.to(model.device), output_hidden_states=True, output_attentions=True)
                # output from o_proj is [batch, seq, hidden_dim]
                head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
                head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).numpy() # [layers, seq, hidden]
                head_wise_hidden_states_list.append(head_wise_hidden_states)
    
    features = []
    num_heads = model.config.num_attention_heads
    for head_wise_hidden_states in head_wise_hidden_states_list:
        # We take the state of the last token in the sequence
        last_token_states = head_wise_hidden_states[:, -1, :] # [layers, hidden]
        # Reshape to [layers, heads, d_head]
        reshaped = rearrange(last_token_states, 'l (h d) -> l h d', h=num_heads)
        features.append(reshaped)
        
    features = np.stack(features, axis=0) # [num_statements, layers, heads, d_head]
    return features

def main():
    df_path = DATA_DIR / "congreso_2022_2026_20intervenciones.parquet"
    if not df_path.exists():
        print(f"Error: No se encuentra {df_path}")
        return

    df = pd.read_parquet(df_path)
    
    # Modelos actualizados que caben en los 24GB de VRAM de una RTX 3090 en fp16.
    models = [
        # 'meta-llama/Meta-Llama-3-8B-Instruct',
        # 'failspy/Meta-Llama-3-8B-Instruct-abliterated-v3',
        # 'mistralai/Mistral-7B-Instruct-v0.3', 
        # 'Qwen/Qwen2.5-7B-Instruct',
        'google/gemma-4-E4B-it',
        'Qwen/Qwen3.5-9B',
        'huihui-ai/Huihui-Qwen3.5-9B-abliterated',
        'meta-llama/Llama-3.1-70B-Instruct',
        'latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0',
    ]

    for model_name in models:
        print(f"\n--- Procesando modelo: {model_name} ---")
        safe_name = model_name.replace("/", "_")
        
        output_path = RESULTS_DIR / f"{safe_name}_politician_features.pkl"
        if output_path.exists():
            print(f"Activaciones ya extraídas para {model_name}. Saltando...")
            continue
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE), token=HUGGINGFACE_TOKEN, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=str(MODEL_CACHE), 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16, 
            token=HUGGINGFACE_TOKEN,
            device_map="auto"
        )

        # Preparar prompts
        print("Preparando prompts...")
        statements = []
        for _, row in df.iterrows():
            intervenciones = row.get('intervenciones')
            # Check if it's a valid list or numpy array with elements
            if isinstance(intervenciones, (list, np.ndarray)) and len(intervenciones) > 0:
                if isinstance(intervenciones, list):
                    texto_intervenciones = "\n".join(intervenciones)
                else: # it's a numpy array
                    texto_intervenciones = "\n".join(intervenciones.astype(str))
            elif pd.notna(intervenciones) and not isinstance(intervenciones, (list, np.ndarray)) and str(intervenciones).strip():
                texto_intervenciones = str(intervenciones)
            else:
                texto_intervenciones = ""
                
            if texto_intervenciones:
                
                # Truncar para evitar OOM con prompts demasiado largos
                texto_intervenciones = texto_intervenciones[:3000]
                
                prompt_intervenciones = f"A continuación, algunas intervenciones parlamentarias suyas:\n{texto_intervenciones}\n"
            else:
                prompt_intervenciones = "No hay intervenciones registradas.\n"
                
            messages = [{"role": "user", "content": f"Genera una declaración de {row['Name']}, parlamentario/a en Chile del partido {row['party_1_P6']}.\n{prompt_intervenciones}"}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_text += f"En 2025, {row['Name']} dijo que"

            statements.append(tokenizer(prompt_text, return_tensors="pt")['input_ids'])
        
        # Guardar prompts (opcional, como en el notebook)
        with open(RESULTS_DIR / f"{safe_name}_politicians.pkl", 'wb') as f:
            pickle.dump(statements, f)

        # Extraer activaciones
        labels = np.array(df['dim_ideology_P6'].astype(float))
        features = extract_attention_head_activations(model, statements)
        
        output_path = RESULTS_DIR / f"{safe_name}_politician_features.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump((features, labels), f)
        
        print(f"Activaciones guardadas en {output_path}")
        
        # Liberar memoria
        del model
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
