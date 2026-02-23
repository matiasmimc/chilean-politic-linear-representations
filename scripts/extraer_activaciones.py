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
RESULTS_DIR = BASE_DIR / "results_replication"
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
    df_path = DATA_DIR / "congreso_2022_2026_con_resumen.csv"
    if not df_path.exists():
        print(f"Error: No se encuentra {df_path}. Ejecuta obtener_resumenes.py primero.")
        return

    df = pd.read_csv(df_path)
    models = ['meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', 'lmsys/vicuna-7b-v1.5']

    for model_name in models:
        print(f"\n--- Procesando modelo: {model_name} ---")
        safe_name = model_name.replace("/", "_")
        
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
            if pd.notna(intervenciones) and intervenciones:
                if isinstance(intervenciones, list):
                    texto_intervenciones = "\n".join(intervenciones)
                else:
                    texto_intervenciones = str(intervenciones)
                
                # Truncar para evitar OOM con prompts demasiado largos
                texto_intervenciones = texto_intervenciones[:3000]
                
                prompt_intervenciones = f"Además, considera las siguientes intervenciones parlamentarias:\n{texto_intervenciones}\n"
            else:
                prompt_intervenciones = ""
            prompt_text = f"""USER: Genera una declaración de {row["Name"]}, parlamentario/a en Chile del partido {row['party_1_P6']}.
            A continuación un resumen de su biografía:
            {row['resumen']}

            {prompt_intervenciones}

            ASSISTANT: En 2019, {row["Name"]} dijo que"""
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
