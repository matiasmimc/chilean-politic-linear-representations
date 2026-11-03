import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
import os

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "congreso_2022_2026_biografias.csv"
OUTPUT_FILE = DATA_DIR / "congreso_2022_2026_con_resumen.csv"

# Configuración del Modelo
# MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"

def load_model(model_id):
    print(f"Cargando modelo {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto",
        low_cpu_mem_usage=True,
        token=HUGGINGFACE_TOKEN
    )
    return model, tokenizer

def generar_resumen(model, tokenizer, nombre, partido, texto_biografia):
    prompt = f"""USER: A continuación te presento una biografía del diputado/a chileno/a {nombre} del partido {partido}. 
Resúmela en una lista corta de puntos con características factuales (profesión, partido, gestiones principales). 
Es crucial que no contenga apreciaciones subjetivas, adjetivos calificativos ni juicios de valor.

Biografía:
{texto_biografia[:3000]}

ASSISTANT: Aquí tienes el resumen factual:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=250, 
            temperature=0.1, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return response.strip()

def main():
    if not INPUT_FILE.exists():
        print(f"Error: No se encuentra el archivo {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Procesando {len(df)} parlamentarios...")

    model, tokenizer = load_model(MODEL_ID)

    resumenes = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        nombre = row['Name']
        partido = row['party_1_P6']
        biografia_html = row['html_biografia']
        
        # Limpieza de HTML
        biografia_texto = BeautifulSoup(str(biografia_html), 'html.parser').get_text(separator=' ', strip=True)

        try:
            resumen = generar_resumen(model, tokenizer, nombre, partido, biografia_texto)
        except Exception as e:
            print(f"\nError procesando a {nombre}: {e}")
            resumen = None
            
        resumenes.append(resumen)

    df["resumen"] = resumenes
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Proceso completado. Resultados guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
