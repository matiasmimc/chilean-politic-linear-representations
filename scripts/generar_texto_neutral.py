import pandas as pd
import numpy as np
import argparse
import random
import os
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "intervenciones_neutrales.parquet"

# Configuración del modelo
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

def get_hf_token():
    if HUGGINGFACE_TOKEN: return HUGGINGFACE_TOKEN
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('HUGGINGFACE_TOKEN='):
                    return line.strip().split('=', 1)[1]
    return None

def generar_textos_sinteticos(model, tokenizer, num_samples=80):
    print(f"Generando {num_samples} intervenciones sintéticas apolíticas con Llama 3...")
    textos = []
    
    # Temas hiper aburridos / apolíticos
    temas = [
        "la estandarización del tamaño de las roscas en los tornillos industriales",
        "el ciclo reproductivo del salmón en aguas frías",
        "la definición del huso horario para la observación astronómica",
        "los beneficios nutricionales de la palta en la dieta humana",
        "el proceso químico de la fotosíntesis en plantas de interior",
        "la historia evolutiva de las hormigas",
        "la clasificación de las nubes según su altitud",
        "el diseño aerodinámico de las alas de los aviones comerciales",
        "el uso del sistema métrico decimal en la ingeniería civil",
        "la geología de los volcanes inactivos",
        "las propiedades acústicas de los teatros antiguos",
        "la domesticación de la papa en los Andes",
        "el funcionamiento de los relojes de péndulo",
        "la arquitectura de los puentes colgantes",
        "el estudio de los dialectos en las zonas rurales"
    ]
    
    for i in tqdm(range(num_samples)):
        tema = random.choice(temas)
        prompt = f"Escribe una breve intervención formal y solemne (de un párrafo) de un parlamentario sobre {tema}. El discurso debe ser completamente apolítico, sin mencionar ideologías, partidos, ni la palabra Chile. Solo una exposición técnica y formal del tema."
        
        messages = [
            {"role": "system", "content": "Eres un asistente que redacta textos técnicos y neutrales."},
            {"role": "user", "content": prompt}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        # Limpiar si el modelo agrega introducciones
        if '"' in generated_text:
            generated_text = generated_text.replace('"', '')
            
        textos.append(generated_text)
        
    return textos

def obtener_textos_ciencias_sociales(num_samples=20):
    print(f"Descargando {num_samples} extractos reales de ciencias sociales desde Wikipedia (es)...")
    try:
        # Descargar un chunk pequeño de Wikipedia en español
        ds = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
        textos = []
        
        # Palabras clave para filtrar artículos de ciencias sociales (Antropología, Lingüística, Sociología, Historia)
        keywords = ["sociología", "antropología", "lingüística", "historia antigua", "arqueología", "psicología cognitiva", "fonética", "etnografía"]
        
        for idx, item in enumerate(ds):
            text = item['text']
            title = item['title'].lower()
            
            # Chequear si es de ciencias sociales
            if any(kw in title for kw in keywords) or any(kw in text[:500].lower() for kw in keywords):
                # Extraer un párrafo del medio para que no sea solo el intro (aprox 400 caracteres)
                paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 200]
                if paragraphs:
                    textos.append(paragraphs[min(1, len(paragraphs)-1)][:800] + "...")
                    
            if len(textos) >= num_samples:
                break
                
        return textos
    except Exception as e:
        print(f"Error descargando dataset: {e}. Generando textos de contingencia...")
        return ["La sociología es el estudio sistemático de la sociedad humana y el comportamiento social."] * num_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=int, default=80)
    parser.add_argument("--real", type=int, default=20)
    args = parser.parse_args()
    
    token = get_hf_token()
    
    textos_sinteticos = []
    if OUTPUT_FILE.exists():
        try:
            df_existente = pd.read_parquet(OUTPUT_FILE)
            if "source" in df_existente.columns:
                textos_sinteticos_existentes = df_existente[df_existente["source"] == "synthetic"]["intervenciones"].tolist()
                if len(textos_sinteticos_existentes) >= args.synthetic:
                    print(f"Recuperando {args.synthetic} textos sintéticos existentes desde {OUTPUT_FILE} (saltando generación).")
                    textos_sinteticos = textos_sinteticos_existentes[:args.synthetic]
        except Exception as e:
            print(f"Error al leer archivo existente: {e}")
            
    if len(textos_sinteticos) < args.synthetic:
        MODEL_CACHE = BASE_DIR / "model"
        
        print(f"Cargando {MODEL_NAME} desde caché {MODEL_CACHE}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token, cache_dir=str(MODEL_CACHE))
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            token=token,
            cache_dir=str(MODEL_CACHE)
        )
        
        textos_sinteticos = generar_textos_sinteticos(model, tokenizer, args.synthetic)
        
        # Liberar VRAM si es posible
        del model
        torch.cuda.empty_cache()
    
    textos_reales = obtener_textos_ciencias_sociales(args.real)
    
    todos_los_textos = textos_sinteticos + textos_reales
    
    # Crear un DataFrame con el mismo formato base esperado
    # Le pondremos un ID falso de diputado, luego al extraer se cruzará con aleatorios.
    df = pd.DataFrame({
        "id": range(len(todos_los_textos)),
        "intervenciones": todos_los_textos,
        "source": ["synthetic"] * len(textos_sinteticos) + ["wikipedia_social_science"] * len(textos_reales)
    })
    
    df.to_parquet(OUTPUT_FILE)
    print(f"✅ Guardado exitosamente en {OUTPUT_FILE} ({len(df)} textos neutrales)")

if __name__ == "__main__":
    main()
