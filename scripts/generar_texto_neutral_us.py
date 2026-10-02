import pandas as pd
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
OUTPUT_FILE = DATA_DIR / "us_intervenciones_neutrales.parquet"

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

def generar_textos_sinteticos_us(model, tokenizer, num_samples=80):
    print(f"Generando {num_samples} intervenciones sintéticas apolíticas (EE.UU.) con Llama 3...")
    textos = []
    
    # Translated boring topics to English
    temas = [
        "the standardization of thread sizes in industrial screws",
        "the reproductive cycle of salmon in cold waters",
        "the definition of time zones for astronomical observation",
        "the nutritional benefits of avocado in the human diet",
        "the chemical process of photosynthesis in indoor plants",
        "the evolutionary history of ants",
        "the classification of clouds according to their altitude",
        "the aerodynamic design of commercial airplane wings",
        "the use of the metric system in civil engineering",
        "the geology of inactive volcanoes",
        "the acoustic properties of ancient theaters",
        "the domestication of the potato in the Andes",
        "the mechanics of pendulum clocks",
        "the architecture of suspension bridges",
        "the study of dialects in rural areas"
    ]
    
    for i in tqdm(range(num_samples)):
        tema = random.choice(temas)
        prompt = f"Write a brief, formal, and solemn speech (one paragraph) by a representative about {tema}. The speech must be completely apolitical, without mentioning ideologies, parties, or the word 'America' or 'United States'. It should just be a technical and formal exposition of the topic."
        
        messages = [
            {"role": "system", "content": "You are an assistant who writes technical and neutral texts."},
            {"role": "user", "content": prompt}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        if '"' in generated_text:
            generated_text = generated_text.replace('"', '')
            
        textos.append(generated_text)
        
    return textos

def obtener_textos_ciencias_sociales_us(num_samples=20):
    print(f"Descargando {num_samples} extractos reales de ciencias sociales desde Wikipedia (en)...")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        textos = []
        
        keywords = ["sociology", "anthropology", "linguistics", "ancient history", "archaeology", "cognitive psychology", "phonetics", "ethnography"]
        
        for idx, item in enumerate(ds):
            text = item['text']
            title = item['title'].lower()
            
            if any(kw in title for kw in keywords) or any(kw in text[:500].lower() for kw in keywords):
                paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 200]
                if paragraphs:
                    textos.append(paragraphs[min(1, len(paragraphs)-1)][:800] + "...")
                    
            if len(textos) >= num_samples:
                break
                
        return textos
    except Exception as e:
        print(f"Error descargando dataset: {e}. Generando textos de contingencia...")
        return ["Sociology is the systematic study of human society and social behavior."] * num_samples

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
        
        textos_sinteticos = generar_textos_sinteticos_us(model, tokenizer, args.synthetic)
        
        del model
        torch.cuda.empty_cache()
    
    textos_reales = obtener_textos_ciencias_sociales_us(args.real)
    
    todos_los_textos = textos_sinteticos + textos_reales
    
    df = pd.DataFrame({
        "id": range(len(todos_los_textos)),
        "intervenciones": todos_los_textos,
        "source": ["synthetic"] * len(textos_sinteticos) + ["wikipedia_social_science"] * len(textos_reales)
    })
    
    df.to_parquet(OUTPUT_FILE)
    print(f"✅ Guardado exitosamente en {OUTPUT_FILE} ({len(df)} textos neutrales US)")

if __name__ == "__main__":
    main()
