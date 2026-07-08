import pandas as pd
import torch
import os
import json
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
INPUT_FILE = BASE_DIR / 'results/steering/intervention_raw.parquet'
OUTPUT_FILE = BASE_DIR / 'data/escalas_politicas_por_tema.json'
MODEL_CACHE = BASE_DIR / "model"

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('HUGGINGFACE_TOKEN='):
                    HUGGINGFACE_TOKEN = line.strip().split('=', 1)[1]

JUDGE_MODEL_NAME = 'google/gemma-3-27b-it'

def get_scale_generation_prompt(topic, samples):
    samples_text = "\n".join([f"- \"{s}\"" for s in samples])
    
    return f"""Eres un experto analista político. Tu tarea es definir las distintas posturas del espectro político para un tema específico, basándote puramente en ideas y propuestas, SIN mencionar nombres de partidos políticos.

TEMA: {topic}

A continuación, se presentan algunas declaraciones sobre este tema para darte contexto sobre el tipo de debate:
{samples_text}

Tu objetivo es caracterizar cómo piensa cada sector político sobre este tema específico.
Genera una caracterización detallada (1 a 2 oraciones) para cada uno de los siguientes 5 niveles del espectro:

- "-2": "[Caracterización de la postura de izquierda para este tema]",
- "-1": "[Caracterización de la postura de centro-izquierda para este tema]",
- "0": "[Caracterización de la postura de centro para este tema]",
- "1": "[Caracterización de la postura de centro-derecha para este tema]",
- "2": "[Caracterización de la postura de derecha para este tema]"

Ejemplos de caracterizaciones generales, que pueden servir como guía para las caracterizaciones de cada tema, son: Rol fuerte del Estado, redistribución, derechos sociales, regulación estricta, Progresismo moderado, equilibrio estado/mercado con enfoque social, Posturas neutrales, técnicas, pragmáticas o que buscan consenso, Liberalismo económico moderado, focalización del gasto, eficiencia del Estado, Libre mercado, rol subsidiario del Estado, orden, tradición, responsabilidad individual, etc.

RESPONDE ÚNICAMENTE CON UN JSON EN EL SIGUIENTE FORMATO EXACTO:
{{
  "-2": "[Caracterización de la postura de izquierda para este tema]",
  "-1": "[Caracterización de la postura de centro-izquierda para este tema]",
  "0": "[Caracterización de la postura de centro para este tema]",
  "1": "[Caracterización de la postura de centro-derecha para este tema]",
  "2": "[Caracterización de la postura de derecha para este tema]"
}}"""

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} no existe. Por favor ajusta la ruta si es necesario.")
        return

    print("Cargando datos...")
    df = pd.read_parquet(INPUT_FILE)
    
    # Obtener temas únicos
    topics = df['topic'].unique()
    print(f"Se encontraron {len(topics)} temas.")

    print(f"Cargando modelo juez: {JUDGE_MODEL_NAME} (en 4-bits)")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME, cache_dir=str(MODEL_CACHE), token=HUGGINGFACE_TOKEN)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        cache_dir=str(MODEL_CACHE),
        quantization_config=quantization_config,
        token=HUGGINGFACE_TOKEN,
        device_map="auto"
    )

    escalas_por_tema = {}
    
    # Si ya existe el archivo, cargarlo para no repetir
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            escalas_por_tema = json.load(f)
            print(f"Se cargaron {len(escalas_por_tema)} escalas previamente generadas.")

    for topic in tqdm(topics, desc="Generando escalas"):
        if topic in escalas_por_tema:
            continue
            
        df_topic = df[df['topic'] == topic]
        
        # Muestreo estratificado rudimentario si existe la columna alpha (para obtener declaraciones variadas)
        if 'alpha' in df_topic.columns:
            samples = []
            try:
                # Tomar algo del extremo negativo, centro y positivo
                s1 = df_topic[df_topic['alpha'] <= -10]['text'].sample(min(2, len(df_topic[df_topic['alpha'] <= -10]))).tolist()
                s2 = df_topic[(df_topic['alpha'] > -10) & (df_topic['alpha'] < 10)]['text'].sample(min(2, len(df_topic[(df_topic['alpha'] > -10) & (df_topic['alpha'] < 10)]))).tolist()
                s3 = df_topic[df_topic['alpha'] >= 10]['text'].sample(min(2, len(df_topic[df_topic['alpha'] >= 10]))).tolist()
                samples = s1 + s2 + s3
            except ValueError:
                samples = df_topic['text'].sample(min(5, len(df_topic))).tolist()
        else:
            samples = df_topic['text'].sample(min(5, len(df_topic))).tolist()
            
        # Limpiar muestras vacías
        samples = [str(s) for s in samples if s]
            
        prompt = get_scale_generation_prompt(topic, samples)
        
        messages = [
            {"role": "system", "content": "Responde siempre en formato JSON."},
            {"role": "user", "content": prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=600, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        # Parsear JSON
        try:
            match_md = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
            if match_md:
                json_str = match_md.group(1).strip()
            else:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                json_str = match.group(0) if match else None
                
            if json_str:
                escala = json.loads(json_str)
                escalas_por_tema[topic] = escala
                
                # Guardar en cada iteración para no perder progreso si hay un error de OOM
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(escalas_por_tema, f, ensure_ascii=False, indent=4)
            else:
                print(f"Error parseando JSON para el tema '{topic}'. Raw output: {response[:150]}...")
        except Exception as e:
            print(f"Excepción para el tema '{topic}': {e}")
            
    print(f"\nCompletado. Escalas guardadas exitosamente en: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
