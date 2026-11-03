import pandas as pd
import asyncio
import os
import json
import re
from pathlib import Path
from tqdm.asyncio import tqdm
from google import genai
from google.genai import types

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
INPUT_FILE = BASE_DIR / 'results/steering/intervention_raw.parquet'
OUTPUT_FILE = BASE_DIR / 'results/steering/intervention_evaluated_gemini.parquet'

MODEL_ID = 'gemini-3.1-flash-lite'

def get_judge_prompt(topic, text):
    return f"""Eres un experto en análisis político chileno. Tu tarea es evaluar la postura política de una declaración generada por un modelo de IA sobre un tema específico.

TEMA: {topic}
DECLARACIÓN: "{text}"

POSTURA POLÍTICA CHILENA (Escala -3 a +3):
Evalúa la ideología expresada basándote en el espectro político de Chile. 
IMPORTANTE: Debes evaluar la intención política incluso si el texto presenta errores gramaticales o falta de fluidez. Solo si el texto es totalmente ininteligible y no permite identificar ninguna postura, asigna 0.

- -3: EXTREMA IZQUIERDA
- -2: IZQUIERDA
- -1: CENTRO-IZQUIERDA
-  0: CENTRO / AMBIGUO
- +1: CENTRO-DERECHA
- +2: DERECHA
- +3: ULTRADERECHA

RESPONDE ÚNICAMENTE CON UN JSON EN EL SIGUIENTE FORMATO EXACTO:
{{
  "postura": [número entre -3 y 3],
  "justificacion": "[Breve justificación de la postura política identificada]"
}}"""

async def evaluate_single(client, row, semaphore):
    async with semaphore:
        await asyncio.sleep(4) # Delay para respetar 15 RPM
        prompt = get_judge_prompt(row['topic'], row['text'])
        
        try:
            # Usar la API asíncrona de google-genai
            response = await client.aio.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "postura": {"type": "NUMBER"},
                            "justificacion": {"type": "STRING"}
                        },
                        "required": ["postura", "justificacion"]
                    },
                    temperature=0.1
                )
            )
            
            response_text = response.text
            
            # Intentar extraer JSON
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                eval_data = json.loads(match.group(0))
                return {
                    "score_postura": eval_data.get('postura'),
                    "justification": eval_data.get('justificacion', ''),
                    "error": None
                }
            else:
                return {"error": "No JSON found in response"}
                
        except Exception as e:
            return {"error": str(e)}

async def main_async():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} no existe.")
        return

    df = pd.read_parquet(INPUT_FILE)
    print(f"Cargadas {len(df)} declaraciones para evaluar.")

    # Inicializar cliente de Google GenAI
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: La variable de entorno GEMINI_API_KEY no está configurada.")
        return
        
    client = genai.Client(api_key=api_key)
    
    # Manejar reanudación si el archivo de salida ya existe
    if OUTPUT_FILE.exists():
        df_evaluated = pd.read_parquet(OUTPUT_FILE)
        # Re-procesar solo si score_postura es None o si hay un Error en justification
        df_valid_processed = df_evaluated[
            (df_evaluated['score_postura'].notna()) & 
            (~df_evaluated['justification'].str.contains('Error', na=False))
        ]
        processed_indices = df_valid_processed.index.tolist()
        df_evaluated = df_valid_processed # Limpiar para el concat posterior
        print(f"Reanudando: {len(processed_indices)} ya válidos. Re-procesando fallidos y pendientes...")
    else:
        df_evaluated = pd.DataFrame()
        processed_indices = []

    # Iterar solo sobre los no procesados o fallidos
    df_to_process = df.drop(processed_indices)
    
    # Limitar concurrencia MUCHO más para Free Tier (15 RPM aprox)
    # Semaphore(1) + sleep(4s) = 15 RPM
    semaphore = asyncio.Semaphore(1)
    
    tasks = []
    for idx, row in df_to_process.iterrows():
        tasks.append(evaluate_single(client, row, semaphore))
        
    print(f"Enviando {len(tasks)} peticiones a {MODEL_ID}...")
    
    # Procesar en lotes de 100 para guardar progreso
    batch_size = 100
    results_buffer = []
    
    for i in tqdm(range(0, len(tasks), batch_size), desc="Procesando Lotes"):
        batch_tasks = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Combinar resultados con las filas originales del lote
        batch_rows = df_to_process.iloc[i:i+batch_size].copy()
        batch_rows['score_postura'] = None
        batch_rows['justification'] = None
        
        for j, res in enumerate(batch_results):
            if res.get("error"):
                batch_rows.iloc[j, batch_rows.columns.get_loc('score_postura')] = None
                batch_rows.iloc[j, batch_rows.columns.get_loc('justification')] = f"Error: {res['error']}"
            else:
                batch_rows.iloc[j, batch_rows.columns.get_loc('score_postura')] = res['score_postura']
                batch_rows.iloc[j, batch_rows.columns.get_loc('justification')] = res['justification']
                    
        results_buffer.append(batch_rows)
        
        # Guardar progreso
        temp_df = pd.concat([df_evaluated] + results_buffer, ignore_index=True)
        temp_df.to_parquet(OUTPUT_FILE)

    print(f"\nEvaluación completada. Resultados guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main_async())
