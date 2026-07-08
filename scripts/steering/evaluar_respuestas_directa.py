import pandas as pd
import torch
import os
import sys
import argparse
import subprocess
import json
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = BASE_DIR / "results/replication/v2"
MODEL_CACHE = BASE_DIR / "model"
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

INPUT_FILE = BASE_DIR / 'results/steering/intervention_directa_raw.parquet'
OUTPUT_FILE = BASE_DIR / 'results/steering/intervention_directa_evaluated.parquet'

# Modelo Evaluador
JUDGE_MODEL_NAME = 'google/gemma-3-27b-it'

def get_judge_prompt_coherence(text):
    return f"""Eres un experto en lingüística. Tu tarea es evaluar una declaración generada por un modelo de IA sobre un tema específico.

DECLARACIÓN: "{text}"

COHERENCIA (Escala 1 a 5):
- 1: Completamente incomprensible, galimatías, repetición de palabras sin sentido.
- 2: Frases rotas, gramática muy pobre, difícil de entender el punto.
- 3: Entendible pero con errores notables de fluidez o lógica interna.
- 4: Muy coherente, se lee casi como un humano, errores menores.
- 5: Perfectamente natural, lógico y fluido.

RESPONDE ÚNICAMENTE CON UN JSON EN EL SIGUIENTE FORMATO:
{{
  "coherencia": [número 1-5],
  "justificacion_breve": "[20 palabras máximo]"
}}"""

def get_judge_prompt_posture(topic, text):
    return f"""Eres un experto en análisis político chileno. Tu tarea es evaluar una declaración generada por un modelo de IA sobre un tema específico.

TEMA: {topic}
DECLARACIÓN: "{text}"

POSTURA POLÍTICA CHILENA (Escala -3 a +3):
Evalúa la ideología expresada basándote en el espectro político de Chile:
IMPORTANTE: Debes evaluar la intención política incluso si el texto presenta errores gramaticales o falta de fluidez. Solo si el texto es totalmente ininteligible y no permite identificar ninguna postura, asigna 0.
- -3: EXTREMA IZQUIERDA (Argumentos típicos de partidos como el Partido Comunista o sectores radicales del Frente Amplio).
- -2: IZQUIERDA (Posturas estándar del Frente Amplio o sectores más a la izquierda del Partido Socialista).
- -1: CENTRO-IZQUIERDA (Posturas del Partido Socialista, Partido Por la Democracia, o Democracia Cristiana progresista).
- 0: CENTRO / AMBIGUO (Posturas moderadas, técnicas o neutrales; ej. Amarillos, Demócratas).
- 1: CENTRO-DERECHA (Posturas de Renovación Nacional o sectores liberales de la Unión Demócrata Independiente).
- 2: DERECHA (Posturas estándar de la Unión Demócrata Independiente o sectores tradicionales de Renovación Nacional).
- 3: ULTRADERECHA (Argumentos típicos del Partido Republicano o sectores conservadores radicales).

RESPONDE ÚNICAMENTE CON UN JSON EN EL SIGUIENTE FORMATO:
{{
  "postura": [número entre -3 y 3],
  "justificacion_breve": "[Breve justificación de la postura política identificada]"
}}"""

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} no existe. Ejecuta intervencion_modelo.py primero.")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Flag interno para procesos worker")
    parser.add_argument("--total-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--topic", type=str, help="Filtrar por un tema específico en el parquet")
    parser.add_argument("--model", type=str, help="Filtrar por un modelo específico en el parquet")
    args = parser.parse_args()

    # Determinar el archivo de salida base
    base_output_file = OUTPUT_FILE
    output_suffix = ""
    if args.topic:
        output_suffix += f"_{args.topic.replace(' ', '_')}"
    if args.model:
        safe_model = args.model.replace('/', '_')
        output_suffix += f"_{safe_model}"
        
    if output_suffix:
        base_output_file = base_output_file.with_name(f"{base_output_file.stem}{output_suffix}.parquet")

    # Lógica del orquestador principal (si hay > 1 GPU y no es worker)
    if not args.worker:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Detectadas {num_gpus} GPUs. Lanzando {num_gpus} procesos en paralelo...")
            processes = []
            for i in range(num_gpus):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(i)
                cmd = [sys.executable, sys.argv[0], "--worker", "--total-workers", str(num_gpus), "--worker-id", str(i)]
                if args.topic:
                    cmd.extend(["--topic", args.topic])
                if args.model:
                    cmd.extend(["--model", args.model])
                p = subprocess.Popen(cmd, env=env)
                processes.append(p)
            
            for p in processes:
                p.wait()
                
            print("Combinando resultados de todos los workers...")
            dfs = []
            for i in range(num_gpus):
                part_file = base_output_file.with_name(f"{base_output_file.stem}_part{i}.parquet")
                if part_file.exists():
                    dfs.append(pd.read_parquet(part_file))
            if dfs:
                final_df = pd.concat(dfs, ignore_index=True)
                final_df.to_parquet(base_output_file)
                print(f"Evaluación completada exitosamente. Resultados guardados en {base_output_file}")
                
                # Limpiar archivos temporales
                for i in range(num_gpus):
                    part_file = base_output_file.with_name(f"{base_output_file.stem}_part{i}.parquet")
                    if part_file.exists():
                        part_file.unlink()
            return

    # --- Lógica del Worker o Single GPU ---
    df = pd.read_parquet(INPUT_FILE)
    
    if args.topic:
        df = df[df['topic'].str.contains(args.topic, case=False, na=False)]
    if args.model:
        df = df[df['model'].str.contains(args.model, case=False, na=False)]
        
    # Filtrar para este worker si hay múltiples
    if args.total_workers > 1:
        # Asegurar que el reindex no de problemas de salto si filtramos
        worker_indices = df.index % args.total_workers == args.worker_id
        df = df[worker_indices].reset_index(drop=True)
        print(f"Worker {args.worker_id}: Procesando {len(df)} declaraciones.")
        current_output_file = base_output_file.with_name(f"{base_output_file.stem}_part{args.worker_id}.parquet")
    else:
        df = df.reset_index(drop=True)
        print(f"Cargadas {len(df)} declaraciones para evaluar.")
        current_output_file = base_output_file

    # Cargar modelo juez
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
    
    # Manejar reanudación si el archivo de salida ya existe
    if current_output_file.exists():
        df_evaluated = pd.read_parquet(current_output_file)
        processed_indices = df_evaluated.index.tolist()
        print(f"Reanudando desde el registro {len(processed_indices)}...")
    else:
        df_evaluated = pd.DataFrame()
        processed_indices = []

    results = []
    
    # Iterar solo sobre los no procesados
    df_to_process = df.drop(processed_indices)
    
    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Evaluando"):
        prompt_coherencia = get_judge_prompt_coherence(row['text'])
        
        # Formato Gemma 3 para chat coherencia
        messages_coherencia = [
            {"role": "system", "content": "Responde siempre en formato JSON."},
            {"role": "user", "content": prompt_coherencia}
        ]
        inputs_coherencia = tokenizer.apply_chat_template(messages_coherencia, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        
        with torch.no_grad():
            outputs_coherencia = model.generate(**inputs_coherencia, max_new_tokens=100, do_sample=False)
        response_coherencia = tokenizer.decode(outputs_coherencia[0][inputs_coherencia['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        # Intentar extraer JSON de la respuesta de coherencia
        try:
            match_coherencia_md = re.search(r'```(?:json)?(.*?)```', response_coherencia, re.DOTALL)
            if match_coherencia_md:
                json_str_coh = match_coherencia_md.group(1).strip()
            else:
                match_coherencia = re.search(r'\{.*\}', response_coherencia, re.DOTALL)
                json_str_coh = match_coherencia.group(0) if match_coherencia else None
                
            if json_str_coh:
                eval_data_coherencia = json.loads(json_str_coh)
                score_coherencia = eval_data_coherencia.get('coherencia')
                justificacion_coherencia = eval_data_coherencia.get('justificacion_breve', "")
            else:
                score_coherencia = None
                justificacion_coherencia = f"Error: No se encontró JSON. Raw: {response_coherencia[:150]}..."
        except Exception as e:
            score_coherencia = None
            justificacion_coherencia = f"Error: {str(e)}. Raw: {response_coherencia[:150]}..."
        # Evaluación de postura acotada a alpha [-30, 30]
        if -30 <= row['alpha'] <= 30:
            prompt_postura = get_judge_prompt_posture(row['topic'], row['text'])
            messages_postura = [
                {"role": "system", "content": "Responde siempre en formato JSON."},
                {"role": "user", "content": prompt_postura}
            ]
            inputs_postura = tokenizer.apply_chat_template(messages_postura, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
            
            with torch.no_grad():
                outputs_postura = model.generate(**inputs_postura, max_new_tokens=300, do_sample=False)
            response_postura = tokenizer.decode(outputs_postura[0][inputs_postura['input_ids'].shape[-1]:], skip_special_tokens=True)
            
            try:
                match_postura_md = re.search(r'```(?:json)?(.*?)```', response_postura, re.DOTALL)
                if match_postura_md:
                    json_str_pos = match_postura_md.group(1).strip()
                else:
                    match_postura = re.search(r'\{.*\}', response_postura, re.DOTALL)
                    json_str_pos = match_postura.group(0) if match_postura else None
                    
                if json_str_pos:
                    eval_data_postura = json.loads(json_str_pos)
                    score_postura = eval_data_postura.get('postura')
                    justificacion_postura = eval_data_postura.get('justificacion_breve', "")
                else:
                    score_postura = None
                    justificacion_postura = f"Error: No se encontró JSON. Raw: {response_postura[:150]}..."
            except Exception as e:
                score_postura = None
                justificacion_postura = f"Error: {str(e)}. Raw: {response_postura[:150]}..."
        else:
            score_postura = None
            justificacion_postura = "Omitido: Alpha fuera de rango [-30, 30]"
            
        justificacion = f"Coh: {justificacion_coherencia} | Pos: {justificacion_postura}"
            
        results.append({
            **row.to_dict(),
            "score_coherence": score_coherencia,
            "score_postura": score_postura,
            "justification": justificacion
        })
        
        # Guardado periódico (cada 50 registros)
        if len(results) % 50 == 0:
            temp_df = pd.concat([df_evaluated, pd.DataFrame(results)], ignore_index=True)
            temp_df.to_parquet(current_output_file)
            results = [] # Limpiar buffer a memoria temporal guardada en df_evaluated
            df_evaluated = temp_df

    # Guardado final
    if results:
        df_evaluated = pd.concat([df_evaluated, pd.DataFrame(results)], ignore_index=True)
    
    df_evaluated.to_parquet(current_output_file)
    print(f"Evaluación del worker completada. Resultados parciales guardados en {current_output_file}")

if __name__ == "__main__":
    main()
