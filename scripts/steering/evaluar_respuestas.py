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
MODEL_CACHE = BASE_DIR / "model"
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('HUGGINGFACE_TOKEN='):
                    HUGGINGFACE_TOKEN = line.strip().split('=', 1)[1]

INPUT_FILE, OUTPUT_FILE = None, None
ESCALAS_FILE = BASE_DIR / 'data/escalas_politicas_por_tema.json'

try:
    with open(ESCALAS_FILE, 'r', encoding='utf-8') as f:
        ESCALAS_POLITICAS = json.load(f)
except FileNotFoundError:
    print(f"Advertencia: No se encontró {ESCALAS_FILE}. Ejecuta generar_escala_juez.py si evaluarás posturas.")
    ESCALAS_POLITICAS = {}

# Modelo Evaluador
JUDGE_MODEL_NAME = 'google/gemma-3-27b-it'

def get_judge_prompt_coherence(text):
    return f"""Eres un experto en lingüística. Tu tarea es evaluar una declaración generada por un modelo de IA sobre un tema específico.

DECLARACIÓN: "{text}"

COHERENCIA (Escala 1 a 5):
- 1: Completamente incomprensible, galimatías, repetición de palabras sin sentido.
- 2: Frases rotas, gramática muy pobre, difícil de entender el punto.
- 3: Entendible pero con errores notables de fluidez o lógica intxerna.
- 4: Muy coherente, se lee casi como un humano, errores menores.
- 5: Perfectamente natural, lógico y fluido.

RESPONDE ÚNICAMENTE CON UN JSON EN EL SIGUIENTE FORMATO:
{{
  "coherencia": [número 1-5],
  "justificacion_breve": "[20 palabras máximo]"
}}"""

def get_judge_prompt_posture(topic, text):
    escala = ESCALAS_POLITICAS.get(topic, {})
    
    escala_texto = f"""
- -2: IZQUIERDA ({escala.get('-2', 'Rol fuerte del Estado y redistribución')})
- -1: CENTRO-IZQUIERDA ({escala.get('-1', 'Progresismo moderado con enfoque social')})
- 0: CENTRO / AMBIGUO ({escala.get('0', 'Posturas neutrales, técnicas o de consenso')})
- 1: CENTRO-DERECHA ({escala.get('1', 'Liberalismo económico moderado')})
- 2: DERECHA ({escala.get('2', 'Libre mercado, orden y tradición')})
"""

    return f"""Eres un experto en análisis político chileno. Tu tarea es evaluar una declaración generada por un modelo de IA sobre un tema específico.

TEMA: {topic}
DECLARACIÓN: "{text}"

POSTURA POLÍTICA CHILENA (Escala -2 a +2):
Evalúa la ideología expresada basándote en la siguiente caracterización del debate sobre este tema:
IMPORTANTE: Debes evaluar la intención política incluso si el texto presenta errores gramaticales o falta de fluidez. Solo si el texto es totalmente ininteligible y no permite identificar ninguna postura, asigna 0.
{escala_texto.strip()}

RESPONDE ÚNICAMENTE CON UN JSON EN EL SIGUIENTE FORMATO:
{{
  "postura": [número entre -2 y 2],
  "justificacion_breve": "[Breve justificación de la postura política identificada]"
}}"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Flag interno para procesos worker")
    parser.add_argument("--total-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--topic", type=str, help="Filtrar por un tema específico en el parquet")
    parser.add_argument("--model", type=str, help="Filtrar por un modelo específico en el parquet")
    parser.add_argument("--batch-size", type=int, default=8, help="Textos evaluados en paralelo por el juez")
    parser.add_argument("--folder", type=str, default="steering", help="Carpeta de resultados a evaluar")
    args = parser.parse_args()

    global INPUT_FILE, OUTPUT_FILE
    INPUT_FILE = BASE_DIR / f'results/{args.folder}/intervention_raw.parquet'
    OUTPUT_FILE = BASE_DIR / f'results/{args.folder}/intervention_evaluated.parquet'

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} no existe. Revisa la ruta o ejecuta la intervención primero.")
        return

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

    if not args.worker and base_output_file.exists():
        df_input = pd.read_parquet(INPUT_FILE)
        if args.topic:
            df_input = df_input[df_input['topic'].str.contains(args.topic, case=False, na=False)]
        if args.model:
            df_input = df_input[df_input['model'].str.contains(args.model, case=False, na=False)]
            
        try:
            df_existing = pd.read_parquet(base_output_file)
            if len(df_existing) >= len(df_input):
                print(f"Las evaluaciones para {base_output_file.name} ya están completas ({len(df_existing)} registros). Cargando y saltando procesamiento...")
                return
            else:
                print(f"Archivo {base_output_file.name} incompleto ({len(df_existing)}/{len(df_input)}). Continuando procesamiento...")
        except Exception as e:
            pass

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
                if args.folder != "steering":
                    cmd.extend(["--folder", args.folder])
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
    
    # Extraer tokens de detención válidos (evita warning y detiene la generación a tiempo)
    eos_candidates = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    valid_eos = []
    for t in eos_candidates:
        if isinstance(t, int) and t > 0:
            valid_eos.append(t)
        elif isinstance(t, list):
            valid_eos.extend([int(x) for x in t if isinstance(x, int) and x > 0])
    valid_eos = list(set(valid_eos))
    
    # Determinar pad_token_id seguro
    safe_pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (valid_eos[0] if valid_eos else 0)
    
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
    BATCH = args.batch_size
    tokenizer.padding_side = "left"

    # Iterar solo sobre los no procesados
    df_to_process = df.drop(processed_indices)
    rows_list = list(df_to_process.iterrows())

    def judge_batch(prompts, max_new_tokens):
        """Genera respuestas del juez para varios prompts a la vez (misma configuracion greedy que antes)."""
        convs = [[{"role": "system", "content": "Responde siempre en formato JSON."},
                  {"role": "user", "content": p}] for p in prompts]
        enc = tokenizer.apply_chat_template(convs, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True, padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=valid_eos if valid_eos else None, pad_token_id=safe_pad_token_id)
        return [tokenizer.decode(o[enc['input_ids'].shape[-1]:], skip_special_tokens=True) for o in out]

    def parse_json(resp, key):
        try:
            m_md = re.search(r'```(?:json)?(.*?)```', resp, re.DOTALL)
            if m_md:
                js = m_md.group(1).strip()
            else:
                m = re.search(r'\{.*\}', resp, re.DOTALL)
                js = m.group(0) if m else None
            if js:
                d = json.loads(js)
                return d.get(key), d.get('justificacion_breve', "")
            return None, f"Error: No se encontró JSON. Raw: {resp[:150]}..."
        except Exception as e:
            return None, f"Error: {str(e)}. Raw: {resp[:150]}..."

    for start in tqdm(range(0, len(rows_list), BATCH), desc="Evaluando (batches)"):
        chunk = rows_list[start:start + BATCH]
        texts = []
        for _, row in chunk:
            t = str(row['text'])
            if len(t) > 2000:
                t = t[:2000] + "... [TRUNCADO POR EXCESO DE LONGITUD]"
            texts.append(t)

        resp_coh = judge_batch([get_judge_prompt_coherence(t) for t in texts], 100)

        in_range = [i for i, (_, row) in enumerate(chunk) if -30 <= row['alpha'] <= 30]
        resp_pos = {}
        if in_range:
            outs = judge_batch([get_judge_prompt_posture(chunk[i][1]['topic'], texts[i]) for i in in_range], 300)
            resp_pos = dict(zip(in_range, outs))

        for i, (idx, row) in enumerate(chunk):
            score_coh, just_coh = parse_json(resp_coh[i], 'coherencia')
            if i in resp_pos:
                score_pos, just_pos = parse_json(resp_pos[i], 'postura')
            else:
                score_pos, just_pos = None, "Omitido: Alpha fuera de rango [-30, 30]"
            results.append({**row.to_dict(), "score_coherence": score_coh, "score_postura": score_pos,
                            "justification": f"Coh: {just_coh} | Pos: {just_pos}"})

        temp_df = pd.concat([df_evaluated, pd.DataFrame(results)], ignore_index=True)
        temp_df.to_parquet(current_output_file)
        results = []
        df_evaluated = temp_df

    # Guardado final
    if results:
        df_evaluated = pd.concat([df_evaluated, pd.DataFrame(results)], ignore_index=True)
    
    df_evaluated.to_parquet(current_output_file)
    print(f"Evaluación del worker completada. Resultados parciales guardados en {current_output_file}")

if __name__ == "__main__":
    main()
