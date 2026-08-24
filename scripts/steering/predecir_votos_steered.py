import torch
import numpy as np
import pandas as pd
import pickle
import os
import sys
import argparse
import os
import sys
import argparse
import re
import warnings
import sqlite3
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from baukit import TraceDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results/replication/v2"
OUTPUT_DIR = BASE_DIR / "results/voting_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE = BASE_DIR / "model"

# --- Integración con Repositorio Externo ---
CHATBOT_DIR = Path("/home/matias/Documentos/GitHub/diputados_chatbot")
DB_PATH = CHATBOT_DIR / 'data' / 'database' / 'parlamento.db'

sys.path.append(str(CHATBOT_DIR))
try:
    from src.core.context_builder import ParlamentarioContextBuilder
except ImportError:
    print("Advertencia: No se pudo importar ParlamentarioContextBuilder.")
    sys.exit(1)

# Variables Globales para Steering
current_alpha = 0
current_method = "ridge"
current_focal_ridge_dict = {}
current_head_dict = {}
current_features = None  # [num_mps, layers, heads, d_head]
current_num_heads = None
precomputed_interventions = {}

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
            
    return heads, prefix

def lt_modulated_vector_add(head_output, layer_name):
    global precomputed_interventions, current_num_heads, current_method
    
    match = re.search(r'\.(\d+)\.', layer_name) or re.search(r'^(\d+)\.', layer_name)
    layer_index = int(match.group(1))
    
    if current_method == "ridge":
        head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=current_num_heads)
        for head_index in current_head_dict.get(layer_index, []):
            intervention_vec = precomputed_interventions.get((layer_index, head_index))
            if intervention_vec is not None:
                if intervention_vec.device != head_output.device:
                    intervention_vec = intervention_vec.to(head_output.device)
                    precomputed_interventions[(layer_index, head_index)] = intervention_vec
                head_output[:, -1, head_index, :] += intervention_vec
        head_output = rearrange(head_output, 'b s h d -> b s (h d)')
    else:
        intervention_vec = precomputed_interventions.get(layer_index)
        if intervention_vec is not None:
            if intervention_vec.device != head_output.device:
                intervention_vec = intervention_vec.to(head_output.device)
                precomputed_interventions[layer_index] = intervention_vec
            head_output[:, -1, :] += intervention_vec
            
    return head_output

def build_prompt(nombre, partido, bill_title, bill_summary):
    prompt = f"""Eres el diputado chileno {nombre} del partido {partido}.
Tu tarea es decidir cómo votarás frente al siguiente proyecto de ley basándote ESTRICTAMENTE en tu ideología y afiliación política histórica.

PROYECTO DE LEY:
Título: {bill_title}
Resumen: {bill_summary}

Decide si votarás A FAVOR o EN CONTRA del proyecto. Responde únicamente con tu voto explícito y una justificación de una oración.
Ejemplo de formato:
VOTO: A FAVOR
JUSTIFICACIÓN: Porque apoya a los trabajadores de nuestra nación.

VOTO:"""
    return prompt

def get_real_bills_for_mp(mp_uid, limit=3):
    """Extrae proyectos de ley reales donde el diputado votó A Favor o En Contra."""
    query = """
        SELECT DISTINCT b.titulo, b.resumen, v.voto 
        FROM bills b 
        JOIN sesiones_votacion sv ON b.bill_id = sv.bill_id 
        JOIN votos_parlamentario v ON sv.sesion_votacion_id = v.sesion_votacion_id 
        WHERE v.mp_uid = ? 
          AND v.voto IN ('A Favor', 'En Contra') 
          AND b.resumen IS NOT NULL 
          AND length(b.resumen) > 50
        ORDER BY RANDOM()
        LIMIT ?
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(query, (int(mp_uid), limit))
        rows = cur.fetchall()
        
    bills = []
    for r in rows:
        bills.append({"title": r[0], "summary": r[1], "real_vote": r[2].upper()})
    return bills

def main():
    global current_method, current_alpha, current_features, precomputed_interventions, current_num_heads
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--steering-method", type=str, choices=["ridge", "pca", "sae", "nmf"], default="nmf")
    parser.add_argument("--k", type=int, default=16, help="Top K componentes a usar")
    parser.add_argument("--limit-mps", type=int, default=10, help="Límite de diputados a evaluar (para pruebas rápidas). 0 = todos.")
    args = parser.parse_args()
    
    current_method = args.steering_method
    model_name = args.model
    safe_name = model_name.replace("/", "_")
    
    print(f"\n{'='*60}")
    print(f"Predicción Steered con Activaciones Base | Modelo: {model_name} | Método: {current_method}")
    print(f"{'='*60}")
    
    # 1. Cargar DataFrame de Diputados
    df_path = DATA_DIR / "congreso_2022_2026_20intervenciones.parquet"
    if not df_path.exists():
        print(f"Error: No se encuentra {df_path}")
        return
    df = pd.read_parquet(df_path)
    
    # 2. Cargar tensores de activaciones base y modelos descubiertos
    feat_path = RESULTS_DIR / f"{safe_name}_politician_features.pkl"
    with open(feat_path, 'rb') as f:
        current_features, _ = pickle.load(f) # current_features tiene len igual a df
        
    if current_method == "ridge":
        dict_path = RESULTS_DIR / f"{safe_name}_ridge.pkl"
        perf_path = RESULTS_DIR / f"{safe_name}_politician_performance.pkl"
        trained_model_dict = pickle.load(open(dict_path, 'rb'))
        performance = pickle.load(open(perf_path, 'rb'))
    else:
        # Los metodos no-supervisados ahora se guardan en su propia carpeta
        dict_path = BASE_DIR / f"results/{current_method}_analysis/{safe_name}_{current_method}_analysis.pkl"
        trained_model_dict = pickle.load(open(dict_path, 'rb'))
        
    # 3. Cargar LLM
    print("\nCargando Modelo LLM y Tokenizador...")
    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE), token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=str(MODEL_CACHE), 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16, 
        token=HUGGINGFACE_TOKEN,
        device_map="auto"
    )
    input_device = getattr(model, "device", None) or next(model.parameters()).device
    
    current_num_heads = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", None))
    heads_list, _ = get_attention_heads(model)
    
    # 4. Preparar Top-K componentes
    layers_to_trace = []
    global current_head_dict, current_focal_ridge_dict
    
    if current_method == "ridge":
        top_indices = np.dstack(np.unravel_index(np.argsort(performance.ravel()), performance.shape))[0][-args.k:, :][::-1]
        for i in top_indices:
            l_idx, h_idx = i
            current_focal_ridge_dict[(l_idx, h_idx)] = trained_model_dict[l_idx][h_idx].coef_
            current_head_dict.setdefault(l_idx, []).append(h_idx)
        layers_to_trace = [heads_list[i] for i in sorted(current_head_dict.keys())]
    else:
        top_components = trained_model_dict["all_components"][:args.k]
        for comp in top_components:
            l_idx = comp['layer']
            if current_method == "pca":
                vec = comp['vector']
            else:
                vec = comp['vector']
            current_head_dict.setdefault(l_idx, []).append(vec)
        layers_to_trace = [heads_list[i] for i in sorted(current_head_dict.keys())]
        
    # Extraer parlamentarios target del DF
    if args.limit_mps > 0:
        # Tomar algunos del PC y otros de Republicanos para contraste rápido
        df_target = pd.concat([
            df[df['party_1_P6'] == 'Partido Comunista'].head(args.limit_mps // 2),
            df[df['party_1_P6'] == 'Partido Republicano'].head(args.limit_mps // 2)
        ])
    else:
        df_target = df
        
    print(f"\nPolíticos a evaluar: {len(df_target)}")
    
    results = []
    # Alphas: 0 = Baseline, 1.0 = Amplificar Ideología
    alphas_to_test = [0, 0.5, 1.0, 1.5, 2, 2.5, 3] 
    
    # Iterar sobre cada diputado en el DataFrame filtrado
    for row_idx, row in tqdm(df_target.iterrows(), total=len(df_target), desc="Diputados"):
        nombre = row['Name']
        partido = row['party_1_P6']
        mp_uid = row.get('mp_uid', row_idx) # Por si mp_uid no existe en el parquet
        
        # Cargar contexto histórico desde BD del Chatbot
        with ParlamentarioContextBuilder() as builder:
            context = builder.build_complete_context(mp_uid)
        
        # Extraer leyes reales votadas por este diputado
        real_bills = get_real_bills_for_mp(mp_uid, limit=3)
        if not real_bills:
            continue
        
        # Obtener activaciones base de este diputado específico [layers, heads, d_head]
        mp_base_act = current_features[row_idx]
        
        for bill in real_bills:
            prompt_text = build_prompt(nombre, partido, bill["title"], bill["summary"])
            messages = [{"role": "user", "content": prompt_text}]
            prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(prompt_formatted, return_tensors="pt")['input_ids'].to(model.device)
            
            for alpha in alphas_to_test:
                current_alpha = alpha
                precomputed_interventions.clear()
                
                # Precomputar el vector específico para ESTE diputado
                if alpha != 0:
                    if current_method == "ridge":
                        for l_idx, h_idx in current_focal_ridge_dict.keys():
                            ridge_coef = current_focal_ridge_dict[(l_idx, h_idx)]
                            
                            # Proyección de LAS ACTIVACIONES DE ESTE DIPUTADO sobre la dirección descubierta
                            # mp_base_act: [layers, heads, d_head]
                            act_mp_head = mp_base_act[l_idx, h_idx, :]
                            
                            # Normalizar la dirección
                            norm_coef = np.linalg.norm(ridge_coef)
                            direction = ridge_coef / (norm_coef if norm_coef > 0 else 1)
                            
                            # Proyección escalar del diputado
                            proj_mp = np.dot(act_mp_head, direction)
                            
                            # Vector de intervención = alpha * (proyección del diputado) * dirección
                            vec = current_alpha * proj_mp * direction
                            
                            t_vec = torch.from_numpy(vec).to(model.dtype).to(input_device)
                            precomputed_interventions[(l_idx, h_idx)] = t_vec
                    else:
                        for l_idx, vecs in current_head_dict.items():
                            layer_total_vec = np.zeros_like(vecs[0])
                            
                            # mp_base_act[l_idx, :, :] es [heads, d_head], aplanado a [hidden_dim]
                            act_mp_layer = mp_base_act[l_idx, :, :].reshape(-1)
                            
                            for vec in vecs:
                                norm_coef = np.linalg.norm(vec)
                                direction = vec / (norm_coef if norm_coef > 0 else 1)
                                
                                # Proyección del diputado en esta componente
                                proj_mp = np.dot(act_mp_layer, direction)
                                
                                # Acumular la intervención ponderada por su propia proyección
                                layer_total_vec += (current_alpha * proj_mp * direction)
                                
                            t_vec = torch.from_numpy(layer_total_vec).to(model.dtype).to(input_device)
                            precomputed_interventions[l_idx] = t_vec
                        
                with torch.no_grad():
                    if alpha == 0:
                        out_tokens = model.generate(input_ids, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    else:
                        with TraceDict(model, layers_to_trace, edit_output=lt_modulated_vector_add):
                            out_tokens = model.generate(input_ids, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                
                new_tokens = out_tokens[0][len(input_ids[0]):]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                
                # Evaluar coincidencia
                predicted_vote = "A FAVOR" if "A FAVOR" in response.upper() else ("EN CONTRA" if "EN CONTRA" in response.upper() else "INDECISO")
                is_match = (predicted_vote == bill["real_vote"])
                
                results.append({
                    "mp_uid": mp_uid,
                    "nombre": nombre,
                    "partido": partido,
                    "ley": bill["title"],
                    "voto_real": bill["real_vote"],
                    "alpha": alpha,
                    "voto_predicho": predicted_vote,
                    "respuesta_completa": response,
                    "acierto": is_match
                })
                
    df_results = pd.DataFrame(results)
    
    # Calcular y mostrar reporte de clasificación por alpha
    best_alpha = 0
    max_acc = -1
    
    if not df_results.empty:
        print("\n--- Resultados de Exactitud (Classification Report) ---")
        for alpha in alphas_to_test:
            subset = df_results[df_results['alpha'] == alpha]
            if not subset.empty:
                acc = subset['acierto'].mean() * 100
                print(f"\n{'='*40}")
                print(f"Alpha: {alpha:4.1f} | Exactitud: {acc:.1f}%")
                print(f"{'='*40}")
                y_true = subset['voto_real']
                y_pred = subset['voto_predicho']
                print(classification_report(y_true, y_pred, zero_division=0))
                
                if acc > max_acc:
                    max_acc = acc
                    best_alpha = alpha
                    
        print(f"\n🏆 Mejor Alpha Automático: {best_alpha} (Exactitud: {max_acc:.1f}%)")
        
        # Guardar registro automático del mejor alpha
        import json
        optimal_json_path = OUTPUT_DIR / "optimal_alphas.json"
        optimal_dict = {}
        if optimal_json_path.exists():
            with open(optimal_json_path, 'r') as f:
                try:
                    optimal_dict = json.load(f)
                except:
                    pass
                    
        if safe_name not in optimal_dict:
            optimal_dict[safe_name] = {}
            
        optimal_dict[safe_name][current_method] = {"best_alpha": float(best_alpha), "accuracy": float(max_acc)}
        
        with open(optimal_json_path, 'w') as f:
            json.dump(optimal_dict, f, indent=4)
    
    out_file = OUTPUT_DIR / f"voting_predictions_{safe_name}_{current_method}.parquet"
    df_results.to_parquet(out_file)
    print(f"\n✅ Resultados guardados en {out_file}")

if __name__ == "__main__":
    main()
