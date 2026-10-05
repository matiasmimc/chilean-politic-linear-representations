import torch
import numpy as np
import pandas as pd
import pickle
import os
import sys
import re
import warnings
import sqlite3
import hydra
from omegaconf import DictConfig, OmegaConf
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
# Intentar buscar en local y en el servidor
if (Path.home() / "Documentos/GitHub/diputados_chatbot").exists():
    CHATBOT_DIR = Path.home() / "Documentos/GitHub/diputados_chatbot"
else:
    CHATBOT_DIR = Path.home() / "diputados_chatbot" # Asumiendo ruta en servidor

DB_PATH = DATA_DIR / 'parlamento.db'

sys.path.append(str(CHATBOT_DIR))
try:
    from src.core.context_builder import ParlamentarioContextBuilder
except ImportError:
    print(f"Advertencia: No se pudo importar ParlamentarioContextBuilder desde {CHATBOT_DIR}.")
    # No detenemos la ejecución porque la variable `context` no se estaba usando en el prompt final.

# Variables Globales para Steering
current_alpha = 0
current_method = "ridge"
current_focal_ridge_dict = {}
current_head_dict = {}
current_features = None  # [num_mps, layers, heads, d_head]
current_num_heads = None
precomputed_interventions = {}

def get_attention_heads(model):
    heads = []
    # Buscar usando regex sobre todos los modulos nombrados, para evitar
    # que wrappers de accelerate (offload a CPU/disco) oculten los atributos.
    for name, _ in model.named_modules():
        if "vision_tower" in name:
            continue
        if re.search(r'\.\d+\.self_attn\.o_proj$', name):
            heads.append(name)
        elif re.search(r'\.\d+\.attn\.c_proj$', name):
            heads.append(name)
        elif re.search(r'\.\d+\.linear_attn\.out_proj$', name):
            heads.append(name)
            
    # Ordenar por el número de capa para asegurar el orden correcto
    def extract_layer_num(name):
        match = re.search(r'\.(\d+)\.', name)
        return int(match.group(1)) if match else 0
        
    heads = sorted(heads, key=extract_layer_num)
    
    if not heads:
        raise ValueError(f"No se encontraron proyecciones de atención en {type(model).__name__}")
        
    # Obtener el prefix asumiendo que todos comparten el mismo
    prefix = heads[0].rsplit('.', 3)[0]
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

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    global current_method, current_alpha, current_features, precomputed_interventions, current_num_heads
    global current_head_dict, current_focal_ridge_dict
    
    current_method = cfg.get("steering_method", "cpca")
    limit_mps = cfg.get("limit_mps", 10)
    
    sota_models = [
        'google/gemma-4-E4B-it',
        'meta-llama/Llama-3.1-8B-Instruct',
        'zai-org/GLM-4-9B-0414',
    ]
    sota_models_safe = [m.replace('/', '_') for m in sota_models]
    
    RESULTS_DIR = BASE_DIR / cfg.paths.results_dir
    feature_files = [f for f in RESULTS_DIR.glob("*_politician_features.pkl") if any(sota in f.name for sota in sota_models_safe)]
    
    if not feature_files:
        print(f"No se encontraron archivos en {RESULTS_DIR}")
        return
        
    # 1. Cargar DataFrame de Diputados
    df_path = BASE_DIR / cfg.paths.mp_path
    if not df_path.exists():
        print(f"Error: No se encuentra {df_path}")
        return
    df = pd.read_parquet(df_path)
    
    for feat_path in feature_files:
        safe_name = feat_path.name.replace("_politician_features.pkl", "")
        model_name = next((m for m in sota_models if m.replace('/', '_') == safe_name), safe_name)
        
        current_head_dict.clear()
        current_focal_ridge_dict.clear()
        precomputed_interventions.clear()
        
        print(f"\n{'='*60}")
        print(f"Predicción Steered con Activaciones Base | Modelo: {model_name} | Método: {current_method}")
        print(f"{'='*60}")
        
        with open(feat_path, 'rb') as f:
            current_features, _ = pickle.load(f)
            
        if current_method == "ridge":
            dict_path = RESULTS_DIR / f"{safe_name}_ridge.pkl"
            perf_path = RESULTS_DIR / f"{safe_name}_politician_performance.pkl"
            trained_model_dict = pickle.load(open(dict_path, 'rb'))
            performance = pickle.load(open(perf_path, 'rb'))
        elif current_method == "cpca":
            cpca_dir = BASE_DIR / "cpca_results"
            dict_path = cpca_dir / f"{safe_name}_cpca_analysis_global.pkl"
            if not dict_path.exists(): 
                dict_path = cpca_dir / f"{model_name.replace('/', '_')}_cpca_analysis_global.pkl"
            trained_model_dict = pickle.load(open(dict_path, 'rb'))
        else:
            dict_path = BASE_DIR / f"results/{current_method}_analysis/{safe_name}_{current_method}_analysis.pkl"
            trained_model_dict = pickle.load(open(dict_path, 'rb'))
        
        # 3. Cargar LLM
        print("\nCargando Modelo LLM y Tokenizador...")
        HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
        if not HUGGINGFACE_TOKEN:
            env_path = BASE_DIR / ".env"
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        if line.startswith('HUGGINGFACE_TOKEN='):
                            HUGGINGFACE_TOKEN = line.strip().split('=', 1)[1].strip("'\"")
                            break
                            
        hf_token_param = HUGGINGFACE_TOKEN if HUGGINGFACE_TOKEN else None
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE), token=hf_token_param)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=str(MODEL_CACHE), 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16, 
            token=hf_token_param,
            device_map="auto"
        )
        input_device = getattr(model, "device", None) or next(model.parameters()).device
        
        current_num_heads = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", None))
        heads_list, _ = get_attention_heads(model)
        
        # 4. Preparar Top-K componentes
        layers_to_trace = []
        
        if current_method == "ridge":
            k_ridge = cfg.get("k", 16)
            top_indices = np.dstack(np.unravel_index(np.argsort(performance.ravel()), performance.shape))[0][-k_ridge:, :][::-1]
            for i in top_indices:
                l_idx, h_idx = i
                current_focal_ridge_dict[(l_idx, h_idx)] = trained_model_dict[l_idx][h_idx].coef_
                current_head_dict.setdefault(l_idx, []).append(h_idx)
            layers_to_trace = [heads_list[i] for i in sorted(current_head_dict.keys())]
        elif current_method == "cpca":
            top_components = trained_model_dict.get("top_90_components", trained_model_dict.get("all_components", []))
            for comp in top_components:
                l_idx = comp['layer']
                vec = comp['vector']
                current_head_dict.setdefault(l_idx, []).append(vec)
            
            print(f"[DEBUG] Model: {model_name}")
            print(f"[DEBUG] len(heads_list) = {len(heads_list)}")
            print(f"[DEBUG] current_head_dict keys = {sorted(current_head_dict.keys())}")
            
            layers_to_trace = [heads_list[i] for i in sorted(current_head_dict.keys())]
        else:
            k_val = cfg.get("k", 16)
            top_components = trained_model_dict["all_components"][:k_val]
            for comp in top_components:
                l_idx = comp['layer']
                vec = comp['vector']
                current_head_dict.setdefault(l_idx, []).append(vec)
            layers_to_trace = [heads_list[i] for i in sorted(current_head_dict.keys())]
        
        # Extraer parlamentarios target del DF
        if limit_mps > 0:
            df_target = pd.concat([
                df[df['party_1_P6'] == 'Partido Comunista'].head(limit_mps // 2),
                df[df['party_1_P6'] == 'Partido Republicano'].head(limit_mps // 2)
            ])
        else:
            df_target = df
            
        print(f"\nPolíticos a evaluar: {len(df_target)}")
        
        results = []
        alphas_to_test = [0, 0.5, 1.0, 1.5, 2, 2.5, 3] 
        
        for row_idx, row in tqdm(df_target.iterrows(), total=len(df_target), desc="Diputados"):
            nombre = row['Name']
            partido = row['party_1_P6']
            mp_uid = row.get('mp_uid', row_idx) 
            
            try:
                with ParlamentarioContextBuilder() as builder:
                    context = builder.build_complete_context(mp_uid)
            except NameError:
                context = ""
            
            if not DB_PATH.exists():
                print(f"ERROR: No se encuentra la BD de proyectos en {DB_PATH}.")
                sys.exit(1)
                
            real_bills = get_real_bills_for_mp(mp_uid, limit=3)
            if not real_bills:
                continue
            
            mp_base_act = current_features[row_idx]
            
            for bill in real_bills:
                prompt_text = build_prompt(nombre, partido, bill["title"], bill["summary"])
                messages = [{"role": "user", "content": prompt_text}]
                prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer(prompt_formatted, return_tensors="pt")['input_ids'].to(model.device)
                
                for alpha in alphas_to_test:
                    current_alpha = alpha
                    precomputed_interventions.clear()
                    
                    if alpha != 0:
                        if current_method == "ridge":
                            for l_idx, h_idx in current_focal_ridge_dict.keys():
                                ridge_coef = current_focal_ridge_dict[(l_idx, h_idx)]
                                act_mp_head = mp_base_act[l_idx, h_idx, :]
                                norm_coef = np.linalg.norm(ridge_coef)
                                direction = ridge_coef / (norm_coef if norm_coef > 0 else 1)
                                proj_mp = np.dot(act_mp_head, direction)
                                vec = current_alpha * proj_mp * direction
                                t_vec = torch.from_numpy(vec).to(model.dtype).to(input_device)
                                precomputed_interventions[(l_idx, h_idx)] = t_vec
                        else:
                            for l_idx, vecs in current_head_dict.items():
                                layer_total_vec = np.zeros_like(vecs[0])
                                act_mp_layer = mp_base_act[l_idx, :, :].reshape(-1)
                                
                                for vec in vecs:
                                    norm_coef = np.linalg.norm(vec)
                                    direction = vec / (norm_coef if norm_coef > 0 else 1)
                                    proj_mp = np.dot(act_mp_layer, direction)
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
                    from sklearn.metrics import classification_report
                    y_true = subset['voto_real']
                    y_pred = subset['voto_predicho']
                    print(classification_report(y_true, y_pred, zero_division=0))
                    
                    if acc > max_acc:
                        max_acc = acc
                        best_alpha = alpha
                        
            print(f"\n🏆 Mejor Alpha Automático: {best_alpha} (Exactitud: {max_acc:.1f}%)")
            
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
        
        # Unload model explicitly
        del model
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
