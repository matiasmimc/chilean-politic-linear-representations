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
current_method = None
current_focal_ridge_dict = {}
current_head_dict = {}
current_features = None  # [num_mps, layers, heads, d_head]
current_num_heads = None
precomputed_interventions_dict = {}  # layer_name -> tensor [batch_size, H*D]

def fast_hook(head_output, layer_name):
    intervention = precomputed_interventions_dict.get(layer_name)
    if intervention is not None:
        head_output[:, -1, :] += intervention
    return head_output

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

def get_real_bills_for_mp(mp_uid, limit=25):
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

def get_mp_uid_by_name(nombre):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT mp_uid FROM dim_parlamentario WHERE nombre_completo = ?", (nombre,))
        res = cur.fetchone()
        if res: return res[0]
        
        parts = nombre.split()
        if len(parts) >= 3:
            first = parts[0]
            last1 = parts[-2]
            last2 = parts[-1]
            like_q = f"%{first}%{last1}%{last2}%"
            cur.execute("SELECT mp_uid FROM dim_parlamentario WHERE nombre_completo LIKE ?", (like_q,))
            res = cur.fetchone()
            if res: return res[0]
            
            like_q2 = f"%{last1}%{last2}%"
            cur.execute("SELECT mp_uid FROM dim_parlamentario WHERE nombre_completo LIKE ?", (like_q2,))
            res = cur.fetchone()
            if res: return res[0]
        return None

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    global current_method, current_alpha, current_features, precomputed_interventions_dict, current_num_heads
    global current_head_dict, current_focal_ridge_dict
    
    current_method = cfg.get("steering_method", "cpca")
    limit_mps = cfg.get("limit_mps", 0)
    
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
        precomputed_interventions_dict.clear()
        
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
        elif current_method.startswith("bcall"):
            bcall_dir = BASE_DIR / "results/probing/bcall/names_only"
            dict_path = bcall_dir / f"{safe_name}_bcall_ridge.pkl"
            perf_path = bcall_dir / f"{safe_name}_bcall_performance.pkl"
            trained_model_dict = pickle.load(open(dict_path, 'rb'))
            performance = pickle.load(open(perf_path, 'rb'))
        else:
            base_method = current_method.split('_')[0]
            if base_method == "cpca":
                cpca_dir = BASE_DIR / "results/cpca_analysis"
                dict_path = cpca_dir / f"{safe_name}_cpca_analysis_global.pkl"
                if not dict_path.exists(): 
                    dict_path = cpca_dir / f"{model_name.replace('/', '_')}_cpca_analysis_global.pkl"
                trained_model_dict = pickle.load(open(dict_path, 'rb'))
            else:
                dict_path = BASE_DIR / f"results/{base_method}_analysis/{safe_name}_{base_method}_analysis_global.pkl"
                if not dict_path.exists():
                    dict_path = BASE_DIR / f"results/{base_method}_analysis/{model_name.replace('/', '_')}_{base_method}_analysis_global.pkl"
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
        elif current_method.startswith("bcall"):
            k_ridge = cfg.get("k", 16)
            dims_to_use = ['d1', 'd2'] if current_method == 'bcall_joint' else [current_method.split('_')[1]]
            current_focal_ridge_dict = {dim: {} for dim in dims_to_use}
            for dim in dims_to_use:
                perf_dim = performance[dim]
                top_indices = np.dstack(np.unravel_index(np.argsort(perf_dim.ravel()), perf_dim.shape))[0][-k_ridge:, :][::-1]
                for i in top_indices:
                    l_idx, h_idx = i
                    current_focal_ridge_dict[dim][(l_idx, h_idx)] = trained_model_dict[dim][l_idx][h_idx].coef_
                    current_head_dict.setdefault(l_idx, []).append(h_idx)
            for l_idx in current_head_dict:
                current_head_dict[l_idx] = list(set(current_head_dict[l_idx]))
            layers_to_trace = [heads_list[i] for i in sorted(current_head_dict.keys())]
        else:
            # Para métodos no supervisados.
            # Si el método termina en "_90var", usamos el subset dinámico.
            # De lo contrario (o si es "_topk"), usamos los top K fijos.
            if current_method.endswith("_90var") and "top_90_components" in trained_model_dict:
                top_components = trained_model_dict["top_90_components"]
                print(f"[INFO] Usando {len(top_components)} componentes del 90% de varianza para {current_method}.")
            else:
                k_val = cfg.get("k", 16)
                top_components = trained_model_dict.get("all_components", [])[:k_val]
                print(f"[INFO] Usando top {len(top_components)} componentes de all_components para {current_method}.")
                
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
        
        # Build tasks for batching
        if not DB_PATH.exists():
            print(f"ERROR: No se encuentra la BD de proyectos en {DB_PATH}.")
            sys.exit(1)
            
        tasks = []
        for row_idx, row in df_target.iterrows():
            nombre = row['Name']
            partido = row['party_1_P6']
            mp_uid = get_mp_uid_by_name(nombre)
            if not mp_uid: continue
            
            real_bills = get_real_bills_for_mp(mp_uid, limit=25)
            if not real_bills: continue
            
            for bill in real_bills:
                tasks.append({
                    "row_idx": row_idx,
                    "mp_uid": mp_uid,
                    "nombre": nombre,
                    "partido": partido,
                    "ley": bill["title"],
                    "voto_real": bill["real_vote"],
                    "prompt": build_prompt(nombre, partido, bill["title"], bill["summary"])
                })
                
        print(f"Total de predicciones a procesar: {len(tasks)}")
        
        # Batch settings
        BATCH_SIZE = cfg.get("batch_size", 8)  # Set a safe default batch size
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        d_head = current_features.shape[-1]
        current_num_heads = current_features.shape[-2]
            
        results = []
        alphas_to_test = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
        
        for alpha in alphas_to_test:
            current_alpha = alpha
            print(f"\n--- Evaluando Alpha = {alpha} ---")
            
            for i in tqdm(range(0, len(tasks), BATCH_SIZE), desc=f"Batches (Alpha={alpha})"):
                batch_tasks = tasks[i : i + BATCH_SIZE]
                b_size = len(batch_tasks)
                
                prompts = [t["prompt"] for t in batch_tasks]
                messages_batch = [[{"role": "user", "content": p}] for p in prompts]
                prompts_formatted = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
                
                inputs = tokenizer(prompts_formatted, return_tensors="pt", padding=True).to(model.device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
                precomputed_interventions_dict.clear()
                
                if alpha != 0:
                    for l_idx in current_head_dict.keys():
                        layer_name = heads_list[l_idx]
                        batch_layer_intervention = np.zeros((b_size, current_num_heads * d_head), dtype=np.float32)
                        
                        for b_idx, task in enumerate(batch_tasks):
                            row_idx = task["row_idx"]
                            mp_base_act = current_features[row_idx]
                            
                            if current_method == "ridge":
                                mp_vec = np.zeros((current_num_heads, d_head), dtype=np.float32)
                                for h_idx in current_head_dict.get(l_idx, []):
                                    if (l_idx, h_idx) in current_focal_ridge_dict:
                                        ridge_coef = current_focal_ridge_dict[(l_idx, h_idx)]
                                        act_mp_head = mp_base_act[l_idx, h_idx, :]
                                        norm_coef = np.linalg.norm(ridge_coef)
                                        direction = ridge_coef / (norm_coef if norm_coef > 0 else 1)
                                        proj_mp = np.dot(act_mp_head, direction)
                                        mp_vec[h_idx, :] = current_alpha * proj_mp * direction
                                batch_layer_intervention[b_idx, :] = mp_vec.reshape(-1)
                                
                            elif current_method.startswith("bcall"):
                                mp_vec = np.zeros((current_num_heads, d_head), dtype=np.float32)
                                dims_to_use = ['d1', 'd2'] if current_method == 'bcall_joint' else [current_method.split('_')[1]]
                                for h_idx in current_head_dict.get(l_idx, []):
                                    act_mp_head = mp_base_act[l_idx, h_idx, :]
                                    vec_total = np.zeros_like(act_mp_head)
                                    for dim in dims_to_use:
                                        if (l_idx, h_idx) in current_focal_ridge_dict[dim]:
                                            ridge_coef = current_focal_ridge_dict[dim][(l_idx, h_idx)]
                                            norm_coef = np.linalg.norm(ridge_coef)
                                            direction = ridge_coef / (norm_coef if norm_coef > 0 else 1)
                                            proj_mp = np.dot(act_mp_head, direction)
                                            vec_total += current_alpha * proj_mp * direction
                                    mp_vec[h_idx, :] = vec_total
                                batch_layer_intervention[b_idx, :] = mp_vec.reshape(-1)
                                
                            else:
                                layer_total_vec = np.zeros(current_num_heads * d_head, dtype=np.float32)
                                act_mp_layer = mp_base_act[l_idx, :, :].reshape(-1)
                                for vec in current_head_dict.get(l_idx, []):
                                    norm_coef = np.linalg.norm(vec)
                                    direction = vec / (norm_coef if norm_coef > 0 else 1)
                                    proj_mp = np.dot(act_mp_layer, direction)
                                    layer_total_vec += (current_alpha * proj_mp * direction)
                                batch_layer_intervention[b_idx, :] = layer_total_vec
                                
                        t_vec = torch.from_numpy(batch_layer_intervention).to(model.dtype).to(input_device)
                        precomputed_interventions_dict[layer_name] = t_vec

                with torch.no_grad():
                    if alpha == 0:
                        out_tokens = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=50,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    else:
                        with TraceDict(model, layers_to_trace, edit_output=fast_hook):
                            out_tokens = model.generate(
                                input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=50,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id
                            )
                
                input_len = input_ids.shape[1]
                new_tokens_batch = out_tokens[:, input_len:]
                responses = tokenizer.batch_decode(new_tokens_batch, skip_special_tokens=True)
                
                for task, response in zip(batch_tasks, responses):
                    response = response.strip()
                    predicted_vote = "A FAVOR" if "A FAVOR" in response.upper() else ("EN CONTRA" if "EN CONTRA" in response.upper() else "INDECISO")
                    is_match = (predicted_vote == task["voto_real"])
                    
                    results.append({
                        "mp_uid": task["mp_uid"],
                        "nombre": task["nombre"],
                        "partido": task["partido"],
                        "ley": task["ley"],
                        "voto_real": task["voto_real"],
                        "alpha": alpha,
                        "voto_predicho": predicted_vote,
                        "respuesta_completa": response,
                        "acierto": is_match
                    })
            
            # Early stopping check
            alpha_results = [r for r in results if r["alpha"] == alpha]
            if alpha_results:
                indecisos = sum(1 for r in alpha_results if r["voto_predicho"] == "INDECISO")
                tasa_fallo = indecisos / len(alpha_results)
                
                if tasa_fallo > 0.5:
                    print(f"⚠️ Modelo colapsado en alpha={alpha} (Tasa de fallos: {tasa_fallo*100:.1f}%). Activando early stopping y descartando alphas mayores.")
                    break
                    
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
