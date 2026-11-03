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
from sklearn.model_selection import train_test_split
import json
import fcntl

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

def fast_hook(output, layer_name):
    intervention = precomputed_interventions_dict.get(layer_name)
    if intervention is not None:
        if isinstance(output, tuple):
            output[0][:, -1, :] += intervention
        else:
            output[:, -1, :] += intervention
    return output

def get_residual_layers(model):
    layers_dict = {}
    for name, module in model.named_modules():
        if "vision_tower" in name:
            continue
        m = re.search(r'^(.*(?:layers|h))\b\.(\d+)$', name)
        if m:
            layer_num = int(m.group(2))
            layers_dict[layer_num] = name
            
    if layers_dict:
        return [layers_dict[i] for i in sorted(layers_dict.keys())]

    heads, _ = get_attention_heads(model)
    residual_layers = []
    for h in heads:
        m = re.search(r'^(.*?\.\d+)\.', h)
        if m:
            residual_layers.append(m.group(1))
        else:
            residual_layers.append(h.rsplit('.', 2)[0])
    return residual_layers

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

def get_contested_bills(min_frac_contra):
    """bill_id con fraccion media de votos En Contra (por sesion) >= min_frac_contra."""
    q = """SELECT bill_id, AVG(1.0 * en_contra_total / NULLIF(a_favor_total + en_contra_total, 0))
           FROM sesiones_votacion GROUP BY bill_id"""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(q).fetchall()
    return {r[0] for r in rows if r[1] is not None and r[1] >= min_frac_contra}

def vote_metrics(df):
    """Exactitud, exactitud balanceada (recall medio de A FAVOR / EN CONTRA; INDECISO cuenta como error)."""
    out = {"accuracy": float(df["acierto"].mean() * 100)}
    recalls = []
    for c, key in (("A FAVOR", "recall_a_favor"), ("EN CONTRA", "recall_en_contra")):
        sub = df[df["voto_real"] == c]
        r = float((sub["voto_predicho"] == c).mean() * 100) if len(sub) else float("nan")
        out[key] = r
        if len(sub):
            recalls.append(r)
    out["balanced_accuracy"] = float(np.mean(recalls)) if recalls else float("nan")
    out["pred_a_favor_rate"] = float((df["voto_predicho"] == "A FAVOR").mean() * 100)
    out["indeciso_rate"] = float((df["voto_predicho"] == "INDECISO").mean() * 100)
    return out

def get_real_bills_for_mp(mp_uid, limit=25, seed=42, bill_filter=None):
    """Proyectos con un único voto A Favor / En Contra por (diputado, proyecto).

    El voto del proyecto es el mayoritario entre sus sesiones de votación; los empates
    se descartan. El muestreo es reproducible (seed + mp_uid) y no depende del orden de bill_id.
    """
    query = """
        SELECT b.bill_id, b.titulo, b.resumen, v.voto
        FROM bills b
        JOIN sesiones_votacion sv ON b.bill_id = sv.bill_id
        JOIN votos_parlamentario v ON sv.sesion_votacion_id = v.sesion_votacion_id
        WHERE v.mp_uid = ?
          AND v.voto IN ('A Favor', 'En Contra')
          AND b.resumen IS NOT NULL
          AND length(b.resumen) > 50
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(query, (int(mp_uid),))
        rows = cur.fetchall()

    per_bill = {}
    for bill_id, titulo, resumen, voto in rows:
        entry = per_bill.setdefault(bill_id, {"title": titulo, "summary": resumen, "votes": []})
        entry["votes"].append(voto.upper())

    bills = []
    for bill_id in sorted(per_bill):
        if bill_filter is not None and bill_id not in bill_filter:
            continue
        e = per_bill[bill_id]
        n_favor, n_contra = e["votes"].count("A FAVOR"), e["votes"].count("EN CONTRA")
        if n_favor == n_contra:
            continue
        bills.append({"bill_id": bill_id, "title": e["title"], "summary": e["summary"],
                      "real_vote": "A FAVOR" if n_favor > n_contra else "EN CONTRA"})

    rng = np.random.RandomState(seed + int(mp_uid))
    if len(bills) > limit:
        idx = np.sort(rng.choice(len(bills), size=limit, replace=False))
        bills = [bills[i] for i in idx]
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

def create_or_load_tasks(df_target, limit_mps, curr_output_dir, calib_ratio=0.2, force=False, bill_filter=None):
    tasks_file_name = f"voting_tasks{f'_limit_{limit_mps}' if limit_mps > 0 else ''}.parquet"
    tasks_cache_path = curr_output_dir / tasks_file_name
    
    if tasks_cache_path.exists() and not force:
        print(f"Cargando tareas fijas precalculadas desde {tasks_cache_path}...")
        df_tasks = pd.read_parquet(tasks_cache_path)
        return df_tasks

    print("Generando tareas deterministas desde la base de datos...")
    if not DB_PATH.exists():
        print(f"ERROR: No se encuentra la BD de proyectos en {DB_PATH}.")
        sys.exit(1)

    tasks = []
    for row_idx, row in df_target.iterrows():
        nombre = row['Name']
        partido = row['party_1_P6']
        mp_uid = get_mp_uid_by_name(nombre)
        if not mp_uid:
            continue
        
        real_bills = get_real_bills_for_mp(mp_uid, limit=25, bill_filter=bill_filter)
        if not real_bills:
            continue
        
        for bill in real_bills:
            tasks.append({
                "row_idx": int(row_idx),
                "mp_uid": int(mp_uid),
                "nombre": nombre,
                "partido": partido,
                "ley": bill["title"],
                "voto_real": bill["real_vote"],
                "prompt": build_prompt(nombre, partido, bill["title"], bill["summary"])
            })
            
    df_tasks = pd.DataFrame(tasks)
    if df_tasks.empty:
        raise ValueError("No se pudieron generar tareas de votación.")

    # Estratificación calibración / test por partido político
    stratify_col = df_tasks['partido'].fillna('Independiente')
    val_counts = stratify_col.value_counts()
    rare_parties = val_counts[val_counts < 2].index
    stratify_labels = stratify_col.replace(rare_parties, 'Otros')

    try:
        calib_idx, test_idx = train_test_split(
            np.arange(len(df_tasks)),
            test_size=(1.0 - calib_ratio),
            train_size=calib_ratio,
            stratify=stratify_labels,
            random_state=42
        )
    except Exception:
        calib_idx, test_idx = train_test_split(
            np.arange(len(df_tasks)),
            test_size=(1.0 - calib_ratio),
            train_size=calib_ratio,
            random_state=42
        )

    df_tasks['split'] = 'test'
    df_tasks.loc[calib_idx, 'split'] = 'calibration'

    df_tasks.to_parquet(tasks_cache_path)
    print(f"✅ Tareas guardadas en {tasks_cache_path} ({len(calib_idx)} calibración, {len(test_idx)} test).")
    return df_tasks

def run_batch_inference(
    tasks_list,
    alpha,
    model,
    tokenizer,
    is_residual,
    d_model,
    d_head,
    current_num_heads,
    target_layers_list,
    layers_to_trace,
    input_device,
    batch_size=8,
    desc="Inferencia"
):
    results = []
    for i in tqdm(range(0, len(tasks_list), batch_size), desc=desc):
        batch_tasks = tasks_list[i : i + batch_size]
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
                layer_name = target_layers_list[l_idx]
                
                if is_residual:
                    batch_layer_intervention = np.zeros((b_size, d_model), dtype=np.float32)
                    
                    for b_idx, task in enumerate(batch_tasks):
                        row_idx = task["row_idx"]
                        mp_base_act = current_features[row_idx]
                        act_mp_layer = mp_base_act[l_idx, :]
                        
                        if current_method == "ridge":
                            if l_idx in current_focal_ridge_dict:
                                ridge_coef = current_focal_ridge_dict[l_idx].reshape(-1)
                                norm_coef = np.linalg.norm(ridge_coef)
                                direction = ridge_coef / (norm_coef if norm_coef > 0 else 1)
                                proj_mp = np.dot(act_mp_layer, direction)
                                batch_layer_intervention[b_idx, :] = alpha * proj_mp * direction
                                
                        elif current_method.startswith("bcall"):
                            dims_to_use = ['d1', 'd2'] if current_method == 'bcall_joint' else [current_method.split('_')[1]]
                            vec_total = np.zeros(d_model, dtype=np.float32)
                            for dim in dims_to_use:
                                if l_idx in current_focal_ridge_dict[dim]:
                                    ridge_coef = current_focal_ridge_dict[dim][l_idx].reshape(-1)
                                    norm_coef = np.linalg.norm(ridge_coef)
                                    direction = ridge_coef / (norm_coef if norm_coef > 0 else 1)
                                    proj_mp = np.dot(act_mp_layer, direction)
                                    vec_total += alpha * proj_mp * direction
                            batch_layer_intervention[b_idx, :] = vec_total
                            
                        else:
                            layer_total_vec = np.zeros(d_model, dtype=np.float32)
                            for vec in current_head_dict.get(l_idx, []):
                                vec = vec.reshape(-1)
                                norm_coef = np.linalg.norm(vec)
                                direction = vec / (norm_coef if norm_coef > 0 else 1)
                                proj_mp = np.dot(act_mp_layer, direction)
                                layer_total_vec += (alpha * proj_mp * direction)
                            batch_layer_intervention[b_idx, :] = layer_total_vec
                else:
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
                                    mp_vec[h_idx, :] = alpha * proj_mp * direction
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
                                        vec_total += alpha * proj_mp * direction
                                mp_vec[h_idx, :] = vec_total
                            batch_layer_intervention[b_idx, :] = mp_vec.reshape(-1)
                            
                        else:
                            layer_total_vec = np.zeros(current_num_heads * d_head, dtype=np.float32)
                            act_mp_layer = mp_base_act[l_idx, :, :].reshape(-1)
                            for vec in current_head_dict.get(l_idx, []):
                                norm_coef = np.linalg.norm(vec)
                                direction = vec / (norm_coef if norm_coef > 0 else 1)
                                proj_mp = np.dot(act_mp_layer, direction)
                                layer_total_vec += (alpha * proj_mp * direction)
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
                "alpha": float(alpha),
                "voto_predicho": predicted_vote,
                "respuesta_completa": response,
                "acierto": is_match
            })
    return results

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
    
    target_mode = cfg.get("target", None)
    RESULTS_DIR = BASE_DIR / cfg.paths.results_dir
    if target_mode == "residual" and not str(RESULTS_DIR).endswith("residual"):
        RESULTS_DIR = RESULTS_DIR / "residual"

    model_filter = cfg.get("model", None)
    if model_filter:
        model_safe = model_filter.replace('/', '_')
        feature_files = [f for f in RESULTS_DIR.glob("*_politician_features.pkl") if model_safe.lower() in f.name.lower()]
    else:
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
        model_name = next((m for m in sota_models if m.replace('/', '_') == safe_name), (model_filter if (model_filter and model_filter.replace('/', '_') == safe_name) else safe_name))
        
        current_head_dict.clear()
        current_focal_ridge_dict.clear()
        precomputed_interventions_dict.clear()
        
        print(f"\n{'='*60}")
        print(f"Predicción Steered con Activaciones Base | Modelo: {model_name} | Método: {current_method}")
        print(f"{'='*60}")
        
        with open(feat_path, 'rb') as f:
            current_features, _ = pickle.load(f)
            
        is_residual = (current_features.ndim == 3) or (target_mode == "residual") or ("residual" in str(feat_path))

        curr_output_dir = OUTPUT_DIR / "residual" if is_residual else OUTPUT_DIR
        task_mode = cfg.get("task_mode", "all")
        bill_filter = None
        if task_mode == "divided":
            curr_output_dir = curr_output_dir / "divided"
            bill_filter = get_contested_bills(cfg.get("min_frac_contra", 0.15))
            print(f"[TASK MODE] divided: {len(bill_filter)} proyectos con >= {cfg.get('min_frac_contra', 0.15):.0%} En Contra")
        curr_output_dir.mkdir(parents=True, exist_ok=True)
            
        if current_method == "ridge":
            dict_path = RESULTS_DIR / f"{safe_name}_ridge.pkl"
            perf_path = RESULTS_DIR / f"{safe_name}_politician_performance.pkl"
            trained_model_dict = pickle.load(open(dict_path, 'rb'))
            performance = pickle.load(open(perf_path, 'rb'))
        elif current_method.startswith("bcall"):
            bcall_dir = BASE_DIR / "results/probing/bcall/names_only"
            if is_residual:
                bcall_dir = bcall_dir / "residual"
            dict_path = bcall_dir / f"{safe_name}_bcall_ridge.pkl"
            perf_path = bcall_dir / f"{safe_name}_bcall_performance.pkl"
            trained_model_dict = pickle.load(open(dict_path, 'rb'))
            performance = pickle.load(open(perf_path, 'rb'))
        else:
            base_method = current_method.split('_')[0]
            if base_method in ("cpca", "random"):
                cpca_dir = BASE_DIR / "results/cpca_analysis"
                if is_residual:
                    cpca_dir = cpca_dir / "residual"
                ca_match = re.search(r'_ca([0-9.]+)', current_method)
                ca_tag = f"_ca{ca_match.group(1)}" if ca_match else ""
                if re.search(r'_np(_|$)', current_method):
                    ca_tag += "_noparty"
                dict_path = cpca_dir / f"{safe_name}_cpca_analysis_global{ca_tag}.pkl"
                if not dict_path.exists(): 
                    dict_path = cpca_dir / f"{model_name.replace('/', '_')}_cpca_analysis_global{ca_tag}.pkl"
                trained_model_dict = pickle.load(open(dict_path, 'rb'))
            else:
                method_dir = BASE_DIR / f"results/{base_method}_analysis"
                if is_residual:
                    method_dir = method_dir / "residual"
                dict_path = method_dir / f"{safe_name}_{base_method}_analysis_global.pkl"
                if not dict_path.exists():
                    dict_path = method_dir / f"{model_name.replace('/', '_')}_{base_method}_analysis_global.pkl"
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
        
        if is_residual:
            target_layers_list = get_residual_layers(model)
        else:
            target_layers_list = heads_list

        # 4. Preparar Top-K componentes
        layers_to_trace = []
        
        if current_method == "ridge":
            k_ridge = cfg.get("k", 16)
            if is_residual:
                top_layers = np.argsort(performance)[-k_ridge:][::-1]
                for l_idx in top_layers:
                    current_focal_ridge_dict[l_idx] = trained_model_dict[l_idx].coef_
                    current_head_dict.setdefault(l_idx, []).append(0)
            else:
                top_indices = np.dstack(np.unravel_index(np.argsort(performance.ravel()), performance.shape))[0][-k_ridge:, :][::-1]
                for i in top_indices:
                    l_idx, h_idx = i
                    current_focal_ridge_dict[(l_idx, h_idx)] = trained_model_dict[l_idx][h_idx].coef_
                    current_head_dict.setdefault(l_idx, []).append(h_idx)
            layers_to_trace = [target_layers_list[i] for i in sorted(current_head_dict.keys())]
        elif current_method.startswith("bcall"):
            k_ridge = cfg.get("k", 16)
            dims_to_use = ['d1', 'd2'] if current_method == 'bcall_joint' else [current_method.split('_')[1]]
            current_focal_ridge_dict = {dim: {} for dim in dims_to_use}
            if is_residual:
                for dim in dims_to_use:
                    perf_dim = performance[dim]
                    top_layers = np.argsort(perf_dim)[-k_ridge:][::-1]
                    for l_idx in top_layers:
                        current_focal_ridge_dict[dim][l_idx] = trained_model_dict[dim][l_idx].coef_
                        current_head_dict.setdefault(l_idx, []).append(0)
                for l_idx in current_head_dict:
                    current_head_dict[l_idx] = list(set(current_head_dict[l_idx]))
            else:
                for dim in dims_to_use:
                    perf_dim = performance[dim]
                    top_indices = np.dstack(np.unravel_index(np.argsort(perf_dim.ravel()), perf_dim.shape))[0][-k_ridge:, :][::-1]
                    for i in top_indices:
                        l_idx, h_idx = i
                        current_focal_ridge_dict[dim][(l_idx, h_idx)] = trained_model_dict[dim][l_idx][h_idx].coef_
                        current_head_dict.setdefault(l_idx, []).append(h_idx)
                for l_idx in current_head_dict:
                    current_head_dict[l_idx] = list(set(current_head_dict[l_idx]))
            layers_to_trace = [target_layers_list[i] for i in sorted(current_head_dict.keys())]
        else:
            if current_method.endswith("_90var") and "top_90_components" in trained_model_dict:
                top_components = trained_model_dict["top_90_components"]
                print(f"[INFO] Usando {len(top_components)} componentes del 90% de varianza para {current_method}.")
            else:
                k_match = re.search(r'_k(\d+)', current_method)
                k_val = int(k_match.group(1)) if k_match else cfg.get("k", 16)
                top_components = trained_model_dict.get("all_components", [])[:k_val]
                print(f"[INFO] Usando top {len(top_components)} componentes de all_components para {current_method}.")
                
            if base_method == "random":
                # Control: mismas capas/numero de componentes que cPCA, direcciones aleatorias unitarias
                s_match = re.search(r'_s(\d+)', current_method)
                rng_dir = np.random.RandomState(int(s_match.group(1)) if s_match else 0)
                top_components = [
                    {'layer': c['layer'], 'vector': (lambda v: (v / np.linalg.norm(v)).astype(np.float32))(rng_dir.randn(*np.shape(c['vector'])))}
                    for c in top_components
                ]
            for comp in top_components:
                l_idx = comp['layer']
                vec = comp['vector']
                current_head_dict.setdefault(l_idx, []).append(vec)
                
            layers_to_trace = [target_layers_list[i] for i in sorted(current_head_dict.keys())]
        
        # Extraer parlamentarios target del DF
        if limit_mps > 0:
            df_target = pd.concat([
                df[df['party_1_P6'] == 'Partido Comunista'].head(limit_mps // 2),
                df[df['party_1_P6'] == 'Partido Republicano'].head(limit_mps // 2)
            ])
        else:
            df_target = df
            
        print(f"\nPolíticos a evaluar: {len(df_target)}")
        
        # 5. Cargar o generar tareas de votación fijas
        force_tasks = cfg.get("force_tasks", False)
        force_baseline = cfg.get("force_baseline", False)
        calib_ratio = cfg.get("calib_ratio", 0.2)
        
        df_tasks = create_or_load_tasks(df_target, limit_mps, curr_output_dir, calib_ratio=calib_ratio, force=force_tasks, bill_filter=bill_filter)
        print(f"Total de predicciones a procesar: {len(df_tasks)}")
        
        # Batch settings
        BATCH_SIZE = cfg.get("batch_size", 8)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        if is_residual:
            d_model = current_features.shape[-1]
            d_head = d_model
            current_num_heads = 1
        else:
            d_head = current_features.shape[-1]
            current_num_heads = current_features.shape[-2]
            d_model = d_head * current_num_heads
            
        # 6. Baseline: Cargar o Ejecutar Alpha = 0 (1 sola vez para todo el dataset)
        baseline_file_name = f"baseline_predictions_{safe_name}{f'_limit_{limit_mps}' if limit_mps > 0 else ''}.parquet"
        baseline_path = curr_output_dir / baseline_file_name
        
        if baseline_path.exists() and not force_baseline:
            print(f"\n[BASELINE] Cargando predicciones baseline (Alpha=0) precalculadas desde:\n  -> {baseline_path}")
            df_baseline = pd.read_parquet(baseline_path)
        else:
            print(f"\n[BASELINE] Ejecutando baseline global (Alpha=0) para {len(df_tasks)} tareas...")
            baseline_tasks = df_tasks.to_dict('records')
            baseline_results = run_batch_inference(
                baseline_tasks,
                alpha=0.0,
                model=model,
                tokenizer=tokenizer,
                is_residual=is_residual,
                d_model=d_model,
                d_head=d_head,
                current_num_heads=current_num_heads,
                target_layers_list=target_layers_list,
                layers_to_trace=layers_to_trace,
                input_device=input_device,
                batch_size=BATCH_SIZE,
                desc="Baseline (Alpha=0)"
            )
            df_baseline = pd.DataFrame(baseline_results)
            df_baseline.to_parquet(baseline_path)
            print(f"✅ Baseline guardado con éxito en: {baseline_path}")
            
        calib_mask = (df_tasks['split'] == 'calibration').values
        df_baseline['split'] = df_tasks['split'].values
        baseline_test_acc = df_baseline[~calib_mask]['acierto'].mean() * 100
        baseline_test_m = vote_metrics(df_baseline[~calib_mask])
        baseline_calib_acc = df_baseline[calib_mask]['acierto'].mean() * 100
        baseline_full_acc = df_baseline['acierto'].mean() * 100
        
        print(f"\n{'='*55}")
        print(f"BASELINE (Alpha=0) | Calibración (20%): {baseline_calib_acc:.1f}% | Total (100%): {baseline_full_acc:.1f}%")
        print(f"{'='*55}")
        
        # 7. Calibración de Alpha en el 20% estratificado
        alphas_to_test = cfg.get("alphas", [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0])
        calib_tasks = df_tasks[calib_mask].to_dict('records')
        print(f"\nIniciando calibración de Alpha sobre {len(calib_tasks)} tareas estratificadas ({int(calib_ratio*100)}%)...")
        
        best_alpha = 0.0
        best_calib_acc = baseline_calib_acc
        baseline_calib_m = vote_metrics(df_baseline[calib_mask])
        best_calib_bacc = baseline_calib_m["balanced_accuracy"]  # criterio de seleccion de alpha
        calib_results_by_alpha = {}
        calib_summary = {0.0: {"accuracy": float(baseline_calib_acc), "balanced_accuracy": best_calib_bacc, "failure_rate": 0.0}}
        
        for alpha in alphas_to_test:
            alpha = float(alpha)
            print(f"\n--- Evaluando Alpha = {alpha} (Calibración) ---")
            alpha_results = run_batch_inference(
                calib_tasks,
                alpha=alpha,
                model=model,
                tokenizer=tokenizer,
                is_residual=is_residual,
                d_model=d_model,
                d_head=d_head,
                current_num_heads=current_num_heads,
                target_layers_list=target_layers_list,
                layers_to_trace=layers_to_trace,
                input_device=input_device,
                batch_size=BATCH_SIZE,
                desc=f"Calibración (Alpha={alpha})"
            )
            calib_results_by_alpha[alpha] = alpha_results
            acc = np.mean([r['acierto'] for r in alpha_results]) * 100
            indecisos = sum(1 for r in alpha_results if r["voto_predicho"] == "INDECISO")
            tasa_fallo = indecisos / len(alpha_results)
            
            m_cal = vote_metrics(pd.DataFrame(alpha_results))
            calib_summary[alpha] = {"accuracy": float(acc), "balanced_accuracy": m_cal["balanced_accuracy"],
                                    "recall_en_contra": m_cal["recall_en_contra"], "failure_rate": float(tasa_fallo)}
            print(f"Alpha: {alpha:4.2f} | Exactitud Calibración: {acc:.1f}% | Bal.Acc: {m_cal['balanced_accuracy']:.1f}% | Recall EN CONTRA: {m_cal['recall_en_contra']:.1f}% | Indecisos: {tasa_fallo*100:.1f}%")
            
            if tasa_fallo > 0.5:
                print(f"⚠️ Modelo colapsado en alpha={alpha} (Tasa de fallos: {tasa_fallo*100:.1f}%). Deteniendo búsqueda de alphas mayores.")
                break
                
            if m_cal["balanced_accuracy"] > best_calib_bacc:
                best_calib_bacc = m_cal["balanced_accuracy"]
                best_calib_acc = acc
                best_alpha = alpha
                
        print(f"\n{'='*60}")
        print(f"🏆 Alpha Óptimo Seleccionado: {best_alpha}")
        print(f"   Bal.Acc en Calibración ({int(calib_ratio*100)}%): {best_calib_bacc:.1f}% (vs Baseline: {baseline_calib_m['balanced_accuracy']:.1f}%) | Exactitud: {best_calib_acc:.1f}%")
        print(f"{'='*60}")
        
        # 8. Evaluación en el 100% con Best Alpha
        if best_alpha == 0.0:
            print("\nNingún alpha superó al baseline en calibración. El modelo final se mantendrá en baseline (Alpha=0).")
            df_final = df_baseline.copy()
            max_acc = baseline_full_acc
            steered_test_acc = baseline_test_acc
            steered_test_m = vote_metrics(df_baseline[~calib_mask])
        else:
            eval_tasks = df_tasks[~calib_mask].to_dict('records')
            print(f"\nIniciando inferencia con Best Alpha={best_alpha} en el 80% restante ({len(eval_tasks)} tareas)...")
            eval_results = run_batch_inference(
                eval_tasks,
                alpha=best_alpha,
                model=model,
                tokenizer=tokenizer,
                is_residual=is_residual,
                d_model=d_model,
                d_head=d_head,
                current_num_heads=current_num_heads,
                target_layers_list=target_layers_list,
                layers_to_trace=layers_to_trace,
                input_device=input_device,
                batch_size=BATCH_SIZE,
                desc=f"Test 80% (Alpha={best_alpha})"
            )
            
            df_steered = pd.concat([
                pd.DataFrame(calib_results_by_alpha[best_alpha]).assign(split='calibration'),
                pd.DataFrame(eval_results).assign(split='test')
            ], ignore_index=True)
            steered_test_acc = df_steered[df_steered['split'] == 'test']['acierto'].mean() * 100
            steered_test_m = vote_metrics(df_steered[df_steered['split'] == 'test'])
            
            steered_full_acc = df_steered['acierto'].mean() * 100
            max_acc = steered_full_acc
            
            print(f"\n{'='*60}")
            print(f"RESULTADO FINAL 100% DATASET | Método: {current_method}")
            print(f"Baseline (Alpha=0):                {baseline_full_acc:.1f}%")
            print(f"Steered (Best Alpha={best_alpha:4.1f}):     {steered_full_acc:.1f}% (Delta: {steered_full_acc - baseline_full_acc:+.1f}%)")
            print(f"TEST bal.acc: baseline {baseline_test_m['balanced_accuracy']:.1f}% -> steered {steered_test_m['balanced_accuracy']:.1f}% | recall EN CONTRA {baseline_test_m['recall_en_contra']:.1f}% -> {steered_test_m['recall_en_contra']:.1f}%")
            print(f"TEST 80% (sin fuga de calibración): baseline {baseline_test_acc:.1f}% -> steered {steered_test_acc:.1f}% (Delta: {steered_test_acc - baseline_test_acc:+.1f}%)")
            print(f"{'='*60}")
            
            # Combinamos baseline (100%) y steered (100%) en df_final
            df_final = pd.concat([df_baseline, df_steered], ignore_index=True)
            
        # 9. Reportes de clasificación
        print("\n--- Reporte de Clasificación Baseline (Alpha=0, 100% Dataset) ---")
        print(classification_report(df_baseline['voto_real'], df_baseline['voto_predicho'], zero_division=0))
        
        if best_alpha > 0.0:
            print(f"\n--- Reporte de Clasificación Steered (Alpha={best_alpha}, 100% Dataset) ---")
            steered_subset = df_final[df_final['alpha'] == best_alpha]
            print(classification_report(steered_subset['voto_real'], steered_subset['voto_predicho'], zero_division=0))
            
        # 10. Guardar Optimal Alphas
        optimal_json_path = curr_output_dir / "optimal_alphas.json"
        lock_file = open(curr_output_dir / ".optimal_alphas.lock", "w")
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # varias corridas en paralelo escriben el mismo JSON
        optimal_dict = {}
        if optimal_json_path.exists():
            try:
                with open(optimal_json_path, 'r') as f:
                    optimal_dict = json.load(f)
            except Exception:
                optimal_dict = {}
                
        if safe_name not in optimal_dict:
            optimal_dict[safe_name] = {}
            
        optimal_dict[safe_name][current_method] = {
            "best_alpha": float(best_alpha),
            "accuracy": float(max_acc),
            "baseline_accuracy": float(baseline_full_acc),
            "test_accuracy": float(steered_test_acc),
            "test_metrics": steered_test_m,
            "baseline_test_metrics": baseline_test_m,
            "baseline_test_accuracy": float(baseline_test_acc),
            "calibration_accuracy": float(best_calib_acc),
            "calibration_summary": calib_summary
        }
        
        with open(optimal_json_path, 'w') as f:
            json.dump(optimal_dict, f, indent=4)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
            
        # 11. Guardar DataFrame final
        out_file = curr_output_dir / f"voting_predictions_{safe_name}_{current_method}.parquet"
        df_final.to_parquet(out_file)
        print(f"\n✅ Resultados guardados en {out_file}")
        
        # Unload model explicitly
        del model
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
