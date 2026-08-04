import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_CACHE = BASE_DIR / "model"

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('HUGGINGFACE_TOKEN='):
                    HUGGINGFACE_TOKEN = line.strip().split('=', 1)[1]

STANDARD_MODELS = [
    'google/gemma-4-E4B-it',
    'meta-llama/Llama-3.1-8B-Instruct',
    'Qwen/Qwen3.5-9B',
    # 'huihui-ai/Huihui-Qwen3.5-9B-abliterated',
    # 'mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated',
]

LARGE_MODELS = [
    'meta-llama/Llama-3.1-70B-Instruct',
    'latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0',
]

def descargar_modelo(model_name, cache_dir):
    print(f"\n--- Iniciando descarga de {model_name} ---")
    try:
        # snapshot_download descarga los pesos sin cargar el modelo en RAM/VRAM
        path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            token=HUGGINGFACE_TOKEN,
            resume_download=True
        )
        print(f"Éxito: Modelo {model_name} descargado en {path}")
    except Exception as e:
        print(f"Error al descargar {model_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Descargar modelos localmente sin utilizar GPU ni RAM para instanciarlos.")
    parser.add_argument("--model", type=str, help="Nombre de un modelo específico a descargar.")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE), 
                        help="Directorio donde descargar los modelos. Para modelos de 70B, puedes usar un directorio como '/data'.")
    parser.add_argument("--all-standard", action="store_true", help="Descargar todos los modelos estándar (7B-9B).")
    parser.add_argument("--all-large", action="store_true", help="Descargar todos los modelos grandes (70B).")
    
    args = parser.parse_args()
    
    if not HUGGINGFACE_TOKEN:
        print("Advertencia: No se encontró HUGGINGFACE_TOKEN. Las descargas de modelos privados (como Llama) fallarán.")
        
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directorio de descarga configurado en: {cache_dir.resolve()}")
    
    modelos_a_descargar = []
    
    if args.model:
        modelos_a_descargar.append(args.model)
    if args.all_standard:
        modelos_a_descargar.extend(STANDARD_MODELS)
    if args.all_large:
        modelos_a_descargar.extend(LARGE_MODELS)
        
    if not modelos_a_descargar:
        print("Por favor, especifica al menos un modelo o flag. Usa --help para ver las opciones.")
        return
        
    # Eliminar duplicados manteniendo el orden
    modelos_a_descargar = list(dict.fromkeys(modelos_a_descargar))
    
    for model_name in modelos_a_descargar:
        descargar_modelo(model_name, cache_dir)
        
    print("\n--- Todas las descargas solicitadas han finalizado ---")

if __name__ == "__main__":
    main()
