import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path

# Usaremos matplotlib para el mapa de colores
try:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
except ImportError:
    print("Instalando matplotlib...")
    os.system(f"{sys.executable} -m pip install matplotlib")
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures" / "token_visualizations"

def load_data(model_name, k, alpha, topic, tokens_dir):
    safe_name = model_name.replace("/", "_")
    
    # Paths
    ridge_path = RESULTS_DIR / "replication" / "v2" / f"{safe_name}_ridge.pkl"
    act_path = BASE_DIR / tokens_dir / safe_name / f"k_{k}" / f"alpha_{alpha}" / f"{topic}.pkl"
    
    if not ridge_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo ridge probe: {ridge_path}")
    if not act_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de activaciones: {act_path}")
        
    with open(ridge_path, 'rb') as f:
        ridge_dict = pickle.load(f)
        
    with open(act_path, 'rb') as f:
        act_data = pickle.load(f)
        
    return ridge_dict, act_data

def generate_html(token_strings, scores, output_path, title):
    # Normalizar scores a [-1, 1] (o según el valor máximo absoluto) para el colormap
    max_abs = max(abs(np.min(scores)), abs(np.max(scores))) if len(scores) > 0 else 1.0
    if max_abs == 0:
        max_abs = 1.0
        
    norm = mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
    # cmap 'coolwarm' va de azul (negativo/izquierda) a rojo (positivo/derecha)
    cmap = cm.get_cmap('coolwarm')
    
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{ font-family: "Courier New", Courier, monospace; font-size: 16px; line-height: 1.8; padding: 30px; background-color: #1e1e1e; color: #f0f0f0; }}
            .container {{ max-width: 900px; margin: auto; background-color: #2d2d2d; padding: 30px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }}
            .token {{ display: inline-block; padding: 2px 0px; border-radius: 4px; margin-bottom: 2px; cursor: default; transition: opacity 0.2s; }}
            .token:hover {{ opacity: 0.8; outline: 1px solid white; }}
            h1 {{ font-family: sans-serif; text-align: center; color: #ffffff; margin-bottom: 5px; }}
            p.subtitle {{ font-family: sans-serif; text-align: center; color: #bbbbbb; margin-bottom: 30px; }}
            .legend-container {{ display: flex; justify-content: center; align-items: center; margin-bottom: 20px; font-family: sans-serif; font-size: 14px; gap: 15px; }}
            .color-box {{ display: inline-block; width: 20px; height: 10px; border-radius: 2px; }}
            .text-content {{ padding: 20px; background: #222222; border-radius: 5px; word-wrap: break-word; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <p class="subtitle">Análisis palabra por palabra del contenido político latente</p>
            <div class="legend-container">
                <div><span class="color-box" style="background-color: {mcolors.to_hex(cmap(0.0))}"></span> Izquierda (Negativo)</div>
                <div><span class="color-box" style="background-color: {mcolors.to_hex(cmap(0.5))}"></span> Neutral (Cero)</div>
                <div><span class="color-box" style="background-color: {mcolors.to_hex(cmap(1.0))}"></span> Derecha (Positivo)</div>
            </div>
            <div class="text-content">
    """
    
    for token, score in zip(token_strings, scores):
        # Limpiar el token para HTML (muchos tokenizers usan 'Ġ' o ' ' para el espacio)
        display_token = token.replace('Ġ', ' ').replace(' ', ' ').replace('<', '&lt;').replace('>', '&gt;')
        
        # Saltos de línea
        if display_token == 'Ċ' or display_token == '\n' or display_token == '<0x0A>':
            html_content += "<br/>"
            continue
            
        # Generar color
        rgba = cmap(norm(score))
        hex_color = mcolors.to_hex(rgba)
        
        # Para mejorar el contraste del texto
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = "white" if luminance < 0.6 else "black"
        
        # Ajustamos el padding si empieza con espacio
        padding_left = "4px" if display_token.startswith(" ") else "0px"
        
        html_content += f'<span class="token" style="background-color: {hex_color}; color: {text_color}; padding-left: {padding_left};" title="Score: {score:.3f}">{display_token}</span>'
        
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description="Generar visualización HTML de proyecciones políticas por palabra.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Nombre del modelo.")
    parser.add_argument("--topic", type=str, default="aborto", help="Tema de la declaración.")
    parser.add_argument("--alpha", type=int, default=10, help="Valor de alpha de la intervención.")
    parser.add_argument("--k", type=int, default=16, help="Número de cabezas k.")
    parser.add_argument("--tokens-dir", type=str, default="results/500_tokens/token_activations", help="Directorio de activaciones.")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Cargando datos para {args.model}, tema: {args.topic}, alpha: {args.alpha}, k: {args.k}...")
    try:
        ridge_dict, act_data = load_data(args.model, args.k, args.alpha, args.topic, args.tokens_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    activations = act_data['activations']
    token_strings = act_data['token_strings']
    
    keys = list(activations.keys())
    
    if not keys:
        print("Error: No se encontraron activaciones guardadas.")
        return

    # Determinar el largo de la secuencia generado
    seq_len = activations[keys[0]].shape[0]
    
    # Prevenir errores de desfase
    min_len = min(seq_len, len(token_strings))
    token_strings = token_strings[:min_len]
    
    scores = np.zeros(min_len)
    
    for (layer, head) in keys:
        acts = activations[(layer, head)][:min_len, :] # shape: (min_len, d_head)
        coef = ridge_dict[layer][head].coef_ # shape: (d_head,)
        
        # Proyección sobre el vector político
        proj = acts @ coef
        scores += proj
        
    # Promedio de la proyección a través de todas las cabezas extraídas
    scores = scores / len(keys)
    
    safe_name = args.model.replace("/", "_")
    output_filename = f"token_viz_{safe_name}_k{args.k}_alpha{args.alpha}_{args.topic}.html"
    output_path = FIGURES_DIR / output_filename
    
    title = f"Proyección Política - {args.topic.capitalize()} (alpha={args.alpha})"
    generate_html(token_strings, scores, output_path, title)
    
    print(f"Éxito! Visualización guardada en: {output_path}")

if __name__ == "__main__":
    main()
