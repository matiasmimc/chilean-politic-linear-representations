import pandas as pd
import google.generativeai as genai
import os
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
import time

# Configuración de Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "congreso_2022_2026_biografias.csv"
OUTPUT_FILE = DATA_DIR / "congreso_2022_2026_con_resumen.csv"

# Configuración de API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def main():
    if not GEMINI_API_KEY:
        print("Error: La variable de entorno GEMINI_API_KEY no está configurada.")
        return

    if not INPUT_FILE.exists():
        print(f"Error: No se encuentra el archivo {INPUT_FILE}")
        return

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    df = pd.read_csv(INPUT_FILE)
    print(f"Procesando {len(df)} parlamentarios usando Gemini 1.5 Flash...")

    resumenes = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        nombre = row['Name']
        partido = row['party_1_P6']
        biografia_html = row['html_biografia']
        
        # Limpieza de HTML
        biografia_texto = BeautifulSoup(str(biografia_html), 'html.parser').get_text(separator=' ', strip=True)

        prompt = f"""
        A continuación te presento una biografía del diputado/a chileno/a {nombre} del partido {partido}. 
        Resúmela en una lista corta de puntos con características factuales (profesión, partido, gestiones principales). 
        Es crucial que no contenga apreciaciones subjetivas, adjetivos calificativos ni juicios de valor.
        
        Biografía:
        {biografia_texto[:10000]}
        """

        try:
            response = model.generate_content(prompt)
            resumen = response.text
            # Pequeño delay para no saturar el rate limit del free tier
            time.sleep(1) 
        except Exception as e:
            print(f"\nError procesando a {nombre}: {e}")
            resumen = None
            
        resumenes.append(resumen)

    df["resumen"] = resumenes
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Proceso completado. Resultados guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
