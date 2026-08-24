"""
Script para extraer intervenciones temáticas usando la librería bcn-scraper.
Permite descargar debates asociados a leyes/boletines específicos (ej. Seguridad, Pensiones, Salud)
y mapearlos a los 155 diputados de nuestro dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm

# Asegurar que bcn-scraper esté en el sys.path si no está instalado globalmente
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
BCN_SCRAPER_SRC = BASE_DIR.parent / "GitHub/bcn-scraper/src"

if BCN_SCRAPER_SRC.exists():
    sys.path.insert(0, str(BCN_SCRAPER_SRC))

try:
    from bcn_scraper.wrapper import historia_dataframe_from_ley_o_boletin
    from bcn_scraper.speech_dataframe import flatten_speech_content
    from bcn_scraper.historia_lookup import HistoriaLookup
except ImportError:
    print("⚠️ Advertencia: No se pudo importar bcn_scraper. Asegúrate de tener instalado el paquete.")

DATA_DIR = BASE_DIR / "data"
TEMAS_DIR = DATA_DIR / "temas"
TEMAS_DIR.mkdir(parents=True, exist_ok=True)

# Leyes emblemáticas del periodo 2022-2026 preconfiguradas por tema
EJEMPLOS_LEYES_POR_TEMA = {
    "seguridad": [
        {"numero_ley": "21560", "nombre": "Ley Naín-Retamal (Protección a Policías)"},
        {"numero_ley": "21577", "nombre": "Ley contra el Crimen Organizado"},
        {"numero_ley": "21633", "nombre": "Ley de Usurpaciones"},
    ],
    "pensiones": [
        {"numero_boletin": "15480-13", "nombre": "Reforma Previsional / Pensiones"},
    ],
    "trabajo": [
        {"numero_ley": "21561", "nombre": "Ley 40 Horas Laborales"},
        {"numero_ley": "21643", "nombre": "Ley Karin (Acoso Laboral)"},
    ],
    "salud": [
        {"numero_ley": "21674", "nombre": "Ley Corta de Isapres"},
    ]
}


def extraer_discursos_de_ley(numero_ley=None, numero_boletin=None):
    print(f"Descargando debate para Ley: {numero_ley} / Boletín: {numero_boletin}...")
    df_tramites = historia_dataframe_from_ley_o_boletin(
        numero_ley=numero_ley,
        numero_boletin=numero_boletin,
        clean_text=True,
        prefer_individual_tramites=True,
        fetch_akn=True
    )
    
    # Aplanar discursos individuales
    df_speeches = flatten_speech_content(df_tramites)
    return df_speeches


def main():
    parser = argparse.ArgumentParser(description="Extracción temática de intervenciones legislativas con bcn-scraper.")
    parser.add_argument("--tema", type=str, choices=list(EJEMPLOS_LEYES_POR_TEMA.keys()) + ["custom"], default="seguridad", help="Tema preconfigurado o 'custom'.")
    parser.add_argument("--numero-ley", type=str, default=None, help="Número de ley (para tema 'custom').")
    parser.add_argument("--numero-boletin", type=str, default=None, help="Número de boletín (para tema 'custom').")
    parser.add_argument("--output-name", type=str, default=None, help="Nombre del archivo parquet de salida.")
    args = parser.parse_args()

    # Cargar dataset de referencia de 155 diputados
    deputies_path = DATA_DIR / "congreso_2022_2026_20intervenciones.parquet"
    if not deputies_path.exists():
        print(f"Error: No se encuentra {deputies_path}")
        return
    df_deputies = pd.read_parquet(deputies_path)

    leyes_a_procesar = []
    if args.tema == "custom":
        if not args.numero_ley and not args.numero_boletin:
            print("Error: Para tema 'custom' debes especificar --numero-ley o --numero-boletin.")
            return
        leyes_a_procesar.append({"numero_ley": args.numero_ley, "numero_boletin": args.numero_boletin, "nombre": "Personalizada"})
    else:
        leyes_a_procesar = EJEMPLOS_LEYES_POR_TEMA[args.tema]

    all_speeches = []
    for item in leyes_a_procesar:
        num_ley = item.get("numero_ley")
        num_bol = item.get("numero_boletin")
        try:
            df_sp = extraer_discursos_de_ley(numero_ley=num_ley, numero_boletin=num_bol)
            df_sp["proyecto_nombre"] = item.get("nombre")
            all_speeches.append(df_sp)
            print(f"  -> Obtenidos {len(df_sp)} párrafos/intervenciones.")
        except Exception as e:
            print(f"⚠️ Error al procesar {item}: {e}")

    if not all_speeches:
        print("No se extrajo ninguna intervención.")
        return

    df_total_speeches = pd.concat(all_speeches, ignore_index=True)
    
    # Guardar discursos planos temáticos
    tema_nombre = args.output_name or f"intervenciones_{args.tema}"
    out_file = TEMAS_DIR / f"{tema_nombre}.parquet"
    df_total_speeches.to_parquet(out_file)
    print(f"\n✅ Total de {len(df_total_speeches)} intervenciones guardadas en: {out_file}")


if __name__ == "__main__":
    main()
