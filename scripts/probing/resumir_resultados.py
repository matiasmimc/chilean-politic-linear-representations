import pickle
import pandas as pd
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"

def summarize_results():
    data = []
    
    # Definir los 4 directorios de experimentación
    configs = [
        ("Nominate", "Solo Nombres", RESULTS_DIR / "replication" / "v2_names_only"),
        ("Nominate", "Intervenciones", RESULTS_DIR / "replication" / "v2"),
        ("B-Call", "Solo Nombres", RESULTS_DIR / "probing" / "bcall" / "names_only"),
        ("B-Call", "Intervenciones", RESULTS_DIR / "probing" / "bcall" / "interventions")
    ]
    
    for metric, datatype, directory in configs:
        if not directory.exists():
            continue
            
        for filepath in directory.glob("*_performance.pkl"):
            # Extraer el nombre del modelo
            model_name = filepath.name.replace("_politician_performance.pkl", "").replace("_bcall_performance.pkl", "")
            
            with open(filepath, 'rb') as f:
                perf = pickle.load(f)
                
            if isinstance(perf, dict):
                # Formato B-Call (diccionario con d1 y d2)
                max_d1 = np.max(perf['d1'])
                max_d2 = np.max(perf['d2'])
                data.append({
                    "Modelo": model_name,
                    "Métrica": metric,
                    "Datos": datatype,
                    "Spearman D1 (Ideología)": round(max_d1, 4),
                    "Spearman D2 (Volatilidad)": round(max_d2, 4)
                })
            else:
                # Formato Nominate (array simple)
                max_perf = np.max(perf)
                data.append({
                    "Modelo": model_name,
                    "Métrica": metric,
                    "Datos": datatype,
                    "Spearman D1 (Ideología)": round(max_perf, 4),
                    "Spearman D2 (Volatilidad)": "N/A"
                })
                
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by=["Modelo", "Métrica", "Datos"])
        print("\n=== RESUMEN DE RESULTADOS DE PROBING ===\n")
        print(df.to_markdown(index=False))
        
        out_csv = RESULTS_DIR / "probing" / "resumen_probing.csv"
        out_md = RESULTS_DIR / "probing" / "resumen_probing.md"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        df.to_markdown(out_md, index=False)
        print(f"\nTablas guardadas en:\n- {out_csv}\n- {out_md}")
    else:
        print("No se encontraron resultados de probing en los directorios.")

if __name__ == "__main__":
    summarize_results()
