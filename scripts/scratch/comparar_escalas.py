import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, required=True, help="Tema a comparar (ej. aborto o inmigración)")
    args = parser.parse_args()
    
    topic = args.topic.replace(' ', '_')
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Asegúrate de que los nombres coincidan con los que tienes localmente
    file_old = BASE_DIR / f'results/steering/intervention_evaluated_{topic}.parquet'
    file_new = BASE_DIR / f'results/steering/intervention_evaluated_{topic}_dinamico.parquet'
    
    if not file_old.exists() or not file_new.exists():
        print(f"Error: Faltan archivos.")
        print(f"Buscando: \n - {file_old}\n - {file_new}")
        return
        
    print(f"Cargando resultados para el tema: {topic}...")
    df_old = pd.read_parquet(file_old)
    df_new = pd.read_parquet(file_new)
    
    # Agregar columna para distinguir la versión de evaluación
    df_old['Metodo_Evaluacion'] = 'Estática (Partidos)'
    df_new['Metodo_Evaluacion'] = 'Dinámica (Ideas por Tema)'
    
    # Combinar dataframes
    df = pd.concat([df_old, df_new], ignore_index=True)
    
    # Limpiar y convertir a numérico
    df['score_postura'] = pd.to_numeric(df['score_postura'], errors='coerce')
    df = df.dropna(subset=['score_postura'])
    
    # Filtrar
    df_slant = df[(df['alpha'] >= -30) & (df['alpha'] <= 30)]
    
    # Crear directorio para las figuras
    FIGURES_DIR = BASE_DIR / "figures" / "comparacion_escalas"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # GRAFICAR
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=df_slant, 
        x='alpha', 
        y='score_postura', 
        hue='Metodo_Evaluacion', 
        palette={'Estática (Partidos)': 'blue', 'Dinámica (Ideas por Tema)': 'red'},
        # style='model', 
        markers=True, 
        dashes=False, 
        errorbar='sd',
        err_style='bars', 
        err_kws={'capsize': 3}
    )
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Centro')
    plt.title(f'Comparación de Rúbricas de Evaluación LLM-as-a-Judge\nTema: {args.topic.capitalize()}', pad=20)
    plt.xlabel('Alpha (Fuerza del Vector de Steering)')
    plt.ylabel('Puntaje Ideológico (Juez LLM)')
    
    # Forzar el eje Y para que abarque al menos de -3 a 3 para ver ambos rangos 
    # (el estático era -3 a 3, el dinámico es -2 a 2)
    plt.ylim(-2, 2)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_file = FIGURES_DIR / f'comparacion_{topic}.png'
    plt.savefig(out_file, dpi=300)
    print(f"Gráfico comparativo guardado exitosamente en: {out_file}")

if __name__ == "__main__":
    main()
