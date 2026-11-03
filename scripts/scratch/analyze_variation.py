import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import numpy as np

BASE_DIR = Path('/home/matias/Documentos/chilean-politic-linear-representations')
FILE_PATH = BASE_DIR / 'intervention_evaluated.parquet'

print("Cargando datos...")
df = pd.read_parquet(FILE_PATH)

# Filtrar solo textos con coherencia razonable según la evaluación anterior
df_valid = df[df['score_coherence'] >= 3]

# Agrupar por dirección del steering
def get_direction(alpha):
    if alpha < -10: return "Izquierda (alpha < -10)"
    if alpha > 10: return "Derecha (alpha > 10)"
    return "Centro (-10 <= alpha <= 10)"

df_valid['steering_dir'] = df_valid['alpha'].apply(get_direction)

print(f"\nAnalizando {len(df_valid)} textos coherentes...")
print(df_valid['steering_dir'].value_counts())

# Análisis TF-IDF por grupo
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
# Nota: idealmente usaríamos stop_words en español, pero esto es un análisis rápido.
# Definimos algunas stop words básicas en español
spanish_stop_words = ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'del', 'las', 'un', 'por', 'con', 'no', 'una', 'su', 'para', 'es', 'como', 'más', 'o', 'pero', 'sus', 'le', 'ha', 'me', 'si', 'sin', 'sobre', 'este', 'ya', 'entre', 'cuando', 'todo', 'esta', 'ser', 'son', 'dos', 'también', 'fue', 'había', 'era', 'muy', 'años', 'hasta', 'desde', 'está', 'mi', 'porque', 'qué', 'solo', 'han', 'yo', 'hay', 'vez', 'puede', 'todos', 'así', 'nos', 'ni', 'parte', 'tiene', 'él', 'uno', 'donde', 'bien', 'tiempo', 'mismo', 'ese', 'ahora', 'cada', 'e', 'vida', 'otro', 'después', 'te', 'otros', 'aunque', 'esa', 'eso', 'hace', 'otra', 'creo', 'tema', 'respecto']

vectorizer = TfidfVectorizer(stop_words=spanish_stop_words, max_features=500, ngram_range=(1, 2))
X = vectorizer.fit_transform(df_valid['text'])
features = np.array(vectorizer.get_feature_names_out())

# Calcular el promedio de TF-IDF por grupo
for direction in ["Izquierda (alpha < -10)", "Centro (-10 <= alpha <= 10)", "Derecha (alpha > 10)"]:
    indices = df_valid[df_valid['steering_dir'] == direction].index
    # Mapear los índices originales a las filas de la matriz X
    row_mask = (df_valid['steering_dir'] == direction).values
    X_group = X[row_mask]
    
    mean_tfidf = np.array(X_group.mean(axis=0)).flatten()
    top_indices = mean_tfidf.argsort()[-15:][::-1]
    
    print(f"\n--- Top 15 términos para: {direction} ---")
    for idx in top_indices:
        print(f"  {features[idx]}: {mean_tfidf[idx]:.4f}")

# Calcular diferencia relativa (Derecha vs Izquierda)
print("\n--- Términos que más diferencian Derecha vs Izquierda ---")
mask_der = (df_valid['steering_dir'] == "Derecha (alpha > 10)").values
mask_izq = (df_valid['steering_dir'] == "Izquierda (alpha < -10)").values

mean_der = np.array(X[mask_der].mean(axis=0)).flatten()
mean_izq = np.array(X[mask_izq].mean(axis=0)).flatten()

diff = mean_der - mean_izq
# Top palabras más "Derecha"
top_der_idx = diff.argsort()[-10:][::-1]
print("Palabras más exclusivas hacia Alpha Positivo (Derecha):")
for idx in top_der_idx:
    print(f"  {features[idx]} (Diff: +{diff[idx]:.4f})")

# Top palabras más "Izquierda"
top_izq_idx = diff.argsort()[:10]
print("\nPalabras más exclusivas hacia Alpha Negativo (Izquierda):")
for idx in top_izq_idx:
    print(f"  {features[idx]} (Diff: {diff[idx]:.4f})")
