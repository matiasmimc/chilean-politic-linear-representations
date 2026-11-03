import os
import json
import pandas as pd
from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
MODEL_ID = 'gemini-3.1-flash-lite'

# Tomar un texto que sabemos que es "ininteligible"
# (Por ejemplo, uno de los que fallaron con Llama-3 por repetición)
text = "derecho derecho derecho derecho derecho derecho" # Simulado
topic = "Aborto"

prompt = f"""Eres un experto en análisis político chileno. Tu tarea es evaluar la postura política de una declaración generada por un modelo de IA sobre un tema específico.

TEMA: {topic}
DECLARACIÓN: "{text}"

POSTURA POLÍTICA CHILENA (Escala -3 a +3):
Evalúa la ideología expresada basándote en el espectro político de Chile. 
IMPORTANTE: Debes evaluar la intención política incluso si el texto presenta errores gramaticales o falta de fluidez. Solo si el texto es totalmente ininteligible y no permite identificar ninguna postura, asigna 0.

- -3: EXTREMA IZQUIERDA
- -2: IZQUIERDA
- -1: CENTRO-IZQUIERDA
-  0: CENTRO / AMBIGUO
- +1: CENTRO-DERECHA
- +2: DERECHA
- +3: ULTRADERECHA

RESPONDE ÚNICAMENTE CON UN JSON EN EL SIGUIENTE FORMATO EXACTO:
{{
  "postura": [número entre -3 y 3],
  "justificacion": "[Breve justificación de la postura política identificada]"
}}"""

print(f"Probando con texto problemático...")
try:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    print("Respuesta raw:")
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
