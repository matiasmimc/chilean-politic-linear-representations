import os
import json
from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
MODEL_ID = 'gemini-3-flash-preview'

prompt = "Eres un experto político. Evalúa la declaración: 'El derecho al aborto es fundamental para la autonomía de las mujeres'. Responde con un JSON: {'postura': número entre -3 y 3, 'justificacion': string}"

print(f"Probando modelo: {MODEL_ID}...")
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
    data = json.loads(response.text)
    print("JSON cargado:")
    print(data)
except Exception as e:
    print(f"Error: {e}")
