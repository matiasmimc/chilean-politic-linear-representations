import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Listando modelos disponibles...")
try:
    for model in client.models.list():
        print(f"ID: {model.name} | Display Name: {model.display_name}")
except Exception as e:
    print(f"Error al listar modelos: {e}")
