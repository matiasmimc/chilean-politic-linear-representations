#!/bin/bash

# Pipeline de ejecución: Representaciones Políticas
# Detiene la ejecución si algún comando falla
set -e

echo "--- Iniciando Pipeline ---"

# El token de HuggingFace se lee desde .env (ver .env.example)
set -a; [ -f .env ] && source .env; set +a
export CUDA_VISIBLE_DEVICES=1 # Forzar el uso de la GPU 1 para todo el proceso
# echo "1/2 Ejecutando obtener_resumenes.py..."
# python scripts/obtener_resumenes.py

# echo "2/2 Ejecutando extraer_activaciones.py..."
# python scripts/extracion_activaciones/extraer_activaciones.py

# echo "--- Pipeline Finalizado con Éxito ---"
# echo "Los resultados están en la carpeta results_replication/"

echo "Ejecutando intervencion_modelos.py..."
python scripts/steering/intervencion_modelo.py