#!/bin/bash
set -e  # Detiene el script si algún comando falla

# Configurar para usar únicamente la GPU 2
export CUDA_VISIBLE_DEVICES=2

echo "Iniciando corrida secuencial de steering..."

echo "1/8: Ejecutando CPCA 90% varianza..."
# uv run python scripts/steering/predecir_votos_steered.py +steering_method=cpca_90var

echo "2/8: Ejecutando CNMF 90% varianza..."
# uv run python scripts/steering/predecir_votos_steered.py +steering_method=cnmf_90var

echo "3/8: Ejecutando CPCA top k..."
# uv run python scripts/steering/predecir_votos_steered.py +steering_method=cpca_topk

echo "4/8: Ejecutando CNMF top k..."
# uv run python scripts/steering/predecir_votos_steered.py +steering_method=cnmf_topk

echo "5/8: Ejecutando Ridge..."
# uv run python scripts/steering/predecir_votos_steered.py +steering_method=ridge

echo "6/8: Ejecutando BCall Joint..."
# uv run python scripts/steering/predecir_votos_steered.py +steering_method=bcall_joint

echo "7/8: Ejecutando BCall D1..."
uv run python scripts/steering/predecir_votos_steered.py +steering_method=bcall_d1

echo "8/8: Ejecutando BCall D2..."
uv run python scripts/steering/predecir_votos_steered.py +steering_method=bcall_d2

echo "¡Todas las corridas terminaron exitosamente!"