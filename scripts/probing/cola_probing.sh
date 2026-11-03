#!/bin/bash
# Probing robusto (CPU). Primero residual (rapido), luego atencion. Espera a la extraccion de controles para las otras condiciones.
cd "$(dirname "$0")/../.."
P="uv run python scripts/probing/probes_robustos.py"
echo "##### [$(date +%H:%M)] residual, condicion v2 (ya disponible)"
$P --reprs residual --conditions v2
echo "##### [$(date +%H:%M)] atencion p6, condicion v2"
$P --reprs attention --targets p6 --conditions v2
echo "##### [$(date +%H:%M)] esperando extraccion de controles"
until grep -q "EXTRACCION TERMINADA" logs/extraer_controles_probing.log 2>/dev/null; do sleep 60; done
echo "##### [$(date +%H:%M)] residual, todas las condiciones"
$P --reprs residual
echo "##### [$(date +%H:%M)] atencion p6, todas las condiciones"
$P --reprs attention --targets p6
echo "PROBING TERMINADO $(date)"
