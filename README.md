# Representaciones Lineales de la Política Chilena

Este repositorio contiene el código y los datos para analizar y manipular la ideología política de los Modelos de Lenguaje Grande (LLMs) en el contexto de la política chilena.

## 📌 Descripción General

Los Modelos de Lenguaje Grande (como Llama 3, Gemma y Qwen) forman representaciones internas complejas de conceptos humanos, incluidas las ideologías políticas. Este proyecto utiliza la **Interpretabilidad Mecanicista** —un campo que mira dentro de los modelos de IA— para descubrir cómo estos representan el espectro político chileno.

En lugar de ajustar finamente (fine-tuning) o reentrenar los modelos, identificamos la "dirección política" interna (un vector lineal) dentro los parámetros del modelo. Al intervenir sobre este vector durante la generación de texto, podemos **guiar (steer)** dinámicamente las respuestas del modelo para que se inclinen más hacia alguna posición en varios temas locales (por ejemplo, inmigración, aborto, cambio climático).

## 🔍 Cómo Funciona

Nuestra metodología consta de cuatro pasos principales:

1. **Leyendo la Mente de la IA (Extracción)**: Alimentamos al modelo con declaraciones hechas por políticos chilenos reales y extraemos sus activaciones internas (cómo se activan las neuronas artificiales).
2. **Encontrando la Brújula Política**: Entrenamos modelos de extracción de características para encontrar la dirección matemática específica en el espacio interno del LLM que corresponde al espectro político chileno.
3. **Guiando la Salida (Intervención)**: Cuando el modelo genera texto nuevo, podemos "empujarlo" a lo largo de este vector político. Esto nos permite controlar la postura política de sus respuestas sobre la marcha.
4. **Evaluación**: Utilizamos la predicción de votos reales de diputados para evaluar si la intervención logra alterar genuinamente el comportamiento político del modelo, simulando cómo votaría bajo diferentes posiciones políticas.

## 📂 Estructura del Repositorio

- `data/`: Conjuntos de datos que contienen declaraciones, biografías y escalas ideológicas de políticos chilenos.
- `scripts/`: La canalización (pipeline) principal para extraer activaciones, entrenar clasificadores (probes), intervenir (guiar) los modelos y evaluar los resultados.
- `notebooks/`: Cuadernos (notebooks) de Jupyter para la exploración de datos, replicación de resultados y generación de visualizaciones.
- `results/`: Las salidas consolidadas de nuestros experimentos, incluidos los datos de intervención (steering) y las declaraciones evaluadas.
- `figures/`: Visualizaciones de nuestros hallazgos.
- `data/derived/`: Tablas intermedias pequeñas (`intervention_*.parquet`, `steering_summary_stats*.csv`).
- `docs/`: Informes y planes de experimentos.
- `scripts/legacy/` y `scripts/scratch/`: Versiones antiguas y scripts exploratorios (no mantenidos).
- `conf/`: Configuración (Hydra).

No versionados (ver `.gitignore`): `model/` (pesos de modelos), `results/`, `results_replication/`, `outputs/`, `logs/`, `.venv/`.
Las credenciales van en `.env` (plantilla en `.env.example`).