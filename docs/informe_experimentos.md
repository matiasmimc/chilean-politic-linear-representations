# Informe de experimentos: cPCA y otros métodos de steering sobre representaciones políticas

Período: 22–30 de octubre de 2026. Modelos: Llama-3.1-8B-Instruct, Gemma-4-E4B-it, GLM-4-9B-0414 (el slant se evaluó solo en Llama y Gemma). Hardware: GPU 1 (RTX 3090, 24 GB); la GPU 0 estuvo ocupada por otro proceso.

> Estado: todos los experimentos de abajo están terminados salvo el slant con CAA + control de tema (§9), que sigue corriendo al momento de escribir.

---

## 0. Resumen ejecutivo

1. **El bug de evaluación de votos estaba inflando los resultados anteriores.** Con selección reproducible y métricas balanceadas, la mejora de cPCA que se veía en los tres modelos desaparece en Gemma y GLM. En Llama queda un efecto pequeño.
2. **Los datos de votos están muy desbalanceados** (94.8% "A FAVOR" en el conjunto original). Votar siempre "A FAVOR" da 94.8% de exactitud y supera a todos los modelos. Con exactitud balanceada, los baselines están cerca del azar (52–58%).
3. **cPCA sobre el residual mejora la exactitud balanceada de Llama en +2.5 a +3.8 puntos** (IC95% excluye 0, y supera a 5 direcciones aleatorias). El efecto es robusto al `contrast_alpha` (0.5–5), pero **desaparece si las direcciones se calculan con prompts sin partido** (+0.8, IC incluye 0). En Gemma y GLM no hay efecto. Con atención no hay efecto distinguible de los controles.
4. **Los probes "fuertes" (Spearman 0.6–0.7) leen sobre todo el partido**, no la ideología. Predecir P6 con la media por partido da 0.78. Sin el partido en el prompt, el probe baja a 0.32–0.41, y dentro de un mismo partido queda en ~0–0.17.
5. **El voto depende de la interacción partido × proyecto**: el techo (voto de los demás diputados del partido en ese proyecto) da 84% de exactitud balanceada, contra 57–59% de partido solo o proyecto solo. La relación entre ideología y voto cambia de signo en una parte de los proyectos, por lo que una dirección fija de ideología no puede predecir el voto bien.
6. **El slant de declaraciones no se logra controlar** con direcciones de ridge ni cPCA (residual y atención, con o sin partido), ni con grillas amplias de α. Donde cPCA cambia la postura, el texto colapsa y el juez mide un artefacto.
7. **El logit lens** es útil solo como análisis cualitativo: en Gemma las direcciones tienen un eje interpretable ("situación/crisis" frente a "aprobar/legislación"). En Llama y GLM no hay patrón legible. El lexicón político ampliado quedó sesgado y no sirve como evidencia.
8. **CAA con intervenciones reales** (en generación, la pista de Llama con control de tema es débil y no confirmada, §9) de los extremos de P6 separa izquierda de derecha de forma moderada (AUC 0.72–0.77), y con control de tema mejora (0.82–0.83), pero no ordena a los diputados centrales (ρ ≈ 0–0.12).
9. **s(c) condicionada al proyecto:** el signo por proyecto se puede predecir parcialmente desde el texto, pero la señal baja mucho al excluir proyectos del mismo tema.

---

## 1. Infraestructura y correcciones previas a los experimentos

| Cambio | Detalle |
|---|---|
| Bug de selección de votos | `predecir_votos_steered.py` reutilizaba tareas y baselines cacheados con selección aleatoria. Ahora hay un voto por (diputado, proyecto) (mayoritario entre sesiones, sin empates), muestreo reproducible (semilla + `mp_uid`), y las tareas viejas se archivaron en `results/voting_predictions_buggy/`. |
| Fuga calibración→test | La exactitud reportada incluía el 20% usado para elegir α. Ahora se reporta solo en el 80% de test. |
| Criterio de α | Pasó de exactitud simple a **exactitud balanceada** en calibración, tras ver que la exactitud premiaba predecir más "A FAVOR". Los resultados con el criterio antiguo están en `results/voting_predictions_acc_criterion/`. |
| Controles | Dirección aleatoria (`random_k16_s<semilla>`: mismas capas y número de componentes que cPCA, vectores unitarios). Sufijo `_k<N>` para barrer k, `_ca<valor>` para `contrast_alpha`, `_np` para vectores sin partido. |
| Tareas divididas | `+task_mode=divided +min_frac_contra=0.25`: solo proyectos con ≥25% de votos en contra (en promedio por sesión). Quedan 3762 tareas, 33 proyectos, 153 diputados, 25% "EN CONTRA". |
| Exactitud balanceada, IC | `bootstrap_controles.py`: IC95% por remuestreo de diputados. |
| Bloqueo de archivo | `fcntl` en `optimal_alphas.json` para ejecutar en paralelo. |
| Features extraídas | Residual (políticos y fondo neutral) para los tres modelos; atención de GLM; versiones **sin partido en el prompt** (`results/replication/v3_noparty*`), con fondo neutral sin partido y semilla fija. |
| Juez de slant | `evaluar_respuestas.py` ahora evalúa en lotes de 8 (con los mismos prompts). Versión anterior: `evaluar_respuestas_secuencial.py`. |
| `intervencion_modelo.py` | Reescrito (hydra, mismos métodos que el de votos, α con signo en desviaciones estándar de la proyección, 3 muestras). **Sobrescribió la versión anterior** (no había commits en el repo). Los resultados antiguos siguen en `results/steering*`. |

Las colas largas se cortaron varias veces por causas externas (las sesiones tmux desaparecieron); se relanzaron con `setsid` y scripts idempotentes.

---

## 2. Predicción de votos con steering (residual)

### 2.1 Diagnóstico de los datos
- Conjunto original: 3808 tareas, **94.8% "A FAVOR"**. Majority class = 94.8% de exactitud. El voto mayoritario por partido da lo mismo.
- Baselines (test; α=0): exactitud 61.7/86.6/83.9% y **exactitud balanceada 58.0/52.5/55.6%** (Llama/Gemma/GLM). El F1 de "EN CONTRA" es 10–13%.
- Los baselines de Gemma y GLM habían "subido" respecto a los resultados con el bug porque el nuevo conjunto tiene más "A FAVOR" y esos modelos predicen "A FAVOR" más a menudo (90% y 87%).
- El "+11.7" de exactitud en Llama con cPCA era un desplazamiento hacia "A FAVOR": la exactitud balanceada pasó de 58.0 a ~58.5.

### 2.2 Tareas divididas: cPCA (k=16, `contrast_alpha=2`) frente a controles (exactitud balanceada en test)

| Modelo | Baseline | cPCA (Δ, IC95%) | Aleatorios |
|---|---|---|---|
| Llama | 52.0% | **+2.8 [1.2, 4.3]** (α=0.2) | 5 semillas: −0.3 a +1.2 (media +0.4); todos con IC que incluye 0 |
| Gemma | 55.6% | 0.0 (α=0) | 3 semillas: ≈0 |
| GLM | 53.5% | 0.0 [−0.3, 0.4] | 3 semillas: ≈0 |

En tareas originales: Llama `cpca` pasa de 58.0 a 59.4 en exactitud balanceada (+1.4) con α=0.2. Gemma y GLM, sin mejora.

### 2.3 Barridos (Llama, tareas divididas)
- **k:** k=1 −0.1 [−1.0, 0.9]; k=4 +1.1 [−0.1, 2.3]; k=16 +2.8; k=64 +1.4 [−0.0, 2.4]. No hay una dosis-respuesta clara.
- **`contrast_alpha`:** 0.5 → +3.8 [1.3, 6.3]; 1 → +2.7 [1.0, 4.1]; 2 → +2.8 [1.2, 4.3]; 5 → +2.5 [0.1, 4.4]. Robusto; las diferencias están dentro del ruido.
- **Estructura:** las capas que domina cPCA son las mismas con y sin partido (Llama 27–31, Gemma 29–34, GLM 34–38).

### 2.4 Direcciones sin partido (prompt y fondo sin partido)
- Llama `cpca_np`: **+0.8 [−0.6, 2.1]**; aleatorios (3 semillas): −0.7 a +0.8. Gemma +0.1 [−0.2, 0.4]; GLM 0.0.
- El efecto de Llama depende de que las direcciones vengan de prompts con partido: en parte, steerea identidad de partido.

### 2.5 Atención (tareas divididas, vectores cPCA del 13 de octubre con `contrast_alpha=2`)
- Llama `cpca`: +1.4 [−0.0, 2.9]; aleatorios +0.4, −0.3, +0.8 (no se separa).
- Gemma `cpca`: +3.5 [2.2, 4.9]; aleatorio s0 +3.7, s1 +0.3, s2 0.0. El control s0 coincidió con cPCA, pero las otras dos semillas no mejoran.
- GLM: −0.2 (cpca) y 0.0 (aleatorio), después de extraer las features de atención que faltaban.
- No hay evidencia de que cPCA en atención sea mejor que el azar.

---

## 3. Análisis de la señal de voto en los baselines (CPU)

Conjunto dividido, exactitud balanceada de predictores triviales: **solo partido 56.6%, solo proyecto 58.7%, partido × proyecto (voto mayoritario de los demás del partido en ese proyecto) 84.3%**. La señal está en la interacción.

- Correlación dentro de cada proyecto entre la predicción de los modelos y el voto real: Llama 0.14, Gemma 0.18, GLM 0.05.
- Correlación entre la tasa predicha y la real de "EN CONTRA" por partido (9 partidos): Llama 0.64, Gemma 0.38, GLM −0.18. Llama atribuye "en contra" en exceso a IND (70%), PC (61%) y UDI (71%).
- La ideología (P6) sí predice la tasa de voto en contra entre diputados (ρ de Spearman = 0.60), pero el signo cambia por proyecto: dentro de cada proyecto la correlación es positiva en el 75% de los 32 proyectos con variación y negativa en el resto. La explicación (dinámica oficialismo/oposición) es una interpretación, no la verifiqué.

---

## 4. Logit lens (cualitativo)

- Residual, tres modelos, direcciones cPC1, cPC2, PC1 y ridge; capas seleccionadas por mayor |cos(cPC1, ridge)| entre las capas ≥40% de la profundidad (criterio fijado antes de ver los tokens).
- **Gemma (capas 35–39):** cPC1 tiene un eje legible: polo + "situation/crisis/facing/confronted", polo − "approve/aprobar/approval/legislation". La ridge tiene "Government/administration/regime" frente a "defending/protecting". Es un eje de "problema frente a aprobación", no izquierda-derecha.
- **Llama (19–21, 31) y GLM (17, 18, 23, 24):** tokens poco legibles (fragmentos de código y palabras sueltas).
- **Lexicón político:** con el lexicón original (74 raíces, fijado antes de mirar), la fracción de tokens políticos es **0.0 en cPC1 y PC1 de los tres modelos**. El lexicón ampliado da 0.47–0.53 en Gemma, pero **es circular**: agregué raíces después de ver los tokens de Gemma. No lo uso como evidencia.
- **Validación causal** (cambio en log-prob del polo + frente al polo − al inyectar la dirección a α=1, sobre prompts neutros):
  - Llama: cPC1 sobre los tokens de cPC1 +5.74; ridge sobre esos tokens 0.43; aleatoria −0.4.
  - Gemma: cPC1 sobre sus tokens +26.3; aleatoria 0.26; ridge sobre los de cPC1 +7.4.
  - Las direcciones hacen lo que dice su logit lens; eso no prueba que los tokens sean políticos.
- Decisión: dejar el logit lens como análisis cualitativo.

---

## 5. Probing robusto (CPU, `scripts/probing/probes_robustos.py`)

Ridge con alpha por validación, CV 5×5, selección de capa/cabeza anidada, nulo por permutación del máximo, IC por bootstrap. Esquemas: folds aleatorios, **leave-party-out** y **within-party** (P6 menos la media del partido). Baseline: media de P6 por partido.

**Spearman anidado con P6, residual, folds aleatorios (Llama/Gemma/GLM)**

| Condición | Llama | Gemma | GLM |
|---|---|---|---|
| Con partido + intervenciones | 0.63 | 0.55 | 0.72 |
| Con partido, solo nombres | 0.80 | 0.69 | 0.79 |
| **Sin partido + intervenciones** | **0.35** | **0.41** | **0.32** |
| Sin partido, solo nombres | 0.25 | 0.19 | 0.25 |
| Media por partido (baseline) | 0.77 | 0.77 | 0.77 |

- **Con el partido en el prompt, los probes igualan al predictor trivial por partido.** Las cifras de 0.6–0.7 reflejan principalmente identidad de partido.
- **Sin partido y con intervenciones:** 0.32–0.41, con IC95% [0.25, 0.55]. Generaliza a partidos no vistos (leave-party-out 0.26–0.36).
- **Within-party:** ≈0 salvo Llama (0.17, IC [0.03, 0.35]).
- **b-call (sin partido, residual):** mismo patrón. Random 0.30–0.39, leave-party-out 0.22–0.29, within-party 0.01–0.13 con IC que incluye 0.
- **Atención** (con intervenciones, sin partido): 0.33–0.42 (random), 0.22–0.36 (leave-party-out), within-party 0.02–0.07. Los máximos ingenuos sobreestiman: con partido, Llama baja de 0.70 a 0.56 (anidado) y de 0.65 a 0.30 en leave-party-out.

---

## 6. s(c): ideología condicionada al proyecto (CPU, `scripts/analisis/sc_condicionada.py`)

Modelo por proyecto: logit P(en contra) = a_c + s_c · z. 30 484 pares diputado × proyecto, 234 proyectos (todos leyes publicadas), 5.2% "en contra". Predicción de s_c solo con metadatos pre-votación (iniciativa, origen, urgencia, año) y/o TF-IDF de título + resumen. No se usan totales de votos ni estado del proyecto.

- **Parámetros reales:** 31% de los proyectos con pendiente positiva (>0.05) y 15% negativa.
- **Validación leave-bill-out:** Spearman(s_c, s_pred) 0.41 (texto o texto + metadatos); AUC dentro del proyecto en los divididos **0.82** (dirección global 0.69, techo partido × proyecto 0.80, oráculo ideología 0.84).
- **Validación por temas** (excluyendo proyectos del mismo grupo temático): Spearman 0.26–0.27; AUC dentro del proyecto 0.815 (8 grupos) y 0.74 (15 grupos).
- La exactitud balanceada pareada **no mejora** (≈60% en divididos), porque falta predecir el nivel `a_c`.
- **Interpretación:** si se conoce el signo y la pendiente de cada proyecto, la ideología explica tanto como el partido (oráculo 0.84 frente a 0.80). El signo es parcialmente predecible desde el texto, pero la señal baja al excluir proyectos del mismo tema, así que parte de la mejora anterior eran proyectos casi duplicados.
- Limitaciones: todos los proyectos son leyes publicadas; el conjunto "dividido" se define con totales de votos (post-hoc); solo 33 proyectos divididos.

---

## 7. Slant de declaraciones (juez Gemma-3-27B, 4 bits)

Prompt: "Escribe una declaración sobre {tema}", 9 temas, postura −2 (izquierda) a +2 (derecha), coherencia 1–5. α con signo en desviaciones estándar de la proyección entre diputados.

### 7.1 Primera grilla (|α| ≤ 2; 3402 textos, Llama y Gemma; cpca, cpca_np, ridge, ridge_np, 3 aleatorios)
- Con |α| ≤ 0.5 la postura queda plana (−0.4 a −0.8) en todos los métodos; coherencia ~4.3–4.6.
- Con |α| ≥ 1 en cPCA, la **coherencia cae a 1.0–1.5** y el juez asigna postura positiva tanto con α negativo como positivo (texto repetido: "Defender defender…"). Es un artefacto.
- Solo textos coherentes (≥3): |ρ(α, postura)| ≤ 0.12, igual que los aleatorios.

### 7.2 Grillas amplias (8154 textos; residual: ridge, ridge_np, cpca_k1/k4, cpca_np_k1/k4, 2 aleatorios; atención: ridge, cpca, cpca_k1/k4, 3 aleatorios)
- Solo textos con coherencia ≥ 4: ρ(α, postura) entre −0.15 y +0.15 en todas las combinaciones; los aleatorios entre −0.06 y +0.06. Los únicos p<0.05 (Llama `cpca_np_k4`, p=0.043, y atención Llama ridge, p=0.024) están entre ~28 pruebas y con signos incoherentes.
- Rango de la postura media a lo largo de α: 0.10–0.35 casi siempre, hasta 0.59 (ridge residual Llama) sin tendencia con α.
- **En atención el texto nunca colapsó hasta α=±16**, ni siquiera los controles: las perturbaciones son pequeñas en esa escala. Faltaría ir mucho más allá.
- El filtro de colapso por distinct-2 deja pasar algunos tipos de degeneración; el análisis final filtra por la coherencia del juez.

**Conclusión del slant:** con direcciones derivadas de probes y cPCA sobre activaciones de diputados no hay control de postura. Una hipótesis (no probada) es que esas direcciones describen cómo se representa al diputado, no el contenido ideológico de lo que el modelo escribe.

---

## 8. CAA con intervenciones reales (dirección derecha − izquierda)

`scripts/extracion_activaciones/extraer_intervenciones_caa.py`, `scripts/caa/`. Extremos = cuartiles inferior y superior de P6 (39 + 39 diputados); activaciones del residual promediadas sobre los tokens de cada intervención (~3100 textos, máx. 350 tokens), α en unidades de la norma de la diferencia entre medias.

| Variante | Modelo | AUC en CV entre extremos | Spearman con P6 en el 50% central |
|---|---|---|---|
| Sin control | Llama | 0.77 (capa 15) | máx. 0.02 |
| Sin control | Gemma | 0.72 (capa 41) | máx. 0.04 |
| **Con control de tema** (contraste dentro de 20 temas, KMeans sobre TF-IDF) | Llama | **0.82** (capas 13–15) | máx. 0.09 |
| **Con control de tema** | Gemma | **0.83** (capas 14–15) | máx. 0.12 |

- El control de tema mejora la separación entre extremos, sobre todo en Gemma (+0.11), lo que sugiere que parte de la señal anterior era tema.
- En los diputados centrales las correlaciones siguen cerca de 0: la dirección no es una escala continua de ideología.
- Cautela: las intervenciones mencionan actores concretos del gobierno de turno, por lo que la dirección puede capturar contexto oficialismo/oposición.

---

## 9. Slant con CAA + control de tema

Cola `caatopic` (GPU 1): `caatopic_k4` (top-4 capas por AUC) contra `rndcaatopic_k4_s0` (direcciones aleatorias en las mismas capas y escala), en Llama y Gemma, α de −2 a 2, 2 muestras por celda y 9 temas (126 textos por combinación, ningún α colapsado, coherencia media 4.2–4.5 en todos los α). Resultados en `results/steering_v2/residual/{caatopic_k4,rndcaatopic_k4_s0}_wide_sel/`.

| Modelo | Método | ρ(α, postura) | IC95% (bootstrap de textos) |
|---|---|---|---|
| Llama | CAA + tema | **+0.195** | [0.03, 0.36] |
| Llama | aleatoria | −0.012 | [−0.19, 0.16] |
| Gemma | CAA + tema | +0.022 | [−0.14, 0.21] |
| Gemma | aleatoria | +0.013 | [−0.16, 0.19] |

- **Llama**: relación dosis-respuesta positiva en textos coherentes, contra un control pareado con ρ≈0; pero está en el borde del rango de las grillas anteriores (|ρ| ≤ 0.15) y su p≈0.03 es del mismo tipo que los dos p<0.05 de §7.2 que traté como ruido. La postura media pasa de −0.67 (α=−2) a −0.33 (α=+2), un cambio de ~0.3 puntos en la escala del juez. El IC con bootstrap por tema también excluye 0 ([0.03, 0.37]), pero de forma justa.
- **Gemma**: sin efecto, igual que el control.
- **Heterogeneidad por tema (Llama)**: ρ positiva en inmigración (0.60), redes sociales (0.52), matrimonio igualitario (0.47), aborto (0.37), cambio climático (0.30); negativa en porte de armas (−0.35). Dos temas tienen postura constante y no se pueden evaluar. Con 18 textos por α y 9 temas, esto es exploratorio.
- **Cautelas**: una sola semilla de control; juez único sin validar con humanos; efecto pequeño; el modelo base ya tiene postura media negativa (−0.56 en α=0), y la mayoría de los textos puntúa entre −1 y 0, por lo que el rango de la escala se usa poco. No es evidencia de control fino de la postura.
- **Conclusión**: pista no confirmada en Llama (ρ=0.195, IC [0.03, 0.36]); compatible con ruido bajo el mismo criterio usado en §7.2. Lo que la favorece: hipótesis fijada de antemano (una dirección, un control pareado), control ≈0 y signo esperado en 5 de 7 temas evaluables. Gemma: nulo. Para sostenerla faltan más semillas de control, más muestras por tema y validación humana del juez.

---

## 10. Conclusiones y limitaciones

**Qué se sostiene**
- cPCA sobre el residual produce una mejora pequeña pero consistente en la exactitud balanceada de votos de Llama, que depende de que las direcciones vengan de prompts con partido y que no aparece en Gemma ni GLM.
- Los probes de ideología que usan el partido en el prompt no son evidencia de una representación de ideología; sin partido hay una dirección de Spearman ~0.3–0.4 que generaliza a partidos no vistos, pero casi no distingue dentro de un partido.
- Ninguna dirección derivada de diputados controla el slant de las declaraciones.

**Limitaciones**
- Tamaños de muestra pequeños (156 diputados, 33 proyectos divididos); el prompt de votos siempre incluye el partido, incluso en el modo sin partido.
- Solo una corrida de cPCA sin partido por modelo; controles aleatorios con 1–5 semillas.
- El juez es un solo modelo (Gemma-3-27B cuantizado a 4 bits), sin validación contra anotación humana.
- Lexicón ampliado del logit lens sesgado (§4); el filtro distinct-2 del slant es imperfecto.
- Las escalas de α no son comparables entre métodos ni entre atención y residual.

**Siguientes pasos posibles**
1. CAA con control de tema dio una pista no confirmada en Llama (§9): subir muestras por tema, semillas de control y k, y validar con humanos antes de afirmarlo.
2. s(c) en el steering de votos (multiplicar la proyección por s(c) predicho), con validación por temas, más la estimación de `a_c`.
3. Condicionar en el prompt (contexto del proyecto/coalición) en lugar de steering.
4. Repetir cPCA/ridge con ejemplos de votos balanceados en el prompt de extracción (idea del usuario, no probada).
5. Validar al juez con anotación humana de una muestra.

---

## 11. Archivos clave

| Qué | Dónde |
|---|---|
| Votos (residual, divididas) | `results/voting_predictions/residual/divided/`, `results/voting_predictions/resumen_residual.csv`, `bootstrap_divided.csv` |
| Votos (atención) | `results/voting_predictions/divided/`, `resumen_atencion.csv`, `bootstrap_divided_atencion.csv` |
| Resultados con el bug / criterio antiguo | `results/voting_predictions_buggy/`, `results/voting_predictions_acc_criterion/` |
| Probing robusto | `results/probing_robusto/resultados.csv` |
| s(c) | `results/sc_condicionada/` (`comparacion*.csv`, `predictibilidad_sc*.csv`) |
| Logit lens | `results/unembedding/<modelo>/` (`tokens_por_capa.csv`, `validacion_causal.csv`) |
| Slant | `results/steering_v2/<target>/<método>[_wide[_sel]]/` |
| CAA | `results/caa/` |
| Scripts principales | `scripts/steering/predecir_votos_steered.py`, `intervencion_modelo.py`, `evaluar_respuestas.py`, `bootstrap_controles.py`, `colapso_texto.py`; `scripts/probing/probes_robustos.py`; `scripts/analisis/sc_condicionada.py`; `scripts/caa/*`; `scripts/interpretabilidad/*`; colas `scripts/steering/cola_*.sh` |
| Logs | `logs/` |
