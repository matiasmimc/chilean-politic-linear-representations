# Informe de experimentos: cPCA y otros métodos de steering sobre representaciones políticas

Período: 22–30 de octubre y 2–8 de noviembre de 2026. Modelos: Llama-3.1-8B-Instruct, Gemma-4-E4B-it, GLM-4-9B-0414 (el slant se evaluó en los tres; segundo juez Qwen2.5-7B en los tres). Hardware: GPU 1 (RTX 3090, 24 GB); la GPU 0 estuvo ocupada por otro proceso.

> Estado: todos los experimentos de abajo están terminados (§14.4, §15b y §15c se agregaron el 6–7 de noviembre). **Los resultados de §14 (slant v3) corrigen la conclusión de §7.2 y §9** de que ninguna dirección controla el slant; §10–§13 son análisis posteriores (CPU y GPU 1).

---

## 0. Resumen ejecutivo

1. **El bug de evaluación de votos estaba inflando los resultados anteriores.** Con selección reproducible y métricas balanceadas, la mejora de cPCA que se veía en los tres modelos desaparece en Gemma y GLM. En Llama queda un efecto pequeño.
2. **Los datos de votos están muy desbalanceados** (94.8% "A FAVOR" en el conjunto original). Votar siempre "A FAVOR" da 94.8% de exactitud y supera a todos los modelos. Con exactitud balanceada, los baselines están cerca del azar (52–58%).
3. **cPCA sobre el residual mejora la exactitud balanceada de Llama en +2.5 a +3.8 puntos** (IC95% excluye 0, y supera a 5 direcciones aleatorias). El efecto es robusto al `contrast_alpha` (0.5–5), pero **desaparece si las direcciones se calculan con prompts sin partido** (+0.8, IC incluye 0). En Gemma y GLM no hay efecto. Con atención no hay efecto distinguible de los controles.
4. **Los probes "fuertes" (Spearman 0.6–0.7) leen sobre todo el partido**, no la ideología. Predecir P6 con la media por partido da 0.78. Sin el partido en el prompt, el probe baja a 0.32–0.41, y dentro de un mismo partido queda en ~0–0.17.
5. **El voto depende de la interacción partido × proyecto**: el techo (voto de los demás diputados del partido en ese proyecto) da 84% de exactitud balanceada, contra 57–59% de partido solo o proyecto solo. La relación entre ideología y voto cambia de signo en una parte de los proyectos, por lo que una dirección fija de ideología no puede predecir el voto bien.
6. **El slant de declaraciones no se logra controlar** con direcciones de ridge ni cPCA derivadas de diputados (residual y atención, con o sin partido), ni con grillas amplias de α. Donde cPCA cambia la postura, el texto colapsa y el juez mide un artefacto. **Corrección (§14):** con CAA + control de tema (direcciones de declaraciones), más textos y una medida continua, la postura **sí** se controla de forma moderada en Llama (pendiente +0.12 [0.07, 0.18] por α con el juez, aleatorias en 0); en Gemma el efecto aparece en el logit pero no se distingue del ruido con el juez. **Con GLM y un segundo juez (§14.1) el resultado no se replica**: en GLM solo el juez Gemma muestra un efecto débil y el logit nada; con Qwen no hay dosis-respuesta en ningún modelo, y los jueces solo coinciden en textos extremos. Un clasificador entrenado en declaraciones reales (§14.2) sí detecta el efecto de `caatopic` en los tres modelos (aleatorias en 0).
7. **El logit lens** es útil solo como análisis cualitativo: en Gemma las direcciones tienen un eje interpretable ("situación/crisis" frente a "aprobar/legislación"). En Llama y GLM no hay patrón legible. El lexicón político ampliado quedó sesgado y no sirve como evidencia.
8. **CAA con intervenciones reales** (en generación, la pista de Llama con control de tema es débil y no confirmada, §9) de los extremos de P6 separa izquierda de derecha de forma moderada (AUC 0.72–0.77), y con control de tema mejora (0.82–0.83), pero no ordena a los diputados centrales (ρ ≈ 0–0.12).
9. **s(c) condicionada al proyecto:** el signo por proyecto se puede predecir parcialmente desde el texto, pero la señal baja mucho al excluir proyectos del mismo tema.
10. **Análisis posteriores (§10–§13).** Las proyecciones cPCA no aportan al voto sobre el partido (A2); la ganancia de cPCA en Llama es recalibración (A3). Por declaración la señal ideológica entre diputados es alta, pero dentro de partido es débil y solo b-call d1 sobrevive a quitar menciones de partido/gobierno (0.31–0.35). Ni la ablación ni el steering oráculo con el P6 verdadero muestran que el modelo use una dirección lineal de ideología para votar distinto dentro de partido (la ablación sin control positivo válido).
11. **`caatopic` consolidado (§14.4).** Con 10 direcciones aleatorias, 15 muestras por tema y dos clasificadores entrenados en texto real, `caatopic` mueve el slant en los tres modelos (pendiente 0.12–0.21 desviaciones por unidad de α con 4 capas; las aleatorias quedan en ±0.08). Crece con el número de capas; separando dosis de capas, en Llama y Gemma repartir en más capas ayuda y en GLM concentrar ayuda más. El efecto es modesto y no se confirmó con jueces. En voto (§13.1, solo Llama) mueve las decisiones entre partidos en el sentido esperado, pero no dentro de partido.
12. **EE.UU. frente a Chile (§15b).** Con la misma medida (logit del propio modelo), ridge controla el slant con claridad en EE.UU. en los tres modelos (pendiente 1.0–1.8 con |α| ≤ 2), unas 5 veces más que en Chile en Llama y GLM (1.8 frente a 0.4; 1.0 frente a 0.2) y casi igual en Gemma (1.0 frente a 0.8); cPCA es débil o inconsistente en ambos países; las aleatorias dan ≈ 0. El efecto solo aparece con α pequeños: con α grandes los textos colapsan (30–60 %). Es menos marcado que lo descrito en el paper original.
13. **Casas de noticias (§15c).** Las direcciones aprendidas con diputados ordenan las 31 casas de noticias chilenas de forma estable (split-half 0.85–0.95; acuerdo entre modelos y métodos, Spearman mediano 0.78), con un orden plausible (izquierda: radionuevomundo, eldesconcierto, cronicadigital; derecha: emol, df, duna). No hay puntaje externo y el análisis es cualitativo. Reparos: TF-IDF ordena casi igual (0.90), y con control de tema el ranking cambia bastante (0.44).

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

**Conclusión del slant (corregida en §14: con direcciones de declaraciones y CAA sí hay control en Llama):** con direcciones derivadas de probes y cPCA sobre activaciones de diputados no hay control de postura. Una hipótesis (no probada) es que esas direcciones describen cómo se representa al diputado, no el contenido ideológico de lo que el modelo escribe.

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
- **Conclusión (confirmada y ampliada en §14)**: pista no confirmada en Llama (ρ=0.195, IC [0.03, 0.36]); compatible con ruido bajo el mismo criterio usado en §7.2. Lo que la favorece: hipótesis fijada de antemano (una dirección, un control pareado), control ≈0 y signo esperado en 5 de 7 temas evaluables. Gemma: nulo. Para sostenerla faltan más semillas de control, más muestras por tema y validación humana del juez.

---

## 10. cPCA dentro de partido, proyecciones como features y desglose de la ganancia (CPU, 2–3 nov)

Scripts: `scripts/analisis/cpca_dentro_partido.py` (A1), `cpca_features_votos.py` (A2), `desglose_ganancia_cpca.py` (A3).

**A1 — ¿la señal de cPCA distingue diputados dentro de un partido?** Se compara cPCA con fondo neutro, PCA, PCA y cPCA con activaciones centradas por partido, y direcciones aleatorias, con ridge sobre las proyecciones (CV 5×5, k ∈ {1, 4, 16}) y objetivo centrado por partido. En Llama (condición v2) el mejor Spearman dentro de partido es 0.35 (b-call, cPCA k=4), 0.31 (PCA k=4) y 0.25 (PCA within k=4); los aleatorios llegan a 0.13–0.23. Con P6 el máximo es 0.17–0.26 en k=16, contra 0.01–0.13 de los aleatorios. cPCA no supera a PCA, y quitar el partido del prompt (v3_noparty) no cambia el orden. **Los p de permutación de b-call no son confiables** (direcciones aleatorias también salen significativas).

**A2 — ¿las proyecciones cPCA aportan al voto más allá del partido?** Regresión logística única (`class_weight="balanced"`) sobre todos los pares (diputado, proyecto) de proyectos divididos (4227 pares, 150 diputados, 33 proyectos); features: one-hot de proyecto y de partido × proyecto, más z × one-hot(proyecto) (una pendiente por proyecto; z = proyección cPCA o P6 estandarizado). Validación leave-MP-out (10 folds) y leave-bill-out (11 folds). Llama, direcciones con partido (exactitud balanceada / AUC medio dentro de proyecto):

| Features | leave-MP-out (techo + pendiente) | leave-bill-out (partido + z) |
|---|---|---|
| Techo partido × proyecto (sin z) | 85.4 / 0.787 | 63.2 / 0.701 (partido solo) |
| + proyección cPCA k=1 | 83.4 / 0.783 | 62.9 / 0.700 |
| + proyección aleatoria k=1 | 83.5 / 0.766 | 63.1 / 0.701 |
| + P6 verdadero (oráculo) | 88.4 / 0.847 | 64.3 / 0.718 |

Las proyecciones cPCA no aportan nada sobre partido; la proyección aleatoria queda por debajo del techo por sobreajuste de las 33 pendientes (el control justo para cPCA es el aleatorio). Ni el P6 verdadero mejora mucho: +3.0 puntos con el techo conocido, ~1 punto en un proyecto nuevo. Una sola corrida, sin IC.

**A3 — ¿de dónde viene la ganancia de cPCA en Llama?** Con el α óptimo, la exactitud balanceada de Llama sube de 52.0 a 54.8 (+2.8); `cpca_np` +0.8; Gemma y GLM ≈ 0. El 94% de los cambios de voto de cPCA en Llama pasan de EN CONTRA a A FAVOR (la exactitud de esos cambios sube de 18.6% a 81.4%) y el 52% de la varianza del cambio es entre celdas partido × proyecto. Es decir, **la ganancia es sobre todo recalibración de un exceso de EN CONTRA** a nivel de partido/proyecto, no predicción entre diputados.

---

## 11. Probes por declaración y control de tema (CPU)

Scripts: `scripts/probing/probes_por_intervencion.py`, `probes_control_tema.py`. Un ridge sobre la activación de **cada declaración** (~3100; media sobre tokens, sin nombre ni partido) con folds de diputados completos; la predicción se promedia por diputado. Esquemas: `random` (5 folds de diputados), `lpo` (leave-party-out) y `within` (objetivo y predicción centrados por partido). Objetivos: P6, b-call d1/d2, `gov_apoyo` (fracción de A FAVOR en proyectos "Mensaje" divididos). p de permutación sobre el máximo en capas × α (200 permutaciones).

**Fuente `anon` (texto solo).** Entre diputados la señal es alta (Llama: random 0.83–0.84, lpo ~0.77) y dentro de partido queda: b-call d1 Llama 0.38 (p=.005), Gemma 0.36 (p=.005); P6 0.23 (p=.09) y 0.27 (p=.04); `gov_apoyo` 0.21 y 0.19 (n.s.).

**Fuente `decl_v2` (prompt con nombre y partido, último token), 3 modelos.** P6 random 0.77–0.84, lpo 0.70–0.78; dentro de partido 0.31 (Llama, p=.03) y 0.12–0.25 con p>.15 en el resto, salvo `gov_apoyo` Gemma 0.32 (p=.02). La señal dentro de partido es frágil y no replica entre modelos.

**Control de tema (anon; Spearman máximo, p entre paréntesis).**

| Modelo | Variante | P6 lpo | P6 within | b-call d1 lpo | b-call d1 within |
|---|---|---|---|---|---|
| Llama | base | 0.77 | 0.23 (.09) | 0.72 | **0.38 (.005)** |
| | vocabulario (TF-IDF→SVD 300) | 0.71 | 0.19 (.04) | 0.64 | 0.12 (.075) |
| | tema restado (50 dim) | 0.21 (n.s.) | 0.12 (n.s.) | 0.22 (n.s.) | 0.22 (n.s.) |
| | sin declaraciones que nombran partido/gobierno (44%) | 0.59 | 0.10 | 0.49 | **0.35 (.01)** |
| Gemma | base | 0.77 | 0.27 (.04) | 0.69 | **0.36 (.005)** |
| | vocabulario | 0.71 | 0.19 (.04) | 0.64 | 0.12 (.075) |
| | tema restado | 0.09 (n.s.) | 0.21 (n.s.) | 0.06 (n.s.) | 0.21 (n.s.) |
| | sin partido/gobierno | 0.53 | 0.20 (n.s.) | 0.47 | **0.31 (.04)** |

- El vocabulario solo explica gran parte de la señal entre partidos (0.64–0.71) pero muy poco dentro de partido (0.12–0.19).
- Lo único que sobrevive a quitar menciones de partido/gobierno dentro de partido es **b-call d1** (0.35 y 0.31, p ≤ .04) en ambos modelos; no replica para P6 ni `gov_apoyo`.
- **Restar el tema por regresión lineal destruye hasta la señal entre partidos** (Spearman medio negativo): tema y partido están muy mezclados, y no usaría esa variante para concluir.

---

## 12. Ablación de direcciones (Llama, GPU 1)

Scripts: `scripts/steering/ablacion_direcciones.py` (v1, rango 1), `ablacion_v2.py`. Se proyecta fuera un subespacio en capas 12–16 (medias), 26–30 (tardías) o 6–29 (amplias) en todas las posiciones, y se mide sin generar: AUC de voto dentro de proyecto (800 tareas, prompt con "VOTO:"), AUC de clasificación izquierda/derecha de textos reales (400) y **brecha de persona** (diferencia de logit EN CONTRA − A FAVOR entre "Eres un diputado chileno de derecha" y "de izquierda", 400 tareas). Direcciones: INLP del ridge a P6 sobre declaraciones, cPCA, persona (derecha − izquierda) y aleatorias.

- v1 (rango 1): nulo en todas las direcciones.
- v2: base AUC voto 0.689, AUC texto 0.788, brecha 0.606. **El control positivo falla:** ablar la dirección de persona de rango 1 no reduce la brecha (×1.58 en capas medias, ×1.04 en tardías).
- Ninguna dirección de ideología reduce la brecha de forma distinguible del azar: aleatorias de rango 8 en capas amplias dan ×0.47–1.19. cPCA k=64 en capas medias la aumenta (×2.7), más bien daño al modelo.
- El AUC de voto queda entre 0.68 y 0.71 en todas las condiciones.
- El AUC de texto cae solo con subespacios grandes: INLP k=8 amplias 0.73, INLP k=64 medias 0.75, cPCA k=64 medias 0.74, contra ~0.79 de los aleatorios.

**Lectura:** no hay evidencia de que estas direcciones sean necesarias para el voto, pero sin control positivo el nulo no es concluyente. Sí hay evidencia de que ablar subespacios de ideología degrada la lectura de ideología en texto.

---

## 13. Steering oráculo en voto (Llama, GPU 1)

Script: `scripts/steering/oraculo_steering.py`. A cada tarea de voto (3687 tareas, 150 diputados, 647 celdas partido × proyecto) se suma en las capas 12–16 `α · z · σ_l · v_l`, con z el **P6 verdadero** del diputado (estandarizado completo o centrado por partido), σ_l la desviación entre diputados de la proyección media de sus declaraciones, y v_l la dirección (INLP a P6, persona, o aleatoria), α = 10 y 40. Métrica: AUC dentro de celdas partido × proyecto (101 con ambas clases y ≥ 4 diputados).

- Referencias: el modelo sin steering da AUC 0.482 (azar: con nombre y partido no distingue diputados dentro de la bancada); el P6 verdadero solo, sin modelo, da 0.598.
- Con el oráculo, el AUC del cambio de score orientado queda en 0.42–0.48 para INLP y 0.45–0.51 para persona y la aleatoria; ninguna supera 0.51. **El score se mueve con z incluso con una dirección aleatoria** (correlación −0.36 a −0.68): es un efecto genérico de la perturbación. A α=40 el modelo se degrada (AUC dentro de proyecto de 0.65 a 0.47–0.60; la persona empuja el score hacia A FAVOR).
- **Nulo:** con el mejor caso posible, no hay indicios de que el modelo use una dirección lineal de ideología para votar distinto dentro de partido. Solo Llama, una semilla aleatoria, sin IC.

### 13.1 Steering de voto con `caatopic` (Llama, 6 nov)

`scripts/steering/voto_caatopic.py`. La dirección que sí mueve el slant (§14), sumada con el mismo α a **todos** los diputados (sin P6 individual) en las capas 8, 13, 14, 15 y en todos los tokens del prompt de voto (α en unidades de `diff_norm` de CAA por capa; α = ±2); `persona` y 3 aleatorias en las mismas capas y escala; 3.762 votos. Medida: pendiente de (Δ logit EN CONTRA)/α sobre la orientación de la celda partido × proyecto, r = correlación entre P6 y voto EN CONTRA en **otros partidos** del mismo proyecto (>0: la derecha vota más EN CONTRA). IC95% por bootstrap de proyectos.

| Dirección | α = +2: β [IC] | α = −2: β [IC] | cambios de voto |
|---|---|---|---|
| caatopic | **+0.154 [0.059, 0.253]** | **+0.221 [0.156, 0.299]** | 6–8% |
| persona | **+0.271 [0.163, 0.396]** | **+0.309 [0.204, 0.430]** | 11–15% |
| aleatoria s0 | +0.008 [−0.055, 0.080] | −0.003 [−0.080, 0.084] | 5–6% |
| aleatoria s1 | −0.127 [−0.291, 0.012] | −0.150 [−0.322, −0.009] | 12–14% |
| aleatoria s2 | −0.063 [−0.220, 0.084] | −0.106 [−0.283, 0.059] | 14–16% |

(β se mide por unidad de α, así que el mismo signo con ±α es lo esperado si el efecto es lineal.) AUC dentro de celda partido × proyecto: 0.48 base y 0.46–0.49 con steering (no se espera mejora: el mismo desplazamiento para todos).

**Lectura.** Por primera vez una dirección derivada de declaraciones mueve el voto del modelo **en el sentido político esperado**: donde en los otros partidos la derecha vota EN CONTRA, `caatopic` (derecha) sube el logit de EN CONTRA; `persona` lo hace con más fuerza pero con más cambios de voto. Las aleatorias quedan entre −0.15 y +0.01, fuera del rango de `caatopic` y `persona`. Es un efecto funcional pequeño (≈ 0.15–0.3 logits por unidad de α, frente a un cambio medio absoluto de 0.33–0.50) y entre partidos; no dice nada sobre la señal individual dentro de partido (§13), y la ausencia de mejora de AUC por celda es esperable por construcción. Pendiente: replicar en Gemma/GLM y con más aleatorias (una de ellas da un efecto negativo cuyo IC casi excluye 0, así que el rango nulo es ancho).

---

## 14. Slant v3: direcciones por declaración, control positivo y versión personal (Llama, Gemma y GLM)

**Corrige §7.2 y §9**, que concluían que ninguna dirección controlaba el slant. Scripts: `scripts/steering/slant_v3.py`, `juez_rapido.py`, `scripts/analisis/slant_v3_analisis.py`; colas `cola_slant_v3_gemma.sh`.

**Diseño.** 20 temas × 5 muestras × α ∈ {−2, −1, 0, 1, 2} (en unidades de `diff_norm` de CAA por capa), suma al último token de cada paso en las 4 capas top por AUC de CAA (Llama 8, 13, 14, 15; Gemma 14–17), 160 tokens. Direcciones: `caatopic` (CAA con control de tema), `within` (ridge a P6 con activaciones y P6 centrados por partido), `persona` (derecha − izquierda) y 3 aleatorias en las mismas capas y escala. Controles: (a) **positivo**, prompt "Eres un diputado chileno de izquierda/derecha" sin steering; (b) **personal**: α = 1.5 · z_dentro_de_partido (recortado a ±2) por diputado, 156 diputados, 8 temas generados (el juez evaluó 3 de ellos). Dos medidas: `slant_logit` (logit derecha − izquierda del mismo modelo, sin steering, sobre el texto generado) y postura del juez (Gemma-3-27B, 4 bits) **solo en textos con coherencia ≥ 4**. El juez rápido usa los mismos prompts y escala que en §7 pero responde solo el número (el original tardaba ~10 s por texto). Ninguna condición colapsó (distinct-2 ≈ 0.94 en Llama, 0.97 en Gemma). IC95% por bootstrap de temas (diputados en personal), efectos fijos por tema.

**Dosis-respuesta (pendiente por unidad de α).**

| Dirección | Llama, juez | Llama, logit | Gemma, juez | Gemma, logit |
|---|---|---|---|---|
| caatopic | **+0.124 [0.074, 0.178]** | **+0.43 [0.25, 0.62]** | +0.054 [−0.003, 0.104] | **+0.40 [0.20, 0.61]** |
| persona | +0.060 [0.021, 0.098] | **+0.39 [0.23, 0.53]** | +0.054 [−0.001, 0.103] | **+0.54 [0.33, 0.74]** |
| within | −0.028 [−0.051, −0.007] | −0.15 [−0.23, −0.07] | −0.028 [−0.058, 0.001] | −0.13 [−0.30, 0.04] |
| aleatorias (3) | −0.008, −0.006, −0.007 | −0.05, +0.02, −0.06 (IC con 0) | +0.035, +0.009, −0.058 | −0.12, −0.08, +0.01 (IC con 0) |

**Control positivo (prompt derecha frente a izquierda).** Llama: AUC 0.85 con el juez (diferencia 1.3 puntos) y 0.90 con el logit. Gemma: AUC 1.0 con el juez (3.2 puntos) y 0.94 con el logit. El efecto de `caatopic` es ~40% del efecto de un prompt de persona en Llama y ~7% en Gemma (juez).

**Versión personal (pendiente de la postura sobre z dentro de partido; 3 temas, juez).** Llama: caatopic +0.31 [0.24, 0.38] (Spearman 0.44), persona +0.25 [0.20, 0.29], within +0.06 [0.005, 0.11], aleatoria +0.02 [−0.04, 0.07]. Gemma: caatopic +0.11 [0.055, 0.159], persona +0.088 [0.037, 0.141], within +0.098 [0.028, 0.168], aleatoria +0.028 [−0.03, 0.08]. Con logit y los 8 temas, Llama caatopic +1.16 [1.04, 1.28], within +0.11 [−0.02, 0.23]; Gemma caatopic +1.19 [0.96, 1.41], within +0.34 [0.10, 0.60].

**Lectura.**
- En **Llama** la postura del texto se controla de forma continua con CAA + control de tema (y, más débil, con la dirección de persona), con las aleatorias en cero y con dos medidas independientes (el juez es otro modelo). Esto se contradice con la conclusión previa porque ahora hay más textos, un control positivo y una medida continua.
- En **Gemma** el efecto aparece en el logit, pero con el juez cae al borde del ruido: dos de las tres aleatorias tienen un IC que excluye 0 y su rango (−0.06 a +0.04) es el mismo que el de `caatopic`. Es consistente con el nulo de §9. Resultado débil o no concluyente.
- La dirección `within` (ridge intrapartido), la que leen los probes dentro de partido, **no** mueve el slant (en Llama) o lo mueve poco; lo que controla el texto es la dirección izquierda/derecha entre diputados con control de tema.
- La versión personal usa la misma dirección con un coeficiente proporcional al P6; no demuestra que el modelo codifique al diputado individual, solo que la postura es controlable de forma continua.

**Cautelas.** Un solo juez y una sola semilla de generación por celda; la medida de logit usa el mismo modelo que se interviene; el juez rápido no entrega justificación; 11 de los 20 temas no están en `escalas_politicas_por_tema.json` y usan la caracterización por defecto; el juez evaluó 3 de 8 temas en personal; en Gemma el logit del control positivo tiene una escala extrema (media −10, desviación 8.5) y solo se interpreta en signo y orden.

### 14.1 GLM-4-9B y segundo juez (Qwen2.5-7B-Instruct) — 5 nov

Mismo diseño que arriba en GLM-4-9B-0414 (capas top por AUC de CAA ≈ 14–17; `cola_slant_v3_glm.sh`). Segundo juez independiente: Qwen2.5-7B-Instruct, mismos prompts, sobre los tres modelos (`__qwen`). Hoja ciega de 120 textos para anotación humana en `results/slant_v3_validacion/anotacion_humana.csv` (**omitida**: la anotación requeriría criterio de politólogo; se reemplazó por el clasificador de §14.2).

**GLM, dosis-respuesta (pendiente por unidad de α).**

| Dirección | juez Gemma | logit |
|---|---|---|
| caatopic | +0.057 [0.025, 0.095] | +0.03 [−0.05, 0.12] |
| persona | +0.033 [0.014, 0.058] | +0.02 [−0.07, 0.12] |
| within | +0.018 [−0.007, 0.041] | −0.02 [−0.12, 0.06] |
| aleatorias (3) | +0.009, +0.003, +0.023 (una con IC [0.002, 0.043]) | −0.07, −0.06, +0.02 (IC con 0) |

Control positivo: AUC 0.98 con el juez (2.2 puntos de diferencia) y 0.87 con el logit (5.2 puntos). Personal (3 temas, juez): caatopic +0.21 [0.16, 0.26] (Spearman 0.35), persona +0.11, within +0.09 [0.04, 0.14], aleatoria +0.01 [−0.04, 0.07]. Con logit: caatopic +0.22 [0.10, 0.35], el resto con IC en 0.

**Segundo juez (Qwen).** Dosis-respuesta: ninguna dirección distinta de las aleatorias en los tres modelos (Llama caatopic +0.016 [−0.03, 0.06]; Gemma +0.034 [−0.02, 0.10]; GLM +0.018 [−0.01, 0.05]; persona en GLM +0.056 [0.003, 0.115], pero una aleatoria sale con −0.025 [−0.046, −0.003]). Control positivo con Qwen: AUC 0.75 (Llama), 0.95 (Gemma), 0.90 (GLM). Personal con Qwen: solo Llama caatopic +0.095 [0.024, 0.165]; Gemma y GLM ≈ 0.

**Acuerdo entre jueces por texto (Spearman de la postura).** Solo concuerdan en los textos extremos del control positivo (0.66 Llama, 0.86 Gemma, 0.81 GLM). En los textos de dosis es bajo (0.30, 0.20, 0.05) y en personal ≈ 0 o negativo (0.18, −0.14, −0.17). La coherencia concuerda mejor (0.2–0.7). Qwen además tiene otra escala de base (media +0.1 a +0.24 contra −0.6 en Gemma/GLM).

**Lectura.**
- **GLM no replica a Llama:** con el logit no hay dosis-respuesta, y con el juez Gemma el efecto de `caatopic` es la mitad que en Llama y se acerca a las aleatorias. La versión personal sí separa `caatopic` del control en el juez, pero la dosis-respuesta de fondo es débil, así que se parece más a una dosis aplicada con otro mapa que a evidencia de codificación individual.
- **El resultado de Llama (§14) no está validado por un segundo juez.** Con Qwen la dosis-respuesta de Llama desaparece. Como los jueces solo coinciden en textos muy marcados, la diferencia puede ser ruido de medición de Qwen (7B) en posturas sutiles y no un efecto inexistente; con la información actual no se puede decidir cuál juez es más válido. La anotación humana es la que lo resolvería.
- Sigue en pie, con el logit del propio modelo en Llama y Gemma, que `caatopic` y `persona` mueven el slant. Pero el logit no se replica en GLM y no es independiente del modelo intervenido.
- Corrección a la redacción de §0 y §16: el slant controlable es un resultado de Llama (juez Gemma y logit), parcial en Gemma (solo logit) y no concluyente en GLM y con el juez Qwen.

### 14.2 Tercera medida: clasificador entrenado en texto real — 5 nov

`scripts/analisis/clasificador_texto_real.py`. Ridge sobre TF-IDF (1–2-gramas) de las 3.089 declaraciones reales (20 por diputado), objetivo P6, sin menciones a partidos/gobierno. Es independiente de los dos jueces y de los modelos intervenidos. Validación por diputado (5 folds): Spearman por declaración 0.42, AUC derecha vs izquierda en terciles extremos 0.77, Spearman por diputado 0.71, **dentro de partido solo 0.11** (no distingue individuos, solo posición entre partidos). Puntaje estandarizado sobre todos los textos generados (sin filtro de coherencia).

| Dirección (pendiente por α, puntaje z) | Llama | Gemma | GLM |
|---|---|---|---|
| caatopic | **+0.198 [0.144, 0.249]** | **+0.105 [0.059, 0.148]** | **+0.155 [0.105, 0.201]** |
| persona | +0.101 [0.055, 0.142] | −0.018 [−0.057, 0.020] | +0.039 [−0.010, 0.091] |
| within | +0.029 [−0.014, 0.084] | +0.017 [−0.022, 0.054] | +0.023 [−0.016, 0.068] |
| aleatorias (3) | +0.002, +0.033, −0.006 | +0.022, −0.013, −0.008 | −0.002, −0.002, +0.017 |

Control positivo (prompt derecha frente a izquierda): AUC 0.63 (Llama), 0.72 (Gemma), 0.70 (GLM); la dirección es correcta pero la separación es modesta, porque el clasificador se entrenó con discursos parlamentarios y los textos generados son más cortos y genéricos. Personal (3 temas, pendiente sobre z dentro de partido): caatopic +0.66 [0.58, 0.74] (Llama), +0.32 [0.22, 0.44] (Gemma), +0.53 [0.42, 0.66] (GLM); aleatoria ≈ 0; within +0.08, +0.02, +0.13.

**Lectura.**
- **`caatopic` mueve la postura en los tres modelos con una medida que no depende de un juez LLM ni del modelo intervenido**, con aleatorias en 0. Esto respalda el resultado de Llama en §14, resuelve a favor del juez Gemma y contra Qwen en el efecto de Llama, y da un resultado positivo en GLM que el logit y el juez no mostraban claramente.
- La dirección `persona` solo se distingue en Llama; `within` no mueve nada, como antes.
- El efecto de `caatopic` en la versión personal sigue siendo una dosis con otro mapa de α: el clasificador mismo no ve diferencias entre diputados de un mismo partido (0.11), así que no sirve para afirmar codificación individual.

**Repetición con embeddings (multilingual-e5-large, ridge, `scripts/analisis/clasificador_embeddings.py`).** Para descartar que el efecto sea solo vocabulario. Validación por diputado: AUC de terciles 0.68, Spearman por diputado 0.61 y dentro de partido 0.09, peor que el TF-IDF en texto real (el mejor alpha quedó en el borde de la grilla, α = 1).

| Dirección (pendiente por α, puntaje z) | Llama | Gemma | GLM |
|---|---|---|---|
| caatopic | **+0.156 [0.111, 0.202]** | **+0.134 [0.091, 0.174]** | **+0.173 [0.144, 0.203]** |
| persona | +0.065 [0.023, 0.100] | +0.083 [0.047, 0.122] | +0.045 [−0.004, 0.091] |
| within | +0.013 | −0.030 | −0.024 |
| aleatorias (3) | −0.032 [−0.059, −0.009], +0.010, +0.015 | 0.000, −0.018, +0.055 [0.013, 0.101] | −0.023, −0.028, +0.003 |

Control positivo: AUC 0.64, 0.76, 0.65. Personal (caatopic): +0.72, +0.25, +0.41; en Gemma una aleatoria da +0.16 [0.07, 0.25], así que allí el resultado personal no es limpio. **El efecto de `caatopic` se mantiene en los tres modelos con ambos clasificadores**, con magnitudes similares (0.13–0.20), por lo que no es solo vocabulario. `persona` es más débil e inconsistente (Gemma pasa de nulo a +0.08), y `within` sigue sin efecto. La cautela sobre clasificadores lineales entrenados con discursos sigue vigente: miden la posición entre partidos, no la ideología individual.

**Cautelas.** Un TF-IDF responde a léxico: parte del efecto podría ser cambio de vocabulario (p. ej. términos asociados a un bloque) y no de postura; los efectos fijos por tema controlan el tema pero no esto. El clasificador es lineal y de baja capacidad (AUC 0.77 en texto real). No filtra por coherencia ni por colapso. La repetición con embeddings (arriba) mantiene el efecto; falta, como mucho, un ajuste fino pequeño.

### 14.3 Steering personal frente a las declaraciones reales del diputado — 5 nov

`scripts/analisis/personal_vs_real.py`. Para cada texto de la versión personal (156 diputados × 8 temas × 4 métodos), similitud coseno (e5-large) con el centroide de las 20 declaraciones reales de cada diputado de su partido (partidos con ≥ 3 diputados; 136 diputados). Dos medidas: **ventaja propia** (similitud con el centroide del propio diputado menos la media de los demás del partido) y **proximidad** (Spearman entre la similitud con cada compañero y −|diferencia de P6 dentro de partido|). IC95% por bootstrap de diputados; nulo de la ventaja propia con 100 permutaciones de los centroides dentro de partido.

| Modelo | Método | Ventaja propia [IC95%] | Proximidad ideológica [IC95%] |
|---|---|---|---|
| Llama | caatopic | +0.0001 [−0.0006, 0.0008] | −0.023 [−0.059, 0.015] |
| Llama | aleatoria | 0.0000 [−0.0007, 0.0007] | −0.030 [−0.062, 0.005] |
| Gemma | caatopic | 0.0000 [−0.0006, 0.0006] | −0.059 [−0.093, −0.026] |
| Gemma | aleatoria | −0.0001 [−0.0007, 0.0006] | −0.052 [−0.086, −0.021] |
| GLM | caatopic | +0.0001 [−0.0005, 0.0008] | −0.017 [−0.049, 0.019] |
| GLM | aleatoria | 0.0000 [−0.0007, 0.0006] | −0.011 [−0.044, 0.023] |

(persona y within son iguales a las aleatorias; ver `results/personal_vs_real/resumen.csv`.) El p95 del nulo de permutación es ≈ 0.0001 en todos los casos.

**Lectura.**
- **Los textos del steering personal no se parecen más a las declaraciones del propio diputado que a las de sus compañeros de partido**, en ningún modelo ni método (ventaja ≈ 0 e indistinguible del nulo), y `caatopic` no se diferencia de la dirección aleatoria.
- **Tampoco se parecen más a diputados ideológicamente cercanos**: la proximidad es ≈ 0 o ligeramente negativa, igual para `caatopic` y para las aleatorias, es decir un artefacto de los textos base y no del steering.
- Esto es consistente con lo visto: el steering personal es una dosis proporcional al P6 que mueve la postura general en el eje izquierda–derecha (§14.2), pero no hace que el texto reproduzca al diputado. Los textos generados son genéricos (voz del modelo, 160 tokens, mismos 8 temas), y las declaraciones reales son discursos de sala; la similitud global de embeddings puede ser un instrumento poco sensible al estilo individual.
- **Limitación:** la medida mide similitud de contenido/estilo con un embedding general, no la postura sobre el tema concreto. Un control más fino sería comparar con las declaraciones reales del diputado **sobre el mismo tema**, pero para la mayoría de los temas generados hay pocas o ninguna.

---

### 14.4 Consolidación del efecto de `caatopic` (7 nov)

Diseño (`cola_slant_caatopic.sh`, `caatopic_consolidacion.py`, `results/slant_caatopic/`): por modelo, 4 capas con 15 muestras por tema y 10 direcciones aleatorias; sensibilidad con 1 y 8 capas (10 muestras, 3 aleatorias). Sin juez: logit del propio modelo y clasificadores TF-IDF y e5 entrenados en texto real. Pendiente por unidad de α con efectos fijos de tema, IC95 bootstrap por tema, solo textos con distinct-2 ≥ 0.5.

Pendiente de `caatopic` con 4 capas [IC95] (rango de las 10 aleatorias entre paréntesis):
| modelo | logit | clasificador TF-IDF | clasificador e5 |
|---|---|---|---|
| Llama | 0.42 [0.25, 0.60] (−0.09, 0.03) | 0.21 [0.16, 0.25] (−0.03, 0.01) | 0.16 [0.13, 0.18] (−0.02, 0.01) |
| Gemma | 0.49 [0.26, 0.76] (−0.20, 0.34) | 0.12 [0.08, 0.15] (−0.03, 0.08) | 0.13 [0.10, 0.16] (−0.04, 0.06) |
| GLM | 0.06 [0.01, 0.12] (−0.02, 0.03) | 0.21 [0.16, 0.26] (−0.03, 0.03) | 0.14 [0.11, 0.17] (−0.03, 0.01) |

- **El efecto se consolida:** en los 3 modelos y con los clasificadores independientes, `caatopic` queda por encima de las 10 aleatorias (ninguna iguala el valor de `caatopic`; con 10 aleatorias el p mínimo posible es 0.09). Los dos lados de α dan pendientes parecidas (por ejemplo Llama 0.16 y 0.24 con TF-IDF), es decir, la respuesta es simétrica y no un artefacto de un solo signo.
- **Sensibilidad a capas:** el efecto crece con el número de capas intervenidas. Llama con TF-IDF: 0.06 (1 capa), 0.21 (4), 0.43 (8); Gemma 0.00, 0.12, 0.20; GLM 0.07, 0.21, 0.37. Con 1 capa casi no hay efecto (en Gemma, nulo con TF-IDF). Parte del aumento puede deberse a que el desplazamiento total es mayor.
- **Excepciones y límites:** GLM con logit es débil (0.06; 61 % de los textos de GLM quedan fuera por distinct-2 bajo, y con 8 capas 0.20); Gemma con logit tiene aleatorias que llegan a 0.34 (ruido), por lo que su logit es poco informativo y vale más el clasificador. Los efectos siguen siendo pequeños en unidades de los clasificadores (0.1–0.4 desviaciones por unidad de α).
- Sigue sin evaluarse con juez (las dos mediciones de juez discrepaban, §14.1).

**Capas frente a dosis.** Para separar ambos factores se repitió con 1, 2 y 8 capas reescalando la intensidad por capa para que la norma total del desplazamiento (raíz de la suma de cuadrados de las escalas) sea la de 4 capas (`--igualar-dosis`, `cola_slant_caatopic_eq.sh`). Pendiente con clasificador TF-IDF (IC95 de 8 capas: Llama [0.24, 0.35], Gemma [0.11, 0.20], GLM [0.20, 0.30]; aleatorias en ±0.09 como máximo):
| modelo | 1 capa, dosis igualada | 2 capas, igualada | 4 capas (referencia) | 8 capas, igualada | 8 capas, sin igualar |
|---|---|---|---|---|---|
| Llama | 0.12 | 0.19 | 0.21 | 0.29 | 0.43 |
| Gemma | 0.04 | 0.05 | 0.12 | 0.15 | 0.20 |
| GLM | 0.47 | 0.63 | 0.21 | 0.24 | 0.37 |
- **Llama y Gemma:** con la misma norma total sigue habiendo más efecto al repartir el desplazamiento en más capas (Llama 0.12 → 0.29; Gemma 0.04 → 0.15). Una parte del aumento anterior era dosis (8 capas sin igualar: 0.43 frente a 0.29 igualado), pero no todo: repartir en más capas ayuda.
- **GLM al revés:** concentrar el desplazamiento en 1–2 capas da más efecto (0.47–0.63), aunque entre el 44 % y el 47 % de esos textos se descartan por colapso, de modo que esa pendiente se mide sobre los textos que sobreviven y está sesgada.
- **Lectura:** el efecto es robusto al número de capas en los tres modelos con direcciones aleatorias en cero, pero la mejor configuración depende del modelo. «Misma norma L2» es una de varias formas de igualar dosis; no se probó otra. Las medidas con logit y e5 dan el mismo orden (`results/slant_caatopic/resumen.csv`, columna `dosis_igualada`).

## 15. Predicción de votos partido × proyecto, leave-bill-out (CPU, 5 nov)

`scripts/analisis/votos_partido_proyecto.py`. Objetivo: estimar cómo vota cada partido en un proyecto **no visto**, el cuello de botella de la predicción global (techo partido × proyecto frente a partido solo). 33 proyectos divididos (23 Mensaje, 10 Moción), 22 partidos, 3.762 votos. Rasgos conocidos antes de votar: iniciativa, y el texto (embeddings `all-MiniLM-L6-v2` de `bills`, vecinos más parecidos). Leave-bill-out anidado (los rasgos de cada proyecto de entrenamiento no usan el proyecto retenido). Segunda etapa: regresión logística sobre celdas partido × proyecto, clases balanceadas, métricas por diputado.

| Modelo | Exactitud balanceada | AUC (global) | AUC medio por proyecto |
|---|---|---|---|
| partido (tasa EN CONTRA histórica) | 63.0 | 61.2 | 68.7 |
| + iniciativa (Mensaje/Moción) | **66.8** | 66.9 | 67.9 |
| + vecinos de texto | 65.5 | **71.4** | 69.7 |
| + dificultad del proyecto (EN CONTRA de los vecinos) | 64.5 | 68.8 | 68.9 |
| techo: mayoría partido × proyecto (leave-MP-out) | 84.3 | 92.1 | — |

**Lectura.** Los rasgos del proyecto suben 2–4 puntos de exactitud balanceada y ~10 de AUC global respecto de solo partido, pero queda a 17–20 puntos del techo. La mayor parte de la ganancia viene de la iniciativa (Mensaje contra Moción, que separa oficialismo y oposición); el texto aporta ordenamiento (AUC) y no exactitud. Con 33 proyectos las diferencias entre filas de 1–2 puntos están dentro del ruido. El techo no se acerca con rasgos de proyecto disponibles aquí; lo que falta es información sobre el posicionamiento del partido en ese proyecto (por ejemplo, el voto de algunos partidos pivote o la postura de comisión), no un mejor modelo de la ideología individual.

**Para la representación.** Esta línea no dice nada sobre si el modelo representa ideología, pero fija el piso que una señal individual debe superar: el aporte de la representación individual (~1–3 puntos con P6 verdadero, §10, §13) se mide sobre el partido × proyecto, no sobre solo partido. Con el piso actual de ~65–67 la ganancia de cPCA (recalibración de EN CONTRA) seguiría explicándose por partido × proyecto.

---

## 15b. Slant en EE.UU. vs Chile con ridge y cPCA (6 nov)

Mismo diseño en ambos países (`slant_us.py`, `slant_logit_chile_v2.py`, `slant_us_vs_chile.py`): 9 temas, direcciones ridge / cPCA (k=1, k=16) / 3 aleatorias, medida = logit izquierda/derecha del propio modelo, pendiente por unidad de α con efectos fijos de tema, solo textos con distinct-2 ≥ 0.5. En EE.UU. las direcciones salen de 156 congresistas (DW-NOMINATE); en Chile, de diputados (P6). Los textos de Chile con GLM se generaron después (`cola_slant_glm_chile.sh`); antes solo existían para Llama y Gemma.

Pendiente con |α| ≤ 2 [IC95 bootstrap] (Spearman):
| método | Llama EE.UU. | Llama Chile | Gemma EE.UU. | Gemma Chile | GLM EE.UU. | GLM Chile |
|---|---|---|---|---|---|---|
| ridge | 1.83 [1.46, 2.27] (0.47) | 0.37 [0.18, 0.58] (0.18) | 0.97 [0.19, 1.71] | 0.81 [0.42, 1.25] | 1.02 [0.85, 1.22] (0.55) | 0.21 [0.08, 0.30] (0.23) |
| cPCA k=1 | 0.46 [0.16, 0.75] | −0.13 [−0.27, 0.00] | 0.13 [0.09, 0.17] | −1.42 | 0.08 [0.04, 0.14] | 0.08 [−0.01, 0.16] |
| cPCA k=16 | — | 0.25 [−0.02, 0.50] | 4.45 | 4.92 | 0.29 [−0.13, 0.60] | −0.02 [−0.20, 0.13] |
| aleatorias | 0.00 | −0.07 | −0.12 | −0.03 | −0.01 | +0.02 |

Lectura: en EE.UU. el ridge mueve el slant de forma clara y con IC lejos de 0 en los 3 modelos, unas 5 veces más que en Chile en Llama. Con |α| ≤ 4 y 8 el efecto desaparece o se vuelve inestable (Llama ridge ≈ 0 a |α| ≤ 8, GLM ≈ 0) porque los textos colapsan: 30–60 % de los textos de EE.UU. quedan fuera por distinct-2 bajo. cPCA k=1 es débil o con signo inconsistente. cPCA k=16 en Gemma da pendientes grandes en ambos países, pero con muchos textos colapsados, por lo que no es concluyente. GLM Chile repite el patrón: ridge responde (0.21) pero unas 5 veces menos que en EE.UU. (1.02), y cPCA no responde. Las aleatorias son ≈ 0. Correlación con α: Spearman 0.2–0.7, menos "muy marcada" que lo descrito en el paper original, y solo para α pequeños.

## 15c. Noticias: ranking ideológico de casas con las representaciones (6 nov)

Pregunta: ¿las direcciones aprendidas con declaraciones de diputados ordenan las casas de noticias chilenas por ideología? No hay puntaje externo de las casas, por lo que el análisis es cualitativo (el ranking lo evalúa un sociólogo). Corpus: `data/noticias/noticias_chile_2022_2026_keywords.parquet` (635 mil artículos por keywords); muestra con keyword en el título, 37 casas, hasta 400 artículos por casa repartidos parejo entre temas (12.895 artículos con ≥ 20 tokens; `muestra_noticias.py`). Las noticias se llevan al mismo espacio que las declaraciones (`extraer_activaciones_noticias.py`) y se puntúan con ridge (P6), CAA con control de tema (`caatopic`), cPCA con ridge y cPCA con kNN contra declaraciones de diputados (`ideologia_casas.py`). Capas elegidas solo con diputados. Variante `resid_tema`: quita el tema (TF-IDF→SVD). Consenso entre métodos y modelos: `ranking_casas.py` → `results/noticias/ranking/`.

Resultado (Llama, Gemma y GLM; 31 casas con ≥ 300 artículos; `results/noticias/ranking/ranking.{md,csv,png}`):
- **Señal entre casas pequeña pero estable:** eta² = 0.02–0.10 (casi toda la varianza es entre artículos de la misma casa); split-half del ranking 0.85–0.95 con los métodos principales.
- **Los modelos y métodos coinciden:** Spearman mediano 0.78 entre las 9 combinaciones modelo × método (rango 0.50–0.92). Entre modelos con el mismo método: ridge 0.78–0.92, caatopic 0.85–0.91, cPCA-kNN 0.86–0.91.
- **Orden de consenso:** izquierda extrema radionuevomundo (−2.3), eldesconcierto, cronicadigital; luego elmostrador, radio.uchile, elciudadano; centro cooperativa, theclinic, lacuarta, latercera; derecha cnnchile, chilevisión, lanacion, duna, df y emol (+1.35, el más extremo). Los IC de artículos son estrechos, pero se solapan entre casas vecinas, así que el orden fino no es fiable y los grupos sí.
- **Casas con desacuerdo alto entre métodos** (SD del z > 0.65): alairelibre, elrancaguino, elamaule, diariolongino, chilevision, radionuevomundo.
- **Reparo principal, vocabulario:** TF-IDF ordena casi igual (Spearman 0.90 con el consenso), así que no se puede afirmar que las representaciones aportan algo más que el vocabulario asociado a cada casa.
- **Reparo, tema:** con control de tema el ranking cambia bastante (Spearman 0.44 con el consenso; eta² cae a la mitad): parte del orden refleja qué temas cubre cada casa y cómo los nombra, no postura. Sobrevive la izquierda extrema (radionuevomundo, eldesconcierto). Posiciones de estilo o sector: `df.cl` (diario económico) queda a la derecha; las casas regionales (elrancaguino, laprensaaustral) cambian de lado con el control de tema.
- Pendiente: revisión del ranking por el profesor (sociólogo).

## 16. Conclusiones y limitaciones

**Qué se sostiene**
- cPCA sobre el residual produce una mejora pequeña pero consistente en la exactitud balanceada de votos de Llama, que depende de que las direcciones vengan de prompts con partido y que no aparece en Gemma ni GLM.
- Los probes de ideología que usan el partido en el prompt no son evidencia de una representación de ideología; sin partido hay una dirección de Spearman ~0.3–0.4 que generaliza a partidos no vistos, pero casi no distingue dentro de un partido.
- Ninguna dirección derivada de diputados (probes y cPCA sobre activaciones de diputados) controla el slant de las declaraciones; sí lo hace de forma moderada CAA con control de tema, derivada de declaraciones, en Llama (§14), en Gemma y GLM con un clasificador entrenado en texto real (§14.2); los jueces LLM discrepan entre sí (§14.1), y la dosis-respuesta de Llama no se replica con Qwen.
- `caatopic` mueve el slant de forma consistente y por encima de 10 aleatorias en los tres modelos con clasificadores independientes (§14.4); el tamaño del efecto es modesto y depende de cuántas capas se intervienen y de cómo se reparte la dosis. Ridge controla el slant en EE.UU. con claridad y en Chile de forma más débil, siempre con α pequeños (§15b).
- Las representaciones aprendidas con diputados ordenan las casas de noticias chilenas con un ranking estable y plausible, pero buena parte del orden es vocabulario y tema (§15c).
- Para el voto no hay evidencia de uso funcional de una dirección lineal de ideología dentro de partido (§12, §13): el techo es partido × proyecto, y ni el P6 verdadero mejora más de ~1–3 puntos.

**Limitaciones**
- Tamaños de muestra pequeños (156 diputados, 33 proyectos divididos); el prompt de votos siempre incluye el partido, incluso en el modo sin partido.
- Solo una corrida de cPCA sin partido por modelo; controles aleatorios con 1–5 semillas.
- El juez es un solo modelo (Gemma-3-27B cuantizado a 4 bits), sin validación contra anotación humana.
- En el slant, las mediciones sin juez (logit y clasificadores) no se contrastaron con anotación humana; los jueces LLM discrepan (§14.1). La comparación EE.UU.–Chile usa solo el logit del propio modelo, y el filtro de colapso descarta entre 30 % y 60 % de los textos de EE.UU. con α grandes (§15b). El ranking de noticias no tiene etiqueta externa (§15c).
- Lexicón ampliado del logit lens sesgado (§4); el filtro distinct-2 del slant es imperfecto.
- Las escalas de α no son comparables entre métodos ni entre atención y residual.

**Siguientes pasos posibles**
1. **Cerrar el slant (§14, §14.1):** más semillas de generación y aleatorias en Gemma, y los 20 temas con escala chilena del juez (`generar_escala_juez.py`).
2. **Replicar el slant** (GLM ya hecho, no replica con logit) con direcciones de capas distintas (el top-4 de CAA cambia con el empate de AUC).
3. **Probar si lo que controla el slant es individual:** hecho en §14.3 (centroides de embeddings por diputado): nulo. Falta, como mucho, comparar con declaraciones del mismo tema.
4. **Predicción global de votos con estructura de partido × proyecto:** hecho en §15 con iniciativa y vecinos de texto (63 → 65–67 de exactitud balanceada, AUC 61 → 71; techo 84). Falta probar información de posicionamiento del partido (voto de partidos pivote, comisión) y patrocinio/autoría (solo 10 de 33 proyectos tienen autores).
5. **Mejorar la representación dentro de partido** (solo si 4 sugiere que ayuda): centrar activaciones y objetivo por partido antes de entrenar (o LEACE/INLP del partido), objetivo de disidencia (voto menos partido × proyecto), probes multidimensionales (PLS/CCA, MLP), y representación del nombre sin texto.
6. **Control positivo para la ablación** (persona con rango alto o distribuida) y extender ablación y steering oráculo a Gemma.
7. **Comparación con EE.UU.:** hecho el slant con ridge y cPCA (§15b). Falta medirlo con un juez o clasificador en inglés, comparar las direcciones de EE.UU. y de Chile (coseno), la prueba individual con DW-NOMINATE y, si se consiguen votaciones de EE.UU., la prueba de votos.
7b. **Noticias (§15c):** revisión del ranking por el profesor y, si se consigue un puntaje externo de las casas (o Ad Fontes en EE.UU.), la evaluación cuantitativa.
7c. **`caatopic` en voto:** replicar §13.1 en Gemma y GLM y con más direcciones aleatorias; analizar los modelos de 70B y LatamGPT cuando lleguen los resultados del otro servidor (`analizar_slant_70b.sh`).
8. s(c) en el steering de votos, condicionar en el prompt en lugar de steering.

---

## 17. Archivos clave

| Qué | Dónde |
|---|---|
| Votos (residual, divididas) | `results/voting_predictions/residual/divided/`, `results/voting_predictions/resumen_residual.csv`, `bootstrap_divided.csv` |
| Votos (atención) | `results/voting_predictions/divided/`, `resumen_atencion.csv`, `bootstrap_divided_atencion.csv` |
| Resultados con el bug / criterio antiguo | `results/voting_predictions_buggy/`, `results/voting_predictions_acc_criterion/` |
| Probing robusto | `results/probing_robusto/resultados.csv` |
| s(c) | `results/sc_condicionada/` (`comparacion*.csv`, `predictibilidad_sc*.csv`) |
| Logit lens | `results/unembedding/<modelo>/` (`tokens_por_capa.csv`, `validacion_causal.csv`) |
| Slant (§7, §9) | `results/steering_v2/<target>/<método>[_wide[_sel]]/` |
| Personal frente a texto real (§14.3) | `scripts/analisis/personal_vs_real.py` → `results/personal_vs_real/` |
| Votos partido × proyecto (§15) | `scripts/analisis/votos_partido_proyecto.py` → `results/votos_partido_proyecto/` |
| Análisis slant 70B | `scripts/analisis/analizar_slant_70b.sh` (clasificadores y dosis-respuesta de los 70B) |
| Clasificador de texto real (§14.2) | `scripts/analisis/clasificador_texto_real.py`, `clasificador_embeddings.py` → `intervention_evaluated_<modelo>__clf.parquet`, `__clfemb.parquet` |
| Slant v3 (§14) | `results/slant_v3/` (Llama), `results/slant_v3_gemma/`; por experimento `dosis/`, `positivo/`, `personal/`, `personal_sub/` (`intervention_raw.parquet`, `intervention_evaluated_<modelo>.parquet`) |
| cPCA dentro de partido, features, desglose (§10) | `results/cpca_dentro_partido/`, `results/cpca_features_votos/`, `results/desglose_ganancia_cpca/` |
| Probes por declaración y control de tema (§11) | `results/probing_por_intervencion/`, `results/probing_control_tema/` |
| Ablación y oráculo (§12, §13) | `results/ablacion/`, `results/oraculo_steering/` |
| CAA | `results/caa/` |
| Scripts principales | `scripts/steering/predecir_votos_steered.py`, `intervencion_modelo.py`, `evaluar_respuestas.py`, `bootstrap_controles.py`, `colapso_texto.py`; `scripts/probing/probes_robustos.py`; `scripts/analisis/sc_condicionada.py`, `cpca_dentro_partido.py`, `cpca_features_votos.py`, `desglose_ganancia_cpca.py`, `slant_v3_analisis.py`; `scripts/probing/probes_por_intervencion.py`, `probes_control_tema.py`; `scripts/steering/ablacion_direcciones.py`, `ablacion_v2.py`, `oraculo_steering.py`, `slant_v3.py`, `juez_rapido.py`; `scripts/caa/*`; `scripts/interpretabilidad/*`; colas `scripts/steering/cola_*.sh` |
| Consolidación caatopic (§14.4) | `scripts/steering/cola_slant_caatopic.sh`, `scripts/analisis/caatopic_consolidacion.py`, `cola_caatopic_analisis.sh` → `results/slant_v3c_*`, `results/slant_caatopic/` |
| Noticias (§15c) | `scripts/noticias/` (`muestra_noticias.py`, `extraer_activaciones_noticias.py`, `ideologia_casas.py`, `ranking_casas.py`, `cola_noticias.sh`) → `results/noticias/` |
| Slant EE.UU. vs Chile (§15b) | `scripts/steering/slant_us.py`, `cola_slant_us.sh`, `slant_logit_chile_v2.py`, `cola_slant_glm_chile.sh`; `scripts/analisis/slant_us_vs_chile.py` → `results/slant_us/`, `results/slant_chile_logit/`, `results/slant_us_vs_chile/` |
| Logs | `logs/` |
