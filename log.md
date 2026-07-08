# Project Log & Workflow: Chilean Politic Linear Representations

## 1. Project Explanation
**What is this project?**
This project focuses on mechanistic interpretability within the Chilean political context. We investigate how large language models (LLMs) like Llama 3, Mistral, and Qwen represent political ideology (left vs. right) linearly in their internal activation space when discussing local issues.

**What is it for?**
The goal is to understand whether these "political vectors" can be identified via probing and whether they can be intervened upon (steered) to modify the model's output posture—all without fine-tuning, just by adding vectors during the forward pass.

**How does it work?**
1. **Feature Extraction**: Activations are extracted from the models as they process statements by Chilean politicians.
2. **Probing**: Linear probes (like Ridge regression and MLP) are trained on these activations to identify the ideological direction.
3. **Intervention (Steering)**: We intervene on the model's self-attention/projection heads at specific layers by injecting a scaled vector (`alpha`) aligned with the probed ideological direction.
4. **Evaluation**: An external judge model (e.g., Gemma 3) evaluates the generated responses for their ideological posture (on a scale from -2 to +2) and coherence (1 to 5).

## 2. Interactive Pipeline Guide
The typical workflow for running an experiment is:
1. `scripts/extraer_activaciones.py` -> Extract internal activations.
2. `scripts/entrenar_probes.py` -> Train the linear probes.
3. `scripts/intervencion_modelo.py` -> Run the steering experiment, generating statements across various alphas. Output is saved to `results/steering/intervention_raw.parquet`.
4. `scripts/steering/generar_escala_juez.py` -> (Optional) Generate the dynamic political scale for evaluating the statements, saved to `data/escalas_politicas_por_tema.json`.
5. `scripts/steering/evaluar_respuestas.py` -> Evaluate the statements using the judge model. Output is saved to `results/steering/intervention_evaluated.parquet`.
6. `scripts/steering/analisis_resultados.py` -> Generate summary statistics (`steering_summary_stats.csv`) and figures (saved to `figures/`).

## 3. Project Roadmap
- [x] Initial Probe Training & Performance Assessment
- [x] Steering Interventions on Llama, Mistral, Qwen
- [x] Judge Model Setup (Gemma 3) & Dynamic Scale Generation
- [x] Evaluation Pipeline Development
- [ ] Thorough hyperparameter sweep (k, alpha) tuning to maximize slant change without losing coherence
- [ ] Compare linear representations across models

## 4. Experiment Log
*(Update this section chronologically after generating new results)*

### Date: 2026-06-15
- **Experiment**: Repository Reorganization & Workflow Implementation.
- **Objective**: Clean up deprecated folders (`scratch/`, `src/`) and establish a structured directory setup (`results/`, `references/`, `data/`) for better collaborative development.
- **Status**: Completed. Paths in scripts were updated.
- **Insights**: All resulting parquets and CSVs are now centrally located in `results/steering/`.

*(Add new experiment outcomes below)*
