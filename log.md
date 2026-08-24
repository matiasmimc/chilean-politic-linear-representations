# Project Log & Workflow: Chilean Politic Linear Representations

## 1. Project Explanation
**What is this project?**
This project investigates the mechanistic interpretability and geometric structure of political representations in Large Language Models (LLMs)—specifically examining models such as Llama 3.1 (8B/70B), Qwen 3.5 (9B), Gemma 4 (E4B-it), LatamGPT (70B-SFT), and abliterated variants—within the Chilean parliamentary context (2022–2026).

**What is it for?**
The research addresses three core questions:
1. **Representational Fidelity & Probing (RQ1)**: Can linear and non-linear probes recover the continuous ideological spectrum (left-to-right) from internal attention head activations when prompted with local parliamentary speeches versus mere politician names?
2. **Spontaneous Alignment & Geometry (RQ2)**: Does the political spectrum emerge as a primary unsupervised geometric axis (via PCA, Linear CKA, and PWCCA) across different model families and fine-tunings?
3. **Causal Controllability (RQ3)**: Can targeted activation steering along identified ideological directions causally shift generated stances on contentious Chilean legislative topics without degrading text coherence?

**How does it work?**
1. **Feature Extraction**: Activations are extracted from the self-attention output projection (`o_proj`) across all layers using `baukit` while the model processes parliamentary statements and speeches.
2. **Supervised Probing**: Ridge regression and MLP probes with 2-Fold Cross-Validation identify the ideological directions and quantify alignment using Spearman rank correlation against official P6 survey scores.
3. **Unsupervised & Geometric Analysis**: Principal Component Analysis (PCA) checks spontaneous ideology emergence; Linear CKA and PWCCA compute layer-to-layer geometric isomorphism between base, instruct, and regionally fine-tuned models.
4. **Causal Intervention (Steering)**: We inject scaled ideological vectors ($\alpha \cdot \mathbf{v}$) into critical attention heads during inference.
5. **Evaluation**: An LLM judge (Gemma) evaluates generated responses for political slant and text coherence.

---

## 2. Interactive Pipeline Guide

The standardized workflow consists of the following modular scripts:

1. **Activation Extraction**:
   - `scripts/extraer_activaciones.py` -> Extracts activations for small/medium models ($\le 9\text{B}$) in `bfloat16`/`float16`.
   - `scripts/extraer_activaciones_70b_8bit.py` -> Extracts activations for 70B models using 8-bit quantization (`BitsAndBytes`) and multi-GPU tensor placement.
2. **Probing & Baseline Comparison**:
   - `scripts/entrenar_probes.py` -> Trains Ridge regression probes across all layers and attention heads.
   - `scripts/comparar_baselines.py` -> Evaluates and visualizes performance differences between *Names-Only* (baseline) and *Full Context* (20 interventions).
3. **Unsupervised & Geometric Representation Analysis**:
   - `scripts/analisis_pca_activaciones.py` -> Performs PCA decomposition per layer and calculates cosine similarity between unsupervised components (PC1/PC2) and supervised probe vectors.
   - `scripts/analisis_geometrico.py` -> Calculates optimized sample-space Linear CKA and PWCCA similarity matrices and heatmaps across layers and models.
4. **Thematic Bill Extraction**:
   - `scripts/extraer_intervenciones_tematicas.py` -> Fetches and cleans legislative debates for specific bills (Security, Pensions, Health, Labor) via `bcn-scraper`.
5. **Causal Steering & Evaluation**:
   - `scripts/steering/intervencion_modelo.py` -> Performs inference with activation steering across a grid of $\alpha$ injection scales.
   - `scripts/steering/evaluar_respuestas.py` -> Evaluates generated statements using an LLM judge on slant and coherence.
   - `scripts/steering/analisis_resultados.py` -> Generates summary statistics and figures.


---

## 3. Project Roadmap

- [x] Initial Probe Training & Performance Assessment on legacy models (Llama 3 8B, Mistral 7B, Qwen 2.5 7B)
- [x] Steering Interventions & LLM Judge Evaluation Pipeline
- [x] Thesis Proposal Defense & Methodological Formalization (Adoption of Linear CKA, PWCCA, and updated model suite)
- [x] Name-Only Baseline Probing (Validation of local knowledge gap in LLMs)
- [x] Technical Benchmark & Architectural Integration of `bcn-scraper` for thematic analysis
- [x] Analytical Suite Refactoring (Fast sample-space CKA, PWCCA, PCA alignment, and CLI automation)
- [ ] Run activation extraction on updated medium models (`Llama-3.1-8B`, `Qwen3.5-9B`, `Gemma-4-E4B-it`) and 70B models (`Llama-3.1-70B`, `LatamGPT-70B`)
- [ ] Train probes and generate comparative baseline reports for the updated model suite
- [ ] Multi-model geometric alignment analysis (Linear CKA & PWCCA heatmaps across base vs. regional models)
- [ ] Hyperparameter sweep (layer targeting, $\alpha$ scaling) for causal steering on updated architectures
- [ ] Thematic analysis (Security vs. Pensions vs. Health) on specialized legislative corpora

---

## 4. Experiment Log

### Date: 2026-06-15
- **Experiment**: Repository Reorganization & Workflow Implementation.
- **Objective**: Clean up deprecated folders (`scratch/`, `src/`) and establish a structured directory setup (`results/`, `references/`, `data/`) for reproducible development.
- **Status**: Completed.
- **Insights**: All resulting parquets and CSVs are centrally managed in `results/steering/` and `results/replication/`.

### Date: 2026-07-10
- **Experiment**: Methodological Revision for Thesis Proposal.
- **Objective**: Formalize geometric evaluation metrics in the thesis document to replace subspace overlap with Linear CKA (Centered Kernel Alignment) and PWCCA (Projection-Weighted CCA).
- **Status**: Completed.
- **Insights**: Linear CKA in sample space ($N \times N$) provides invariance to orthogonal rotations, allowing direct comparison of activation geometries between non-identical architectures (e.g., LLaMA vs. Qwen).

### Date: 2026-08-04
- **Experiment**: Name-Only Baseline Probing (`v2_names_only`).
- **Objective**: Determine whether LLMs encode political ideology based solely on Chilean politician names and party labels without intervention context.
- **Status**: Completed.
- **Insights**: Spearman correlation dropped significantly across models when given only politician names (Qwen 3.5 9B: 0.33, Llama 3.1 8B: 0.40, Gemma 4: 0.42) compared to the ~0.60–0.66 achieved with 20 real parliamentary interventions. This confirms that pre-trained weights lack granular local political priors and that in-context legislative interventions are essential for rich ideological representation.

### Date: 2026-08-05
- **Experiment**: Evaluation of `bcn-scraper` vs. Curated Intervention Dataset.
- **Objective**: Assess whether to replace the existing 20-intervention parquet with live scraped data from `bcn-scraper`.
- **Status**: Completed.
- **Insights**: The curated dataset (`congreso_2022_2026_20intervenciones.parquet`) guarantees 100% coverage (155 deputies) and high ideological variance. `bcn-scraper` was assigned specifically to the upcoming thematic analysis phase (extracting specific bills such as Ley Naín-Retamal or Reforma Previsional).

### Date: 2026-08-06
- **Experiment**: Pipeline Modularization, Fast Sample-Space CKA/PWCCA & PCA Ideology Alignment.
- **Objective**: Standardize CLI arguments for batch and single-model execution, optimize geometric calculations from $O(D^2)$ to $O(N^2)$, and test unsupervised PCA alignment against supervised probe weights.
- **Status**: Completed.
- **Insights**: Precomputing Gram matrices in sample space reduced CKA computation time across 32 layers from ~15 minutes to <0.1 seconds. PCA alignment shows that unsupervised PC1 captures significant variance while aligning with supervised ideological vectors in mid-to-late transformer layers.
