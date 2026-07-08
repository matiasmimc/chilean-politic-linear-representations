# Chilean Politic Linear Representations

This repository contains the code and data for analyzing and manipulating the political ideology of Large Language Models (LLMs) in the context of Chilean politics.

## 📌 Overview

Large Language Models (like Llama 3, Mistral, and Qwen) form complex internal representations of human concepts, including political ideologies. This project uses **Mechanistic Interpretability**—a field that looks "under the hood" of AI models—to discover how these models represent the Chilean political spectrum (from left to right).

Instead of fine-tuning or retraining the models, we identify the internal "political direction" (a linear vector) within the model's artificial brain. By intervening on this vector during text generation, we can dynamically **steer** the model's responses to be more left-leaning or right-leaning on various local topics (e.g., immigration, abortion, climate change).

## 🔍 How It Works 

Our methodology consists of four main steps:

1. **Reading the AI's Mind (Extraction)**: We feed the model statements made by real Chilean politicians and extract its internal activations (how the artificial neurons fire).
2. **Finding the Political Compass (Probing)**: We train simple classifiers to find the specific mathematical direction in the AI's internal space that corresponds to the left-right political spectrum.
3. **Steering the Output (Intervention)**: When the model generates new text, we can "nudge" it along this political vector. This allows us to control the political posture of its answers on the fly.
4. **Evaluation**: We use a separate judge model (like Gemma 3) to evaluate if the new statements genuinely reflect the intended political posture while keeping the text coherent and natural.

## 📂 Repository Structure

- `data/`: Datasets containing statements, biographies, and ideological scales of Chilean politicians.
- `scripts/`: The core pipeline for extracting activations, training probes, intervening (steering) the models, and evaluating the results.
- `notebooks/`: Jupyter notebooks for data exploration, results replication, and generating visualizations.
- `results/`: The consolidated outputs of our experiments, including steering data and evaluated statements.
- `figures/`: Visualizations of our findings.
- `log.md`: A technical, living document tracking the detailed project workflow, experiment history, and roadmap for developers and collaborators.

## 🚀 Getting Started

If you are a collaborator or want to run the pipeline, please refer to the **Interactive Pipeline Guide** inside [`log.md`](log.md) for step-by-step instructions on how the scripts connect with each other. 