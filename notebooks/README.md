# AlignForge English — Standalone Training Notebooks (Google Colab T4)

This directory provides self-contained, interactive Jupyter Notebooks (`.ipynb`) implementing the full **AlignForge English** pipeline (Track B: Pre-training, LoRA SFT, DPO Alignment, Guardrails & Evaluation).

## Architectural Design: 100% Self-Contained Primitives

As required by the standalone training contract:
- **Zero package dependencies:** These notebooks **do not** call or import from `alignforge.*`. All layers, models, tokenizers, datasets, trainers, loss functions, and guardrails are copied 1:1 and defined directly inside notebook cells.
- **Hardware Agnostic:** Automatically detects and utilizes NVIDIA GPUs (such as Google Colab **T4 GPU**, V100, or A100) via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Public Datasets Only:** Pulls directly from Hugging Face Datasets (`TinyStories`, `CoEdIT`, `paraphrasing-preferences-orpo-dpo`, `JFLEG`) with local offline fallbacks.

---

## Notebook Structure

| Notebook | Stage | Description & Key Artifacts Produced |
|---|---|---|
| [`01_pretrain.ipynb`](./01_pretrain.ipynb) | **Stage 1: Pre-training** | From-scratch BPE Tokenizer (vocab 2048), Causal Transformer Decoder (~3.67M params, RMSNorm, RoPE, GELU FFN), Pretrainer loop on TinyStories. Produces `c0_base.pt`. |
| [`02_sft.ipynb`](./02_sft.ipynb) | **Stage 2: LoRA vs. Full SFT** | Low-Rank Adaptation (LoRA $r=8, \alpha=16$) on $W_q, W_v$ (>99% parameter reduction), CoEdIT GEC dataset with prompt masking, SFTTrainer. Produces `c1_full.pt` and `c1_lora.pt`. |
| [`03_dpo.ipynb`](./03_dpo.ipynb) | **Stage 3: DPO Alignment** | Closed-form Direct Preference Optimization on Paraphrasing Preferences with response-masked logps and frozen reference model. Produces `c2_dpo.pt`. |
| [`04_eval.ipynb`](./04_eval.ipynb) | **Stage 4: Guardrails & Evals** | Deterministic input/output guardrails (prompt injection defense, repetition loop suppression), multi-length KV Cache profiling, JFLEG multi-reference GLEU, and Blinded Judge. |
| [`pipeline.ipynb`](./pipeline.ipynb) | **Turnkey All-in-One** | Complete single-click pipeline from BPE tokenization to pretraining, LoRA SFT, DPO, and interactive inference demo in a single notebook for Google Colab. |

---

## How to Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload `pipeline.ipynb` (or any individual stage notebook).
3. Switch runtime to GPU: **Runtime > Change runtime type > T4 GPU**.
4. Click **Runtime > Run all** (`Ctrl + F9`).
5. All models train with fast GPU acceleration and save checkpoints directly to the Colab environment.
