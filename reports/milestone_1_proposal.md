# AlignForge English — System Proposal (Milestone 1)

**Capstone Project: Building, Optimizing, and Aligning Large Language Models from Scratch**

- **Track:** Track B — Parameter-Efficient Fine-Tuning (PEFT), Direct Preference Optimization (DPO) &amp; Model Safety  
- **Student/Author:** AlignForge Project  
- **Date:** 2026-10-31  
- **Status:** Milestone 1 Submission (Week 1)

---

## 1. Executive Summary &amp; Research Question

**AlignForge English** is an inspectable, lightweight English Writing Coach built upon a custom causal decoder-only Transformer (~3.67M parameters) implemented entirely from scratch in pure PyTorch (`torch.nn.Module`). The system assists learners with short English sentences via two distinct, isolated functional journeys:

1. **Fix Grammar (Primary):** Proposes a grammatical correction for a given sentence without altering intended meaning.
2. **Paraphrase (Experimental Alignment Study):** Proposes a more natural alternative phrasing; used specifically to study preference alignment.

### Core Research Question

> *In a small decoder-only language model trained from scratch, what are the empirical trade-offs of full fine-tuning versus self-written LoRA for grammatical error correction, and does Direct Preference Optimization (DPO) shift the model toward preferred paraphrases without corrupting meaning preservation or grammaticality?*

---

## 2. Model Architecture &amp; Hyperparameter Specifications

The architecture follows modern open-weights LLM conventions (pre-normalization, rotary embeddings, swish/GELU activations) written strictly with PyTorch primitives. Pre-packaged library models (such as `AutoModelForCausalLM`) are strictly omitted.

```text
[Input Token IDs]
       │
       ▼
[Token Embeddings (Vocab 2048, d_model 256)] ──(Weight Tied)──► [LM Head Logits]
       │                                                               ▲
       ▼                                                               │
[Transformer Decoder Block x 4] ──────────────────────────────► [Final RMSNorm]
   ├── RMSNorm Pre-Normalization
   ├── Causal Multi-Head Attention (n_heads=4, d_head=64, RoPE Positional Encoding)
   │     └── Dynamic Tensor KV Cache Hook
   ├── Residual Connection (+)
   ├── RMSNorm Pre-Normalization
   ├── Feed-Forward Network (d_ff=1024, GELU Activation)
   └── Residual Connection (+)
```

### Hyperparameter Registry


| Parameter           | Specification                  | Mathematical Notation | Rationale                                                      |
| ------------------- | ------------------------------ | --------------------- | -------------------------------------------------------------- |
| **Parameters**      | 3,672,320 (\~3.67M)            | $N_{\text{params}}$   | Fits examination budget; allows rapid training on Colab T4     |
| **Layers**          | 4 blocks                       | $n_{\text{layers}}$   | Sufficient depth for hierarchical syntactic representation     |
| **Hidden Size**     | 256                            | $d_{\text{model}}$    | Balanced capacity and training throughput on single GPU        |
| **Attention Heads** | 4 heads                        | $n_{\text{heads}}$    | $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}} = 64$   |
| **FFN Dimension**   | 1024                           | $d_{\text{ff}}$       | Standard $4 \times d_{\text{model}}$ expansion factor          |
| **Vocabulary**      | 2048                           | $V$                   | Byte-level BPE trained from scratch; guarantees 100% roundtrip |
| **Context Length**  | 128 / 256                      | $T_{\text{max}}$      | Audited against sentence length distribution                   |
| **Position Scheme** | Rotary Position Embedding      | $\text{RoPE}$         | Relative positional awareness; smooth extrapolation            |
| **Normalization**   | RMSNorm ($\epsilon = 10^{-5}$) | $\text{RMSNorm}$      | Lower computational overhead than LayerNorm                    |


---

## 3. Dynamic KV Cache &amp; Profiling Design (Milestone 2 Preparation)

Autoregressive inference without caching recomputes key and value vectors across all historical tokens, incurring $O(T^2)$ time complexity. AlignForge implements a per-layer dynamic tensor cache:

- **Prefill Step:** Evaluates prompt tokens in parallel, pre-populating $K_{\text{past}}, V_{\text{past}} \in \mathbb{R}^{B \times n_h \times T_{\text{prompt}} \times d_h}$.
- **Decoding Step:** Evaluates single incoming token $x_t$, concatenating new $K_t, V_t$ to cache, reducing attention compute to $O(1)$ per generated token.
- **Profiling Matrix:** Measures Tokens/second, Time-To-First-Token (TTFT), Time-Per-Output-Token (TPOT), and Peak VRAM across $L \in \{128, 512, 1024, 2048\}$ with Cache ON vs. OFF.
- **Equivalence Contract:** $\max |z_{\text{cached}} - z_{\text{uncached}}| < 10^{-5}$.

---

## 4. Track B Alignment &amp; Safety Strategy (Milestone 3 &amp; 4)

### 4.1 PEFT: Hand-Crafted LoRA from Scratch

- Decomposes projection updates: $W = W_0 + \frac{\alpha}{r}(B \cdot A)$ where $W_0 \in \mathbb{R}^{d \times d}$ is frozen, $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, with rank $r=8$ and $\alpha=16$.
- Attached exclusively to $W_q$ and $W_v$ attention projections.
- Controlled comparison: Full SFT vs LoRA SFT on identical `grammarly/coedit` GEC split, comparing trainable parameter count, peak VRAM, and validation loss.

### 4.2 Direct Preference Optimization (DPO from Scratch)

- Closed-form objective without a separate reward model:
  $$
  \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right) \right]
  $$
- Reference policy $\pi_{\text{ref}}$ is frozen C1-lora checkpoint.
- Loss is strictly response-masked (prompt tokens $x$ excluded from log-probabilities).

### 4.3 Deterministic Guardrails

- Input length truncation ($\le 64$ tokens), control character filtering, and prompt injection detection evaluated against `deepset/prompt-injections`.
- Output sanitization: repetitive loop detection, empty string suppression, and status labeling (`candidate_unverified`, `invalid_generation`).

---

## 5. Data Governance &amp; Empirical Evaluation

**Strict No-Labeling Policy:** Zero synthetic data generation and zero manual labeling. All training pairs derive strictly from existing fields of public datasets:

- **Pretraining:** `roneneldan/TinyStories` (~1–3M tokens).
- **SFT GEC:** `grammarly/coedit` (`task == 'gec'`, `src`/`tgt`).
- **DPO Paraphrasing:** `alecccdd/paraphrasing-preferences-orpo-dpo` (852 train / 95 val pairs).
- **External Evaluation:** `jhu-clsp/jfleg` (multi-reference GLEU, edit rate).
- **Blinded LLM Judge:** Order-randomized pairwise comparison on held-out prompts.
- **Failure Analysis:** At least 5 genuine model errors documented with root cause and engineering mitigations.

---

## 6. Milestone 1 Verification Status

- [x] Custom BPE Tokenizer (`src/alignforge/tokenizer.py`) trained and round-trip verified.
- [x] Pure PyTorch Decoder-Only Transformer (`src/alignforge/model.py`, ~3.67M params) implemented.
- [x] Strict causality masking verified (zero forward attention leakage).
- [x] Autograd gradient flow verified (zero NaNs or Infs).
- [x] Tiny-batch overfitting verified (loss converges from 4.83 to &lt; 0.01 in 80 steps).
- [x] Public dataset pipeline and manifest generated (`reports/dataset_manifest.json`).
- [x] Full test suite: 13 unit tests passing in 2.37s.