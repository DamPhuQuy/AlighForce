# Task TASK-NOTEBOOKS-PIPELINE: Self-Contained Jupyter Notebooks Pipeline for Colab GPU Training

## Intent and acceptance

User outcome and observable success:
- Author self-contained Jupyter Notebooks in `notebooks/` reflecting the full AlignForge pipeline.
- Strictly adhere to user constraint: all primitives (Tokenizer, Causal Transformer, KV Cache, LoRA, DPO, Guardrails, Evaluators) MUST be copied 1:1 into notebook cells from `src/`, with ZERO imports from `alignforge` (`import alignforge` or `from alignforge`).
- Support turnkey training on Google Colab with T4 GPU (device-agnostic `cuda` vs `cpu`).
- Observable check: `uv run pytest tests/test_notebooks.py -v && uv run pytest` passes 100%, verifying valid JSON format, code cell compilation without syntax errors, and zero `alignforge` imports.

## Scope

<allowed_files>
  <file>.agents/TASK.md</file>
  <file>README.md</file>
  <file>process/features/active/self-contained-notebooks-pipeline/spec.json</file>
  <file>process/features/active/self-contained-notebooks-pipeline/task.json</file>
  <file>process/features/active/self-contained-notebooks-pipeline/task.md</file>
  <file>notebooks/README.md</file>
  <file>notebooks/01_pretrain_c0.ipynb</file>
  <file>notebooks/02_sft_alignment.ipynb</file>
  <file>notebooks/03_dpo_preference.ipynb</file>
  <file>notebooks/04_eval_guardrails.ipynb</file>
  <file>notebooks/alignforge_end_to_end_colab.ipynb</file>
  <file>tests/test_notebooks.py</file>
</allowed_files>

## Machine contracts and research artifacts

- Machine spec & task DAG: `spec.json` and `task.json`.
- Validation: `python3 .agents/harness.py validate --spec process/features/active/self-contained-notebooks-pipeline/spec.json --task process/features/active/self-contained-notebooks-pipeline/task.json`.

## Architecture & Implementation Strategy

1. **Jupyter Notebook Format (nbformat v4)**:
   - Construct valid JSON representations with cell type `markdown` and `code`.
2. **1:1 Code Cell Replication**:
   - Every stage contains the complete implementation copied from `src/alignforge/`:
     - Tokenizer (`ByteLevelBPETokenizer`, `SPLIT_PATTERN`)
     - Model (`RMSNorm`, `RotaryEmbedding`, `CausalSelfAttention`, `FeedForward`, `TransformerBlock`, `CausalTransformerModel`)
     - KV Cache (`TensorKVCache`, `DynamicKVCache`)
     - LoRA (`LoRALinear`, `apply_lora_to_model`, `save_lora_checkpoint`, `load_lora_checkpoint`)
     - Training loops & datasets (`TextDataset`, `Pretrainer`, `SFTDataset`, `SFTTrainer`, `DPODataset`, `DPOTrainer`, `dpo_loss`)
     - Guardrails & Evaluation (`CompositeSafetyGuard`, `BaseSafetyRule`, `GLEUEvaluator`, `PairwiseJudgeEvaluator`)
3. **Execution on Colab**:
   - First cell sets up `!pip install datasets pyyaml tqdm` if run in Colab.
   - Device detection: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
