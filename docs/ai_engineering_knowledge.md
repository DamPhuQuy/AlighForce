# AI Engineering Knowledge Map — ML · DL · NLP · LLM

> **Version:** 1.0 — 2026-11-04  
> **Audience:** Software / Backend / AI Application Engineers.  
> **Purpose:** A compact, decision-oriented core reference. This is **not** a research encyclopedia, a model catalogue, or an end-to-end AI textbook.

## 0. Operating contract

**Definition of core knowledge.** A concept belongs here when it (1) explains a broadly reused mechanism, (2) changes a real engineering decision, (3) helps diagnose a failure, or (4) is a prerequisite for another core concept. Do not include an item solely because it appears in a popular paper or framework.

**A concept is considered understood only if the reader can:**
1. **Explain** what problem it solves and the mechanism in plain language.
2. **Trace** inputs, outputs, assumptions, and—in relevant cases—the minimal mathematics.
3. **Choose** when to use it and when a simpler or different alternative is preferable.
4. **Diagnose** a concrete failure mode or an invalid evaluation.
5. **Verify** its value through a baseline, test, or measurement.

**Priority classes.** **P0 — Required:** daily-use concepts or reasoning foundations. **P1 — Practical:** common methods and production trade-offs. **P2 — Contextual:** useful in selected workloads; know what it does and where to learn more. New research variants are excluded unless they meet the above criteria.

**Relationship between domains.** Deep Learning is a subset of Machine Learning. Large Language Models usually use Deep Learning. NLP is a *problem domain* that can be solved with traditional ML, DL, and LLMs. Therefore the following are **four independent reference sections**, not four mutually exclusive or strictly sequential layers.

**How to read entries.** Each item states `Mechanism → Use → Limitation/check`. Follow the **Decision Labs** at the end to convert definitions into judgment.

---

## 1. Machine Learning (ML) — Learning, generalization, evaluation

**Core question:** What signal can be learned from the available data, and how do we know it will generalize to the actual operating environment?

### 1.1 P0 — Foundations & evaluation

| Concept | Mechanism / first principle | When it changes a decision / common trap |
|---|---|---|
| Problem framing | Map a business question to classification, regression, ranking, anomaly detection, clustering, or forecasting. | Start with the outcome and action, not an algorithm. A prediction metric may be irrelevant if no decision changes. |
| Dataset, feature, label, target | Distinguish observed input from desired outcome, acquisition time, and missing information. | Check whether a feature exists **at inference time**; otherwise the model cannot be used as evaluated. |
| Baseline & data-only heuristic | Compare against majority class, simple rules, or simple statistical forecasts before complex models. | No strong baseline means no evidence that additional complexity is worth its cost. |
| Train / validation / test | Learn parameters on train; choose models/hyperparameters on validation; estimate final performance on untouched test. | 80/10/10 is not universal. Use time-aware, group-aware, or entity-aware splits as appropriate. |
| Data leakage & preprocessing pipeline | A transformation must learn its statistics only from training folds; the test set remains independent. | Fitting scaling, imputation, PCA, or feature selection on all data inflates estimated performance. |
| Generalization & overfitting | Training fit does not imply performance on new samples. Compare train/validation behavior and changes in data distribution. | Low training loss is not a quality guarantee; overfitting is broader than literal memorization. |
| Bias–variance / model capacity | Insufficient flexibility underfits; excess flexibility can overfit under finite data. | Increase complexity only when error analysis justifies it. |
| Objective / loss / metric | Loss drives fitting; evaluation metrics reflect business costs. Common losses: MSE/MAE (regression), log loss/cross-entropy (classification). | Do not confuse loss and business KPI; reducing train loss is not the universal objective. |
| Confusion matrix, precision, recall, F1 | Different error types imply different operating costs. F1 balances precision/recall, but hides their individual values. | Accuracy can be misleading for rare events; choose metrics based on false-positive/false-negative impact. |
| Classification threshold & calibration | Predicted scores may not represent calibrated probabilities; threshold defines the action policy. | Change thresholds on validation data, not the final test set; compare precision-recall and calibration. |
| Cross-validation & reproducibility | Repeat evaluation across appropriate folds, with seeds, dataset versions and configuration logged. | Time-series / grouped observations often invalidate random k-fold assumptions. |
| Distribution shift, drift & monitoring | Training data and live data can differ; measure input quality, predictive performance and operational impact over time. | A static offline score does not guarantee long-term production performance. |

### 1.2 P1 — Small set of model families worth recognizing

| Model family | What it does | Prefer / avoid |
|---|---|---|
| Linear regression, logistic regression | Simple, interpretable parametric baselines; logistic maps scores to class probabilities. | Excellent starting points for tabular classification/regression. Check feature assumptions and calibration. |
| Decision tree | Learns rules by recursive partitioning. | Easy to inspect but a deep single tree often overfits. |
| Random Forest | Bagging of diversified trees reduces variance. | Robust tabular baseline; can be larger and less transparent than one tree. |
| Gradient-boosted trees (GBDT) | Sequential trees correct prior prediction errors. | Strong candidate for tabular data; benchmark rather than assuming neural nets win. |
| k-means / clustering | Groups observations under a distance metric and chosen representation. | For exploratory segmentation, not evidence of naturally meaningful clusters. |
| PCA | Finds orthogonal directions capturing variance. | For dimensionality reduction/visualization when appropriate; does not guarantee best features for downstream labels. |

**P2 contextual:** SVM, k-NN, Naive Bayes, anomaly-specific models, causal inference, time-series families. Learn deeply when the problem demands them rather than memorizing every variant.

**Practical ML verification:** Given a skewed fraud dataset, explain why a 99% accuracy result can be worthless; construct a group/time-safe split, baseline, PR curve and business-oriented threshold.

---

## 2. Deep Learning (DL) — Representation, gradients, computation

**Core question:** How does a differentiable model transform tensors, learn internal representations, and use compute/memory effectively?

### 2.1 P0 — Training mechanics

| Concept | Mechanism / first principle | When it changes a decision / common trap |
|---|---|---|
| Tensor, shape, dtype, device | Numeric data has axes, precision, and placement (CPU/GPU). | Track `batch`, `sequence`, `features`, `heads`; shape/broadcast bugs may silently corrupt training. |
| MatMul & broadcasting | Batched linear algebra transforms representations. | Trace dimensions before optimizing; broadcast errors can be *valid Python* but invalid math. |
| Parameters, forward pass, computation graph | A model maps inputs to outputs using learned weights and differentiable operations. | Distinguish parameters from activations and optimizer state in memory analysis. |
| Chain rule, backpropagation, autograd | Gradients propagate through composed operations; autograd records differentiable operations. | Know what `.backward()` computes and how `.grad` is accumulated. |
| Loss functions & reduction | Per-example objectives are aggregated by mean/sum/masks. | Check label alignment, ignore indices, reduction and scaling across batch sizes. |
| SGD, momentum, Adam, AdamW | Different update rules use gradient history; AdamW decouples weight decay from adaptive moments. | AdamW is a common choice, **not** a universal winner. SGD remains relevant and must be understood. |
| Learning rate, schedule, warmup | Update scale changes optimization dynamics; schedules are training design choices. | Warmup + cosine is common but not mandatory. Choose using stability and measured convergence. |
| Batch size & gradient accumulation | Aggregate gradients over examples; accumulation approximates larger effective batches with smaller microbatches. | Scale loss correctly; BatchNorm and stochastic operations can make accumulation non-equivalent to a single large batch. |
| Train/eval mode & no-grad | Dropout, BatchNorm and graph creation have mode-dependent behavior. | `model.eval()` does not itself disable autograd; use `torch.no_grad()`/inference mode when appropriate. |
| Initialization, residual connections, normalization | These affect signal/gradient propagation and stability in deep networks. | Learn what failure each helps with; do not infer one normalization type is best everywhere. |
| Regularization & validation | Weight decay, dropout, data augmentation, early stopping can reduce overfitting. | Select by workload and validation evidence; never use all by rote. |

### 2.2 P1 — Architecture and systems

| Concept | Mechanism / first principle | When it changes a decision / common trap |
|---|---|---|
| Embeddings | Learned lookup/mapping from discrete IDs to continuous vectors. | Vectors are **learned representations**, not automatically human-interpretable semantics. |
| Activation functions: ReLU, GELU, SiLU | Nonlinearity permits function classes beyond composed linear mappings. | Modern transformer FFNs often use GELU/SiLU variants; architecture-specific. |
| Attention | Queries weight values according to key similarity, subject to masks. | Understand `QKᵀ/√d` before memorizing attention variants. |
| CNN / RNN / Transformer | Different inductive biases for local patterns, sequences, and context interactions. | Choose according to modality, latency, data scale, and requirements. |
| Automatic mixed precision (FP16/BF16) | Lower-precision operations reduce memory/bandwidth, with numerical trade-offs. | Monitor NaNs, loss scaling as appropriate and accuracy changes. |
| Checkpointing, optimizer memory, profiling | Activation/optimizer/parameter memory compete with throughput; checkpointing recomputes activations. | Diagnose OOM using memory breakdown, not only lower batch size blindly. |
| Transfer learning | Start from learned representations instead of training from zero. | Usually an efficient baseline when labeled data and compute are limited. |

**Practical DL verification:** Explain why doubling gradient accumulation steps can help avoid OOM yet may not double throughput; identify which tensors must remain in VRAM and why.

---

## 3. Natural Language Processing (NLP) — Text, search, language tasks

**Core question:** What representation and evaluation are appropriate for the specific language problem—without assuming every text task needs an LLM?

### 3.1 P0 — Text fundamentals & baselines

| Concept | Mechanism / first principle | When it changes a decision / common trap |
|---|---|---|
| Unicode, normalization, token boundaries | Text is encoded bytes/code points; visually similar strings can tokenize differently. | Handle Vietnamese accents, emoji, whitespace, case, and normalization deliberately. |
| Sentence/word/subword tokenization | Segmentation defines model units and cost. | Word segmentation matters for some Vietnamese pipelines; tokenizer-specific units are not always words. |
| Vocabulary, OOV, special tokens | Maps units to IDs, with model-specific control and fallback conventions. | Distinguish ordinary text, token IDs, and chat-control tokens. Not all tokenizers share the same symbols. |
| Bag of Words & n-grams | Sparse count vectors retain lexical evidence but little word order/semantics. | Excellent inexpensive baseline for text classification. |
| TF-IDF | Downweights common terms and highlights locally informative terms in sparse retrieval/classification. | Strong baseline; exact lexical matches are often valuable for IDs and rare words. |
| BM25 | Lexical relevance model combining term frequency, document frequency and length normalization. | A serious baseline for document search; semantic embeddings do not automatically outperform it. |
| Text classification & sequence labeling | Predict document classes or token-level spans (e.g. NER). | Evaluation unit matters: token-level accuracy can hide poor entity-level performance. |
| Train/test contamination for text | Near-duplicate documents/templates can leak across folds. | Split by source/author/time/entity where justified and deduplicate appropriately. |
| Task-appropriate evaluation | Classification: precision/recall/F1; retrieval: Recall@k, MRR, nDCG; generation: task-specific human/automatic checks. | BLEU/ROUGE/PPL are not interchangeable universal measures of language quality. |

### 3.2 P1 — Modern text representations and retrieval

| Concept | Mechanism / first principle | When it changes a decision / common trap |
|---|---|---|
| Static vs contextual embeddings | Static word vectors assign one representation per unit; contextual encoders condition on surrounding text. | Understand polysemy and why contextual representations help. |
| Encoder vs encoder–decoder vs decoder-only | Bidirectional encoding, conditional sequence transformation, and autoregressive next-token generation solve different tasks. | Don't describe all NLP as decoder-only next-token prediction. |
| Dense retrieval & similarity | Embedding vectors enable approximate semantic matching. | Requires suitable embedding model, chunking, evaluation and often ANN indexing. |
| Hybrid retrieval & reranking | Combine lexical/dense candidates; reranker scores query–document pairs more precisely. | Often preferable to treating vector search as a magic replacement for keyword search. |
| Dataset quality, annotation agreement, error analysis | Labels can be subjective; inspected false positives/negatives reveal systematic errors. | A metric cannot repair an ambiguous label definition. |

**P2 contextual:** CRF, POS tagging, parsing, translation-specific metrics, classical language models; deepen only for relevant tasks.

**Practical NLP verification:** Build two text-search baselines (BM25 vs embedding retrieval) over the same labeled queries and compare Recall@k, latency and failure cases.

---

## 4. Large Language Models (LLM) — Architecture, application, evaluation, serving

**Core question:** What is the least complex, measurable and secure way to use a language model for the required task?

### 4.1 P0 — Mental model, application choices, evaluation

| Concept | Mechanism / first principle | When it changes a decision / common trap |
|---|---|---|
| Autoregressive modeling & cross-entropy | A common decoder LLM predicts next-token distributions from context. | Next-token loss is a training objective, not a direct guarantee of factuality or task success. |
| Self-attention, causal masking, positional information | Token representations attend to accessible context; decoder masks prevent future-token leakage. | The position can typically attend to **itself and earlier positions** during causal training. |
| Tokens, context window, truncation | Prompts and outputs consume a model-specific token budget. | Count tokens with the **actual tokenizer**; protect important evidence from truncation. |
| Chat templates & tool schemas | Structured message boundaries/roles are serialized into model-specific tokens/text. | Use the model's documented template; malformed templates can degrade tool calls. |
| Prompting & structured output | Task instruction, examples, context, and output contracts shape generation. | Use schema validation; a model producing JSON-looking text does not guarantee correct or safe JSON. |
| Sampling: temperature, top-p, stop rules | Generation samples from transformed probability distributions. | Lower temperature reduces sampling randomness, **not** hallucination risk to zero. |
| RAG as a pipeline | Retrieve relevant evidence, construct context, generate grounded answers and cite/check provenance. | Prefer for private/frequently updated knowledge, but evaluate retrieval and generation separately. |
| Tool/function calling | Model suggests a structured action; the application validates and executes it. | Authorization and side effects belong to deterministic backend code, not to a prompt. |
| Evaluation dataset, task metrics, regression tests | Compare outputs against concrete cases, reference evidence and expected behavior. | Benchmarks alone cannot establish quality on domain-specific real requests. |
| Hallucination, grounding, uncertainty | Fluent text may contain unsupported statements. | Require evidence or abstention for knowledge-critical tasks; measure groundedness and factual errors. |
| Cost, latency, quality trade-off | Model selection is multi-objective: model quality, price, throughput, TTFT and failure rate. | Select smallest sufficient solution with actual workload measurements. |
| Security trust boundaries | Prompts, documents, tool results and user content may contain adversarial instructions. | Treat retrieved text as untrusted input; prompt injection is not solved by an input regex filter. |

### 4.2 P1 — Model internals, adaptation, inference systems

| Concept | Mechanism / first principle | When it changes a decision / common trap |
|---|---|---|
| Transformer block: attention, FFN, residual, norm | Repeated blocks transform contextual hidden states. | Trace tensor shape through one block before worrying about architectural variants. |
| RoPE & position scaling | Rotary transformations encode relative positional information in Q/K interactions. | Longer context is not automatically higher-quality long-context reasoning. |
| Prefill vs decode | Prefill processes input context; decode generates tokens autoregressively. | TTFT and inter-token latency have different bottlenecks. |
| KV cache | Reuses previous tokens' K/V states instead of recomputing them. | For standard dense attention, new-token attention still reads O(context length) cached positions; memory grows with cached sequence length. **Not O(1)** per-token attention. |
| Batching, continuous batching, prefix caching | Amortize server work, schedule requests, and reuse shared prefixes. | Optimization affects throughput, per-user latency and memory differently. |
| Quantization | Lower-precision weight/activation/cache formats trade memory or speed against quality and hardware support. | Measure the specific hardware and tasks; INT4 is not always faster. |
| Fine-tuning: SFT & LoRA/PEFT | Adjust behavior/task performance using labeled examples; LoRA learns low-rank updates to selected weights. | Adapter size, trainable parameter ratio and quality are configuration-dependent. |
| Preference optimization: DPO | Learns from preferred vs rejected responses against a reference policy in a specific optimization setup. | Does not automatically guarantee safety, correctness or hallucination removal. |
| Prompt vs RAG vs fine-tune | Prompt changes instruction/context, RAG supplies external evidence, fine-tuning changes parameters/behavior. | Choose by root cause: unclear instructions, missing knowledge, or insufficient behavior/capability. |
| Monitoring & traceability | Capture prompts/versions, retrieval results, latency, cost, safety events and evaluated errors with privacy controls. | Avoid logging secrets or raw personal data; define observability and retention policies. |

**P2 contextual:** MoE, GQA/MQA, speculative decoding, distributed parallelism, RLHF/PPO, advanced preference algorithms, specialized long-context architectures, distillation. These are optional unless implementing/troubleshooting those systems.

**Practical LLM verification:** Diagnose an internal Q&A assistant with accurate retrieval but unsupported answers: determine whether the issue is indexing, context assembly, generation, instruction hierarchy, or evaluation—not reflexively fine-tune.

---

## 5. Corrected misconceptions from the original playbook

| Oversimplification to avoid | Correct engineering formulation |
|---|---|
| “Forget Decision Tree, Random Forest, SVM, PCA.” | Know the role of simple baselines, tree ensembles, and representative transformations. Deep dive only by project need. |
| “Cross-entropy is the only loss worth knowing.” | The loss depends on task: e.g. cross-entropy for classification/language modeling, MSE or MAE for regression. |
| “Always AdamW, always cosine warmup.” | Common configurations, not laws. Compare SGD/AdamW and schedules based on workload and evidence. |
| “Always split 80/10/10.” | Respect temporal, group, entity and leakage constraints; proportions depend on data size and validation strategy. |
| “Low train loss means success.” | Train/validation/test behavior and real-world metrics establish generalization, not train loss alone. |
| “Deep Learning is just chained matrix multiplication.” | Matrix operations matter, but nonlinearity, learning dynamics, data, architecture, and computation are essential. |
| “KV cache makes attention O(1).” | KV cache avoids past-K/V recomputation; dense attention per generated token still scans the context, roughly O(T) in sequence length. |
| “A low perplexity means accurate answers.” | PPL measures token prediction on a distribution; not a comprehensive factuality/task-quality metric. Comparisons depend on evaluation setup/tokenization. |
| “DPO fixes hallucination/safety.” | Preference optimization can shape behavior, but needs independent task and safety evaluation. |
| “Guardrails can block prompt injection.” | Defense-in-depth reduces risk; enforce privilege boundaries and validate actions outside the model. |

---

## 6. Decision Labs — turn definitions into engineering judgment

### Lab A — Predict churn from a small tabular dataset
- **Baseline:** rule/majority class; logistic regression; boosted trees.
- **Data:** inspect imbalance, leakage, missing data, acquisition time and repeated customers.
- **Evaluation:** PR/F1 or expected business cost as applicable, calibration, out-of-time or group split.
- **Question:** What measurable benefit justifies a neural network?

### Lab B — Search an internal technical knowledge base
- **Baseline:** exact/keyword search, TF-IDF or BM25.
- **Upgrade if needed:** dense retrieval, hybrid retrieval, reranking; then LLM-based answer synthesis.
- **Evaluation:** Recall@k, citation correctness, end-to-end answer quality, P95 latency, cost.
- **Question:** Is retrieval bad, or is generation using good evidence badly?

### Lab C — Generate strictly structured support-ticket summaries
- **Baseline:** instruction + JSON schema + deterministic validation; include no-action fallback.
- **Upgrade if needed:** few-shot examples, better model, task-specific SFT after collecting validated examples.
- **Evaluation:** schema pass rate, field accuracy, unsupported claims, latency, cost, manual-review rate.
- **Question:** Why would fine-tuning be preferable to better prompts or validation?

### Lab D — Serve a local decoder LLM on limited GPU memory
- **Baseline:** record model size, dtype, context, batch sizes, TTFT, tokens/sec and memory.
- **Choices:** quantization, smaller model, KV-cache strategy, batching, shorter context, offload.
- **Evaluation:** the *same* representative prompts and quality tests before/after every change.
- **Question:** Is the bottleneck compute, bandwidth, weight memory, KV memory, or concurrency?

---

## 7. Standard knowledge-card template

Use the following template for concepts that deserve a deeper page. Do not create a page for every keyword by default.

```markdown
# <Concept name>
- **Domain / priority:** ML | DL | NLP | LLM — P0 | P1 | P2
- **Problem:** What failure or constraint makes this concept necessary?
- **Mechanism:** What actually happens? Explain the minimal causal steps.
- **Inputs / outputs / assumptions:** Shapes, data types, boundaries, constraints.
- **Minimal math:** Only equations that clarify the mechanism or implementation.
- **When to use:** Trigger conditions and a concrete task.
- **When NOT to use:** Simpler alternative / counterexample.
- **Trade-offs:** Quality, time, cost, memory, maintainability, risks.
- **Failure modes:** Symptoms, likely root causes, diagnostic checks.
- **Verification:** Baseline, experiment, test or metric.
- **Related concepts:** At most 3 direct dependencies or alternatives.
- **Sources:** Primary docs or stable educational references.
```

**Knowledge acceptance gate:** An entry should fail review if it only defines a term; asserts “always/never” without proven conditions; lacks a failure case; gives no comparison to a baseline; or quotes an uncited, context-free percentage.

## 8. Suggested maintenance and expansion rules

1. **One canonical entry per concept.** Cross-link from other domains rather than duplicating definitions.
2. **Fundamentals first.** Prefer enduring principles over version-specific framework tricks.
3. **Separate facts from defaults.** Mark algorithms and configurations as examples, not universal standards.
4. **Prefer primary documentation.** Cite stable references for contentious claims.
5. **Review by decision usefulness.** Keep a topic only if it changes what you build, measure or debug.
6. **Research parking lot.** New papers/features go into a separate queue; promote to core only after repeatable operational value emerges.
7. **At least one counterexample per chapter.** Ask when the simpler method is better.

## 9. Reference starting points

These sources support the core engineering categories; they are not required reading in full.

- scikit-learn User Guide (models, preprocessing, metrics): https://scikit-learn.org/stable/user_guide.html
- scikit-learn Common Pitfalls (data leakage): https://scikit-learn.org/stable/common_pitfalls.html
- scikit-learn Precision–Recall (imbalanced classification): https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
- PyTorch optimizers and scheduling: https://docs.pytorch.org/docs/stable/optim.html
- Hugging Face Transformers KV cache: https://huggingface.co/docs/transformers/main/kv_cache
- Hugging Face cache computation explanation: https://github.com/huggingface/transformers/blob/main/docs/source/en/cache_explanation.md
- Microsoft: RAG and knowledge sources: https://learn.microsoft.com/azure/ai-studio/concepts/retrieval-augmented-generation
- OWASP GenAI Security Project (LLM risks): https://genai.owasp.org/llm-top-10/

---

**Bottom line:** Learn **why → how → when → when not → how to test**. An engineer who can reject a fashionable technique for the right reason understands it better than one who can recite its architecture.
