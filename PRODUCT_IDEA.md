# AlignForge English — Product Idea

> **Project:** AlignForge English  
> **Working subtitle:** A Lightweight English Writing Coach Built on a From-Scratch Decoder-Only Language Model  
> **Document:** `PROCUCT_IDEA.md` (tên file giữ theo yêu cầu; tên quy ước thường dùng là `PRODUCT_IDEA.md`)  
> **Version:** 1.0.0 · **Status:** Proposed / scope frozen for 4-week MVP · **Date:** 2026-10-31  
> **Owner:** 1 student · **Duration:** 4 weeks · **Track:** B — Alignment & Safety  
> **Audience:** Backend Engineer → AI Application Engineer  
> **Authority order:** `EXAMINATION.md` > approved `LITE_EXECUTION_PLAN.md` > this product idea > prior `PROJECT_SOURCE_OF_TRUTH.md` for optional expansion. Where the documents conflict, the examination governs.

---

## 0. Executive decision / Quyết định chốt

**Build:** A narrowly scoped **English Writing Coach** that helps learners improve **short English sentences** through two user-facing actions:

1. **Fix Grammar (primary):** Given a sentence, propose a corrected sentence without intentionally changing its meaning.
2. **Paraphrase (experimental):** Given a sentence, propose a more natural alternative that preserves meaning; used specifically to study preference alignment with DPO.

**Do not build:** A general English tutor, conversational speaking partner, pronunciation scorer, CEFR certifier, or GPT-quality grammar editor in 4 weeks.

**Scientific core:** Implement a small causal Transformer from scratch in PyTorch; measure KV Cache; compare full SFT vs. self-written LoRA; compare SFT vs. DPO; add explicit input/output guardrails and quantitative evaluation.

**Data policy:** **No custom dataset generation or hand-labeling.** Load publicly available datasets, choose subsamples, filter by length/quality, and convert existing fields into training contracts. *This preprocessing is not creating new labels.*

**Product honesty:** A model of ~3M parameters trained for a short time may fail to fix grammar correctly. The deliverable is an **experimental AI application with transparent measurements**, not a production-grade tutor. No metric targets below imply guaranteed model quality.

### Elevator pitch

> AlignForge English is a small, inspectable English sentence editor that demonstrates how an AI application is built **from tokenizer and Transformer to model adaptation, preference alignment, guardrails, and monitoring**, using only public datasets and a reproducible 4-week research workflow.

---

## 1. Problem / Vấn đề cần giải quyết

Learners frequently write short English sentences but do not know whether the grammar is correct or whether an alternative phrasing sounds more natural. Existing LLM products can help, but for an AI/Backend engineering capstone the important question is different:

- How is the language model actually implemented and decoded?
- Does adapting a tiny model with **LoRA** save trainable parameters while retaining useful editing behavior compared with **full fine-tuning**?
- Does **DPO** shift the model toward preferred paraphrases, and is that change visible on held-out data?
- What happens when the model produces malformed, nonsensical, or unsafe output?
- How do we **evaluate, version, and monitor** the model as a software component rather than trust plausible-looking responses?

### Product problem statement

> Offer an understandable, limited English-editing interaction while turning its underlying Transformer, training pipeline and evaluation into a reproducible engineering artifact.

### Research question

> In a small decoder-only LM trained from scratch, what are the trade-offs of full SFT versus LoRA for sentence editing, and can DPO improve **paraphrase preference** without excessive loss of grammaticality or meaning preservation?

**Important:** Grammar correction and paraphrasing are related **but distinct tasks**. Evidence that DPO improves paraphrasing **does not establish** an improvement in grammatical error correction (GEC).

---

## 2. Audience & Jobs-to-Be-Done

| Persona | User need | 4-week product response |
|---|---|---|
| English learner (A2–B1 as a use-case assumption, **not** measured CEFR coverage) | Correct a short sentence | Enter sentence → see candidate correction or an honest failure state |
| Student writing short English messages | Try a different phrasing | Enter sentence → paraphrase candidate + warning to review meaning |
| AI/Backend engineering examiner | Understand internals, compare checkpoints | Show exact architecture, tokenizer, training logs, cache benchmarks and 5 failure cases |
| Developer extending the project | Integrate a model safely | Stable request/response contract, model version, reproducible evaluation, offline replay |

**Product value:** Useful early-stage prototype for learning English; the strongest measurable value within 4 weeks is **engineering evidence** and a reusable model-evaluation workflow.

---

## 3. Product principles

1. **Exam-first:** Scope is cut by dataset size, UI, and model capacity—not by dropping mandatory methods.
2. **One model, two related editing tasks:** Grammar correction is the primary user journey; paraphrase preference is a distinct DPO experiment.
3. **Grounded claims:** Show outputs and metric confidence honestly; do not present illustrative examples as observed model responses.
4. **No invented labels:** Do not manufacture chosen/rejected pairs or synthetic grammar tags for training.
5. **Minimal moving parts:** Python CLI + local artifacts + offline benchmarks; no frontend or microservice requirement.
6. **Safety outside the weights:** Input boundaries, output checks and resource limits are deterministic application behavior.
7. **Test before optimize:** Unit/property tests for Transformer and DPO are a hard prerequisite for empirical claims.

---

## 4. Product scope: MVP vs. future

### P0 — Required for capstone (4 weeks)

| ID | Feature | Description / Acceptance |
|---|---|---|
| P0-01 | Fix grammar | Accept short English text, generate edited text using the custom causal LM, report errors/timeouts distinctly |
| P0-02 | Paraphrase | Generate candidate rephrasing; report this is an experimental mode, not verified semantic equivalence |
| P0-03 | Model selection | CLI can run `base`, `full_sft`, `lora_sft`, `dpo` checkpoint where applicable |
| P0-04 | Tokenizer & Transformer | Train tokenizer; implement decoder-only Transformer, attention and autoregressive engine from scratch |
| P0-05 | KV Cache | Implement per-layer cache, correctness tests and profiling across 128/512/1024/2048 tokens |
| P0-06 | SFT + LoRA | Implement LoRA on Wq/Wv; compare controlled full SFT and LoRA on the same dataset/task |
| P0-07 | DPO | Implement chosen/rejected DPO objective with a frozen SFT reference; measure paraphrase preference effects |
| P0-08 | Guardrails | Limit input length, reject invalid inputs, enforce generation limits, inspect output shape and suspicious instructions |
| P0-09 | Evaluation | Automated GEC metrics, paraphrase preference metrics, judge comparisons, runtime profiling and at least 5 failures |
| P0-10 | Demo and deliverables | CLI demo, config/checkpoints/logs, reproducible commands, 6–8-page paper, GitHub README |

### P1 — Only if P0 passes

- Lightweight Gradio page with two modes and model dropdown.
- Optional comparison view (`base` vs `SFT` vs `DPO`) for a single input.
- JSON/CSV export of offline evaluation results.
- Basic local metrics viewer (artifact, not live production monitoring).

### P2 — Roadmap **after** the examination

- Error category explanations and grammar knowledge base; learner exercise bank.
- Personalized revision history, progress tracking and spaced repetition.
- Better pretrained foundation model **in a separate research experiment**, respecting examination restrictions for the original main model.
- Real HTTP API and OpenTelemetry traces; queueing/batching, auth and persistent storage.
- Longer passages, level-adjusted feedback, speaking/pronunciation and voice learning.

### Explicit non-goals

Real-time multiuser deployment, conversational AI, autonomous grammar pedagogy, CEFR/IELTS/TOEIC score prediction, training billion-parameter LLMs, RAG/vector DB/agents, self-produced preference labels, mobile application, elaborate dashboard.

---

## 5. User journeys & interaction design

### Flow A — Grammar correction (MVP primary)

1. User chooses `fix_grammar`.
2. User enters a short sentence (e.g., `She go to school every day.`).
3. Application checks input length, control sequences, token budget and selected model.
4. `Tokenizer → Decoder-only LM → Autoregressive generation (with or without KV Cache)`.
5. Application checks generation bounds, empty output, malformed terminal states and simple preservation heuristics.
6. UI/CLI displays **candidate correction** and warnings; it does not claim correctness if unverified.

**Illustrative desired response — not measured model output:**

```json
{
  "task": "fix_grammar",
  "input": "She go to school every day.",
  "candidate": "She goes to school every day.",
  "status": "candidate_unverified",
  "model_version": "lora_sft-v1",
  "latency_ms": null
}
```

In the actual runtime `latency_ms` is a measured number; `null` above explicitly indicates a product-contract example.

### Flow B — Paraphrasing (DPO experiment)

1. User chooses `paraphrase`.
2. User enters a sentence, optionally constrained by max length.
3. Application serializes the **same prompt semantics used in DPO training**.
4. Generate with `lora_sft` and `dpo` checkpoints on the same input.
5. Evaluation compares meaningfulness, grammar, semantic preservation and preference **without** implying that DPO fixes grammar.

**Illustrative desired response:**

```json
{
  "task": "paraphrase",
  "input": "I cannot attend the meeting today.",
  "candidate": "I won't be able to join today's meeting.",
  "status": "candidate_unverified",
  "model_version": "dpo-v1"
}
```

### Failure flow

- Oversized/invalid input → `INPUT_REJECTED` (no model call).
- Model outputs blank/overlength/control text → `INVALID_GENERATION` (do not silently replace output with a reference sentence).
- Suspected semantic drift → `REVIEW_REQUIRED` if a deterministic check supports the flag; do not fabricate confidence.
- Timeout/OOM → `MODEL_UNAVAILABLE` with sanitized diagnostic logging.

**No automatic assertion that a rewrite preserves meaning.** Such a guarantee would require an additional reliable validation method outside MVP scope.

---

## 6. UX surface: CLI first

```text
$ alignforge edit --task fix_grammar --model lora_sft \
    --text "She go to school every day."

Task:         fix_grammar
Model:        lora_sft-v1
Candidate:    <actual decoded model output>
Status:       candidate_unverified | invalid_generation | ...
Latency:      <measured>
```

CLI must also support a benchmark/evaluation mode and export machine-readable results. The primary UX is **honest and inspectable**, not visually polished.

---

## 7. Data source of truth — NO self-created datasets

> All links below refer to publicly available external datasets. Exact revision/hash, field mapping, accepted sample count and license must be captured in a `dataset_manifest.json` during implementation. **Downloadable does not mean automatically free of licensing or data-use restrictions.**

| Stage | Exact source | Native fields / use | Selection rule | Status |
|---|---|---|---|---|
| Pre-training | [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) | `text` | Fixed small train token budget; use upstream validation separately | **PRIMARY** |
| SFT – GEC | [`grammarly/coedit`](https://huggingface.co/datasets/grammarly/coedit) | `_id`, `task`, `src`, `tgt` | Prefer `task == "gec"`; pre-existing corrected `tgt` | **PRIMARY** |
| SFT – Paraphrase | [`grammarly/coedit`](https://huggingface.co/datasets/grammarly/coedit) | Same | If `paraphrase` task value is present on loaded revision, sample it; otherwise use `best` from preference dataset as SFT target on **train only** | **SMALL SUPPORT SPLIT** |
| DPO – Paraphrase | [`alecccdd/paraphrasing-preferences-orpo-dpo`](https://huggingface.co/datasets/alecccdd/paraphrasing-preferences-orpo-dpo) | `chosen`, `rejected` message arrays; `best`, `worst`, `text` | 852 train pairs; 95 validation pairs; filter out sequences that exceed context without truncating preference asymmetrically | **PRIMARY DPO** |
| GEC eval | [`jhu-clsp/jfleg`](https://huggingface.co/datasets/jhu-clsp/jfleg) | `sentence`, `corrections` (4 references) | Frozen evaluation; do not train on this benchmark | **EXTERNAL EVAL** |
| Guardrail eval | [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) | `text`, `label` | Evaluate input-injection detection; not an exhaustive jailbreak/safety test | **SUPPLEMENTAL** |

### Provenance and license notes (verified 2026-10-31)

- TinyStories dataset card displays **CDLA-Sharing-1.0**. It is much larger than MVP needs; **do not download the full corpus by default**.
- CoEdIT card displays **Apache-2.0** and lists **`task`, `src`, `tgt`**. The public release contains ~69K samples; not every example is grammar correction. Its `src` already includes an editing instruction.
- Paraphrasing Preference dataset card describes **852 train / 95 validation** and `chosen`/`rejected` as chat-message arrays, not plain strings. The user's input prompt can vary by `excellent_in`; preserve this conditional text. **License terms must be reviewed from the exact dataset version** before distributing a derived dataset or weights.
- JFLEG card describes **754 validation / 747 test**, each with four human corrections, **CC BY-NC-SA 4.0**. Use as a distinct evaluation benchmark; inspect possible overlap with CoEdIT's underlying public GEC sources.
- `deepset/prompt-injections` provides `text`, `label` with **546 train / 116 test**. Dataset metadata includes inconsistent license hints; verify the actual applicable terms before redistribution.

### Why these sources, and why not combine them indiscriminately?

- **TinyStories:** teaches simple English next-token prediction; it does **not** directly teach editing instructions.
- **CoEdIT GEC:** teaches **grammar edits** and editing instruction format.
- **Paraphrase data:** provides a separate **preference problem** for DPO. Preferred paraphrases can be more natural but **are not equivalent to GEC corrections**.
- **JFLEG:** gives multi-reference correction evaluation, useful because more than one rewrite can be acceptable.
- **Prompt Injection set:** tests one security dimension of an application reading untrusted text.

### No-labeling preparation pipeline

```text
Hugging Face public datasets
   -> load pinned revisions
   -> verify actual columns/splits
   -> filter supported tasks & max token length
   -> normalize to prompt/completion or chosen/rejected contracts
   -> dedupe / check overlap / freeze dev-test
   -> write dataset manifest & cache (not a new labeled dataset)
```

**Mandatory preprocessing rules:**

1. Use only existing `tgt`, `best`, `worst`, `chosen`, `rejected` labels; do not generate synthetic negative examples.
2. Preserve the source dataset's task meaning. For DPO, policy and reference score **the same user prompt** paired with both responses.
3. **Do not cut off one side of a preference pair** to squeeze it into context; drop the full pair instead and report how many survive.
4. Keep CoEdIT upstream validation separate from train. Use JFLEG as an external test and run exact/near-duplicate overlap checks where feasible.
5. Freeze sample IDs and seeds. Record source revision, size before/after filters, token-length distribution, task ratios and any residual limitations.

### Starter dataset budgets (NOT accuracy guarantees)

| Dataset | Planned MVP use |
|---|---|
| TinyStories | ~1–3M pre-training tokens (subject to GPU budget) |
| CoEdIT GEC SFT | ~600–1,200 eligible train examples |
| CoEdIT paraphrase SFT | ~100–300 eligible train examples, only if available and context-fitting |
| Paraphrase DPO | Up to 852 train pairs, with a smaller length-filtered pilot; 95 validation pairs kept for evaluation as applicable |
| JFLEG | Frozen external eval, optionally a fixed subset **declared before** comparison |
| Prompt injections | Frozen official test split for input-guardrail evaluation |

**Critical feasibility gate:** Count retained examples after tokenization **before** committing to `context_length=128`. If too few usable preference pairs survive, raise the **training** length to 256 *if hardware allows*, shrink prompt template only when semantics remain equivalent, or report that the DPO experiment lacks sufficient coverage. Never hide this limit.

---

## 8. Model & learning pipeline (high-level product architecture)

```text
                       OFFLINE / TRAINING
 TinyStories ─────────> BPE tokenizer + decoder-only pre-training ─> C0 base
 CoEdIT (GEC/paraphrase) ──> Full SFT from C0 ────────────────> C1-full
                          └─> LoRA SFT from same C0 ──────────> C1-lora
 Paraphrase chosen/rejected ─> DPO with frozen C1-lora ref ────> C2-dpo
 JFLEG + preference val + safety ─> Benchmark/Failure Analysis

                          ONLINE / DEMO
 CLI input -> request validation -> tokenizer -> custom decoder-only model
           -> autoregressive generator (KV Cache) -> output checks
           -> candidate/warning + metrics log
```

### Baseline architecture (start point)

- A small Transformer Decoder-Only using **`torch.nn.Module`** and PyTorch tensor operations for attention and generation; **do not** use prebuilt `AutoModelForCausalLM` or `nn.Transformer` as the primary implementation.
- Approximate **3M parameter scale**: 4 decoder layers, `d_model=256`, 4 attention heads, BPE vocab 2048, 2-layer FFN, causal mask, RMSNorm, RoPE or documented position implementation.
- Initially `training_context_length=128`, subject to the **token-length feasibility gate** above.
- `KVCache` stores dynamic K/V tensors per layer; `generate` supports cache ON/OFF and both prefill/decode.
- LoRA applied to `Wq` and `Wv`, configurable rank `r` and alpha `α`; SFT uses response-only cross-entropy.
- DPO: reference policy is frozen C1-lora snapshot; train policy from same checkpoint; use response-token log-likelihoods for both chosen and rejected with a correctly signed objective.

**Caution:** A 3M model on a small budget is a mechanism demonstration. Do **not** claim it performs at the level of published CoEdIT models; those use much stronger foundation models.

---

## 9. Core research experiments / Evidence

| Experiment | Controlled comparison | Data | Primary evidence |
|---|---|---|---|
| E0 | Before vs after base pre-training | TinyStories | train/val NLL, perplexity, generation samples |
| E1 | KV Cache ON vs OFF | Fixed token prefixes at **128, 512, 1024, 2048** | tokens/s, TTFT, TPOT, allocated/reserved VRAM; logits equivalence |
| E2 | Full SFT vs self-written LoRA | Same CoEdIT SFT set and C0 | trainable params, optimizer state, peak VRAM, runtime, GEC quality |
| E3 | C1-lora vs C2-dpo | Same held-out paraphrase inputs and DPO validation | preference win rate, reference-relative margin, meaning/grammar regression |
| E4 | Output checks OFF vs ON | Same benign/boundary inputs | invalid output %, false rejects, overhead |
| E5 | Blinded LLM-as-a-Judge pair comparison | Fixed identical prompts and raw model outputs | pairwise wins/losses/ties, rubric, judge version and cost |
| E6 | Failures / regressions | Frozen external examples | >=5 genuine examples with input, expected, observed, cause hypothesis, mitigation |

### Metrics by task

**Grammar correction:** Exact-match is a weak supplement; report at least a multi-reference metric where practical (e.g. GLEU on JFLEG), **edit rate**, invalid-output rate, and human-reviewed failures. A correct paraphrase may differ from any reference; do not overstate exact-match.

**Paraphrase preference:** held-out chosen-vs-rejected conditional log-prob preference accuracy, DPO margin, generated-answer pairwise judge outcome, and manual checks for semantic drift. Chosen/rejected pairwise scoring is *not equivalent to* evaluating the quality of independently generated paraphrases.

**Application runtime:** request latency p50/p95, throughput, token counts, cache hit/active state in the local experiment, timeouts, generation failure rate and current model checkpoint.

**Guardrails:** injection detection rate **on the selected dataset**, false-positive rate, input/output policy violations; never claim broad safety coverage from one small corpus.

### Judge protocol

The exam explicitly calls for LLM-as-a-Judge. Use an examiner-accepted external judge, e.g. an available current equivalent to the models named in the examination, **if accessible**. Blind the two checkpoint identities; randomize A/B order; allow tie/both-invalid; score grammar, meaning preservation, relevance and safety. Record exact model identifier, date, judge prompt, outputs and counts. A judge is a secondary assessment—not ground truth. **If no judge API/access is available, request instructor approval for a substitute rather than silently skipping it.**

### Integrity rules

- Every claimed speedup requires paired measurements with same weights, prompt lengths, batch, dtype, GPU and benchmark implementation.
- Cache correctness tested first (cached vs uncached logits); `2048` is a **mechanical** context benchmark if trained at only 128/256, not proof of long-context linguistic skill.
- DPO improvement **must be limited to measured preference metrics**; no inferred improvement in grammar without a direct GEC eval.
- Experimental plots must come from real exported logs; never prefill expected results.

---

## 10. Guardrails & user trust

### Input policy

- Reject empty/oversized input; cap input tokens and generated tokens.
- Sanitize log display and JSON encoding, not the original linguistic meaning.
- Treat all user text as untrusted *content to edit*, never as a replacement for app/system instructions.
- For prompt-injection testing, inputs may include instructions such as “ignore previous rules”; the edit task must not grant them higher authority.

### Output policy

- Return bounded plain-text candidate with explicit status and checkpoint identity.
- Reject empty, repeated, binary/control-heavy or grossly malformed generations; log reason without printing sensitive data.
- Mark uncertain/failed outputs for review; do not auto-replace text with a reference solution and then score it as a model success.
- Do not generate invented grammar explanations in P0; consider deterministic feedback templates only as a later feature with clearly separated provenance.

### Privacy

- Default to **local CLI** and no raw-user-input retention; store only benchmark records from public datasets.
- Never claim a user's sentence was validated by a human unless a human actually reviewed it.
- Any future user history, analytics, or profile requires explicit consent, retention rules and access control.

---

## 11. Observability & application engineering

Even a small CLI should generate structured events that support replay and regression tracking:

```json
{
  "event": "inference_completed",
  "request_id": "demo-001",
  "task": "fix_grammar",
  "model_version": "lora_sft-v1",
  "tokenizer_hash": "<from actual manifest>",
  "input_tokens": null,
  "output_tokens": null,
  "latency_ms": null,
  "status": "candidate_unverified"
}
```

`null` is an unmeasured placeholder in this **contract example**, not a monitoring reading. Actual requests must populate real measurements.

### Minimum operational signals

- Model/hash/config versions, data split/revision and active task.
- Request counts, p50/p95 latency, generation length, errors, timeout, output-validation failures.
- Offline: GEC evaluation by checkpoint, paraphrase win/tie rates, guardrail false positives.
- Later: shadow evaluation, feedback collection and drift monitoring; **not** part of 4-week deliverable.

**Backend transfer:** clean module boundaries, artifact versioning, request validation, reproducible benchmarks, safe failure handling and regression-testing discipline.

---

## 12. Four-week execution roadmap

| Week | Examination milestone | Product/engineering output | Hard gate |
|---|---|---|---|
| **1 — Foundation** | Proposal + tokenizer + decoder-only architecture | Trained BPE; causal attention/forward/backward tests; dataset viability check and research question | Tokenizer round-trip; model overfits tiny batch; source manifests resolved |
| **2 — C0 + Cache** | Self-written autoregressive model + KV profiling | Base checkpoint; generation CLI; KV Cache correctness; four-length timing/VRAM charts | Cached/uncached outputs agree within numerical tolerance; logs and plots saved |
| **3 — Alignment** | SFT, LoRA, DPO training + loss curves | CoEdIT full/LoRA comparison; paraphrase DPO with fixed reference and preference eval | Adapter frozen-base tests; DPO objective tests; checkpoints and comparisons recorded |
| **4 — Product & Evidence** | Guardrails + judge + report + demo | Two CLI modes, JFLEG eval, safety tests, five failures, IEEE/ICLR-style 6–8-page report | One-command documented reproduction, raw data/plots, candid limitations |

### Stop rules

- If foundational causality/gradient tests fail, do not spend GPU budget on long training.
- If the 128-token context leaves too few valid DPO preference pairs, decide on 256-token training **early** (or document DPO task limitation) rather than silently truncating.
- If C0 is poor, narrow prompts/outputs and report model-quality limitations; **do not replace custom architecture with a prohibited model class**.
- If generated corrections are poor, still deliver valid core algorithms, comparative metrics and defensible negative results; do not claim a successful learner product.
- Never add UI, database, RAG, accounts or multi-agent features at the expense of exam compliance.

---

## 13. Examination traceability & acceptance artifacts

| Exam criterion | Weight | Evidence file or demo |
|---|---:|---|
| Tokenizer self-trained | 10% | tokenizer artifact, encode/decode tests, vocabulary manifest |
| Correct Transformer | 10% | source, shapes/causality/autograd tests, forward pass |
| KV Cache from scratch | 10% | cache source, equivalence tests, 4-length TTFT/TPOT/tokens/s/VRAM plots |
| Objective evaluation + error analysis | 10% | GEC and preference benchmarks, five analyzed failures, guardrail test report |
| Scientific report | 10% | IEEE/ICLR-style PDF (6–8 pages), methods, limitations and sources |
| Presentation | 10% | CLI demo that shows actual outputs/checkpoint and profiling charts |
| Defense | 10% | explain tokenizer, attention, loss masking, cache, LoRA, DPO and task distinctions |
| Track B: specialized work | 20% | full SFT vs LoRA; DPO vs SFT; input/output guardrails; pairwise judge |
| Improvement | 10% | one quantified improvement or insight (e.g. trainable-parameter reduction with bounded GEC quality loss), never assumed |

**Important:** The `EXAMINATION.md` is binding. The use of CoEdIT/Paraphrase datasets instead of the sample datasets in the exam appears consistent with the exam giving examples, but **confirm with examiner if they have course-specific restrictions**. Neither a tiny model nor a short product demo is automatic evidence of full compliance.

---

## 14. Product success criteria & definition of done

### Engineering success (must pass)

- [ ] All main network/attention/generation code is written with PyTorch primitives.
- [ ] Tokenizer is trained and integrated; vocabulary/checkpoint compatibility recorded.
- [ ] Decoder-only causal model trains, saves/loads, generates autoregressively.
- [ ] KV Cache correctness proven and all four benchmark lengths run and plotted.
- [ ] Full SFT and self-written LoRA implemented with fair controlled comparison.
- [ ] DPO objective unit-tested, reference frozen, training run and before/after eval captured.
- [ ] Input/output guardrails and an honest safety evaluation are present.
- [ ] External judge comparison completed or an authorized alternative documented.
- [ ] At least five **real model failures** identified, explained and reproduced.
- [ ] Paper and one-command run documentation submitted.

### Learner-product success (best effort; NOT guaranteed)

- [ ] Can enter one sentence and receive a candidate edit or explicit failure.
- [ ] Can compare `lora_sft` with `dpo` on paraphrase mode.
- [ ] A small, frozen set demonstrates whether corrections beat a copy-input baseline.
- [ ] UX never disguises hallucinated edits or syntactic output failures.

### Minimum baselines

- **Copy-input baseline:** retain original sentence unchanged; useful to identify overediting.
- **Simple edit heuristic baseline (optional):** a deterministic grammar rule only if it is readily available; no new labeled data required.
- **C0, C1-full, C1-lora, C2-dpo:** separated checkpoints with identical evaluation code and input split.

---

## 15. Key risks, constraints, and trade-offs

| Risk | Why it matters | Mitigation |
|---|---|---|
| 3M model does not learn good GEC | Tiny model and tiny pretraining are underpowered | Clear research framing; low-complexity inputs; report negative results; no production claims |
| SFT and DPO train on different editing modes | Improvement may be falsely attributed to grammar | Explicitly separate GEC and paraphrase metrics and checkpoint comparisons |
| Context 128 discards DPO pairs | Long chat prompts + two completions | Count valid pairs first; train length 256 if budget; never asymmetric truncation |
| JFLEG overlaps with CoEdIT source data | Inflated apparent generalization | Exact/near dedup or document suspected contamination; freeze eval; do not train on JFLEG |
| Preference labels are not equally strong | Chosen/rejected quality can be noisy | Preserve dataset prompt; report pairwise margins and manual failures |
| LLM-as-a-Judge access or cost | Explicit track requirement | Approve judge/model/cost early; fallback only with instructor approval |
| Prompt injection evaluation is too narrow | Bench only measures selected attack distribution | Limit claim; pair with deterministic bounds and false-positive measurements |
| 2048 KV run is slow | Naive non-cache decoding repeats work | Smaller fixed output tokens; use controlled GPU profiling; report time honestly |
| License or redistribution uncertainty | Third-party data limitations | Pin cards/revisions, cite and avoid redistributing raw data; review restrictions before public release |

---

## 16. Proposed repo boundaries (not a full implementation spec)

```text
alignforge/
├── docs/
│   ├── PROCUCT_IDEA.md
│   ├── EXAMINATION.md
│   ├── LITE_EXECUTION_PLAN.md
│   └── PROJECT_SOURCE_OF_TRUTH.md
├── configs/
│   ├── model.yaml
│   ├── sft.yaml
│   └── dpo.yaml
├── src/alignforge/
│   ├── tokenizer.py
│   ├── model.py
│   ├── kv_cache.py
│   ├── generate.py
│   ├── data.py
│   ├── train_pretrain.py
│   ├── train_sft.py
│   ├── lora.py
│   ├── train_dpo.py
│   ├── safety.py
│   ├── evaluate.py
│   └── cli.py
├── tests/
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_cache.py
│   ├── test_lora.py
│   ├── test_dpo.py
│   └── test_contracts.py
├── artifacts/             # generated; large files gitignored
├── reports/               # plots, CSVs, paper
├── pyproject.toml
└── README.md
```

Keep the codebase a **modular monolith**. No extra backend service is required to prove application engineering.

---

## 17. Open decisions / Assumptions to validate first

| Decision | Default | Deadline to settle |
|---|---|---|
| Task choice | Grammar correction primary; paraphrase experimental | Before dataset preprocessing |
| CoEdIT `paraphrase` labels & data count | Inspect actual available `task` values | Day 1 |
| Training context | 128, optionally 256 if DPO data retention is inadequate | Day 2 |
| DPO dataset license | Inspect exact revision and terms | Before first DPO download/publication |
| JFLEG overlap | Run cross-dataset de-duplication audit | Before final benchmark |
| External LLM judge | Instructor-approved current model and small sample | Week 1 |
| Hardware | Single GPU/Colab; baseline time + memory pilot | Week 1 |
| Model competence | Unknown until measured; do not set inflated target | Week 2 onward |

---

## 18. External sources & research links

**Public datasets:**

- TinyStories: https://huggingface.co/datasets/roneneldan/TinyStories
- CoEdIT: https://huggingface.co/datasets/grammarly/coedit
- Paraphrasing Preferences: https://huggingface.co/datasets/alecccdd/paraphrasing-preferences-orpo-dpo
- JFLEG: https://huggingface.co/datasets/jhu-clsp/jfleg
- Prompt Injections: https://huggingface.co/datasets/deepset/prompt-injections

**Relevant literature:**

- TinyStories (2023): https://arxiv.org/abs/2305.07759
- CoEdIT (2023): https://arxiv.org/abs/2305.09857
- LoRA (2021): https://arxiv.org/abs/2106.09685
- DPO (2023): https://arxiv.org/abs/2305.18290
- JFLEG benchmark (2017): https://arxiv.org/abs/1702.04066

**Internal source documents:** `EXAMINATION.md`, `PROJECT_SOURCE_OF_TRUTH.md`, `LITE_EXECUTION_PLAN.md`. This product idea does **not** replace the examination or technical specification; it freezes the product scope and application-facing goals.

---

## Final decision

**AlignForge English is an experimental English Writing Coach, not a general English-learning platform.** Implement the main academic architecture from scratch, adapt it to public short-text editing data, study LoRA and paraphrase DPO separately, and build an honest CLI with evaluation and safety boundaries. **Only expand the product after all four examination milestones have evidence.**
