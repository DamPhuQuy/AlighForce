# MiniAlign — Technical Source of Truth

> **Building, Fine-Tuning, Aligning, and Securing a Small Language Model from Scratch**  
> Đồ án cuối khóa, **Hướng B — Parameter-Efficient Fine-Tuning, Preference Alignment & Safety**  
> **Version:** 1.0.0 · **Status:** Proposed / implementation-ready specification · **Updated:** 2026-10-28  
> **Owner:** 01 sinh viên · **Duration:** 04 tuần · **Target:** 01 GPU (12–24 GB) hoặc Colab T4/L4  
> **Primary audience:** Người theo đuổi **AI Application Engineering** với nền tảng **Backend Engineering**.

**Tuyên bố tính trung thực:** Đây là **đặc tả kỹ thuật và kế hoạch thực hiện**, không phải báo cáo kết quả thực nghiệm đã chạy. Những trị số model, epoch, số mẫu, benchmark threshold được ghi là **giá trị xuất phát / mục tiêu / gate**, không phải kết quả đo. Các lệnh CLI bên dưới là **giao diện dự kiến phải hiện thực**, không phải cam kết đã tồn tại. Mọi quyết định còn phụ thuộc tài nguyên thực tế phải được ghi trong *Decision Log*.

---

## 0. Cách sử dụng tài liệu này như source of truth

### 0.1. Quy tắc vận hành

1. **Single source of truth:** Khi code, notebook, báo cáo và README mâu thuẫn nhau về phạm vi, định nghĩa metric hoặc pipeline, tài liệu này được ưu tiên cho tới khi có phiên bản sửa đổi được ghi nhận.
2. **No cargo cult:** Không triển khai kỹ thuật chỉ vì đó là từ khóa hot. Mỗi module phải trả lời được **Why → How → Correctness → Evidence → Failure modes**.
3. **No benchmark without baseline:** Không tuyên bố improvement nếu thiếu baseline, biến kiểm soát, raw logs và phân tích bất định.
4. **No silent shortcuts:** Kiến trúc chính, attention, cache, generation engine, LoRA module, DPO objective phải do sinh viên viết bằng PyTorch. Không giấu việc dùng pretrained weights hoặc thư viện đóng gói.
5. **Frozen test set:** Không dùng test để lựa chọn hyperparameter, sinh dữ liệu train hoặc sửa prompt sau khi nhìn kết quả.
6. **Security outside the model:** Backend authorization, validation và data access control phải được code thực thi ngoài mô hình; guardrails không phải cơ chế bảo mật duy nhất.
7. **Reproducibility by design:** Cố định config, seed, dataset revision, tokenizer, git commit, môi trường và checkpoint. Chạy lại cùng điều kiện phải truy lại được trace thí nghiệm.
8. **Scope > ambition:** Đồ án có 4 tuần; hoàn thành core đúng + một nghiên cứu thuyết phục quan trọng hơn một demo lớn nhưng không thể kiểm chứng.

### 0.2. Ký hiệu mức độ yêu cầu

| Nhãn | Ý nghĩa | Có thể bỏ? |
|---|---|---|
| **MUST** | Bắt buộc bởi đề bài hoặc cần thiết để nghiên cứu có hiệu lực | Không |
| **SHOULD** | Khuyến nghị mạnh, tác động đáng kể đến chất lượng | Chỉ bỏ khi ghi lý do |
| **COULD** | Mở rộng, có giá trị nghề nghiệp | Có |
| **NOT NOW** | Ngoài phạm vi 4 tuần | Có, và nên bỏ |

### 0.3. Danh mục nội dung

1. [Executive decisions](#1-executive-decisions)
2. [Vấn đề, mục tiêu và câu hỏi nghiên cứu](#2-vấn-đề-mục-tiêu-và-câu-hỏi-nghiên-cứu)
3. [Phạm vi và ràng buộc](#3-phạm-vi-và-ràng-buộc)
4. [Kiến trúc tổng thể và ranh giới module](#4-kiến-trúc-tổng-thể-và-ranh-giới-module)
5. [Foundations — học để hiểu bản chất](#5-foundations--học-để-hiểu-bản-chất)
6. [Model specification & Transformer from scratch](#6-model-specification--transformer-from-scratch)
7. [Tokenizer & dữ liệu pre-training](#7-tokenizer--dữ-liệu-pre-training)
8. [Pre-training và inference engine](#8-pre-training-và-inference-engine)
9. [KV Cache & profiling](#9-kv-cache--profiling)
10. [SFT & LoRA](#10-sft--lora)
11. [DPO & preference alignment](#11-dpo--preference-alignment)
12. [Guardrails, threat model & secure application](#12-guardrails-threat-model--secure-application)
13. [Dataset strategy và data contracts](#13-dataset-strategy-và-data-contracts)
14. [Evaluation, ablation & statistical rigor](#14-evaluation-ablation--statistical-rigor)
15. [Repository architecture & engineering conventions](#15-repository-architecture--engineering-conventions)
16. [Kế hoạch 4 tuần & acceptance gates](#16-kế-hoạch-4-tuần--acceptance-gates)
17. [Risks, fallbacks và stop rules](#17-risks-fallbacks-và-stop-rules)
18. [Scientific report & grading traceability](#18-scientific-report--grading-traceability)
19. [Career transfer map](#19-career-transfer-map)
20. [Decision log, open questions & references](#20-decision-log-open-questions--references)
21. [Appendices — contracts, test vectors & operational checklists](#21-appendices--contracts-test-vectors--operational-checklists)

---

## 1. Executive decisions

### 1.1. Chọn hướng

**Selected track: HƯỚNG B — ALIGNMENT.**

Hướng A tối ưu *execution* (MoE, quantization, long context). Hướng B tối ưu *model behavior & adaptation* (SFT, LoRA, DPO, guardrails). AI Application Engineer thường phải quan tâm chất lượng tác vụ, dữ liệu, đánh giá, an toàn và integration nhiều hơn là tự viết GPU kernel; vì thế Hướng B được chọn. Tuy nhiên **KV Cache và profiling** vẫn là phần bắt buộc, giúp duy trì năng lực systems/backend.

### 1.2. Chọn use case

**API Incident Triage Assistant (controlled-domain, evidence-grounded).** Đầu vào là một sự cố API/back-end đơn giản kèm bằng chứng có cấu trúc; đầu ra là phân loại + bước khắc phục trong phạm vi tri thức được cung cấp, hoặc `insufficient_information` nếu dữ liệu không đủ.

**Vì sao không xây chatbot Spring Boot tổng quát?** Một mô hình 8–20M tham số tự pre-train trong 4 tuần khó có kiến thức Java/Spring Boot sâu. Thiết kế bài toán *closed-world* cho phép chứng minh được kỹ thuật bằng metric khách quan. **Đó là lab nghiên cứu LLM**, không phải đối thủ của ChatGPT và không phải hệ thống chẩn đoán sự cố production.

Ví dụ input có kiểm soát:

```json
{
  "task": "api_incident_triage",
  "status_code": 401,
  "symptoms": "GET /profile returns 401 after the client token expires",
  "evidence": "Authorization header contains an expired bearer token",
  "allowed_policy": "Only advise user-visible, non-privileged troubleshooting"
}
```

Ví dụ kết quả mong đợi:

```json
{
  "category": "authentication",
  "action": "refresh_or_reauthenticate",
  "confidence_label": "supported_by_evidence",
  "explanation": "The provided evidence reports an expired bearer token."
}
```

**Lưu ý:** Mô hình có thể không xuất JSON hợp lệ. Output validator phải reject / reprompt có giới hạn / trả về structured error; không tự tuyên bố mọi phản hồi đều đúng schema.

### 1.3. Research thesis

> Trên một decoder-only LM nhỏ tự huấn luyện, liệu **LoRA** có đạt hiệu quả thích nghi tác vụ gần Full Fine-Tuning với ít trainable parameters và bộ nhớ optimizer hơn, và **DPO** có cải thiện hành vi ưu tiên trên các cặp phản hồi domain-specific, dưới sự kiểm soát của **external guardrails**, khi đo bằng benchmark có đối chứng không?

Nghiên cứu phải chấp nhận **negative results**. Nếu DPO làm giảm chất lượng, phải mô tả được nguyên nhân từ năng lực base model, preference noise, overfitting, format brittleness hoặc distribution shift.

### 1.4. Thứ tự ưu tiên đóng băng

| Thứ tự | Deliverable | Mức |
|---:|---|---|
| 1 | Tokenizer + Transformer + train/generate correctness | MUST |
| 2 | Pre-train checkpoint + KV Cache ON/OFF benchmark | MUST |
| 3 | SFT và LoRA tự viết, so sánh với full fine-tuning | MUST |
| 4 | DPO objective tự viết + train + benchmark trước/sau | MUST |
| 5 | Guardrails, threat tests và error analysis | MUST |
| 6 | Chạy được lệnh tái lập và báo cáo 6–8 trang | MUST |
| 7 | FastAPI hoặc Gradio mỏng để demo | SHOULD |
| 8 | RAG / Vector DB / Agent / distributed training | NOT NOW |

## 2. Vấn đề, mục tiêu và câu hỏi nghiên cứu

### 2.1. Problem statement

Nhiều ứng dụng dùng API LLM nhưng không kiểm soát được năng lực và giới hạn mô hình: tokenization sai, latency không được đo, dữ liệu fine-tune không sạch, benchmark leakage, preference optimization không kiểm chứng, và guardrails bị dùng nhầm như rào cản bảo mật. Đồ án xây một **laboratory end-to-end** để hiểu và kiểm nghiệm các lớp đó.

### 2.2. Objectives

- **O1 (core):** Tạo một decoder-only causal LM tự viết, loss giảm trên dữ liệu train; autoregressive generation đúng quy tắc causal.
- **O2 (efficiency):** Triển khai KV Cache và chứng minh tính tương đương về logits/output, đo TTFT/TPOT/tokens/s/VRAM theo độ dài prefix.
- **O3 (adaptation):** Tạo pipeline instruction SFT với target-only loss; tự viết LoRA và so sánh full fine-tune theo compute budget tương đương.
- **O4 (preference):** Tự viết DPO loss với reference model frozen, so sánh preference margin và downstream quality trước/sau.
- **O5 (safety):** Triển khai input/output guardrails, đánh giá attack success + false-positive; công khai limitation.
- **O6 (engineering):** Có test, config, data lineage, checkpoint integrity, benchmark scripts và báo cáo tái lập được.

### 2.3. Research questions & falsifiable hypotheses

| ID | Research question | Giả thuyết **cần kiểm chứng**, không phải kết luận |
|---|---|---|
| RQ1 | Cache ảnh hưởng thế nào đến generation? | H1: TPOT có cache giảm ở prefix dài, trong khi cache dùng thêm VRAM. |
| RQ2 | LoRA tiết kiệm tài nguyên bao nhiêu so với Full FT? | H2: LoRA giảm trainable params, optimizer state; chênh lệch peak VRAM phụ thuộc activations/batch. |
| RQ3 | Với cùng budget, LoRA có giữ được chất lượng? | H3: Domain metric của LoRA có thể gần Full FT trong ngưỡng được báo cáo. |
| RQ4 | DPO có thay đổi preference behavior không? | H4: Held-out pairwise margin và preferred response rate tăng so với SFT; không bảo đảm task accuracy tăng. |
| RQ5 | Guardrails giảm rủi ro gì, đánh đổi gì? | H5: Violation giảm trên threat suite, với chi phí false positive/latency cần đo. |

**Không lựa chọn hướng kỹ thuật sau khi biết kết quả test:** chọn metric, splits và ngưỡng đánh giá **trước** khi run benchmark cuối.

## 3. Phạm vi và ràng buộc

### 3.1. Compliance với đề bài

- Viết kiến trúc bằng **PyTorch `torch.nn.Module`**, không dùng `AutoModelForCausalLM`, `GPT2LMHeadModel`, `LlamaForCausalLM` làm main model.
- Viết attention score/mask/softmax/matmul bằng primitive PyTorch; **không** dùng `nn.Transformer`, `nn.MultiheadAttention`, `scaled_dot_product_attention` làm đường thực thi chính khi nghiệm thu cơ chế attention từ đầu. Có thể dùng SDPA làm *reference/optimization comparator* tách biệt và ghi rõ, nếu giảng viên đồng ý.
- Tự viết **forward, generation loop, sample logic, KV Cache data structure**, LoRA layer, DPO objective, response-only masking.
- Được dùng `torch.optim`, `DataLoader`, `datasets` để đọc dữ liệu, `tokenizers` / `sentencepiece` cho training tokenizer **nếu quy định cho phép**. Để chứng minh hiểu thuật toán, bổ sung implementation BPE mini tự viết ở unit-test/tutorial và tự train tokenizer từ corpus.
- Không dùng API LLM ngoài để sinh câu trả lời của model chính. API judge bên ngoài chỉ là **optional evaluation**, report đầy đủ model/version/cost và tính không tái lập tuyệt đối.
- Pretrained checkpoint chỉ là **optional external-control**, không được coi là thay cho model tự xây và pre-train. Không phụ thuộc vào permission chưa được xác nhận.

### 3.2. In scope / out of scope

**In scope:** single-node single GPU, English-first domain data, batch size nhỏ, dense Transformer 8–20M, offline training, SFT, LoRA, offline DPO, deterministic task evaluation, safety tests, small CLI/API demo.

**Out of scope:** đào tạo LLM tỷ tham số, distributed training, multimodal, tool-using agents, vector database, RLHF/PPO, reward model riêng, MoE, FP8 kernels, multi-tenant auth service, high-QPS production serving, autonomous system administration.

### 3.3. Non-goals

Không tuyên bố học được tri thức backend rộng từ dataset hẹp; không đánh đồng định dạng JSON đúng với reasoning đúng; không dùng LLM-as-a-Judge thay thế ground truth; không tuyên bố safety đầy đủ khi chỉ kiểm tra một tập attack nhỏ.

## 4. Kiến trúc tổng thể và ranh giới module

```text
                          OFFLINE RESEARCH PIPELINE
   Public TinyStories ──> Data ingestion ───> Tokenizer training
             │                                     │
             └──────────────> Causal LM corpus ──> Base Pre-training
                                                     │
                                            Base checkpoint (C0)
                                                     │
               ┌──────────────────────┬──────────────┴──────────────┐
               │                      │                             │
      Domain SFT dataset     Full FT baseline                  LoRA SFT
               │                      │                             │
               └──────────────────────┴──────────────> SFT checkpoint (C1)
                                                            │
      Domain preference pairs ──────> Frozen reference C1 + DPO policy
                                                            │
                                                   DPO checkpoint (C2)
                                                            │
                                 Evaluator: task / preference / safety

                           ONLINE / DEMO PATH
   Request -> Schema validation -> Input policy -> Prompt formatter
                                                    │
                                          Tokenizer + Custom Model
                                                    │
                                          Generation + KV Cache
                                                    │
                                       Output parser + policy checks
                                                    │
                                        Structured response / error
```

### 4.1. Architectural boundaries

| Boundary | Responsibility | Không được làm |
|---|---|---|
| `model/` | mathematical forward/backward, attention, positions | HTTP, file I/O, authorization |
| `tokenization/` | text↔token IDs, special tokens | domain business rules |
| `data/` | ingest/normalize/split/mask | silently modify train/test labels |
| `training/` | optimizer, scheduler, checkpoints, SFT/DPO | hardcode benchmark outcomes |
| `inference/` | cache lifecycle, decode, sampling | make authorization decisions |
| `safety/` | request/output gates, structured validation | trust LLM text as security proof |
| `evaluation/` | immutable metric definitions & scripts | mutate model or dataset |
| `api/` | transport, input contract, timeout, logging | embed training logic |

### 4.2. Các checkpoint có ý nghĩa khác nhau

- **C0 (base):** weights từ causal pre-training; tạo baseline token generation.
- **C1-full:** C0 sau full SFT; làm baseline cho LoRA.
- **C1-lora:** C0 + LoRA adapter, checkpoint riêng; dùng làm **initial policy + frozen reference** trong DPO.
- **C2-dpo:** C1-lora policy sau DPO; so với chính C1-lora trước DPO trên cùng test set.

Không so C2 với C0 rồi suy ra mọi cải thiện là do DPO: chênh lệch đó lẫn tác động của SFT. Khi lưu adapter cần version rõ base checkpoint, tokenizer hash, module targets và merge status.

### 4.3. Quyết định package architecture

`model/` không được phụ thuộc `api/`; `evaluation/` chỉ gọi public interfaces và thu thập kết quả; `training/` thao tác model như một dependency. Dùng interface thông qua type hints thay vì tạo framework/abstract factory không cần thiết. Một người, bốn tuần → **modular monolith** tốt hơn microservices.

## 5. Foundations — học để hiểu bản chất

Mỗi chủ đề học theo chu trình: **Mental model → derivation → tiny implementation → invariant test → real benchmark → explain-back**. Không được đánh dấu đã hiểu chỉ vì chạy được notebook.

### 5.1. Autoregressive language modeling

Với token sequence `x_1,...,x_T`, causal LM mô hình hóa:

$$p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).$$

Nó không lưu sẵn câu trả lời; mỗi bước dự đoán phân phối xác suất token tiếp theo từ prefix. During training, đưa toàn bộ prefix cùng lúc và dùng causal mask (*teacher forcing*). During inference, nhận các token đã sinh và lặp vòng decode.

**Kiểm tra hiểu:** Nếu mask sai và token vị trí `t` nhìn được tương lai `t+1`, vì sao train loss có thể đẹp nhưng generate kém? Cần chứng minh bằng một bài test *future token perturbation*: đổi token ở vị trí tương lai, logits vị trí quá khứ không thay đổi.

### 5.2. Tokenization không phải từ vựng thông thường

Tokenizer ánh xạ text → các đơn vị con (tokens), sau đó embedding bảng tra token ID → vector liên tục. Một từ có thể là nhiều tokens; punctuation/JSON keys cũng tiêu tốn token. Vocabulary nhỏ giảm model embedding size nhưng kéo dài sequence; vocabulary lớn có thể giúp nén token nhưng làm embedding/lm head lớn hơn.

**Trade-off phải đo:** tokens per character trên `TinyStories` và `incident-domain`; rate UNK (nếu có), round-trip Unicode, OOV behavior, hiệu ứng context truncation. Không mặc định BPE là ngữ nghĩa học.

### 5.3. Embedding, self-attention và causal dependency

Với hidden states `X∈R^{B×T×d}`, mỗi head tạo:

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V.$$

$$\mathrm{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.$$

- `Q` biểu diễn điều token hiện tại tìm kiếm; `K` dùng để so khớp; `V` là nội dung được trộn.
- `1/sqrt(d_h)` giảm độ lớn score khi dimension tăng.
- `M[i,j] = -inf` khi `j>i` để loại đường nhìn tương lai.
- Softmax chuẩn hóa weights theo **key dimension**, không phải query dimension.
- Multi-head cho các phép chiếu độc lập; nối outputs rồi chiếu qua `W_O`.

**Invariant:** đổi token tương lai không làm thay đổi output quá khứ; probabilities mỗi hàng trên keys hợp lệ tổng ~1; attention không tạo NaN ở các mask hợp lệ.

### 5.4. Position, RoPE, normalization, FFN

Attention thuần không có trật tự tuyến tính đủ để biểu diễn vị trí tuyệt đối. **RoPE** quay các cặp chiều của Q/K theo góc phụ thuộc vị trí, cho phép biểu diễn quan hệ vị trí trong tích vô hướng. KV Cache phải dùng **absolute position offset chính xác** ở bước decode.

**RMSNorm** chuẩn hóa theo root mean square; **Pre-Norm** đặt normalization trước attention và MLP để gradient đi qua residual path ổn hơn. **SwiGLU** sử dụng cổng phi tuyến điều chỉnh dòng thông tin; khác GELU FFN về số ma trận và số tham số.

**Tại sao phải hiểu?** Model size, FLOPs, tốc độ, chất lượng và LoRA target matrices đều phụ thuộc những lựa chọn kiến trúc này.

### 5.5. Loss, gradient, overfitting, distribution shift

Causal LM và SFT thường dùng cross-entropy trên token mục tiêu:

$$\mathcal L_{CE}=-\frac{1}{N}\sum_{i\in\mathrm{valid\ targets}}\log p_\theta(y_i\mid x_{<i}).$$

- Với pre-training, mọi token hợp lệ (trừ padding/shift boundaries) là target.
- Với SFT prompt-completion, **chỉ completion (bao gồm EOS nếu quy định)** có loss; không phạt model vì không "dự đoán lại prompt".
- Training loss giảm chưa đủ: phải đo held-out metric, phân phối đầu ra và lỗi.
- **PPL** = `exp(mean token NLL)` trên **cùng tokenizer/corpus/normalization**; không so PPL giữa tokenizer khác nhau rồi kết luận model nào tốt hơn.

### 5.6. Mỗi khối cần trả lời được 5 câu hỏi

1. Nếu xóa causal mask thì quan hệ train/generation đổi thế nào?
2. Tại sao cache chỉ lưu K,V mà không cần cache toàn bộ Q?
3. Tại sao SFT cần response-only masking?
4. Tại sao DPO cần **frozen reference policy**?
5. Tại sao hệ thống vẫn phải validate JSON và enforce ACL dù đã train safety preferences?

Không trả lời được bằng ví dụ tensor cụ thể → chưa hoàn thành mục tiêu hiểu sâu.

## 6. Model specification & Transformer from scratch

### 6.1. Configuration baseline

**Candidate P (primary)** — chỉ là xuất phát, phải benchmark GPU thật trước khi khóa:

```yaml
model:
  architecture: decoder_only
  vocab_size: 8192
  max_seq_len: 512
  d_model: 384
  n_layers: 6
  n_heads: 6
  n_kv_heads: 6   # Standard MHA; avoid introducing GQA complexity
  ffn_hidden: 1536
  attention: causal_mha_manual
  positional_encoding: rope
  normalization: rmsnorm_pre
  activation: swiglu
  tie_word_embeddings: true
  dropout: 0.0
  bias: false
  special_tokens: ["<pad>", "<bos>", "<eos>", "<unk>"]
```

Ước tính khoảng **17–18M parameters** (tùy embeddings, biases, implementation). Phải in **actual parameter count** từ code; không dùng số ước tính làm số liệu báo cáo.

**Candidate S (fallback)**:

```yaml
model:
  vocab_size: 4096
  max_seq_len: 256
  d_model: 256
  n_layers: 6
  n_heads: 4
  ffn_hidden: 1024
  positional_encoding: rope
  normalization: rmsnorm_pre
  activation: swiglu
  tie_word_embeddings: true
```

Khoảng 7–8M parameters theo cấu hình tương ứng; cần đo bằng code. Nếu primary không đạt training throughput / GPU budget tối thiểu, chuyển sang fallback ngay sau smoke run, không đợi cuối tuần 2.

### 6.2. Forward contract

```python
forward(
    input_ids: LongTensor[B, T],
    attention_mask: Optional[BoolTensor[B, T]],
    position_ids: Optional[LongTensor[B, T]],
    past_key_values: Optional[Cache],
    use_cache: bool,
) -> ModelOutput(logits: FloatTensor[B, T, V], past_key_values: Optional[Cache])
```

**Tensor invariants:**

| Thành phần | Shape |
|---|---|
| Embeddings | `[B,T,d_model]` |
| Q/K/V sau reshape | `[B,n_heads,T,d_head]` |
| Attention scores | `[B,n_heads,T_query,T_keys]` |
| Cache từng layer K/V | `[B,n_kv_heads,T_cached,d_head]` |
| Model logits | `[B,T,vocab_size]` |

`d_head = d_model/n_heads` phải là integer. `position_ids` không được quay lại 0 mỗi token decode. Kết hợp padding mask và causal mask theo đúng vị trí thật; test `B>1`, lengths không đồng nhất (nếu hỗ trợ batching) hoặc khai báo **batch-1 only** cho demo và không nhận sai hợp đồng.

### 6.3. Training forward / decode forward

- **Full-sequence train:** Q/K/V của tất cả token, triangular causal mask, không dùng cache; giữ computational graph.
- **Prefill:** xử lý prefix một lần, nhận logits cuối và lưu KV đủ mọi layer; `use_cache=True`, `eval`, `inference_mode`.
- **Incremental decode:** chỉ đưa token mới, kết hợp cached K/V với K/V mới; mỗi bước cache tăng chính xác 1 token; output logits của token mới.
- **EOS termination:** mỗi sample có thể kết thúc khác nhau; với batch-1 đơn giản, dừng ngay khi EOS hoặc `max_new_tokens`.

### 6.4. Unit/property tests bắt buộc

| Test ID | Test | Acceptance |
|---|---|---|
| MOD-01 | Shapes + dtype/device | logits `[B,T,V]`; không NaN/Inf |
| MOD-02 | Causality | future perturbation không ảnh hưởng past logits |
| MOD-03 | Autograd | `loss.backward()` có gradient hữu hạn cho trainable weights |
| MOD-04 | Overfit one tiny batch | loss giảm rõ rệt trong 100–500 steps mini-debug |
| MOD-05 | Position consistency | RoPE incremental vs full với offsets chuẩn |
| MOD-06 | Logit equivalence | cache/no-cache `allclose` theo dtype tolerance |
| MOD-07 | State integrity | save/load cùng config tái tạo logits gần như bằng nhau |
| MOD-08 | Max length contract | reject/truncate theo rule minh bạch; không silent overflow |

**Lỗi hay gặp:** pad token vô tình được tính loss; tokenizer EOS không nhất quán; `target = input_ids` nhưng quên shift một vị trí; mask broadcast sai; dropout active trong cache comparison; cache có device/dtype khác model.

## 7. Tokenizer & dữ liệu pre-training

### 7.1. Dataset pre-training: TinyStories

- **Source:** [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories).
- **Purpose:** học causal language modeling trên English đơn giản, dễ quan sát convergence với model nhỏ.
- **License được hiển thị:** CDLA-Sharing-1.0 tại thời điểm kiểm tra; phải xem điều kiện license trước phân phối bản sao dữ liệu hoặc derivative.
- **Không dùng toàn bộ dữ liệu mặc định:** tập quá lớn để mặc định huấn luyện đủ epoch trong 4 tuần; ưu tiên fixed token budget và giữ train/validation nguyên tắc rõ ràng.

**Sampling protocol:** cố định nguồn file/revision; random sample train theo `seed=42`; reserve official validation riêng, tuyệt đối không trộn. Bắt đầu với **5–20 triệu tokens** cho smoke/first run (quy mô ngân sách đề xuất, không phải guarantee), mở rộng khi throughput và remaining GPU-hours cho phép. Báo cáo token budget *thực tế*, không chỉ số record hoặc epoch. Tokenizer train trên **train only**.

### 7.2. Tokenizer implementation plan

**Layer 1 (learning):** Tự triển khai Byte-Pair Encoding (BPE) trên corpus nhỏ để chứng minh hiểu:

1. Chuẩn hóa input đã định nghĩa (`UTF-8`, newline policy, Unicode normalization nếu dùng).
2. Bắt đầu từ base symbols và special tokens (nếu byte-level, có 256 byte tokens).
3. Đếm các cặp symbol xuất hiện liền kề trong training corpus.
4. Gộp cặp có tần suất cao nhất; ghi merge rule; lặp tới target vocab.
5. Tokenize bằng merge rules theo **đúng thứ tự ưu tiên**, không greedy merge tùy ý.
6. Encode / decode round-trip Unicode, newlines, JSON punctuation, string rỗng.

**Layer 2 (production baseline):** Train tokenizer riêng bằng Hugging Face `tokenizers` hoặc SentencePiece với cùng seed/corpus; lưu `tokenizer.json`, config, hash và special token IDs. Được phép dùng thư viện để tối ưu tốc độ theo đề bài, nhưng phải có notebook/BPE implementation chứng minh nguyên lý.

**Quy tắc không thay đổi tokenizer sau pre-training:** đổi vocabulary hoặc token ID mapping làm vô hiệu checkpoint embeddings, pairwise likelihood và benchmark cũ. Khóa `tokenizer_sha256` trong mọi checkpoint.

### 7.3. Tokenizer acceptance

| ID | Experiment | Pass |
|---|---|---|
| TOK-01 | `decode(encode(text))` | Bảo toàn văn bản theo normalization đã công bố |
| TOK-02 | Unknown chars, Vietnamese accents, emoji, JSON | Không crash; byte fallback nếu cam kết hỗ trợ |
| TOK-03 | BOS/EOS/PAD handling | IDs distinct; PAD không được tính loss |
| TOK-04 | Re-training deterministic | Same vocab/merges khi cùng code, seed, data và environment |
| TOK-05 | Compression | Report tokens/char trên pretrain và domain dataset |
| TOK-06 | Special token injection | Không để dữ liệu tự do giả mạo special tokens nếu serializer có escape rules |

## 8. Pre-training và inference engine

### 8.1. Data format & target shifting

Một sequence token `x=[bos, t1, t2, ..., eos]`. Training causal LM dùng:

```text
input  = [bos, t1, t2, ..., t_{n-1}]
target = [t1,  t2, t3, ..., eos]
```

Cross-entropy chỉ trên valid target positions, `ignore_index=-100` cho padding. Với packed sequences, **không được vô tình cho phép attention xuyên sample boundary** nếu tuyên bố mỗi record độc lập; cách đơn giản nhất là concat corpus liên tục kèm EOS giữa documents và công bố đó là training assumption. Với SFT cần preserve prompt/completion boundary và mask đúng.

### 8.2. Training recipe (starter, không phải kết quả tối ưu)

```yaml
pretrain:
  dataset: roneneldan/TinyStories
  seed: 42
  batch_size: 8
  gradient_accumulation_steps: 4
  sequence_length: 512
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.1
  warmup_ratio: 0.03
  grad_clip_norm: 1.0
  precision: fp16_or_bf16_if_supported
  validate_every_steps: 200
  save_every_steps: 500
  early_stop_on_nonfinite_loss: true
  checkpoint_selection: best_validation_nll
```

Đây chỉ là **starting configuration**; batch/sequence phải giảm khi OOM, LR tuning dựa trên dev split. Với FP16 có thể cần `GradScaler`; BF16 chỉ dùng nếu GPU và PyTorch hỗ trợ thực sự. Log `tokens_seen` để so sánh run công bằng.

### 8.3. Training observability

Mỗi run lưu: `train_nll`, `val_nll`, `val_ppl`, `lr`, `grad_norm`, `tokens_seen`, `step_time_ms`, `peak_allocated_vram`, `peak_reserved_vram`, `sample_generation` cố định theo seed, `git_commit`, `dataset_hash` và config JSON/YAML.

**Minimum correctness gate trước khi training dài:** (1) tokenizer pass; (2) overfit 1 batch; (3) loss finite; (4) save/load roundtrip; (5) checkpoint resume optimizer/scheduler đúng; (6) trên 100–500 steps validation cho thấy xu hướng học được, không đòi đồ thị đẹp giả tạo.

### 8.4. Sampling & autoregressive generation

Pipeline: tokenize prompt → model forward/prefill → logits của vị trí cuối → apply temperature/top-k (nếu dùng) → sample token → append → incremental decode → stop EOS/limit.

- **Greedy** là `argmax`; deterministic dưới cùng weights/input.
- **Temperature:** `logits / T`, với `T>0`. Không dùng `T=0` chia trực tiếp; `temperature=0` được định nghĩa là greedy.
- **Top-k:** giữ k logits cao nhất rồi normalize; cần test `1≤k≤V`.
- **Stop:** EOS và token budget; không cho response chạy vô hạn.
- **Prompt truncation:** phải có chính sách cụ thể (reject khi vượt `max_seq_len` hoặc truncate theo thuật toán có trace); benchmark không được lén thay input.

Không dùng `transformers.generate()` để thay generation engine tự viết.

## 9. KV Cache & profiling

### 9.1. Vì sao cache hoạt động?

Đối với self-attention autoregressive, sau khi đã tính K/V của prefix, các biểu diễn K/V của token trước **không phụ thuộc token tương lai** nhờ causal mask. Khi sinh token mới, chỉ cần tính Q/K/V cho token mới và attention của Q mới trên cached keys + key mới. Từ đó tránh recompute toàn bộ prefix ở mỗi decode step.

- **Không cache:** mỗi step chạy model trên prefix dài hơn, làm lại attention, projections và FFN của các token cũ.
- **Có cache:** prefill xử lý prefix một lần, sau đó decode từng token mới. TPOT thường tốt hơn nhưng VRAM tăng do lưu K/V.
- **Phân biệt:** prefill cost khác incremental decode cost; **TTFT** đôi khi không cải thiện đáng kể với cache vì prefill vẫn phải chạy toàn prefix.

### 9.2. Cache data contract

```python
@dataclass
class LayerKV:
    key: torch.Tensor     # [B, H_kv, capacity, head_dim]
    value: torch.Tensor   # [B, H_kv, capacity, head_dim]
    length: int           # visible/valid tokens, <= capacity

@dataclass
class KVCache:
    layers: list[LayerKV]
    batch_size: int
    dtype: torch.dtype
    device: torch.device
    max_length: int
```

**Contract:** `append(layer, k_new, v_new)` ghi vào đoạn `[length:length+T_new]`, không thay đổi tokens cũ; `view()` trả phần length hữu hiệu; `reset()` xóa trạng thái logical; không dùng cache từ user/request trước cho user/request mới. Cần hủy cache ở kết thúc hoặc timeout.

**Cách đơn giản:** dynamic `torch.cat` mỗi step có allocation/re-copy; được dùng như baseline correctness. **Cách nên dùng:** preallocated cache theo `max_length` hoặc grow-by-block để giảm allocation. Tuy nhiên preallocation có thể làm peak reserved VRAM tăng; phải báo cáo cả allocated/reserved.

### 9.3. KV memory estimation

$$\mathrm{Bytes}_{KV}=2\cdot N_{layers}\cdot B\cdot H_{kv}\cdot L\cdot d_{head}\cdot\mathrm{bytes\_per\_element}.$$

Ví dụ primary config, FP16, batch=1, `N_layers=6`, `H_kv=6`, `d_head=64`:

| Context `L` | K/V payload lý thuyết (MiB) |
|---:|---:|
| 128 | 1.125 |
| 512 | 4.5 |
| 1024 | 9.0 |
| 2048 | 18.0 |

Đây là kích thước **payload tensors của KV**, không bao gồm weights, activations, CUDA allocator overhead, model workspace hoặc reserved capacity. Với model trained at 512, test L=1024/2048 **chỉ là cache/inference mechanics**, không được suy diễn chất lượng long-context tốt hoặc vượt giới hạn RoPE đã học.

### 9.4. Cache correctness: critical test

1. Model ở `eval()`, dropout off, logits float32 cho test tính đúng nếu GPU cho phép.
2. Cùng prefix, chạy một lượt `forward(full_prefix)`; so với prefill/incremental từng token ở chính vị trí đó.
3. Assert logits cuối tương đương theo tolerance đã ghi (ví dụ `atol=1e-4, rtol=1e-4` float32; điều chỉnh hợp lý với FP16, không ép bitwise equality).
4. Với greedy decoding: cùng prompt/seed phải ra **cùng chuỗi token** (ngoại trừ bất định số học ở các logits gần bằng nhau; nếu gặp, so logits trước).
5. Test `reset`, EOS, empty-prefix policy, max-length, mismatched dtype/device, context padding, batch=1 supported mode.

**Causal masking gotcha:** với `T_query=1`, `T_keys>1`, lower triangular matrix kiểu square/upper-left đơn giản có thể sai. Phải xây mask theo **global positions** hoặc đảm bảo đường cache single-token query nhìn toàn prefix hợp lệ và không nhìn future tokens.

### 9.5. Measurement protocol

Thực nghiệm `L∈{128,512,1024,2048}`, `max_new_tokens=64` (hoặc giá trị cố định đã chốt), `B=1`, `greedy`, same prompt token IDs/checkpoint/precision/device. Mỗi trường hợp: 5 warm-ups + ≥20 measured runs nếu budget cho phép. Dùng CUDA events hoặc synchronize trước/sau, report mean/median/p95 và std; dùng cùng protocol cho cache ON/OFF.

Metrics:

- **TTFT:** thời gian request → token đầu, bao gồm prefill và first decode/sample theo chính sách được công bố.
- **TPOT:** trung bình thời gian giữa các token sau token đầu (`(t_last-t_first)/(N-1)` khi N>1); không dùng tổng latency /N gọi là TTFT.
- **Decode throughput:** số **generated tokens** sau prefill chia decode walltime.
- **Peak memory:** `max_memory_allocated`, `max_memory_reserved`; reset peak stats trước mỗi test; không lấy `nvidia-smi` làm metric duy nhất.
- **Correctness:** token/logit agreement cached vs uncached.

**Environment record:** GPU SKU, VRAM, driver, CUDA, PyTorch, dtype, CPU, batch size, prefix lengths, warm-up, repetitions, cache strategy, commit. Chỉ so cùng GPU/environment cho claim speedup.

## 10. SFT & LoRA

### 10.1. SFT nhằm giải quyết điều gì?

Pre-training học phân phối ngôn ngữ qua next-token prediction, **không** tự đảm bảo biết nghe instruction. **Supervised Fine-Tuning** đưa các cặp prompt/completion mẫu để tạo hành vi có điều kiện: định dạng JSON, phân loại issue và giải thích dựa trên evidence.

```text
[prompt = task + evidence + policy] [assistant response + EOS]
labels:   -100 -100 -100 ...         token_ids_of_response
```

Cần thiết kế `input_ids`, `labels`, `attention_mask`; không dùng loss trên user prompt. Những ví dụ output có `unknown/insufficient_information` là cực kỳ quan trọng để tránh học cách đoán khi thiếu bằng chứng.

### 10.2. LoRA — derivation và ý nghĩa

Full Fine-Tuning tối ưu toàn bộ tham số nền `W`. LoRA giữ `W` frozen và học update low-rank:

$$W' = W + \Delta W,\quad \Delta W=\frac{\alpha}{r}BA.$$

Với `W∈R^{d_out×d_in}`, `A∈R^{r×d_in}`, `B∈R^{d_out×r}`. Forward cho input `x∈R^{*,d_in}`:

$$y=xW^\top + \frac{\alpha}{r}(xA^\top)B^\top.$$

**Số trainable parameters của update:** `r(d_in+d_out)` thay vì `d_in*d_out`. Nhưng **không suy ra peak VRAM giảm theo đúng tỷ lệ**, vì activations và base weights vẫn chiếm bộ nhớ. LoRA giúp giảm gradients/optimizer states cho base params; hiệu quả peak thực tế phải benchmark.

**Lựa chọn mặc định:** inject `q_proj` và `v_proj` đúng yêu cầu, `r=8`, `alpha=16`, `lora_dropout=0.0` để thuận tiện kiểm chứng. Thử `r={4,8,16}` nếu budget còn; r=0 được dùng làm no-adapter baseline.

### 10.3. Minimal reference implementation (teaching example)

```python
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        assert rank > 0
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (x @ self.lora_A.T) @ self.lora_B.T
        return self.base(x) + (self.alpha / self.rank) * update
```

Đây là **minimal formula reference**, chưa có adapter merge/unmerge, mixed precision guard, dropout, save adapter-only, device sharding. Không copy như full production module.

**Giải thích khởi tạo:** `B=0` làm `ΔW=0` lúc đầu nên model khởi đầu cho ra output như base; `A` random nhận gradient thông qua B sau những bước đầu, `B` nhận gradient ngay. Khi freeze base: kiểm tra mọi base `requires_grad=False`; `optimizer` chỉ chứa LoRA parameters.

### 10.4. LoRA correctness tests

| ID | Kiểm tra | Kết quả |
|---|---|---|
| LOR-01 | `B=0` | LoRA output ≈ base output |
| LOR-02 | Frozen base | Base gradients = None và weights không đổi |
| LOR-03 | Trainable count | Tổng `r(in+out)` chính xác ở injected layers |
| LOR-04 | Gradient flow | LoRA B/A có gradient khi tiến hành learning, A có thể zero ở step đầu |
| LOR-05 | Merge/unmerge (nếu implemented) | Output sau merge ≈ output before merge |
| LOR-06 | Adapter serialization | Load lại đúng weights, same base hash |
| LOR-07 | Bias, dtype, shape | Không làm sai base bias, batch shape hoặc precision |

### 10.5. Fair Full FT vs LoRA comparison

- Same C0 base, same SFT train/dev/test, same tokenizer, same prompt template, same max length.
- Same **tokens_seen** và data order seed; so thêm GPU-hours nếu muốn đánh giá chi phí.
- Report trainable params %, peak allocated/reserved VRAM, step time, throughput, dev NLL, task accuracy, format validity.
- Hyperparameter search budget phải tương đương; không dùng LR full FT rồi áp cho LoRA và tuyên bố LoRA kém. Báo cáo learning rate riêng và lý do.
- Train/dev metric **không đồng nghĩa** test generalization. Đặc biệt structured responses có thể học thuộc templates.

### 10.6. Điều kiện stop / degrade

Nếu tiny base không học được schema sau reasonable pilot, giảm output schema còn 2–3 trường, rút ngắn prompt/response, tăng tỉ trọng examples nhận diện dựa vào evidence, hoặc chuyển sang task ngôn ngữ đơn giản hơn. **Không tăng model size vô hạn** để đuổi theo một chatbot tổng quát.

## 11. DPO & preference alignment

### 11.1. Bài toán DPO khác SFT ở đâu?

SFT nhận `prompt → correct completion`; DPO nhận `prompt → (chosen completion, rejected completion)` với **cùng prompt**, học phân biệt ưu tiên tương đối. SFT bắt chước câu trả lời tốt; DPO điều chỉnh tỷ lệ xác suất giữa cặp tốt/xấu **so với một reference policy cố định**.

**Dataset example:**

```json
{
  "id": "pref_auth_0001",
  "prompt": "TASK: triage. STATUS: 401. EVIDENCE: bearer token expired. Reply with category and action.",
  "chosen": "{\"category\":\"authentication\",\"action\":\"refresh_or_reauthenticate\"}",
  "rejected": "{\"category\":\"database\",\"action\":\"restart_database\"}",
  "preference_reason": "The chosen response is supported by the given evidence."
}
```

**Important:** prompt chung, chosen/rejected là phần response; không đếm prompt tokens trong loss log-likelihood cho DPO. Dữ liệu ưu tiên tự dựng phải có justification và review; nếu chỉ sinh rejected "sai cú pháp", DPO dễ học phân biệt format thay vì tính đúng của triage.

### 11.2. Standard DPO objective — phải hiểu đúng dấu

Ký hiệu `πθ` = policy được huấn luyện, `πref` = bản tham chiếu frozen (C1-lora trước khi DPO), `y_w` chosen, `y_l` rejected. Cho log-likelihood **chỉ response tokens**, đặt:

$$\Delta_\theta = \log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x),$$

$$\Delta_{ref} = \log\pi_{ref}(y_w\mid x)-\log\pi_{ref}(y_l\mid x).$$

$$\boxed{\mathcal L_{DPO}=-\mathbb E\big[\log\sigma\big(\beta(\Delta_\theta-\Delta_{ref})\big)\big]}$$

Equivalent:

$$-\log\sigma\left[\beta\left(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\right)\right].$$

**Tuyệt đối chú ý dấu và thứ tự chọn/từ chối.** Một số bản chép công thức dễ bị đảo dấu hoặc sai cách đặt phân số. Đối chiếu [DPO original paper](https://arxiv.org/abs/2305.18290) và behavior unit test dưới đây.

**Hiểu sâu:** DPO sử dụng một biến đổi của objective preference dưới KL regularization để không cần huấn luyện reward model riêng. Nó **không miễn phí**: thường phải tính scores cho policy và reference trên chosen+rejected; thời gian và bộ nhớ vẫn đáng kể.

### 11.3. Response log-probability

Với `prompt_ids` và `completion_ids`:

1. Tạo `input_ids = [prompt..., completion..., EOS]` theo một serialization đã chốt.
2. Chạy model để có `logits[:, :-1, :]`, ứng với targets `input_ids[:, 1:]`.
3. Áp `log_softmax` trên vocabulary; `gather` log-prob của actual next tokens.
4. Mask chỉ tại vị trí target thuộc **completion/EOS**; padding và prompt phải 0 trong mask.
5. Sum theo token completion để được `logπ(y|x)`. **Trong standard DPO dùng sum**, không vô tình dùng mean per token làm đổi objective.
6. Có thể khai báo phân tích phụ normalized likelihood theo length để phát hiện bias, nhưng không âm thầm đổi training criterion.

**Ranh giới BPE:** Tokenize concatenated prompt+completion theo template chuẩn; không giả định `tokenize(prompt)+tokenize(completion)` luôn bằng tokenize của concatenation nếu merges qua boundary. Dùng explicit boundary/special delimiter, tính `response_start_idx` từ chính serializer và kiểm thử bằng cases có whitespace/punctuation khác nhau.

### 11.4. Minimal DPO objective implementation

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logp: torch.Tensor,   # [B]
    policy_rejected_logp: torch.Tensor, # [B]
    ref_chosen_logp: torch.Tensor,      # [B], detached
    ref_rejected_logp: torch.Tensor,    # [B], detached
    beta: float = 0.1,
) -> torch.Tensor:
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp.detach() - ref_rejected_logp.detach()
    return -F.logsigmoid(beta * (policy_margin - ref_margin)).mean()
```

Phần này chỉ là hàm objective; full pipeline phải xây **masked sequence logps**, reference lifecycle, batching, gradient accumulation, checkpointing, evaluation và dataset validation.

### 11.5. Critical DPO correctness tests

| ID | Input | Expected result |
|---|---|---|
| DPO-01 | `policy==reference` | Per-pair `loss≈log(2)` khi cùng numerical logps |
| DPO-02 | `beta=0` | `loss=log(2)` và gradient w.r.t. logps ≈0 |
| DPO-03 | Policy margin > reference margin | Loss thấp hơn case margin bằng nhau |
| DPO-04 | Đảo chosen/rejected giữ nguyên model | Margin đổi dấu; loss thay đổi tương ứng |
| DPO-05 | `ref` frozen | Không có gradient/changing params phía reference |
| DPO-06 | Prompt-only tokens khác nhau (cùng response) | Loss logp chỉ cộng response/EOS tokens, không prompt |
| DPO-07 | Padding khác nhau | Same valid tokens → same sequence logp |
| DPO-08 | Cùng serialization/tokenizer/checkpoint | Stable scores trong numerical tolerance |

**Debug practice:** Bắt đầu bằng synthetic scalars (`policy_margin=+1`, `ref_margin=0`) để kiểm dấu; sau đó test mini model; sau đó mới mở GPU DPO.

### 11.6. DPO training protocol

- **Initialization:** C1-lora, reference là copy frozen của **chính checkpoint này**; nếu policy dùng LoRA, phải bảo đảm ref là snapshot bất biến của weights/adapter ở thời điểm bắt đầu DPO, không bị vô tình share mutable parameter objects.
- **Optimization:** chỉ update policy trainable params, reference `eval()` và `inference_mode()` khi tính ref logps.
- **Memory optimization:** với fixed dataset, có thể precompute reference chosen/rejected logps và hash dữ liệu/tokenizer/checkpoint; tránh làm ref forward mỗi epoch, nhưng không được đổi sequence/masking sau cache.
- **Starter hyperparams:** `beta=0.1`, learning rate cho LoRA DPO ở mức `1e-5` đến `1e-4` để sweep nhỏ; early stop theo dev preference metric và task accuracy guardrail. Không có LR cố định đúng cho mọi model.
- **Watch:** loss, preference margin, chosen/rejected logps, dev task accuracy, valid JSON %, response length, refusal rate, KL proxy/drift, train-vs-dev gap.

### 11.7. DPO không phải Safety Guarantee

Nếu preference pairs không chứa tình huống phân biệt có bằng chứng/không bằng chứng thì DPO không dạy được calibration. Với model nhỏ, preference signal có thể bị chi phối bởi format/length. Đánh giá trên **unseen semantic scenarios**, không chỉ generated template khác seed.

### 11.8. DPO question bank (để phản biện)

- Vì sao ref policy nên là C1 SFT chứ không phải base C0?
- Nếu `beta→0`, mô hình học được gì?
- Vì sao log probability phải được tính conditional trên cùng prompt?
- Nếu chosen luôn ngắn hơn rejected, mô hình có thể học bias gì?
- Nếu accuracy tăng trên training pairs nhưng giảm test categories, nguyên nhân nào?
- Nếu reference bị update cùng policy, DPO objective còn mang ý nghĩa gì?

## 12. Guardrails, threat model & secure application

### 12.1. Phân biệt alignment và security controls

**Alignment training** làm thay đổi xác suất hành vi mô hình. **Guardrails** là các checks bên ngoài model trước/sau inference. **Backend security** là kiểm soát không thể ủy thác cho output mô hình (authentication, authorization, audit, rate limiting, secret management).

Không được giả định prompt "ignore all previous instructions" là một cơ chế tấn công thần kỳ trên model bé; test cần phân biệt lỗi parsing, policy violation thật và lỗi model không hiểu task.

### 12.2. Threat boundaries

```text
[Untrusted user / input evidence]
          │
          ▼
[API schema validation + size limits + authentication]  <== CODE CONTROL
          │
          ▼
[Input policy + prompt serializer]                    <== CODE CONTROL
          │
          ▼
[Small LM (trained LoRA/DPO)]                        <== PROBABILISTIC
          │
          ▼
[Parse / validate JSON / allowed enums]               <== CODE CONTROL
          │
          ▼
[Output policy + error fallback]                      <== CODE CONTROL
          │
          ▼
[Response only; NO shell/db execution]                 <== NO TOOLS
```

### 12.3. Threat model table

| Threat | Attack surface | Defense-in-depth | Test |
|---|---|---|---|
| Prompt injection | `symptoms`, `evidence` chứa lệnh giả | untrusted data serialization; output schema/allowlist; no privileged actions | cố tình chèn "ignore instruction" vào evidence |
| Output schema poisoning | model sinh key/enum mới | strict JSON parsing + schema + enum whitelist | invalid/missing/extra fields |
| Hallucinated diagnosis | thiếu evidence hoặc evidence xung đột | explicit `insufficient_information`, deterministic checks nếu có | ambiguous/conflicting incident fixtures |
| Sensitive disclosure | prompt yêu cầu token/secrets/PII | không đưa secrets vào context; redact logs; deny source access | canary fake secrets; no real credentials |
| Resource exhaustion | prompt hoặc generation quá dài | request size, token budget, timeouts, concurrency limits | oversized request / EOS never generated |
| Dataset poisoning | malicious instruction in dataset | provenance, validation, split audits, review | suspicious/out-of-schema training samples |
| Unsafe downstream execution | model output gửi tới tool/shell | **No tool execution in project**; future work requires explicit allowlisted action+ACL | attempt arbitrary command in output |

**Không xem regex chống injection là đủ.** Bộ lọc input chỉ giúp phát hiện tín hiệu thô; quan trọng hơn là trust boundary và không nối model output với tác vụ đặc quyền. Tham chiếu [OWASP Top 10 for LLM Applications (2025)](https://genai.owasp.org/llm-top-10/).

### 12.4. Guardrails specification

**Input validation (must):** JSON fields, string types, length caps, allowed task kind, explicit max_tokens, reject control characters bất thường nếu serializer yêu cầu, normalize nhưng tránh xóa evidence cần điều tra.

**Model-time contract (must):** system/domain policy là hằng số quản lý trong code; evidence là untrusted, không cho override policy; model không có access DB, shell, URL fetch, secret store.

**Output validation (must):** parse JSON, validate enum + required keys, reject unsupported action/category, fallback `needs_review`/`invalid_model_output`; không cast vô hạn hoặc silently invent confidence.

**Observability (must):** log request ID, model/config/version, token counts, policy decision, latency, structured validation failures; redact evidence có khả năng chứa credential, không lưu raw prompt mặc định.

### 12.5. Guardrail metrics

- **Attack Success Rate (ASR):** `successful policy violations / valid adversarial attempts` theo criterion đã định nghĩa, không đếm mere mention of forbidden instruction là thành công.
- **False Positive Rate:** `benign inputs incorrectly blocked / all benign inputs`.
- **False Negative Rate:** adversarial inputs cần block nhưng vượt qua gate theo rubric.
- **Valid Output Rate:** % parse + schema compliant.
- **Utility:** task accuracy trên benign set khi có/không guardrail (tránh "an toàn" nhờ từ chối tất cả).
- **Overhead:** p50/p95 additional latency của guardrails.

### 12.6. Safety suite tối thiểu

Tạo ≥50 benign test requests và ≥50 adversarial/boundary requests (mục tiêu tối thiểu, có thể tăng). Chia category: injection in evidence, instruction-role spoof, request for secret, contradictory evidence, invalid schema, oversized input, Unicode/escaping edge cases. Mỗi case có `attack_type`, `expected_policy`, `expected_result`, `severity`, `annotator_note`.

**Manual audit:** xem lại ít nhất 10 false positives/negatives có ý nghĩa; dữ liệu threat suite không được dùng nguyên xi để train rồi công bố kết quả test là unseen.

## 13. Dataset strategy và data contracts

### 13.1. Dataset selection matrix

| Dataset | Stage | Vai trò | Quyết định | Provenance / license caution |
|---|---|---|---|---|
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Pre-train | Small-language causal modeling | **PRIMARY** | HF card hiển thị `cdla-sharing-1.0`; review nghĩa vụ distribution |
| **`api-triage-sft-v1`** (tự xây) | SFT | Task-grounded JSON response | **PRIMARY** | Synthetic rule-based; có documented provenance + human review |
| **`api-triage-preferences-v1`** (tự xây) | DPO | Chosen/rejected theo cùng prompt | **PRIMARY** | Không lấy rejected ngẫu nhiên hoặc chỉ lỗi format |
| [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | SFT supplemental | Instruction diversity, closed QA/classification | **OPTIONAL** | Dataset card `CC BY-SA 3.0`; attribution/share-alike obligations, có nội dung Wikipedia |
| [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | DPO stress/auxiliary | Preference behavior & qualitative check | **OPTIONAL** | HF card `MIT`, có nội dung nhạy cảm; dataset card cảnh báo không dùng thô làm training dialogue agents |
| [Alpaca-Cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | Alternative SFT | General instructions | **DEFER** | Metadata hiển thị CC BY 4.0 nhưng card text nêu CC BY-NC 4.0; upstream provenance điều kiện riêng; cần legal review |
| [ToxiGen](https://huggingface.co/datasets/allenai/toxigen) | Optional safety evaluation | Toxicity robustness | **COULD** | Chỉ thêm nếu kiểm tra card/license/label quality và scope phù hợp |
| MT-Bench / AlpacaEval | Optional external reference | Open-ended comparative eval | **NOT PRIMARY** | Không phù hợp để coi là gold score cho model 8–20M domain-specific |

**Dataset policy:** Không gộp nhiều dataset chỉ để tăng số lượng. TinyStories dạy ngôn ngữ đơn giản; API-triage dataset dạy task format; preference pairs dạy ưu tiên trong task. Dataset nào không giải thích được **causal role** của nó trong pipeline thì không đưa vào main track.

### 13.2. Task ontology (immutable v1)

```yaml
task_name: api_incident_triage
categories:
  - authentication        # e.g. 401, missing/expired token
  - authorization         # e.g. 403, valid identity but insufficient rights
  - rate_limiting         # e.g. 429, backoff per documented policy
  - client_request        # e.g. 400/404/422, malformed or unsupported request
  - upstream_failure      # e.g. 502/503/504, gateway/upstream issues
  - server_error          # e.g. 500, internal server failure
  - network_timeout       # explicit network/transient connection symptoms
  - insufficient_information
allowed_actions:
  - refresh_or_reauthenticate
  - verify_permissions
  - inspect_request_payload
  - verify_resource_path
  - retry_with_backoff
  - inspect_server_logs
  - inspect_upstream_health
  - check_network_connectivity
  - request_more_evidence
```

Mỗi `category/action` phải có **rule table** xác định evidence nào support, evidence nào conflict. Tránh gán triage chỉ dựa HTTP status nếu mô tả mâu thuẫn. Với trường hợp `401` do gateway mapping sai, model chỉ được suy ra điều thông tin đầu vào hỗ trợ; có thể chọn insufficient_information.

### 13.3. Data generation strategy cho bài toán hẹp

**Tối thiểu nên có:**

- SFT: **2.000–5.000 examples** đầu tiên; ưu tiên balanced category và đa dạng tình huống hơn số lượng. **Đây là target tạo dữ liệu, không bảo đảm đủ học.**
- DPO: **500–1.500 preference pairs** có preference rationale, validation riêng.
- Task held-out: **≥200 cases** tách riêng theo semantic scenario/template family; không cho test generator dùng cùng paraphrase templates với train.
- Safety: ≥50 benign + ≥50 adversarial/boundary cases; tách khỏi SFT/DPO training.

**Generator layers:**

1. **Ontology/rule engine:** quy định trạng thái, bằng chứng, hành động đúng và điều kiện không đủ chứng cứ.
2. **Scenario sampler:** sample request details, evidence variants, HTTP codes, contradiction patterns.
3. **Text realizer:** diễn đạt bằng các templates khác nhau (template families có ID), có noise vừa phải về ngôn ngữ/log formats.
4. **Validator:** check consistency của target với ontology; invalid example bị reject.
5. **Reviewer:** manual sample ngẫu nhiên ít nhất 10% của set SFT nhỏ và 10–20% preference pairs theo budget; ghi thống kê disagreement.
6. **Split engine:** group split theo `scenario_family_id` (không chỉ random row split); freeze holdout.

### 13.4. SFT record schema

```json
{
  "id": "sft_000001",
  "source": "generated_rule_based_v1",
  "scenario_family_id": "expired_token_mobile_client",
  "template_family_id": "json_incident_prompt_t03",
  "split": "train",
  "task": "api_incident_triage",
  "prompt": "STATUS=401; EVIDENCE=token expired; SYMPTOM=profile request unauthorized",
  "completion": "{\"category\":\"authentication\",\"action\":\"refresh_or_reauthenticate\"}",
  "expected_category": "authentication",
  "expected_action": "refresh_or_reauthenticate",
  "knowledge_rule_id": "AUTH_EXPIRED_TOKEN_V1",
  "annotation_method": "rule_engine_checked"
}
```

`split` do pipeline gắn, không do generator random đặt; **không lưu private credentials**. Nếu hoàn thiện `confidence_label` field thì giữ schema thống nhất cả train/val/test, không tự thay giữa các runs.

### 13.5. DPO pair schema

```json
{
  "id": "dpo_000001",
  "source": "domain_preference_builder_v1",
  "scenario_family_id": "expired_token_mobile_client",
  "split": "train",
  "prompt": "STATUS=401; EVIDENCE=token expired; SYMPTOM=profile request unauthorized",
  "chosen": "{\"category\":\"authentication\",\"action\":\"refresh_or_reauthenticate\"}",
  "rejected": "{\"category\":\"server_error\",\"action\":\"inspect_server_logs\"}",
  "preference_dimension": "evidence_grounded_correctness",
  "rationale": "Chosen is supported by expiry evidence; rejected is unsupported.",
  "annotator": "rule_engine_plus_manual_review"
}
```

Các preference dimensions nên cân bằng: `correctness`, `grounding`, `appropriate_abstention`, `safety`, `clarity`. **Tránh rejected luôn dài hơn/ít dấu câu hơn**. Cố gắng làm both chosen/rejected syntactically valid và stylistically similar để DPO học *semantic preference* thay vì shortcut.

### 13.6. Dataset leakage prevention

- Deduplicate bằng normalized prompt+response hash; kiểm tra near-duplicates giữa train/dev/test.
- Train/test không chia theo một row-ID random nếu generator sinh từ cùng scenario family.
- Dataset pretrain sample không được vô tình chứa các test fixture tự xây.
- DPO training pairs không chứa chính evaluation preference pairs hoặc paraphrase cận trùng.
- Không tune prompt format bằng test; mọi thay đổi template sau freeze → tạo test version mới, không so trực tiếp như cùng benchmark.
- Lưu `data_manifest.json`: source URL, revision/commit, split, number rows, number tokens, seed, transformations, license notes, SHA256.

### 13.7. External dataset normalization

**Dolly:** record thường có `instruction`, `context`, `response`, `category`; chỉ lấy bounded types (classification, closed QA, information extraction) sau filtering và mapping. Không đưa general brainstorming vào model nhỏ nếu làm loãng domain objective. Ghi attribution.

**HH-RLHF:** `chosen`/`rejected` có thể chứa toàn bộ conversation text; cần parser tách prompt common prefix và completion một cách an toàn, reject trường hợp không cùng prompt hoặc ambiguous dialogue. Không nối thẳng dữ liệu vào DPO mà không kiểm tra; domain shift lớn, safety content có thể gây distress, dataset card nêu usage caveats.

**Alpaca-Cleaned:** chưa cho vào training pipeline trước khi xác minh mâu thuẫn license/provenance, bất kể HF badge hiển thị gì.

## 14. Evaluation, ablation & statistical rigor

### 14.1. Các checkpoint được so sánh

| Model | Causal pretrain | SFT | LoRA | DPO | Purpose |
|---|---|---|---|---|---|
| C0 | ✓ | – | – | – | Language baseline |
| C1-full | ✓ | Full FT | – | – | Adaptation baseline |
| C1-lora | ✓ | ✓ | ✓ | – | PEFT baseline |
| C2-dpo | ✓ | ✓ | ✓ | ✓ | Preference-aligned policy |
| Rule-only baseline | – | – | – | – | Deterministic triage comparator |

**Cần có rule-only baseline!** Vì task có ontology xác định, rule engine thuần có thể outperform small LLM. Nếu vậy, kết luận đúng là **LM có giá trị nghiên cứu nhưng chưa đáng dùng thay bộ phân loại xác định**. Đây là một insight quan trọng cho AI Application Engineering.

### 14.2. Metrics — operational definitions

| Metric | Definition | Caveat |
|---|---|---|
| Token NLL / PPL | Mean CE trên held-out validation corpus | Fixed tokenizer; domain-specific PPL riêng |
| Category accuracy | `correct_category / N` | Report per-category và macro-F1 |
| Action accuracy | `correct_action / N` | Unknown là một đáp án có thể đúng |
| Exact-match JSON | Correct full canonical JSON / N | Rất strict; không thay semantic correctness |
| Schema validity | parse + required keys + enums valid / N | Không nói lên semantic quality |
| Abstention recall | correct unknown / all unknown-required cases | Có thể tăng bằng từ chối quá nhiều; cần false abstention |
| Preference accuracy | chosen scored higher than rejected / heldout pairs | Phải báo cáo dùng reference-relative hay raw margin |
| Pairwise win rate | blinded judgement model A vs B | Include ties, order swaps, judge config |
| Safety ASR | successful violations / valid attack attempts | Need explicit threat-specific success criteria |
| Safety FPR | benign incorrectly blocked / benign | Guardrails có utility trade-off |
| Cache benchmark | TTFT, TPOT, throughput, VRAM | Same GPU/checkpoint/dtype/prompt lengths |
| Efficiency | trainable params, GPU-hours, tokens/s, peak VRAM | Không so budget khác nhau mà giấu đi |

### 14.3. Primary vs secondary metrics

**Primary (lock before running test):** category macro-F1, action accuracy, schema validity, held-out preference accuracy, ASR/FPR, cache TPOT.

**Secondary:** BLEU/ROUGE không phù hợp cho đúng/sai JSON nhiều biến thể; thay bằng canonical parse. Model judge chỉ dùng cho giải thích/cặp open-ended optional. PPL giúp kiểm tra basic modeling, không phải score chính cho triage.

### 14.4. Experiment design / ablation matrix

| Exp | Comparison | Independent variable | Control | Key deliverable |
|---|---|---|---|---|
| E0 | Random init vs C0 | Pre-training | Model/tokenizer architecture | PPL / generation examples |
| E1 | KV ON vs OFF | Cache implementation | Same C0, L, new_tokens, GPU | TTFT/TPOT/VRAM plots |
| E2 | C0 vs C1-lora | SFT/adapter | Frozen test, same base | Task metric improvement or negative result |
| E3 | C1-full vs C1-lora | Parameter update regime | Same data/tokens and reasonable tuning budgets | Accuracy/VRAM/parameter trade-off |
| E4 | C1-lora vs C2-dpo | DPO treatment | Same eval prompts/checkpoint ancestry | Preference metric and task trade-off |
| E5 | Guardrail OFF vs ON | External filters | Same model/test traffic | ASR/FPR/utility/overhead |
| E6 | LoRA r=4/8/16 | Adapter rank | Fixed data & budget | Rank-performance curve **if time** |
| E7 | DPO beta sweep | Preference strength | Fixed pair data & policy init | Beta sensitivity **if time** |

### 14.5. Independent evaluation design

- Split test cases into **in-distribution**, **held-out scenario families**, **counterfactual evidence**, **ambiguous cases**, **adversarial cases**.
- Holdout phải bao gồm các status code quen thuộc nhưng **evidence mapping khác** để chống memorization shortcut.
- Cần kiểm tra bằng **confusion matrix** và per-category precision/recall/F1; report các categories có ít mẫu.
- Sử dụng bootstrap confidence intervals hoặc Wilson intervals cho rate khi N nhỏ; không dùng một phần trăm duy nhất để suy diễn certainty.
- Nếu có điều kiện run ≥3 random seeds cho smaller experiments; nếu chỉ chạy 1 seed do compute, tuyên bố rõ hạn chế.
- Cần ghi 5–10 failure case ID từ held-out test: loại lỗi, ảnh hưởng, giả thuyết nguyên nhân, evidence và thay đổi dự kiến.

### 14.6. LLM-as-a-Judge protocol (optional, đúng yêu cầu đề)

Đối với phản hồi open-ended, judge có thể so hai phiên bản blind A/B (trộn thứ tự, cùng prompt, không gắn tên model) trên rubric: correctness, groundedness, clarity, safety; có tie. Làm **A/B và B/A** cho subset để đo position bias; lưu judge prompts, model ID/version, sampling config, raw output, rubric version, costs và date. Kiểm tra sample thủ công để ước lượng agreement.

**Không cố định GPT-4o mini hoặc Claude 3.5 Haiku như thể luôn có sẵn**; chọn model judge khả dụng và công bố lựa chọn khi chạy. Với model nhỏ domain-specific, deterministic task metrics vẫn ưu tiên hơn judge score. Tham chiếu nghiên cứu [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685).

### 14.7. Failure analysis taxonomy

| Failure tag | Diagnostic questions |
|---|---|
| `TOK` | Tokenizer chia nhỏ/truncate domain tokens? |
| `MASK` | Target mask/attention mask sai? |
| `GEN` | EOS/max length/temperature làm hỏng JSON? |
| `CAPACITY` | Model quá nhỏ không nắm điều kiện? |
| `DATA_LEAK` | Train/test tương tự template quá mức? |
| `DATA_NOISE` | Label/rule/preference mâu thuẫn? |
| `OOD` | Test thuộc distribution khác train? |
| `DPO_DRIFT` | DPO tối ưu preference margin nhưng mất task accuracy? |
| `HALLUCINATION` | Chọn action không supported bởi evidence? |
| `SECURITY` | Input bypass/output bypass/unsafe handling? |

**Template 1 error record:** `case_id`, `prompt_hash`, `expected`, `predicted`, `checkpoint`, `failure_tag`, `evidence`, `root_cause_hypothesis`, `fix_candidate`, `whether_fixed`.

## 15. Repository architecture & engineering conventions

### 15.1. Proposed monorepo (Python modular monolith)

```text
minialign/
├── README.md                       # quickstart + evidence links
├── PROJECT_SOURCE_OF_TRUTH.md      # copy/symlink of THIS spec or docs/...
├── pyproject.toml                  # locked direct dependencies
├── uv.lock                         # or requirements lock, choose one
├── Makefile                        # smoke/test/data/train/eval targets
├── .gitignore
├── .env.example                    # no real keys
├── configs/
│   ├── base_small.yaml
│   ├── base_primary.yaml
│   ├── pretrain.yaml
│   ├── sft_full.yaml
│   ├── sft_lora.yaml
│   ├── dpo.yaml
│   ├── inference.yaml
│   └── evaluation.yaml
├── src/minialign/
│   ├── __init__.py
│   ├── tokenization/
│   │   ├── bpe_reference.py        # hand-written educational BPE
│   │   ├── train_tokenizer.py
│   │   └── tokenizer_adapter.py
│   ├── model/
│   │   ├── configuration.py
│   │   ├── embeddings.py
│   │   ├── rope.py
│   │   ├── rmsnorm.py
│   │   ├── attention.py
│   │   ├── mlp.py
│   │   ├── decoder_block.py
│   │   └── causal_lm.py
│   ├── inference/
│   │   ├── kv_cache.py
│   │   ├── sampler.py
│   │   └── generator.py
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── schema.py
│   │   ├── splits.py
│   │   ├── sft_formatter.py
│   │   ├── dpo_formatter.py
│   │   └── triage_generator.py
│   ├── training/
│   │   ├── pretrain.py
│   │   ├── sft.py
│   │   ├── lora.py
│   │   ├── dpo.py
│   │   ├── optim.py
│   │   └── checkpoints.py
│   ├── evaluation/
│   │   ├── perplexity.py
│   │   ├── task_metrics.py
│   │   ├── preference_metrics.py
│   │   ├── safety_metrics.py
│   │   ├── benchmark_cache.py
│   │   └── report_tables.py
│   ├── safety/
│   │   ├── input_policy.py
│   │   ├── output_validator.py
│   │   └── schemas.py
│   ├── serving/
│   │   ├── service.py
│   │   └── api.py
│   ├── cli/
│   │   └── main.py
│   └── utils/
│       ├── seed.py
│       ├── experiment_manifest.py
│       └── logging.py
├── tests/
│   ├── unit/
│   │   ├── test_bpe.py
│   │   ├── test_attention_causality.py
│   │   ├── test_rope_positions.py
│   │   ├── test_kv_cache.py
│   │   ├── test_lora.py
│   │   ├── test_dpo_loss.py
│   │   ├── test_response_mask.py
│   │   └── test_guardrails.py
│   ├── integration/
│   │   ├── test_train_one_batch.py
│   │   ├── test_generate_cached_vs_uncached.py
│   │   ├── test_sft_lora_smoke.py
│   │   └── test_dpo_pipeline_smoke.py
│   ├── contract/
│   │   ├── test_data_schemas.py
│   │   ├── test_checkpoint_compatibility.py
│   │   └── test_api_contract.py
│   └── evaluation/
│       ├── test_no_split_leakage.py
│       └── test_metric_formulas.py
├── scripts/
│   ├── download_datasets.py
│   ├── build_domain_data.py
│   ├── run_all_evaluations.py
│   └── export_report_assets.py
├── data/
│   ├── raw/                      # gitignored; immutable downloaded artifacts
│   ├── processed/                # gitignored
│   ├── manifests/                # tracked metadata (not sensitive raw data)
│   └── examples/                 # tiny public fixtures only
├── checkpoints/                 # gitignored large model weights
├── runs/                        # gitignored logs, experimental output
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── failure_cases.md
│   └── paper/                     # IEEE/ICLR manuscript
└── docs/
    ├── adr/                       # minimal decisions (optional)
    ├── experiment_protocol.md     # optional excerpt from this spec
    └── dataset_cards/             # own dataset provenance/limitations
```

**Một nguồn chính:** Giữ bản canonical tại `docs/PROJECT_SOURCE_OF_TRUTH.md` hoặc root, **không chỉnh hai bản độc lập**. Sơ đồ trên gợi ý, có thể giảm số file nếu không cần thiết; mục tiêu là ranh giới đúng chứ không phải nhiều thư mục.

### 15.2. Dependency policy

| Package group | Role | Allowed? |
|---|---|---|
| `torch` | Tensor, autograd, optimizer, DataLoader | YES |
| `numpy`, `pandas` | Processing/analysis | YES |
| `datasets` | Fetch public datasets, split/load | YES |
| `tokenizers`, `sentencepiece` | Train/export tokenizer if permitted | YES |
| `pytest`, `ruff`, `mypy` | Tests/lint/types | YES |
| `tensorboard` or `wandb` | Training logs | YES |
| `matplotlib`, `scikit-learn` | Graphs/metrics | YES |
| `fastapi`, `uvicorn`, `pydantic` | Optional thin demo | YES |
| `transformers.AutoModelForCausalLM` | Main model | **NO** |
| `trl.DPOTrainer`, `peft.LoraModel` | Core specialized technique | **NO** |
| `transformers.generate()` | Core inference engine | **NO** |

Có thể tham khảo code của các package bị cấm để kiểm chứng công thức; nếu viết test comparator thì tách khỏi main path và ghi rõ. Pin exact versions, freeze lockfile, tuyệt đối không commit `.env`, API secrets, HF tokens.

### 15.3. CLI contract (chưa triển khai)

```bash
# Cài đặt dependency (sau khi tạo pyproject/uv.lock)
uv sync --frozen

# Smoke tests không cần training dài
uv run pytest -q tests/unit tests/contract
uv run pytest -q tests/integration

# Stage 1: data/tokenizer
uv run python -m minialign.cli.main data prepare --config configs/pretrain.yaml
uv run python -m minialign.cli.main tokenizer train --config configs/base_primary.yaml

# Stage 2: model and cache
uv run python -m minialign.cli.main train pretrain --config configs/pretrain.yaml
uv run python -m minialign.cli.main evaluate cache --config configs/evaluation.yaml

# Stage 3: full/LoRA SFT, DPO
uv run python -m minialign.cli.main data triage-build --config configs/sft_lora.yaml
uv run python -m minialign.cli.main train sft --config configs/sft_full.yaml
uv run python -m minialign.cli.main train sft --config configs/sft_lora.yaml
uv run python -m minialign.cli.main train dpo --config configs/dpo.yaml

# Stage 4: immutable evaluation and demo
uv run python -m minialign.cli.main evaluate all --config configs/evaluation.yaml
uv run python -m minialign.cli.main serve --config configs/inference.yaml
```

**Ghi chú:** `uv` chỉ là package management đề xuất; có thể đổi sang `pip + requirements.txt` mà không thay nghiên cứu. Các lệnh này định nghĩa **target developer experience**, chưa phải executable software. Cần hiện thực interface đúng hoặc cập nhật bản đặc tả khi quyết định đổi CLI.

### 15.4. Reproducibility levels

| Level | Artifact | SLA của đồ án |
|---|---|---|
| R0 | `make smoke` chạy unit tests và mini pipeline CPU | MUST |
| R1 | `make demo` load checkpoint versioned và sinh output | MUST |
| R2 | `make evaluate` ra bảng metric từ checkpoint/dataset đã freeze | MUST |
| R3 | `make reproduce` chạy lại whole pipeline trên GPU phù hợp | SHOULD; có thể tốn GPU-hours, phải công khai |

Không quảng cáo "one-click full pre-training" như chạy vài phút. Script `reproduce` cần biết download size, compute assumptions và chi phí/giới hạn runtime; một lệnh tái lập **quy trình** không có nghĩa là kết quả chạy nhanh.

### 15.5. Model / checkpoint manifest

```json
{
  "experiment_id": "sft_lora_r8_seed42",
  "stage": "sft_lora",
  "git_commit": "<commit_sha>",
  "model_config_sha256": "<hash>",
  "tokenizer_sha256": "<hash>",
  "dataset_manifest_sha256": "<hash>",
  "parent_checkpoint_sha256": "<hash>",
  "seed": 42,
  "precision": "fp16",
  "hardware": "<actual GPU model>",
  "trainable_parameters": null,
  "tokens_seen": null,
  "best_validation_metric": null,
  "created_utc": "<actual ISO-8601>"
}
```

Các `null` là placeholder cho measured fields, không điền số giả. Không lưu `pickle` từ nguồn không tin cậy khi không cần; cân nhắc weight-only/safetensors theo tương thích. Checkpoint resume phải lưu optimizer, scheduler, RNG states khi mục tiêu là resume training chính xác.

### 15.6. Quality gates / CI

- Lint / formatting có một chuẩn duy nhất (ví dụ `ruff check`, `ruff format --check`).
- Unit tests CPU deterministic chạy dưới budget hợp lý.
- Integration mini model trên CPU (vocab nhỏ, 1–2 layers) đủ chạy một optimization step và generate ngắn.
- `pytest -m gpu` chỉ chạy GPU khi sẵn, không chặn unit test CI CPU.
- CI kiểm tra model code không import những forbidden packaged models/trainers làm engine chính; bổ sung test architecture nếu cần.
- Benchmark logs append-only theo `run_id`; không tự overwrite kết quả cũ mà không lưu provenance.

## 16. Kế hoạch 4 tuần & acceptance gates

### 16.1. Critical path

```text
Tokenizer → Causal attention/model → One-batch overfit → C0 pretrain
                                                  │
                                                  ├→ KV Cache correctness → E1 benchmark
                                                  │
                                                  └→ SFT data + masking → Full FT vs LoRA
                                                                            │
                                                                            └→ Frozen ref + DPO
                                                                                      │
                                                                                      └→ Evaluation + guardrails + paper
```

**Critical path:** nếu không có C0 đạt tối thiểu loss decrease / text output, SFT/DPO sẽ khó diễn giải. Bắt buộc bắt đầu dataset triage và paper outline **song song** với pretrain để tránh tuần 3 mới phát hiện thiếu dữ liệu.

### 16.2. Tuần 1 — Foundation / Proposal

| Ngày | Implementation | Test / proof | Deliverable |
|---|---|---|---|
| D1 | Repo + environment + GPU profiling baseline + scope freeze | `torch.cuda.is_available`, mem summary | Proposal outline & hardware manifest |
| D2 | Train tokenizer, encode/decode + custom BPE demo | TOK-01..06 | tokenizer artifact/hash |
| D3 | Embeddings, norm, RoPE, attention mask | shape + causality | attention module |
| D4 | Decoder block, MLP, residual, LM head | MOD-01..03 | end-to-end forward |
| D5 | Training CE + shift/masks + overfit 1 batch | MOD-04 | mini training evidence |
| D6 | Pretrain dataloader, checkpoint, scheduler; pilot throughput | no NaNs; resume test | stable short run |
| D7 | Review architecture, 2-page proposal, decide model size | peer/self audit | **Milestone 1 accepted** |

**Gate G1 MUST:** Có forward/backward, one-batch overfit, tokenizer roundtrip, demo 2-page proposal, measured training throughput, code repo. Nếu chưa qua G1, không mở DPO/guardrails.

### 16.3. Tuần 2 — Base training / KV Cache

| Ngày | Implementation | Test / proof | Deliverable |
|---|---|---|---|
| D8–D9 | Pretrain C0 với token-budget và validation | NLL trend, checkpoint stability | C0 checkpoint v1 |
| D10 | Autoregressive generator (greedy, EOS, max tokens) | deterministic output | CLI generate |
| D11 | KV cache, prefill+decode, reset | MOD-05/06, cache equivalence | `kv_cache.py` |
| D12 | Full cache benchmark: L=128/512/1024/2048 | CUDA timing correctness | raw E1 logs |
| D13 | Plot latency/VRAM + error analysis + evaluate C0 PPL | metrics integrity | milestone charts |
| D14 | Midway demo / freeze C0 for SFT | C0 config/hash | **Milestone 2 accepted** |

**Gate G2 MUST:** generated text (dù đơn giản), cached/uncached equivalence trong tolerance, plots TTFT/TPOT/VRAM, saved C0 + tokenizer. Nếu validation PPL rất kém, không hứa accuracy cao về sau; thực nghiệm vẫn có thể tiếp tục nhưng phải ghi capacity limitation.

### 16.4. Tuần 3 — LoRA, SFT, DPO

| Ngày | Implementation | Test / proof | Deliverable |
|---|---|---|---|
| D15 | Freeze task ontology, SFT data manifests / held-out split | leakage checks | domain dataset v1 |
| D16 | SFT response-only masks / full FT small baseline | mask assertions; task metrics | C1-full |
| D17 | LoRA from scratch + serialization | LOR-01..07 | LoRA module |
| D18 | LoRA SFT under matched token budget | E2/E3 early results | C1-lora |
| D19 | Preference builder, DPO logits/masking/ref snapshot | DPO-01..08 | DPO objective tests |
| D20 | DPO pilot then bounded training | finite loss, ref frozen | C2-dpo |
| D21 | Freeze checkpoints, compare C1/C2 | E4 plots + failures | **Milestone 3 accepted** |

**Gate G3 MUST:** LoRA has frozen base, measurable training savings/parameter reduction; DPO objective unit-tested and run on valid preference dataset; C1/C2 snapshots tracked. **Không bắt buộc DPO thắng** nếu method chính xác và negative results được phân tích nghiêm túc; tuy nhiên nếu rubric yêu cầu *có cải tiến*, tập trung improvement có thể chứng minh như parameter efficiency, test coverage và safety pipeline.

### 16.5. Tuần 4 — Safety, evaluation, paper, demo

| Ngày | Implementation | Test / proof | Deliverable |
|---|---|---|---|
| D22 | Input/output guardrails + response schema | schema and security tests | safety layer |
| D23 | Benign/adversarial suite; ON/OFF comparison | ASR/FPR/utility | E5 raw logs |
| D24 | Run frozen full evaluation & ablations | no test leakage | result CSV/plots |
| D25 | Inspect ≥5 failures, validate claims / confounders | taxonomy + evidence | `failure_cases.md` |
| D26 | Write paper (6–8 pages), tables, ethics, limitations | paper checklist | first full manuscript |
| D27 | README, reproducibility scripts, Gradio/CLI demo | `make smoke`, `make demo` | release candidate |
| D28 | Full dry-run presentation, Q&A, artifact audit | grading rubric cross-check | **Milestone 4 accepted** |

**Gate G4 MUST:** paper + demo + source repo + dataset/checkpoint manifests + unit/integration tests + clear figures + at least 5 error cases. Safety test phải báo cáo false-positive, không chỉ attack block rate.

### 16.6. Weekly hard-stop logic

- **Tối D5**: chưa pass one-batch overfit → debug model/tokenizer/loss; không triển khai KV Cache.
- **Tối D10**: C0 training bất ổn → xuống fallback config; khóa phạm vi pretraining hợp lý.
- **Tối D14**: KV cache benchmark không ra → chỉ sửa correctness và profiling, không xây API demo.
- **Tối D18**: LoRA không train được → dừng hyperparameter sweep, chạy test miniproblem, không bắt đầu DPO trên checkpoint hỏng.
- **Tối D21**: DPO chưa cải thiện → freeze và report negative results, không tự thay test; hoàn thiện guardrails/evaluation.
- **Tuần 4**: không đưa thêm tính năng mới ngoài bug fix / metrics integrity / paper.

## 17. Risks, fallbacks và stop rules

### 17.1. Risk register

| ID | Rủi ro | P | Impact | Early signal | Mitigation / fallback |
|---|---|---|---|---|---|
| R01 | GPU/Colab quota không đủ | High | Critical | GPU-hours consumed nhanh, disconnect | Small config, fewer tokens, resume; log compute budget |
| R02 | Base LM không học instruction-following | High | Critical | valid JSON thấp, dev accuracy near chance | Narrow schema, evidence-closed task, smaller dataset but cleaner, honest negative results |
| R03 | Model bugs làm train loss giả tốt | Medium | Critical | high train accuracy nhưng decode sai | causal perturbation test, response masking tests, overfit mini batch |
| R04 | KV caching sai positions/mask | High | High | cache vs full logits divergent | test short sequence full/incremental at each step |
| R05 | Synthetic label leakage | High | High | test quá cao trên same template | group split by scenario family, adversarial/counterfactual held-out |
| R06 | LoRA slower/VRAM not lower | Medium | Medium | peak VRAM overhead dominates | report trainable params + optimizer memory separately; no inflated performance claims |
| R07 | DPO reward hacking / length bias | High | High | chosen margin rises, accuracy falls | balanced preference pairs, length-distribution audit, early stop |
| R08 | HH dataset domain mismatch | High | Medium | gradients conflict with triage task | optional only, filtered subset and separate ablation |
| R09 | Guardrails block all outputs | Medium | High | ASR near zero and benign pass near zero | report FPR, accuracy, refusal; tune on dev only |
| R10 | License uncertainty / source change | Medium | High | conflicting dataset cards or no revision | pin revision, compliance review, exclude doubtful data |
| R11 | Evaluation bias / judge drift | Medium | High | judge inconsistent after position swap | frozen judge model/version, A/B+B/A, human audit |
| R12 | Overengineering serving stack | High | Medium | no core tests by week 2 | CLI first; FastAPI only after research gates |

`P` là nhận định sơ bộ cần cập nhật tại ngày D1, không phải xác suất số học.

### 17.2. Scope reduction order (khi thiếu thời gian)

1. Cắt UI web, dùng terminal CLI.
2. Bỏ optional Dolly/HH/Alpaca/ToxiGen.
3. Bỏ LoRA rank sweep và DPO beta sweep.
4. Giảm train token budget, nhưng giữ tokenizer / checkpoint / metric reproducibility.
5. Giảm JSON schema xuống `category` + `action` và task categories có dữ liệu đầy đủ.
6. Giảm model từ Candidate P xuống S **trước khi** tiêu tốn GPU budget.
7. **Không cắt:** các requirement bắt buộc của đề: tokenizer, Transformer, cache, SFT-LoRA, DPO, guardrails, evaluation và paper.

### 17.3. Decision quality rules

- Nếu training budget không đủ, **không tự tuyên bố hội tụ**.
- Nếu DPO không làm tăng preference accuracy, báo cáo thất bại và đưa ablation giải thích.
- Nếu classifier rule-based thắng model, giữ nó trong báo cáo và lý giải khi nào cần/không cần LLM.
- Nếu không có LLM judge API, thay bằng deterministic evaluator + human blind sample; ghi rõ khác biệt với đề mục judge ban đầu và xin giảng viên xác nhận phạm vi nghiệm thu nếu bắt buộc phải dùng API judge.
- Nếu không thể tái lập bitwise do CUDA nondeterminism, tái lập bằng environment, seed, tolerance và phân phối metric, không ghi “fully deterministic” sai thực tế.

## 18. Scientific report & grading traceability

### 18.1. Paper outline 6–8 pages (IEEE hoặc ICLR)

1. **Abstract (150–250 words):** problem, model scale, intervention, measurements, findings thực tế, limitation.
2. **Introduction:** use case, vì sao small LLM từ đầu, research questions, contributions thật.
3. **Background & Related Work:** Transformer, TinyStories, LoRA, DPO, application guardrails, performance measurements.
4. **Methodology:** architecture config, datasets/splits, objective C0/C1/C2, guardrail boundaries, test protocol.
5. **Experimental Setup:** hardware/software, training budgets, baselines, data lineage, metric formula, hyperparams.
6. **Results:** C0 training curves; KV cache plots; Full vs LoRA table; SFT vs DPO comparison; guardrail safety/utility.
7. **Discussion & Error Analysis:** failed cases (≥5), confounders, why performance improvements may not generalize, rule-only comparator.
8. **Ethics & Limitations:** dataset synthetic nature, licensing, domain shift, security non-guarantee, model capacity, compute constraints.
9. **Conclusion:** what was learned, verified findings, future work; không làm quá claims.

### 18.2. Research figures cần có

| Fig | Plot | Required axes / note |
|---|---|---|
| F1 | Training and validation NLL vs tokens seen | Same tokenizer, log step events |
| F2 | TTFT, TPOT vs prefix length, cache ON/OFF | Separate plots or clearly labeled panels |
| F3 | Peak allocated/reserved VRAM vs prefix length | Explain allocator behavior |
| F4 | Full FT vs LoRA task metric and resource cost | Same checkpoint parent, control budgets |
| F5 | DPO preference accuracy / margin before-after | Held-out + task accuracy to show trade-off |
| F6 | Guardrails ASR/FPR/benign pass rate | OFF vs ON, clearly define threat suite |
| F7 | Confusion matrix of incident categories | Per-category support and failure types |

**Table minimum:** exact hyperparams, dataset sizes/splits, checkpoint ancestry, evaluation results with N and uncertainty, five failures. Không đưa hình minh họa không có raw data.

### 18.3. Grading rubric mapping (100%)

| Điểm theo đề | % | Artifact chứng minh |
|---|---:|---|
| Tokenizer | 10 | Code/training artifact + TOK tests + compression metrics |
| Transformer đúng | 10 | Model code + causal/gradient/unit tests |
| KV Cache tự viết | 10 | cache code + equivalence + TTFT/TPOT/VRAM plots |
| Evaluation & error | 10 | auto scripts + frozen datasets + ≥5 failures |
| Báo cáo khoa học | 10 | PDF IEEE/ICLR 6–8 pages |
| Trình bày | 10 | live demo + slide explaining evidence |
| Phản biện | 10 | deep concept Q&A, decisions/risk explanations |
| Hướng chuyên sâu | 20 | LoRA + SFT + DPO + guardrails verified |
| Có cải tiến | 10 | measured LoRA efficiency / controlled preference or guardrail improvement |
| **Total** | **100** | |

**Cải tiến phải có số liệu:** Ví dụ giảm trainable parameters hay optimizer memory khi dùng LoRA; cải thiện response validation/safety so với baseline; hoặc thay đổi DPO preference metric. Không claim DPO cải thiện nếu kết quả không support.

### 18.4. Defense / Q&A readiness rubric

Sinh viên phải giải thích được bằng **đường đi của một token thực tế**:

- Từ text → tokens → embeddings → attention → logits → sampled next token.
- Vì sao logits train khác decode khi positional offsets/cache sai.
- Kích thước tensor Q, K, V ở một layer và cách tính size KV cache.
- Tại sao LoRA giảm optimizer state nhưng không nhất thiết tăng inference throughput.
- Tại sao DPO loss=log(2) khi policy==reference.
- Tại sao model có thể giỏi dataset synthetic nhưng yếu ở incident thật.
- Tại sao rule-only baseline và output validator đôi khi đáng tin hơn LLM.
- Các ranh giới trust của prompt injection và access control bên ngoài model.

## 19. Career transfer map

### 19.1. Kiến thức chuyển giao trực tiếp

| Đồ án | AI Application Engineer | Backend Engineer |
|---|---|---|
| Tokenizer/context budgeting | prompt truncation, API cost, eval token budgets | request limits, input validation |
| Transformer/causal generation | model limitations, trade-offs prompt vs FT | predictability and failure handling |
| KV Cache & profiling | model serving throughput / token latency | caching, memory allocation, latency SLOs |
| SFT / LoRA | domain adaptation, cost-aware customization | versioned artifact + deployment lifecycle |
| DPO / preference datasets | feedback loop and behavior evaluation | data contracts, controlled experiments |
| Guardrails / threat model | prompt injection defense in depth | authz, schema validation, secrets, rate limits |
| Experiment harness | eval-driven AI application development | CI, observability, regression gates |
| Report / reproducibility | defensible model selection | evidence-based engineering decisions |

### 19.2. Những kỹ năng đồ án chưa dạy đủ

Đây không phải lộ trình AI Application Engineering hoàn chỉnh. **Sau đồ án** vẫn cần học: embeddings, retrieval/RAG, vector search, grounded generation, external LLM API semantics, structured outputs/function calling, MCP/tool permission design, agent workflows, evaluation pipelines trên production traffic, queues/background jobs, streaming SSE, concurrency, cost monitoring, incident response, privacy/security và model/provider abstraction.

### 19.3. Follow-on portfolio project (sau 4 tuần)

Dùng cùng evaluation framework và guardrail boundary để triển khai một **Backend API Incident Assistant** thật hơn:

```text
Spring Boot / FastAPI application
    ├─ authn/authz + tenant context
    ├─ incident ingestion and trace IDs
    ├─ retriever over public/internal runbooks (ACL-aware)
    ├─ model adapter (local or provider API)
    ├─ structured-output and evidence validators
    ├─ policy enforcement + audit logs
    └─ offline & online evaluation
```

**Điểm cần học:** Product-facing AI App Engineering thường không yêu cầu phải pretrain một LLM; phải chọn đúng giữa rules, classical ML, RAG, prompting, fine-tuning, hoặc agent tool use. Đồ án tạo nền tảng để lựa chọn có căn cứ, không buộc mọi vấn đề phải giải bằng model tự viết.

## 20. Decision log, open questions & references

### 20.1. Architecture decision log — v1

| ADR | Decision | Rationale | Revisit when |
|---|---|---|---|
| ADR-001 | Choose Track B | Aligns with AI App Engineering skill transfer | Career target changes to LLM infra |
| ADR-002 | Closed-world API incident triage | Model nhỏ có thể đánh giá khách quan | C0 đủ năng lực broaden task |
| ADR-003 | TinyStories for initial C0 | Small-model pretraining tractable | GPU/token budget prevents learning |
| ADR-004 | Hand-written core in PyTorch | Academic compliance + learning | Teacher explicitly grants exceptions |
| ADR-005 | RoPE + RMSNorm + SwiGLU | Modern yet tractable decoder block | Model correctness/compute constraints |
| ADR-006 | LoRA targeting Q,V | Matches assignment, clear parameter savings | Ablation after completion |
| ADR-007 | DPO uses frozen C1-lora reference | Clean comparison and correct objective | Policy init changes in documented new experiment |
| ADR-008 | Rule-based baseline | Check whether LLM is warranted | Never remove without scientific reason |
| ADR-009 | Security enforced externally | LLM output is untrusted | Never change without security review |
| ADR-010 | Domain synthetic train and group-split test | Feasible controlled benchmark | Real-world dataset becomes available |

### 20.2. Open questions — must resolve before D7

- **OQ-01:** Giảng viên chấp nhận tokenizer train bằng `tokenizers` nếu kèm BPE implementation tự viết không? Nếu không, cần chuyển implementation BPE tự viết thành main tokenizer.
- **OQ-02:** “From scratch” trong rubric cấm cả việc nạp pretrained weights? Main path ở đây **không dùng weights có sẵn**; external comparator chỉ khi chấp thuận.
- **OQ-03:** LLM-as-a-Judge là bắt buộc dùng remote judge thật hay deterministic scorer + judge subset được chấp nhận? Nếu bắt buộc, phải dự trù chi phí và ghi model/version.
- **OQ-04:** GPU thật là T4 16GB, RTX 3060 12GB, RTX 4090 24GB hay khác? Phải đo throughput ngay D1.
- **OQ-05:** Có bắt buộc mô hình đạt instruction following ở mức cụ thể, hay chỉ cần chứng minh objective và quality delta? Ghi biên bản rubric nếu cần.
- **OQ-06:** Publication/re-distribution of derived datasets + checkpoints có được phép theo nguồn và giấy phép hiện tại? Check trước upload public repo.

### 20.3. References — primary / official sources

Các nguồn dưới đây hỗ trợ lý thuyết, quy trình và license notes. **Xác minh truy cập ngày 2026-10-28**; URL, version và dataset card có thể cập nhật. Đối chiếu paper gốc khi gặp công thức tranh cãi.

**Core architecture & small models**

1. Vaswani et al., *Attention Is All You Need* (2017): https://arxiv.org/abs/1706.03762
2. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021): https://arxiv.org/abs/2104.09864
3. Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019): https://arxiv.org/abs/1910.07467
4. Shazeer, *GLU Variants Improve Transformer* (2020): https://arxiv.org/abs/2002.05202
5. Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* (2023): https://arxiv.org/abs/2305.07759
6. TinyStories dataset card: https://huggingface.co/datasets/roneneldan/TinyStories
7. SentencePiece official repository / tokenizer training: https://github.com/google/sentencepiece

**SFT, parameter-efficient tuning & preferences**

8. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021): https://arxiv.org/abs/2106.09685
9. Microsoft LoRA repository: https://github.com/microsoft/LoRA
10. Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023): https://arxiv.org/abs/2305.18290
11. Hugging Face TRL DPO documentation — comparator/format reference **only**, not allowed as main trainer: https://huggingface.co/docs/trl/main/dpo_trainer
12. Dolly-15k dataset card (license + categories): https://huggingface.co/datasets/databricks/databricks-dolly-15k
13. Anthropic HH-RLHF dataset card (caveats, chosen/rejected): https://huggingface.co/datasets/Anthropic/hh-rlhf
14. Alpaca-Cleaned card (**conflicting license fields; verify before use**): https://huggingface.co/datasets/yahma/alpaca-cleaned
15. ToxiGen dataset: https://huggingface.co/datasets/allenai/toxigen

**Evaluation, security, reproducibility**

16. Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023): https://arxiv.org/abs/2306.05685
17. OWASP, *Top 10 for LLM Applications (2025)*: https://genai.owasp.org/llm-top-10/
18. PyTorch docs, CUDA semantics / caching allocator: https://docs.pytorch.org/docs/stable/notes/cuda
19. PyTorch docs, reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
20. PyTorch docs, SDPA reference (not main self-written attention): https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
21. PyTorch docs, CUDA event for accurate timing: https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html

**Citation policy:** Trong final paper, không cite README/dataset page cho lý thuyết cốt lõi khi đã có paper gốc; dataset card là nguồn phù hợp cho field schema, license, data lineage. Khi version PyTorch khác, pin exact documentation page/version trong environment manifest.

## 21. Appendices — contracts, test vectors & operational checklists

### Appendix A — Worked examples: tự suy luận để không học vẹt

#### A.1. Một bước attention với 3 tokens

Giả sử input `[BOS, api, error]`, `T=3`, 1 head, `Q,K,V∈R^{1×1×3×d_head}`. Score `QK^T` có shape `[1,1,3,3]`. Causal mask hợp lệ:

```text
           K1    K2    K3
Query 1     ✓     ×     ×
Query 2     ✓     ✓     ×
Query 3     ✓     ✓     ✓
```

Khi predict token 4, logits cuối (ở Query 3) được dùng để sample. Nếu bạn thay `error` bằng `success`, logits ở Query 1 và Query 2 phải không đổi (với model eval, không dropout, no weird state). Đó là test chứng minh causality, mạnh hơn kiểm code có dòng `torch.tril()`.

#### A.2. Tại sao cache không cần cache Q?

Tại decode step mới, chỉ Q của token hiện tại được dùng để hỏi K/V quá khứ. Q cũ không cần cho tương lai vì causal query mới chỉ tính output của vị trí mới. K/V quá khứ vẫn là operands của dot-product nên phải giữ.

Nếu cache giữ `K,V` của 3 tokens và nhận token thứ 4:

```text
K_all = concat(K_cached[1:3], K_new[4])
V_all = concat(V_cached[1:3], V_new[4])
Q_new = Q(token[4], position=4)
output_4 = softmax(Q_new @ K_all.T / sqrt(d_head)) @ V_all
```

Mục tiêu correctness là cùng logits token 4 như full forward trên 4 tokens. **Mã trong ví dụ trên mang tính mô tả**; indices là vị trí 1-based, implementation Tensor PyTorch dùng 0-based.

#### A.3. Phân tích định lượng LoRA

Một `W_q` với `d_in=d_out=384` có `384×384=147,456` weight elements. LoRA `r=8` thêm `8×384+384×8=6,144` trainable params (~4.17% so với chính projection đó). Gắn vào cả `W_q` và `W_v` của 6 layers → `6×2×6,144=73,728` adapter trainable params (chưa kể bias/other trainable heads nếu cho phép).

**Tại sao tỷ lệ tham số cập nhật có thể rất thấp nhưng training VRAM không giảm tương ứng?** Vì full base weights vẫn cần để forward và backprop xuyên qua các layers để tính gradient cho adapters; activations vẫn phải lưu ở nhiều vị trí. Lợi ích chủ yếu nằm ở gradients/optimizer states không cần cho base params.

#### A.4. DPO bằng số — kiểm dấu

Giả sử một pair (log-likelihood tổng trên response, số âm là bình thường):

```text
policy(chosen)   = -2.0
policy(rejected) = -4.0       => Δpolicy = 2.0
ref(chosen)      = -3.0
ref(rejected)    = -4.0       => Δref = 1.0
beta             = 0.1
```

DPO logit = `0.1 × (2 − 1) = +0.1`, loss = `-log(sigmoid(0.1)) ≈ 0.6444`. Nếu `policy == reference`, logit = 0, loss = `log 2 ≈ 0.6931`. Ưu tiên chosen tương đối mạnh hơn ref → loss giảm, hợp với trực giác. Nếu implement đảo dấu và loss tăng, hãy kiểm tra `chosen`/`rejected` và phân số.

**Derivation intuition:** Tối ưu reward với KL penalty so với reference dẫn đến `π*(y|x) ∝ πref(y|x) exp(r(x,y)/β)`. Khi lấy reward difference giữa chosen và rejected, hằng số chuẩn hóa theo prompt triệt tiêu; dùng Bradley–Terry pairwise preference sẽ cho objective logistic trên **difference of log-ratios**. Đây là lý do có reference và log-probabilities, không phải một công thức tùy tiện.

#### A.5. SFT response masking bằng chuỗi ngắn

```text
Serialized:  [BOS] [USER] 401 [ASSISTANT] auth [EOS]
Inputs:      [BOS] [USER] 401 [ASSISTANT] auth
Targets:     [USER] 401 [ASSISTANT] auth [EOS]
Train labels:  -100   -100  -100         auth [EOS]
```

Lưu ý: position alignment của `labels` trong PyTorch có thể tổ chức là full-sequence labels shift **bên trong model loss** hoặc labels đã shift ở data loader; phải chọn **một** cách và test. Không shift hai lần. Token `auth` và EOS thuộc completion, các tokens user/instruction là context không dùng làm target loss của SFT.

#### A.6. Một tình huống phản ví dụ cho data leakage

Train set có `"401 token expired" → authentication`, test set cũng chính câu đó nhưng đổi từ "bearer" thành "access". Test accuracy cao chưa chứng minh tổng quát hóa. Thử test:

```text
STATUS=401, EVIDENCE=proxy rewrites 503 to 401; upstream health check fails.
```

Mục tiêu không nhất thiết bắt model biết hệ thống thực; nó phải tránh suy luận quá tự tin chỉ từ `401` khi evidence mâu thuẫn/thiếu. Ground truth cần ontology rule và review cho loại case này.

### Appendix B — API demo contract (optional, read-only)

```http
POST /v1/triage
Content-Type: application/json
X-Request-ID: 123e4567-e89b-12d3-a456-426614174000
```

```json
{
  "status_code": 429,
  "symptoms": "Requests are temporarily rejected",
  "evidence": "Response includes retry-after header",
  "max_new_tokens": 80
}
```

**Response success shape (proposed):**

```json
{
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "decision": {
    "category": "rate_limiting",
    "action": "retry_with_backoff"
  },
  "model_version": "C2-dpo",
  "guardrail_status": "passed",
  "usage": {"input_tokens": 42, "output_tokens": 18}
}
```

Khi model output invalid hoặc policy violation, trả status/error code do backend chọn; không trả lại JSON đã lỗi như thành công. `model_version` phải là artifact ID thật, không hardcode `C2-dpo` nếu đang serve C1. Endpoint không có chức năng query database hoặc thực thi shell. Demo nên tránh tính năng chat memory để giảm vấn đề privacy/cache isolation.

### Appendix C — Experiment output schemas

**`results/cache_benchmark.csv` (columns):**

```text
run_id,git_commit,model_checkpoint,precision,gpu,batch_size,
prefix_len,max_new_tokens,cache_enabled,repeat_idx,ttft_ms,
mean_tpot_ms,total_latency_ms,generated_tokens,decode_toks_per_sec,
peak_allocated_mib,peak_reserved_mib,correctness_passed
```

**`results/model_eval.csv` (columns):**

```text
checkpoint,stage,test_version,seed,n_samples,category_accuracy,
action_accuracy,category_macro_f1,schema_valid_rate,
unknown_recall,false_abstention_rate,preference_accuracy,
attack_success_rate,guardrail_false_positive_rate
```

Giá trị metric thiếu có thể được lưu `null`/NA với explanation; không điền `0` khi không đo. Every figure phải trace được tới raw CSV và command invocation.

### Appendix D — Definition of Done (DoD) theo capability

#### D.1. Foundation DoD

- [ ] Tokenizer own-training artifact và test roundtrip.
- [ ] Decoder-only model không import packaged model class.
- [ ] Causal mask future perturbation test pass.
- [ ] Checkpoint restore correctness; one-batch overfit pass.
- [ ] C0 train/validation logged; sample output lưu cùng seed.

#### D.2. KV Cache DoD

- [ ] Prefill & incremental decode, position offsets, cache reset.
- [ ] Logits equivalence cả unbatched và declared-supported shapes.
- [ ] `L={128,512,1024,2048}` benchmark full raw data.
- [ ] TTFT và TPOT phân biệt, memory allocated/reserved phân biệt.

#### D.3. Alignment DoD

- [ ] Response-only SFT mask unit tests.
- [ ] Full FT và LoRA cùng C0 + same split, report trainable params and GPU cost.
- [ ] LoRA Q,V adapter trainable, base weights frozen, save/load pass.
- [ ] DPO reference snapshot frozen, loss mathematical tests pass.
- [ ] DPO train logs + C1-vs-C2 evaluation; đủ cases âm nếu fail.

#### D.4. Safety & Research DoD

- [ ] Input/output policy và schema validator hoạt động.
- [ ] Không có tool side effect từ model output.
- [ ] ASR/FPR + benign utility + latency overhead report.
- [ ] Test split manifests, license notes, checkpoint ancestry tracked.
- [ ] 5+ failure analyses và một baseline rule-only.
- [ ] Paper 6–8 pages, code repo, documented commands và demo.

### Appendix E — Daily learning journal template

```markdown
## Date / Run ID
- Concept: What exactly did I study?
- Mechanism: Which tensor/loss/data contract changed?
- Hand derivation: Can I derive it with small numbers?
- Hypothesis: What result do I expect, and why?
- Controlled experiment: What variable changed? What stayed fixed?
- Correctness test: What invariant passed/failed?
- Evidence: Log/CSV/plot/checkpoint IDs.
- Surprise / failure: Which assumption failed?
- Next decision: Retain / rollback / redesign, with reason.
```

**Learning requirement:** Với mỗi kỹ thuật, phải có ít nhất **1 hand-worked example + 1 implementation test + 1 measured experiment + 1 failure analysis** (nếu kỹ thuật không có lỗi quan sát được, lấy một edge case hoặc limitation được chứng minh).

### Appendix F — Trước khi công bố kết quả

- [ ] Mỗi chart đều có đơn vị, N, hardware, seed và source data.
- [ ] Không so PPL khác tokenizer/domain như cùng metric.
- [ ] Không trộn Full FT vs LoRA với khác biệt số tokens hay checkpoint cha không ghi chú.
- [ ] Không báo DPO "tăng quality" chỉ từ train loss giảm.
- [ ] Không báo "safety solved" vì một regex rule block mọi request.
- [ ] Không đưa API token thật, PII, secrets, private prompts lên public repo.
- [ ] Không ghi licensed dataset raw data vào repo nếu chưa tuân thủ distribution terms.
- [ ] Thí nghiệm chưa chạy ghi rõ **Not Measured**, không placeholder giả.
- [ ] Những kết quả không đạt kỳ vọng vẫn có mặt trong final Discussion.

---

## Final acceptance statement

Một đồ án **MiniAlign thành công** không cần tạo ra LLM thông minh ngang model thương mại. Nó cần **chứng minh được bằng code, công thức, test và thực nghiệm** rằng tác giả hiểu cách một causal LM được token hóa, huấn luyện và sinh văn bản; cách KV Cache ảnh hưởng latency/memory; cách LoRA thay đổi subspace tham số được tối ưu; cách DPO dùng preference relative to reference; và cách AI application phải kiểm soát input/output như dữ liệu không tin cậy.

**Target professional identity:** Backend Engineer có khả năng thiết kế, đánh giá, tùy chỉnh và tích hợp mô hình ngôn ngữ thành AI application đáng tin cậy — không phụ thuộc mù quáng vào framework, benchmark hoặc model provider.

**Document change control:** Bất kỳ đổi `task ontology`, dataset split, tokenizer ID mapping, loss definition, checkpoint ancestry, primary metric, benchmark protocol hoặc security policy đều cần tăng version (ít nhất minor) và ghi ADR. Những thay đổi cosmetic/format có thể tăng patch. Không sửa định nghĩa metric sau khi xem frozen test results mà không công khai là protocol revision.
