# ĐỀ TÀI ĐỒ ÁN CUỐI KHÓA: XÂY DỰNG, TỐI ƯU VÀ CĂN CHỈNH MÔ HÌNH NGÔN NGỮ LỚN TỪ SỐ 0
*(Capstone Project: Building, Optimizing, and Aligning Large Language Models from Scratch)*

* **Thời lượng:** 4 tuần
* **Quy mô:** 1 sinh viên/nhóm

---

## 1. YÊU CẦU NỀN TẢNG BẮT BUỘC

Mọi nhóm sinh viên bắt buộc phải tự tay lập trình và nghiệm thu khối kiến trúc lõi trước khi đi vào các bài toán chuyên sâu:

* **Kiến trúc Transformer Decoder-Only Hiện đại:** Lập trình từ đầu toàn bộ các khối cơ bản của mô hình.
* **Hệ thống Quản lý Bộ nhớ & Tự xây dựng KV Cache:** 
  * Lập trình cấu trúc dữ liệu lưu trữ KV Cache dạng tensor động phục vụ quá trình sinh từ tự hồi quy (Autoregressive Generation).
  * **Đo đạc và lập biểu đồ thực nghiệm (Profiling):** Đo tốc độ sinh từ (Tokens/giây), độ trễ *First Token Latency* vs. *Next Token Latency*, và mức độ chiếm dụng VRAM khi **Bật vs. Tắt KV Cache** với các độ dài ngữ cảnh tăng dần: $L \in \{128, 512, 1024, 2048\}$.
* **Tự xây dựng mô hình Tokenizer:** Huấn luyện và tích hợp bộ tách từ cho mô hình.

---

## 2. PHÂN HƯỚNG CHUYÊN SÂU (Sinh viên chọn 1 trong 3 Hướng)

Sau khi hoàn thành hệ thống cốt lõi, mỗi nhóm chọn **duy nhất 1 bài toán chuyên sâu** để tạo đóng góp thực nghiệm rõ nét.

```text
                           [Khung Cốt Lõi Bắt Buộc]
                                      |
       +------------------------------+------------------------------+
       |                              |                              |
  [HƯỚNG A: SYSTEMS]          [HƯỚNG B: ALIGNMENT]          [HƯỚNG C: REASONING]
  Tối ưu Hệ thống,            Tinh chỉnh Tham số,           Kích hoạt Suy luận,
  MoE & Ngữ cảnh Dài          DPO & An toàn Mô hình         CoT & Test-Time Compute
```

### Hướng A: Tối ưu Hệ thống, Kiến trúc Nâng cao & Mở rộng Ngữ cảnh
* **Trọng tâm:** Đào sâu vào hiệu năng phần cứng, kiến trúc thưa và xử lý chuỗi dài.
* **Nhiệm vụ nghiên cứu:**
  1. **Kiến trúc Mixture of Experts (MoE):** Thay thế Feed-Forward Network (FFN) thông thường bằng kiến trúc MoE với $N$ experts (ví dụ 4 hoặc 8 experts) và cơ chế Top-2 Softmax Gating. Lập trình thêm *Auxiliary Load Balancing Loss* để tránh hiện tượng sập định tuyến (routing collapse).
  2. **Mở rộng Ngữ cảnh Dài (Long-Context):** Triển khai kỹ thuật mở rộng tần số cơ sở của RoPE (như *RoPE Base Frequency Scaling* hoặc *Linear Interpolation*) để kéo dài context length gấp 2–4 lần so với thiết lập ban đầu.
  3. **Lượng hóa (Quantization):** Viết hàm lượng hóa trọng số hoặc kích hoạt sang định dạng INT8/FP8 (Dynamic hoặc Static Quantization) và đánh giá độ suy giảm Perplexity (PPL).
* **Đánh giá thực nghiệm:** Thực hiện bài kiểm tra *Needle In A Haystack (NIAH)* để đo độ chính xác truy xuất thông tin ở các vị trí khác nhau trong văn bản dài (đặc biệt là vùng giữa context).

### Hướng B: Tinh chỉnh Hiệu quả, Căn chỉnh Hành vi & An toàn (Alignment & Safety)
* **Trọng tâm:** Hậu huấn luyện, điều khiển hành vi và xây dựng rào chắn an toàn.
* **Nhiệm vụ nghiên cứu:**
  1. **Lập trình PEFT (LoRA from Scratch - Bài 10):** Tự lập trình module LoRA (Low-Rank Adaptation) với các tham số rank $r$ và hệ số $\alpha$ tùy chỉnh, gắn vào các ma trận chiếu $W_q, W_v$. Thực hiện SFT mô hình trên tập dữ liệu chỉ dẫn (ví dụ: `tatsu-lab/alpaca` hoặc `databricks-dolly-15k`).
  2. **Căn chỉnh Sở thích Không cần Reward Model (DPO - Bài 12):** Tự viết hàm mất mát Direct Preference Optimization (DPO) theo công thức chuẩn:
     $$L_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
     Căn chỉnh mô hình trên tập dữ liệu so sánh ưu tiên (ví dụ: `Anthropic/hh-rlhf` subset).
  3. **Guardrails & An toàn (Bài 14):** Xây dựng bộ lọc an toàn đầu vào/đầu ra (Input/Output Guardrails) chống Prompt Injection, Jailbreak và nội dung độc hại.
* **Đánh giá thực nghiệm:** So sánh chất lượng phản hồi bằng phương pháp *LLM-as-a-Judge* (sử dụng GPT-4o mini hoặc Claude 3.5 Haiku làm trọng tài chấm cặp theo thang điểm MT-Bench/AlpacaEval).

### Hướng C: Năng lực Suy luận & Tính toán Thời điểm Suy luận (Reasoning & Test-Time Compute)
* **Trọng tâm:** Kích hoạt tư duy từng bước, giải quyết bài toán phức tạp và mở rộng suy luận.
* **Nhiệm vụ nghiên cứu:**
  1. **SFT với Dữ liệu Chuỗi Tư duy (Chain-of-Thought - Bài 9, 13):** Tinh chỉnh mô hình trên tập dữ liệu suy luận toán học hoặc logic (như `GSM8K` hoặc `MATH` subset) có định dạng reasoning rõ ràng: phân tách rõ phần `<think>...</think>` và phần kết luận cuối cùng.
  2. **Tối ưu hóa Thời điểm Suy luận (Test-Time Scaling - Bài 13):** Lập trình và so sánh các chiến lược sinh từ lúc inference:
     * *Majority Voting (Self-Consistency):* Sinh $K$ chuỗi suy luận độc lập ở nhiệt độ cao ($T > 0$) và dùng thuật toán bỏ phiếu chọn kết quả xuất hiện nhiều nhất.
     * *Best-of-N Re-ranking:* Sử dụng một hàm chấm điểm tự viết hoặc mô hình kiểm tra (Verifier) để lọc ra chuỗi suy luận có tính nhất quán cao nhất.
  3. **Phân tích Quá trình Suy luận:** Đo lường mối tương quan giữa độ dài chuỗi tư duy (số lượng token reasoning sinh ra) và độ chính xác của câu trả lời cuối cùng.
* **Đánh giá thực nghiệm:** Đánh giá độ chính xác tuyệt đối ($\text{Pass}@1$, $\text{Pass}@k$) trên tập kiểm thử chuẩn GSM8K bằng regex trích xuất đáp án số tự động.

---

## 3. DỮ LIỆU & TÀI NGUYÊN HUẤN LUYỆN KHUYẾN NGHỊ

Đồ án được tính toán để chạy mượt mà trên phần cứng đơn GPU (RTX 3060 12GB, RTX 4090 24GB, hoặc Google Colab T4/L4):

| Hạng mục | Hướng A: Systems | Hướng B: Alignment | Hướng C: Reasoning |
| :--- | :--- | :--- | :--- |
| **Tập dữ liệu huấn luyện** | TinyStories (Pre-train) / C4 | Alpaca-Cleaned, hh-rlhf (DPO) | GSM8K (split train), Open-R1 sub |
| **Benchmark đánh giá** | NIAH Benchmark, Perplexity (WikiText-2) | MT-Bench (subset), ToxiGen | GSM8K Test Split (1.319 câu) |

---

## 4. TIẾN ĐỘ THỰC HIỆN & BÀN GIAO (MILESTONES)

* **Cột mốc 1 (Tuần 1) - Bản Đề cương Kỹ thuật (System Proposal):**
  * Nộp đề cương (2 trang): Sinh viên chọn những kỹ thuật gì, sơ đồ kiến trúc module dự kiến, các siêu tham số định hướng ($d_{\text{model}}$, $n_{\text{heads}}$, $n_{\text{layers}}$, block size).
* **Cột mốc 2 (Tuần 2) - Nghiệm thu Khung Cốt lõi (Midway Checkpoint):**
  * Demo mã nguồn PyTorch thuần: Code forward pass của Transformer, nạp trọng số thành công, sinh được văn bản theo kiểu autoregressive.
  * Nộp báo cáo đo đạc profiling định lượng: Tốc độ và bộ nhớ khi Có KV Cache vs. Không có KV Cache.
* **Cột mốc 3 (Tuần 3) - Hoàn thành Huấn luyện & Tối ưu Chuyên sâu:**
  * Nộp nhật ký huấn luyện (W&B/TensorBoard log): Đường cong Loss hội tụ của kỹ thuật chuyên sâu (MoE routing / SFT-DPO / Reasoning).
* **Cột mốc 4 (Tuần 4) - Báo cáo Khoa học & Demo Hoàn thiện:**
  * Nộp báo cáo khoa học chuẩn format IEEE hoặc ICLR (độ dài 6–8 trang).
  * Nộp GitHub repo có tài liệu hướng dẫn (`README.md`) tái lập kết quả một lệnh (one-click run).
  * Trình diễn Demo tương tác (qua terminal hoặc giao diện web Gradio).

---

## 5. THANG ĐIỂM ĐÁNH GIÁ (GRADING RUBRIC)

### Cơ cấu điểm số

| Tiêu chí đánh giá | Tỷ trọng |
| :--- | :---: |
| 1. Tự xây dựng mô hình Tokenizer | 10% |
| 2. Tính Chuẩn xác của Kiến trúc Transformer | 10% |
| 3. KV Cache (From Scratch) | 10% |
| 4. Đo lường Đánh giá Khách quan & Phân tích Lỗi (Evaluation & Error) | 10% |
| 5. Báo cáo Kỹ thuật Chuẩn Format Khoa học | 10% |
| 6. Kỹ năng Trình bày | 10% |
| 7. Tư duy Phản biện | 10% |
| 8. Phát triển được hướng chuyên sâu | 20% |
| 9. Có cải tiến sáng tạo | 10% |
| **Tổng cộng** | **100%** |

### Yêu cầu chi tiết cho các hạng mục đánh giá

* **Thực nghiệm & Phân tích chuyên sâu:** Thực hiện bài kiểm thử so sánh rõ ràng:
  * *Hướng A:* So sánh throughput & VRAM giữa Dense Model vs. MoE Model, hoặc hiệu năng mở rộng ngữ cảnh.
  * *Hướng B:* So sánh Full Fine-tuning vs. LoRA tự viết; mô hình gốc vs. sau khi chạy DPO.
  * *Hướng C:* So sánh Greedy Search thông thường vs. Majority Voting / Best-of-N lúc Test-Time.
* **Đo lường & Phân tích Thất bại:**
  * Có script đánh giá tự động trên benchmark định lượng.
  * Có mục *"Error Analysis"*: Phân tích ít nhất 5 trường hợp mô hình cho ra kết quả sai (do lý do gì: tokenization, attention suy giảm ở context dài, hay ảo giác kiến thức).
* **Báo cáo Kỹ thuật:**
  * Cấu trúc bài báo khoa học chuẩn chỉnh: *Abstract, Methodology, Experimental Setup, Results, Discussion, Ethics & Limitations*.

> **⚠️ QUY ĐỊNH BẤT BIẾN:** 
> Toàn bộ kiến trúc mạng, cơ chế chú ý (Attention) và công cụ giải mã (Inference Engine) **phải được tự lập trình bằng PyTorch thuần (`torch.nn.Module`)**. Tuyệt đối cấm sử dụng các class đóng gói sẵn toàn bộ mô hình (như `AutoModelForCausalLM` của Hugging Face) để làm kiến trúc chính. Sinh viên được phép dùng thư viện chuẩn để xử lý Tokenizer, DataLoader, tính toán số liệu và nạp các tập dữ liệu mở có sẵn.