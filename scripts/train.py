"""Turnkey Google Colab Training Runner with T4 GPU Acceleration & Mixed Precision for AlignForge English."""

from __future__ import annotations

import os
from pathlib import Path
import time
import torch
from alignforge.infrastructure.evaluation import (
    run_full_evaluation_pipeline,
    run_kv_cache_profiling,
)
from alignforge.infrastructure.training import (
    train_dpo,
    train_pretrain,
    train_sft,
)


def main():
    print("=" * 70)
    print("AlignForge English: Training Pipeline (AMP FP16 Enabled)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_gpu = (device.type == "cuda")
    print(f"Active Device: {device}")
    if is_gpu:
        print(f"GPU Name:     {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print("Tensor Cores: Enabled (Automatic Mixed Precision FP16)")
    else:
        print("[Note] Running on CPU. For high-speed training, activate T4 GPU under Runtime > Change runtime type.")

    # Configure scaled hyper-parameters based on compute accelerator
    pretrain_steps = 1000 if is_gpu else 50
    pretrain_bs = 32 if is_gpu else 8

    sft_steps = 300 if is_gpu else 40
    sft_bs = 16 if is_gpu else 8

    dpo_steps = 200 if is_gpu else 30
    dpo_bs = 8 if is_gpu else 4

    # 1. Base Pre-training C0
    print(f"\n>>> Stage 1: Base Pretraining C0 on TinyStories ({pretrain_steps} steps, batch_size={pretrain_bs}, FP16={is_gpu})...")
    pretrain_metrics = train_pretrain(
        max_steps=pretrain_steps,
        batch_size=pretrain_bs,
        use_amp=is_gpu,
        output_checkpoint="artifacts/checkpoints/c0_base.pt",
    )
    print(f"C0 Pretrain Loss: {pretrain_metrics['final_loss']:.4f}")

    # 2. KV Cache Multi-Length Profiling
    print("\n>>> Stage 2: Profiling Dynamic KV Cache (128, 512, 1024, 2048)...")
    run_kv_cache_profiling()

    # 3. Supervised Fine-Tuning: LoRA vs Full SFT
    print(f"\n>>> Stage 3A: Supervised Fine-Tuning with LoRA (C1-lora) ({sft_steps} steps, batch_size={sft_bs})...")
    lora_metrics = train_sft(
        mode="lora",
        max_steps=sft_steps,
        batch_size=sft_bs,
        use_amp=is_gpu,
        output_checkpoint="artifacts/checkpoints/c1_lora.pt",
    )

    print(f"\n>>> Stage 3B: Controlled Full SFT Comparison (C1-full) ({sft_steps} steps, batch_size={sft_bs})...")
    full_metrics = train_sft(
        mode="full",
        max_steps=sft_steps,
        batch_size=sft_bs,
        use_amp=is_gpu,
        output_checkpoint="artifacts/checkpoints/c1_full.pt",
    )

    # 4. Direct Preference Optimization (DPO)
    print(f"\n>>> Stage 4: DPO Preference Alignment on Paraphrase Pairs (C2-dpo) ({dpo_steps} steps, batch_size={dpo_bs})...")
    dpo_metrics = train_dpo(
        max_steps=dpo_steps,
        batch_size=dpo_bs,
        use_amp=is_gpu,
        output_checkpoint="artifacts/checkpoints/c2_dpo.pt",
    )

    # 5. Objective Evaluation Benchmarks & Failure Analysis
    print("\n>>> Stage 5: Objective Evaluation & Error Analysis...")
    eval_metrics = run_full_evaluation_pipeline()

    print("\n" + "=" * 70)
    print("COLAB TRAINING & BENCHMARK PIPELINE COMPLETE!")
    print(f"- C0 Base Checkpoint:      artifacts/checkpoints/c0_base.pt (Final Loss: {pretrain_metrics['final_loss']:.4f})")
    print(f"- C1 LoRA Checkpoint:      artifacts/checkpoints/c1_lora.pt (Final Loss: {lora_metrics['final_loss']:.4f})")
    print(f"- C1 Full SFT Checkpoint:  artifacts/checkpoints/c1_full.pt (Final Loss: {full_metrics['final_loss']:.4f})")
    print(f"- C2 DPO Checkpoint:       artifacts/checkpoints/c2_dpo.pt (Final Loss: {dpo_metrics['final_loss']:.4f})")
    print(f"- Profiling Report:        reports/kv_cache_profiling.csv")
    print(f"- Evaluation Report:       reports/eval_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
