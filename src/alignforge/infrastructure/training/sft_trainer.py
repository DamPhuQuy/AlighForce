"""Supervised Fine-Tuning (SFT) Infrastructure implementation.

Upholds OOP Inheritance: SFTTrainer subclasses BaseTrainer to specialize
gradient steps on prompt-masked instruction-completion pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from alignforge.infrastructure.data.dataset_repository import load_coedit_gec_samples
from alignforge.infrastructure.models.lora import apply_lora_to_model, save_lora_checkpoint
from alignforge.infrastructure.models.transformer import CausalTransformerModel
from alignforge.infrastructure.tokenization.bpe_tokenizer import ByteLevelBPETokenizer
from alignforge.infrastructure.training.base_trainer import BaseTrainer


class SFTDataset(Dataset):
    """Dataset with prompt-masked labels for response-only cross entropy."""

    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer: ByteLevelBPETokenizer,
        max_seq_len: int = 128,
    ) -> None:
        self.items: List[Tuple[torch.Tensor, torch.Tensor]] = []
        ignore_index = -100

        for s in samples:
            prompt = s["prompt"]
            completion = s["completion"]

            prompt_ids = tokenizer.encode(prompt, add_bos=True)
            comp_ids = tokenizer.encode(completion, add_eos=True)

            total_ids = prompt_ids + comp_ids
            labels = [ignore_index] * len(prompt_ids) + list(comp_ids)

            if len(total_ids) > max_seq_len:
                total_ids = total_ids[:max_seq_len]
                labels = labels[:max_seq_len]
            elif len(total_ids) < max_seq_len:
                pad_len = max_seq_len - len(total_ids)
                total_ids = total_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [ignore_index] * pad_len

            input_t = torch.tensor(total_ids, dtype=torch.long)
            target_t = torch.tensor(labels, dtype=torch.long)
            self.items.append((input_t, target_t))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.items[idx]


class SFTTrainer(BaseTrainer):
    """Supervised Fine-Tuning Trainer inheriting from BaseTrainer with FP16 support."""

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        with self.autocast_context():
            out = self.model(x)
            logits = out[0] if isinstance(out, tuple) else out

            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()

            vocab_size = getattr(self.model, "vocab_size", shift_logits.shape[-1])
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        self.backward_and_step(loss)
        return float(loss.item())


def train_sft(
    mode: str = "lora",  # "full" or "lora"
    config_path: str | Path = "configs/sft.yaml",
    base_checkpoint: str | Path = "artifacts/checkpoints/c0_base.pt",
    output_checkpoint: Optional[str | Path] = None,
    max_steps: Optional[int] = None,
    device_override: Optional[str] = None,
    batch_size: Optional[int] = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Train SFT model either fully or via LoRA on CoEdIT GEC dataset."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tok_dir = Path("artifacts/tokenizer")
    tokenizer = ByteLevelBPETokenizer.load(tok_dir)

    model = CausalTransformerModel(
        vocab_size=2048,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=128,
        dropout=float(cfg["training"]["dropout"]),
    )

    base_path = Path(base_checkpoint)
    if base_path.exists():
        ckpt = torch.load(base_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    if mode == "lora":
        apply_lora_to_model(
            model,
            r=int(cfg["lora"]["r"]),
            alpha=float(cfg["lora"]["alpha"]),
            dropout=float(cfg["lora"]["dropout"]),
            target_modules=tuple(cfg["lora"]["target_modules"]),
        )
        default_out = "artifacts/checkpoints/c1_lora.pt"
    else:
        default_out = "artifacts/checkpoints/c1_full.pt"

    out_ckpt = Path(output_checkpoint or default_out)

    raw_samples = load_coedit_gec_samples(max_samples=cfg["data"]["max_samples"])
    bs = batch_size or (16 if device.type == "cuda" else cfg["training"]["batch_size"])
    dataset = SFTDataset(raw_samples, tokenizer, max_seq_len=128)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    trainer = SFTTrainer(
        model=model,
        dataloader=loader,
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        max_grad_norm=float(cfg["training"]["max_grad_norm"]),
        device=device,
        use_amp=use_amp,
    )

    steps = max_steps or cfg["training"]["max_steps"]
    metrics = trainer.train(max_steps=steps)

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if mode == "lora":
        save_lora_checkpoint(model, out_ckpt, config=cfg)
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "loss": metrics["final_loss"],
                "steps": steps,
            },
            out_ckpt,
        )

    return metrics
