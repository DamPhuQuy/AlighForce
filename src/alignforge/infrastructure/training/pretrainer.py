"""Pretrainer Infrastructure implementation.

Inherits from BaseTrainer to execute causal language modeling pretraining on TinyStories.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from alignforge.infrastructure.models.transformer import CausalTransformerModel
from alignforge.infrastructure.tokenization.bpe_tokenizer import ByteLevelBPETokenizer
from alignforge.infrastructure.training.base_trainer import BaseTrainer


class TextDataset(Dataset):
    """Tokenized tensor dataset for next-token prediction."""

    def __init__(self, token_ids: List[int], seq_len: int = 128) -> None:
        self.seq_len = seq_len
        num_chunks = len(token_ids) // (seq_len + 1)
        self.chunks: List[torch.Tensor] = []
        for i in range(num_chunks):
            chunk = token_ids[i * (seq_len + 1) : (i + 1) * (seq_len + 1)]
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class Pretrainer(BaseTrainer):
    """Causal Language Model Pretrainer implementing BaseTrainer with FP16 support."""

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        with self.autocast_context():
            out = self.model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.view(-1, self.model.vocab_size), y.view(-1))

        self.backward_and_step(loss)
        return float(loss.item())


def train_pretrain(
    config_path: str | Path = "configs/model.yaml",
    max_steps: Optional[int] = None,
    output_checkpoint: str | Path = "artifacts/checkpoints/c0_base.pt",
    device_override: Optional[str] = None,
    batch_size: Optional[int] = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Train base causal language model C0 and export checkpoint."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tok_dir = Path("artifacts/tokenizer")
    if (tok_dir / "tokenizer.json").exists():
        tokenizer = ByteLevelBPETokenizer.load(tok_dir)
    else:
        starter_stories = [
            "Once upon a time, there was a little girl named Lily. She loved to run in the park.",
            "Tim had a red car. He played with his car every day in his room.",
            "The sun was bright and warm. A little puppy barked happily in the green garden.",
            "A bird flew high into the blue sky. It sang a sweet song for all the animals.",
            "She helped her mother bake a cake. The cake smelled delicious and sweet.",
            "The cat sat on the soft mat. It purred softly as the boy petted its fur.",
        ] * 20
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(starter_stories, target_vocab_size=cfg["model"]["vocab_size"])
        tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tok_dir)

    model = CausalTransformerModel(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"],
        norm_eps=float(cfg["model"]["norm_eps"]),
        rope_base=float(cfg["model"]["rope_base"]),
    ).to(device)

    corpus_text = (
        "Once upon a time, a little girl named Lily lived in a small house near the woods. "
        "Lily had a fluffy dog named Spot. Spot loved to run and jump. "
        "Every morning, Lily and Spot walked to school together. "
        "They saw birds singing in the trees and flowers blooming in the yard. "
        "One sunny afternoon, Spot found a shiny golden key under an old oak tree. "
        "Lily picked up the key and smiled. She wondered what magical door it would open."
    ) * 50
    tokens = tokenizer.encode(corpus_text, add_bos=True, add_eos=True)
    dataset = TextDataset(tokens, seq_len=cfg["model"]["max_seq_len"])
    bs = batch_size or (32 if device.type == "cuda" else cfg["training"]["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
    )

    trainer = Pretrainer(
        model=model,
        dataloader=loader,
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        max_grad_norm=float(cfg["training"]["max_grad_norm"]),
        device=device,
        use_amp=use_amp,
    )

    steps = max_steps or 50
    metrics = trainer.train(max_steps=steps)

    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "step": steps,
            "loss": metrics["final_loss"],
            "vocab_size": model.vocab_size,
        },
        out_path,
    )

    return metrics
