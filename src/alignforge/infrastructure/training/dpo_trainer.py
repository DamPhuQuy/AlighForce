"""Direct Preference Optimization (DPO) Infrastructure implementation.

Upholds OOP Inheritance: DPOTrainer subclasses BaseTrainer, encapsulating
frozen reference model comparisons and response-masked loss formulation.
"""

from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from alignforge.domain.services import DPOLossService
from alignforge.infrastructure.data.dataset_repository import load_paraphrase_dpo_samples
from alignforge.infrastructure.models.lora import apply_lora_to_model, load_lora_checkpoint, save_lora_checkpoint
from alignforge.infrastructure.models.transformer import CausalTransformerModel
from alignforge.infrastructure.tokenization.bpe_tokenizer import ByteLevelBPETokenizer
from alignforge.infrastructure.training.base_trainer import BaseTrainer


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_pad_token_id: int = -100,
) -> torch.Tensor:
    """Compute per-sequence log probabilities strictly on response tokens."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_mask = shift_labels != label_pad_token_id
    clean_labels = shift_labels.clone()
    clean_labels[~loss_mask] = 0

    per_token_logps = torch.gather(
        shift_logits.log_softmax(-1),
        dim=2,
        index=clean_labels.unsqueeze(2),
    ).squeeze(2)

    return (per_token_logps * loss_mask).sum(-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Compute DPO loss, margin, and win_rate."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    margin = chosen_rewards - rejected_rewards
    win_rate = (logits > 0).float().mean().item()

    return losses.mean(), margin.mean(), win_rate


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute vector DPO loss."""
    loss, margin, win_rate = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return loss, chosen_rewards.mean(), rejected_rewards.mean()


class DPODataset(Dataset):
    """Dataset packaging prompt + chosen and prompt + rejected pairs with prompt masking."""

    def __init__(
        self,
        pairs: List[Dict[str, Any]],
        tokenizer: ByteLevelBPETokenizer,
        max_seq_len: int = 128,
    ) -> None:
        self.items: List[Dict[str, torch.Tensor]] = []
        ignore_index = -100

        for p in pairs:
            prompt = p["prompt"]
            chosen = p["chosen"]
            rejected = p["rejected"]

            prompt_ids = tokenizer.encode(prompt, add_bos=True)
            chosen_ids = tokenizer.encode(f" {chosen}", add_eos=True)
            rejected_ids = tokenizer.encode(f" {rejected}", add_eos=True)

            c_input = prompt_ids + chosen_ids
            c_label = [ignore_index] * len(prompt_ids) + list(chosen_ids)

            r_input = prompt_ids + rejected_ids
            r_label = [ignore_index] * len(prompt_ids) + list(rejected_ids)

            def pad_or_truncate(tokens: List[int], labels: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                    labels = labels[:max_seq_len]
                elif len(tokens) < max_seq_len:
                    diff = max_seq_len - len(tokens)
                    tokens = tokens + [tokenizer.pad_token_id] * diff
                    labels = labels + [ignore_index] * diff
                return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

            c_in_t, c_lbl_t = pad_or_truncate(c_input, c_label)
            r_in_t, r_lbl_t = pad_or_truncate(r_input, r_label)

            self.items.append({
                "chosen_input": c_in_t,
                "chosen_label": c_lbl_t,
                "rejected_input": r_in_t,
                "rejected_label": r_lbl_t,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


class DPOTrainer(BaseTrainer):
    """Direct Preference Optimization Trainer inheriting from BaseTrainer with FP16 support."""

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        dataloader: DataLoader,
        beta: float = 0.1,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
    ) -> None:
        super().__init__(
            model=policy_model,
            dataloader=dataloader,
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            device=device,
            use_amp=use_amp,
        )
        self._ref_model = ref_model.to(self.device).eval()
        for p in self._ref_model.parameters():
            p.requires_grad = False
        self._beta = beta

    @property
    def ref_model(self) -> nn.Module:
        return self._ref_model

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        c_in = batch["chosen_input"].to(self.device)
        c_lbl = batch["chosen_label"].to(self.device)
        r_in = batch["rejected_input"].to(self.device)
        r_lbl = batch["rejected_label"].to(self.device)

        self.optimizer.zero_grad()
        with self.autocast_context():
            # Policy forward pass
            out_c = self.model(c_in)
            out_r = self.model(r_in)
            logits_pi_c = out_c[0] if isinstance(out_c, tuple) else out_c
            logits_pi_r = out_r[0] if isinstance(out_r, tuple) else out_r

            pi_chosen_logp = get_batch_logps(logits_pi_c, c_lbl)
            pi_rejected_logp = get_batch_logps(logits_pi_r, r_lbl)

            # Frozen reference forward pass
            with torch.no_grad():
                out_ref_c = self._ref_model(c_in)
                out_ref_r = self._ref_model(r_in)
                logits_ref_c = out_ref_c[0] if isinstance(out_ref_c, tuple) else out_ref_c
                logits_ref_r = out_ref_r[0] if isinstance(out_ref_r, tuple) else out_ref_r

                ref_chosen_logp = get_batch_logps(logits_ref_c, c_lbl)
                ref_rejected_logp = get_batch_logps(logits_ref_r, r_lbl)

            loss, _, _ = compute_dpo_loss(
                pi_chosen_logp,
                pi_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                beta=self._beta,
            )

        self.backward_and_step(loss)
        return float(loss.item())


def train_dpo(
    config_path: str | Path = "configs/dpo.yaml",
    sft_checkpoint: str | Path = "artifacts/checkpoints/c1_lora.pt",
    output_checkpoint: str | Path = "artifacts/checkpoints/c2_dpo.pt",
    max_steps: Optional[int] = None,
    device_override: Optional[str] = None,
    batch_size: Optional[int] = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Train DPO policy model from SFT baseline and save checkpoint."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tok_dir = Path("artifacts/tokenizer")
    tokenizer = ByteLevelBPETokenizer.load(tok_dir)

    def create_model() -> CausalTransformerModel:
        m = CausalTransformerModel(
            vocab_size=2048,
            d_model=256,
            n_heads=4,
            n_layers=4,
            d_ff=1024,
            max_seq_len=128,
            dropout=0.0,
        )
        apply_lora_to_model(
            m,
            r=int(cfg["lora"]["r"]),
            alpha=float(cfg["lora"]["alpha"]),
            dropout=0.0,
            target_modules=tuple(cfg["lora"]["target_modules"]),
        )
        ckpt_path = Path(sft_checkpoint)
        if ckpt_path.exists():
            load_lora_checkpoint(m, ckpt_path)
        return m

    policy_model = create_model()
    ref_model = create_model()

    bs = batch_size or (8 if device.type == "cuda" else cfg["training"]["batch_size"])
    pairs = load_paraphrase_dpo_samples(max_samples=cfg["data"]["max_samples"])
    dataset = DPODataset(pairs, tokenizer, max_seq_len=128)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    trainer = DPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        dataloader=loader,
        beta=float(cfg["dpo"]["beta"]),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        max_grad_norm=float(cfg["training"]["max_grad_norm"]),
        device=device,
        use_amp=use_amp,
    )

    steps = max_steps or cfg["training"]["max_steps"]
    metrics = trainer.train(max_steps=steps)

    out_ckpt = Path(output_checkpoint)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_lora_checkpoint(policy_model, out_ckpt, config=cfg)

    return metrics
