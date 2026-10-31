"""Base Trainer Infrastructure class.

Encapsulates optimization state, mixed precision scaling, device management, and step execution.
Upholds Template Method Pattern (OOP Inheritance & Polymorphism).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from alignforge.domain.interfaces import ITrainer


class BaseTrainer(ITrainer, ABC):
    """Abstract Base Trainer orchestrating training loops and gradient updates."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
    ) -> None:
        self._device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = model.to(self._device)
        self._dataloader = dataloader
        self._max_grad_norm = max_grad_norm
        self._lr = lr
        self._weight_decay = weight_decay

        trainable_params = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay
        )

        # Mixed Precision support via CUDA AMP
        self._use_amp = use_amp and (self._device.type == "cuda")
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def use_amp(self) -> bool:
        return self._use_amp

    def autocast_context(self):
        """Context manager for FP16 mixed precision forward pass."""
        return torch.cuda.amp.autocast(enabled=self._use_amp, dtype=torch.float16)

    def backward_and_step(self, loss: torch.Tensor) -> None:
        """Encapsulated gradient scaling, backward pass, clipping, and optimizer step."""
        if self._use_amp:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
            nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()

    @abstractmethod
    def train_step(self, batch: Any) -> float:
        """Abstract step computing loss, executing backwards pass, and returning scalar loss."""
        pass

    def train(self, max_steps: int = 50, log_interval: int = 10) -> Dict[str, float]:
        """Execute template training loop for max_steps iterations."""
        self._model.train()
        step_count = 0
        final_loss = 0.0
        t0 = time.time()

        while step_count < max_steps:
            for batch in self._dataloader:
                loss_val = self.train_step(batch)
                step_count += 1
                final_loss = loss_val

                if step_count % log_interval == 0 or step_count == max_steps:
                    amp_tag = " (FP16)" if self._use_amp else ""
                    print(f"Step {step_count:4d}/{max_steps}{amp_tag} | Loss: {final_loss:.4f}")

                if step_count >= max_steps:
                    break

        elapsed = time.time() - t0
        return {
            "final_loss": final_loss,
            "steps": float(step_count),
            "elapsed_seconds": elapsed,
        }
