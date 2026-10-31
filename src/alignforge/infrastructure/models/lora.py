"""Low-Rank Adaptation (LoRA) module and Adapter pattern implementation.

Adheres to Open/Closed Principle (OCP) and Liskov Substitution Principle (LSP):
Adapts linear projections and models without altering base architecture definitions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn

from alignforge.domain.entities import ModelArchitectureConfig
from alignforge.domain.interfaces import ICausalLM, IKVCache


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear layer with trainable low-rank matrices A and B."""

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze base linear layer
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + lora_out * self.scaling


class LoRAModelAdapter(nn.Module, ICausalLM):
    """Object-Oriented Adapter wrapping ICausalLM with LoRA fine-tuning capabilities."""

    def __init__(
        self,
        base_model: ICausalLM,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
    ) -> None:
        super().__init__()
        assert isinstance(base_model, nn.Module), "base_model must be a PyTorch nn.Module"
        self._base_model = base_model
        self._r = r
        self._alpha = alpha
        self._dropout = dropout
        self._target_modules = target_modules

        self._attach_lora()

    @property
    def config(self) -> ModelArchitectureConfig:
        return self._base_model.config

    @property
    def base_model(self) -> ICausalLM:
        return self._base_model

    def _attach_lora(self) -> None:
        """Freeze base model and attach LoRALinear modules."""
        for param in self._base_model.parameters():
            param.requires_grad = False

        # In modern AlignForgeModel / CausalTransformerModel
        if hasattr(self._base_model, "layers"):
            for block in self._base_model.layers:
                attn = block.attn
                for mod_name in self._target_modules:
                    if hasattr(attn, mod_name):
                        orig_layer = getattr(attn, mod_name)
                        if isinstance(orig_layer, nn.Linear):
                            lora_layer = LoRALinear(
                                orig_layer, r=self._r, alpha=self._alpha, dropout=self._dropout
                            )
                            setattr(attn, mod_name, lora_layer)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[Any] = None,
        use_cache: bool = False,
    ) -> Any:
        return self._base_model.forward(input_ids, kv_caches=kv_caches, use_cache=use_cache)

    def get_num_params(self, non_embedding: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters(self) -> int:
        return self.get_num_params()

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: param
            for name, param in self._base_model.named_parameters()
            if "lora_" in name and param.requires_grad
        }

    def save_weights(self, path: Union[str, Path]) -> None:
        save_lora_checkpoint(self._base_model, path)

    def load_weights(self, path: Union[str, Path]) -> None:
        load_lora_checkpoint(self._base_model, path)


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
) -> Dict[str, int]:
    """Freeze all base model weights and attach LoRALinear to specified target projections."""
    for param in model.parameters():
        param.requires_grad = False

    replaced_count = 0
    if hasattr(model, "layers"):
        for block in model.layers:
            attn = block.attn
            for mod_name in target_modules:
                if hasattr(attn, mod_name):
                    orig_layer = getattr(attn, mod_name)
                    if isinstance(orig_layer, nn.Linear):
                        lora_layer = LoRALinear(orig_layer, r=r, alpha=alpha, dropout=dropout)
                        setattr(attn, mod_name, lora_layer)
                        replaced_count += 1

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "replaced_layers": replaced_count,
    }


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only the trainable LoRA adapter parameters."""
    return {
        name: param
        for name, param in model.named_parameters()
        if "lora_" in name and param.requires_grad
    }


def save_lora_checkpoint(
    model: nn.Module,
    output_path: Union[str, Path],
    config: Optional[Dict] = None,
) -> None:
    """Save LoRA adapter weights and configuration to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lora_state_dict": get_lora_state_dict(model),
        "config": config or {},
    }
    torch.save(payload, out)


def load_lora_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
) -> None:
    """Load LoRA adapter weights into a model with matching LoRALinear layers."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    lora_weights = payload["lora_state_dict"]
    model.load_state_dict(lora_weights, strict=False)
