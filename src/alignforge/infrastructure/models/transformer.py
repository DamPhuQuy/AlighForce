"""Causal Transformer Decoder Architecture implementation.

Implements the ICausalLM domain interface, encapsulating weights, RoPE embeddings,
and transformer layers into a cohesive domain entity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from alignforge.domain.entities import ModelArchitectureConfig
from alignforge.domain.interfaces import ICausalLM, IKVCache
from alignforge.infrastructure.models.layers import RMSNorm, RotaryEmbedding, TransformerBlock


class CausalTransformerModel(nn.Module, ICausalLM):
    """From-scratch Causal Decoder-Only Language Model.

    Adheres strictly to the ICausalLM interface while maintaining full
    backward compatibility with AlignForgeModel.
    """

    def __init__(
        self,
        config: Optional[ModelArchitectureConfig] = None,
        vocab_size: int = 2048,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()

        if config is not None:
            self._config = config
        else:
            self._config = ModelArchitectureConfig(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                dropout=dropout,
                eps=norm_eps,
            )

        self.vocab_size = self._config.vocab_size
        self.d_model = self._config.d_model
        self.n_heads = self._config.n_heads
        self.n_layers = self._config.n_layers
        self.d_ff = self._config.d_ff
        self.max_seq_len = self._config.max_seq_len

        # Token embedding
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.d_model)
        self.drop = nn.Dropout(self._config.dropout)

        # RoPE provider
        self.rope = RotaryEmbedding(
            self.d_model // self.n_heads,
            max_seq_len=self.max_seq_len,
            base=rope_base,
        )

        # Transformer decoder blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self._config.dropout,
                norm_eps=self._config.eps,
            )
            for _ in range(self.n_layers)
        ])

        # Final normalization & language model head
        self.norm = RMSNorm(self.d_model, eps=self._config.eps)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Weight tying between token embeddings and LM head
        self.tok_embeddings.weight = self.lm_head.weight

        self._init_weights()

    @property
    def config(self) -> ModelArchitectureConfig:
        return self._config

    def _init_weights(self) -> None:
        """Initialize parameters with scaled normal distribution."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(param)

    def count_parameters(self) -> int:
        """Count total trainable parameters (accounting for tied weights)."""
        seen = set()
        total = 0
        for p in self.parameters():
            if p.requires_grad and p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    def get_num_params(self, non_embedding: bool = False) -> int:
        return self.count_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[Union[list, IKVCache]] = None,
        use_cache: bool = False,
    ) -> Any:
        """Unified forward pass handling tuple list caches, domain IKVCache, or uncached."""
        # Check if kv_caches is an IKVCache instance
        if isinstance(kv_caches, IKVCache):
            return self._forward_with_ikvcache(input_ids, kv_caches)

        B, T = input_ids.shape
        x = self.drop(self.tok_embeddings(input_ids))

        past_len = (
            kv_caches[0][0].shape[2]
            if (kv_caches is not None and kv_caches[0] is not None)
            else 0
        )
        total_len = past_len + T
        cos, sin = self.rope(x, total_len)
        cos = cos[past_len:total_len]
        sin = sin[past_len:total_len]

        new_caches = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_cache = kv_caches[idx] if kv_caches is not None else None
            x, new_cache = layer(
                x, cos, sin, kv_cache=layer_cache, use_cache=use_cache
            )
            if use_cache:
                new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits, new_caches

    def _forward_with_ikvcache(
        self, input_ids: torch.Tensor, cache: IKVCache
    ) -> torch.Tensor:
        """Forward pass utilizing encapsulated IKVCache domain object."""
        B, T = input_ids.shape
        x = self.drop(self.tok_embeddings(input_ids))

        past_len = cache.current_seq_len
        total_len = past_len + T
        cos, sin = self.rope(x, total_len)
        cos = cos[past_len:total_len]
        sin = sin[past_len:total_len]

        for idx, layer in enumerate(self.layers):
            # Compute Q, K, V for this layer
            normed_x = layer.attn_norm(x)
            q = layer.attn.q_proj(normed_x).view(B, T, self.n_heads, layer.attn.head_dim).transpose(1, 2)
            k = layer.attn.k_proj(normed_x).view(B, T, self.n_heads, layer.attn.head_dim).transpose(1, 2)
            v = layer.attn.v_proj(normed_x).view(B, T, self.n_heads, layer.attn.head_dim).transpose(1, 2)

            q, k = layer.attn.apply_rope(q, k, cos, sin) if hasattr(layer.attn, 'apply_rope') else (q, k)

            full_k, full_v = cache.update(idx, k, v)
            scores = torch.matmul(q, full_k.transpose(-2, -1)) / (layer.attn.head_dim ** 0.5)

            if T > 1:
                total_k_len = full_k.shape[2]
                q_idx = torch.arange(total_k_len - T, total_k_len, device=x.device).unsqueeze(1)
                k_idx = torch.arange(total_k_len, device=x.device).unsqueeze(0)
                causal_mask = (k_idx > q_idx).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(causal_mask, float("-inf"))

            attn_weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn_weights, full_v)
            context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
            attn_out = layer.attn.out_proj(context)

            x = x + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))

        x = self.norm(x)
        return self.lm_head(x)

    def save_weights(self, path: Union[str, Path]) -> None:
        """Save model weights to path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)

    def load_weights(self, path: Union[str, Path]) -> None:
        """Load model weights from path."""
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


# Backward compatibility alias
AlignForgeModel = CausalTransformerModel
