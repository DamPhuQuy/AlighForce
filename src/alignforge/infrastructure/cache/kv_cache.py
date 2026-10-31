"""Tensor-backed Dynamic KV Cache implementation.

Strictly encapsulates key and value state tensors, enforcing the IKVCache domain interface.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from alignforge.domain.interfaces import IKVCache


class TensorKVCache(IKVCache):
    """Encapsulated Dynamic Tensor KV Cache.

    Maintains preallocated (or dynamically expanded) key and value state tensors
    across transformer decoding layers.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._max_seq_len = max_seq_len
        self._device = device or torch.device("cpu")
        self._dtype = dtype

        self._current_seq_len: int = 0
        self._batch_size: Optional[int] = None

        # Protected encapsulated cache storage
        self._k_cache: list[Optional[torch.Tensor]] = [None] * n_layers
        self._v_cache: list[Optional[torch.Tensor]] = [None] * n_layers

    @property
    def current_seq_len(self) -> int:
        return self._current_seq_len

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    @property
    def n_layers(self) -> int:
        return self._n_layers

    def _allocate_layer(self, layer_idx: int, batch_size: int) -> None:
        """Internal encapsulated helper allocating storage tensors for a specific layer."""
        shape = (batch_size, self._n_heads, self._max_seq_len, self._head_dim)
        self._k_cache[layer_idx] = torch.zeros(
            shape, dtype=self._dtype, device=self._device
        )
        self._v_cache[layer_idx] = torch.zeros(
            shape, dtype=self._dtype, device=self._device
        )

    def update(
        self, layer_idx: int, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update and retrieve key/value tensors for a given layer.

        Args:
            layer_idx: index of transformer block.
            k_val: new key tensor of shape (B, n_heads, seq_len, head_dim).
            v_val: new value tensor of shape (B, n_heads, seq_len, head_dim).

        Returns:
            Tuple of (full_k, full_v) up to current sequence length.
        """
        if layer_idx < 0 or layer_idx >= self._n_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range for cache with {self._n_layers} layers."
            )

        batch_size, _, seq_len, _ = k_val.shape

        if self._k_cache[layer_idx] is None:
            self._batch_size = batch_size
            self._allocate_layer(layer_idx, batch_size)

        start_pos = self._current_seq_len
        end_pos = start_pos + seq_len

        if end_pos > self._max_seq_len:
            raise ValueError(
                f"Cache capacity exceeded: requested {end_pos} tokens, max {self._max_seq_len}."
            )

        # Place new entries in pre-allocated buffers
        self._k_cache[layer_idx][:, :, start_pos:end_pos, :] = k_val
        self._v_cache[layer_idx][:, :, start_pos:end_pos, :] = v_val

        # Increment seq_len pointer on final layer
        if layer_idx == self._n_layers - 1:
            self._current_seq_len = end_pos

        full_k = self._k_cache[layer_idx][:, :, :end_pos, :]
        full_v = self._v_cache[layer_idx][:, :, :end_pos, :]
        return full_k, full_v

    def reset(self) -> None:
        """Reset the sequence position pointer back to zero."""
        self._current_seq_len = 0


class DynamicKVCache(IKVCache):
    """Dynamic per-layer Key and Value cache maintaining variable tensor sizes."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.k_cache: list[Optional[torch.Tensor]] = [None] * num_layers
        self.v_cache: list[Optional[torch.Tensor]] = [None] * num_layers

    @property
    def current_seq_len(self) -> int:
        return self.get_seq_len(0)

    @property
    def max_seq_len(self) -> int:
        return 2048

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """Return the current sequence length stored in the cache."""
        if self.k_cache[layer_idx] is not None:
            return self.k_cache[layer_idx].shape[2]
        return 0

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new key and value states to the specified layer's cache."""
        past_k = self.k_cache[layer_idx]
        past_v = self.v_cache[layer_idx]

        if past_k is None or past_v is None:
            self.k_cache[layer_idx] = k_new
            self.v_cache[layer_idx] = v_new
        else:
            self.k_cache[layer_idx] = torch.cat([past_k, k_new], dim=2)
            self.v_cache[layer_idx] = torch.cat([past_v, v_new], dim=2)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self) -> None:
        """Reset all cached states."""
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers

