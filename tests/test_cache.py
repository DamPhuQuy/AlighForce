import pytest
import torch
from alignforge.infrastructure.models import AlignForgeModel
from alignforge.infrastructure.cache import DynamicKVCache
from alignforge.infrastructure.generation import generate


def test_dynamic_kv_cache_structure():
    cache = DynamicKVCache(num_layers=4)
    assert cache.get_seq_len() == 0

    # Simulate layer 0 update with prefill (B=1, n_heads=4, T=5, head_dim=64)
    k1 = torch.randn(1, 4, 5, 64)
    v1 = torch.randn(1, 4, 5, 64)
    cache.update(layer_idx=0, k_new=k1, v_new=v1)
    assert cache.get_seq_len(0) == 5

    # Simulate decode step (T=1)
    k2 = torch.randn(1, 4, 1, 64)
    v2 = torch.randn(1, 4, 1, 64)
    cache.update(layer_idx=0, k_new=k2, v_new=v2)
    assert cache.get_seq_len(0) == 6

    # Test reset
    cache.reset()
    assert cache.get_seq_len(0) == 0


def test_cached_vs_uncached_generation_equivalence():
    """Verify that cached generation yields identical token predictions to uncached generation."""
    torch.manual_seed(42)
    model = AlignForgeModel(
        vocab_size=256,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.0,
    )
    model.eval()

    prompt = torch.tensor([[12, 45, 78, 92, 105, 33]])
    max_new_tokens = 12

    # 1. Generate with KV Cache ON
    out_cached, metrics_cached = generate(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=0.0,
        eos_token_id=None,
    )

    # 2. Generate with KV Cache OFF
    out_uncached, metrics_uncached = generate(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        use_cache=False,
        temperature=0.0,
        eos_token_id=None,
    )

    # Exact token equality assertion
    assert torch.equal(out_cached, out_uncached), (
        f"Mismatch between cached and uncached tokens!\nCached: {out_cached}\nUncached: {out_uncached}"
    )

    # Generated tokens count assertion
    assert out_cached.shape[1] == prompt.shape[1] + max_new_tokens
    assert metrics_cached["use_cache"] == 1.0
    assert metrics_uncached["use_cache"] == 0.0


def test_cache_across_multi_token_contexts():
    """Test that caching works cleanly across different prompt context lengths."""
    torch.manual_seed(100)
    model = AlignForgeModel(
        vocab_size=256,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
    )
    model.eval()

    for prompt_len in [8, 32, 64]:
        prompt = torch.randint(0, 256, (1, prompt_len))
        out_cached, _ = generate(model, prompt, max_new_tokens=4, use_cache=True, temperature=0.0)
        out_uncached, _ = generate(model, prompt, max_new_tokens=4, use_cache=False, temperature=0.0)
        assert torch.equal(out_cached, out_uncached)
