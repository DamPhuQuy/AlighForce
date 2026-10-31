"""Dynamic KV Cache Automated Profiling Infrastructure implementation.

Measures:
- Tokens per second (Throughput)
- First Token Latency (TTFT)
- Next Token Latency (TPOT)
- Memory consumption (Peak Allocated RAM / VRAM)
- Comparison: KV Cache ON vs OFF
Exports: reports/kv_cache_profiling.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Union
import torch

from alignforge.infrastructure.generation.generator import generate
from alignforge.infrastructure.models.transformer import CausalTransformerModel


def get_peak_memory_mb(device: torch.device) -> float:
    """Return peak memory usage in megabytes (CUDA VRAM or process RAM)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return usage / 1024.0
        except Exception:
            return 0.0


def run_kv_cache_profiling(
    context_lengths: List[int] = [128, 512, 1024, 2048],
    new_tokens: int = 16,
    output_csv: Union[str, Path] = "reports/kv_cache_profiling.csv",
) -> List[Dict[str, Any]]:
    """Profile KV Cache ON vs OFF across 4 context lengths and export CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting KV Cache profiling on device: {device}")

    model = CausalTransformerModel(
        vocab_size=2048,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=max(context_lengths) + new_tokens + 64,
        dropout=0.0,
    ).to(device)
    model.eval()

    records: List[Dict[str, Any]] = []

    for length in context_lengths:
        print(f"--- Profiling Context Length: {length} tokens ---")
        prompt = torch.randint(4, 2000, (1, length), device=device)

        # 1. Profile with KV Cache ON
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _, metrics_on = generate(
            model,
            prompt,
            max_new_tokens=new_tokens,
            use_cache=True,
            temperature=0.0,
            eos_token_id=None,
        )
        mem_on = get_peak_memory_mb(device)

        # 2. Profile with KV Cache OFF
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _, metrics_off = generate(
            model,
            prompt,
            max_new_tokens=new_tokens,
            use_cache=False,
            temperature=0.0,
            eos_token_id=None,
        )
        mem_off = get_peak_memory_mb(device)

        tpot_on = metrics_on["tpot_seconds"]
        tpot_off = metrics_off["tpot_seconds"]
        speedup = (tpot_off / tpot_on) if tpot_on > 0 else 1.0

        record_on = {
            "context_length": length,
            "cache_mode": "ON",
            "new_tokens": new_tokens,
            "ttft_seconds": metrics_on["ttft_seconds"],
            "tpot_seconds": metrics_on["tpot_seconds"],
            "tokens_per_second": metrics_on["tokens_per_second"],
            "peak_memory_mb": mem_on,
            "speedup_factor": speedup,
        }
        record_off = {
            "context_length": length,
            "cache_mode": "OFF",
            "new_tokens": new_tokens,
            "ttft_seconds": metrics_off["ttft_seconds"],
            "tpot_seconds": metrics_off["tpot_seconds"],
            "tokens_per_second": metrics_off["tokens_per_second"],
            "peak_memory_mb": mem_off,
            "speedup_factor": 1.0,
        }

        records.extend([record_on, record_off])
        print(
            f"L={length:4d} | Cache ON: {metrics_on['tokens_per_second']:.1f} tok/s (TPOT={tpot_on*1000:.1f}ms) | "
            f"Cache OFF: {metrics_off['tokens_per_second']:.1f} tok/s (TPOT={tpot_off*1000:.1f}ms) | "
            f"Speedup: {speedup:.2f}x"
        )

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"Exported profiling results to {csv_path}")
    return records


if __name__ == "__main__":
    run_kv_cache_profiling()
