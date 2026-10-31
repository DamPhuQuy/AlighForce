"""Generation Engine Infrastructure implementation.

Implements the ITextGenerator domain port, supporting both cached (dynamic KV Cache)
and uncached autoregressive decoding with strict latency metrics.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from alignforge.domain.entities import GenerationRequest, GenerationResult
from alignforge.domain.interfaces import ICausalLM, ITextGenerator, ITokenizer


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample next token from logits using greedy, temperature, or top-p nucleus sampling."""
    if temperature <= 0.0 or temperature < 1e-4:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / temperature

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        scaled_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class AutoregressiveGenerator(ITextGenerator):
    """Encapsulated generator adhering to ITextGenerator port."""

    def __init__(self, model: ICausalLM, tokenizer: ITokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute text generation from structured request."""
        device = next(self._model.parameters()).device if hasattr(self._model, "parameters") else torch.device("cpu")
        prompt_ids = self._tokenizer.encode(request.prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        out_ids, metrics = generate(
            model=self._model,
            input_ids=input_ids,
            max_new_tokens=request.max_new_tokens,
            use_cache=request.use_cache,
            temperature=request.temperature,
            eos_token_id=self._tokenizer.eos_id,
        )

        gen_tokens = out_ids[0, len(prompt_ids):].tolist()
        gen_text = self._tokenizer.decode(gen_tokens)

        return GenerationResult(
            generated_text=gen_text,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(gen_tokens),
            latency_ms=metrics["total_latency_seconds"] * 1000.0,
            cached=request.use_cache,
        )


def generate(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    use_cache: bool = True,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = 3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Autoregressive generation function supporting backward compatibility."""
    model.eval()
    B, prompt_len = input_ids.shape
    output_ids = input_ids.clone()
    caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    t_start = time.perf_counter()
    ttft: Optional[float] = None
    decode_times: List[float] = []

    with torch.no_grad():
        if use_cache:
            t_prefill_start = time.perf_counter()
            logits, caches = model(input_ids, use_cache=True)
            ttft = time.perf_counter() - t_prefill_start

            next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_p=top_p)
            output_ids = torch.cat([output_ids, next_token], dim=1)

            for _ in range(max_new_tokens - 1):
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

                t_step_start = time.perf_counter()
                logits, caches = model(next_token, kv_caches=caches, use_cache=True)
                step_duration = time.perf_counter() - t_step_start
                decode_times.append(step_duration)

                next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_p=top_p)
                output_ids = torch.cat([output_ids, next_token], dim=1)

        else:
            for step_idx in range(max_new_tokens):
                t_step_start = time.perf_counter()
                logits = model(output_ids, use_cache=False)
                if isinstance(logits, tuple):
                    logits = logits[0]
                step_duration = time.perf_counter() - t_step_start

                if step_idx == 0:
                    ttft = step_duration
                else:
                    decode_times.append(step_duration)

                next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_p=top_p)
                output_ids = torch.cat([output_ids, next_token], dim=1)

                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

    t_total = time.perf_counter() - t_start
    num_generated = output_ids.shape[1] - prompt_len
    tokens_per_sec = num_generated / t_total if t_total > 0 else 0.0
    tpot = sum(decode_times) / len(decode_times) if decode_times else 0.0

    metrics = {
        "num_generated_tokens": float(num_generated),
        "total_latency_seconds": float(t_total),
        "ttft_seconds": float(ttft or 0.0),
        "tpot_seconds": float(tpot),
        "tokens_per_second": float(tokens_per_sec),
        "use_cache": float(1.0 if use_cache else 0.0),
    }

    return output_ids, metrics
