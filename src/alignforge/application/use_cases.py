"""Application Layer Use Cases for AlignForge.

Adheres strictly to Clean Architecture and SOLID:
- Single Responsibility Principle (SRP): Each use case represents one observable business action.
- Dependency Inversion Principle (DIP): Depends strictly on domain abstract contracts (ITokenizer, ICausalLM, ISafetyGuard, ITextGenerator, ITrainer, IEvaluator).
- Inversion of Control (IoC): Dependencies are injected via constructor.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import torch

from alignforge.domain.entities import (
    EvaluationReport,
    GenerationRequest,
    GenerationResult,
    SafetyEvaluationResult,
)
from alignforge.domain.interfaces import (
    ICausalLM,
    IEvaluator,
    ISafetyGuard,
    ITextGenerator,
    ITokenizer,
    ITrainer,
)


class CorrectGrammarUseCase:
    """Orchestrates Grammar Error Correction (GEC) with input & output guardrails."""

    def __init__(
        self,
        model: ICausalLM,
        tokenizer: ITokenizer,
        generator: ITextGenerator,
        safety_guard: ISafetyGuard,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._generator = generator
        self._safety_guard = safety_guard

    def execute(
        self,
        text: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> GenerationResult:
        """Execute full grammatical error correction workflow with safety checks."""
        # 1. Input Guardrail Inspection
        input_eval = self._safety_guard.inspect_input(text)
        if not input_eval.is_safe:
            return GenerationResult(
                generated_text="",
                prompt_tokens=0,
                generated_tokens=0,
                latency_ms=0.0,
                cached=use_cache,
                risk_score=input_eval.risk_score,
                safe=False,
                flagged_reason=f"Input rejected by {input_eval.flagged_rule}: {input_eval.message}",
            )

        # 2. Format Canonical Task Prompt
        prompt = f"Fix the grammar in this sentence: {text.strip()}\nCorrection: "

        # 3. Autoregressive Generation
        req = GenerationRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=use_cache,
        )
        gen_result = self._generator.generate(req)

        # Parse candidate correction
        raw_text = gen_result.generated_text
        if "Correction: " in raw_text:
            candidate = raw_text.split("Correction: ", 1)[-1].strip()
        else:
            candidate = raw_text.strip()

        # 4. Output Guardrail Inspection
        output_eval = self._safety_guard.inspect_output(text, candidate)
        is_safe = output_eval.is_safe and (output_eval.flagged_rule is None)
        reason = output_eval.message if not is_safe else None

        return GenerationResult(
            generated_text=candidate if is_safe else text.strip(),
            prompt_tokens=gen_result.prompt_tokens,
            generated_tokens=gen_result.generated_tokens,
            latency_ms=gen_result.latency_ms,
            cached=gen_result.cached,
            risk_score=output_eval.risk_score,
            safe=is_safe,
            flagged_reason=reason,
        )


class ParaphrasePreferenceUseCase:
    """Orchestrates natural paraphrase generation with alignment validation."""

    def __init__(
        self,
        model: ICausalLM,
        tokenizer: ITokenizer,
        generator: ITextGenerator,
        safety_guard: ISafetyGuard,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._generator = generator
        self._safety_guard = safety_guard

    def execute(
        self,
        text: str,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> GenerationResult:
        input_eval = self._safety_guard.inspect_input(text)
        if not input_eval.is_safe:
            return GenerationResult(
                generated_text="",
                prompt_tokens=0,
                generated_tokens=0,
                latency_ms=0.0,
                cached=use_cache,
                safe=False,
                flagged_reason=input_eval.message,
            )

        prompt = f"Paraphrase this sentence to make it more natural: {text.strip()}\nParaphrase: "
        req = GenerationRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=use_cache,
        )
        gen_result = self._generator.generate(req)
        raw_text = gen_result.generated_text
        candidate = raw_text.split("Paraphrase: ", 1)[-1].strip() if "Paraphrase: " in raw_text else raw_text.strip()

        output_eval = self._safety_guard.inspect_output(text, candidate)
        return GenerationResult(
            generated_text=candidate,
            prompt_tokens=gen_result.prompt_tokens,
            generated_tokens=gen_result.generated_tokens,
            latency_ms=gen_result.latency_ms,
            cached=gen_result.cached,
            safe=output_eval.is_safe,
            flagged_reason=output_eval.message if not output_eval.is_safe else None,
        )


class PretrainModelUseCase:
    """Executes base language model pretraining through an injected trainer."""

    def __init__(self, trainer: ITrainer) -> None:
        self._trainer = trainer

    def execute(self, max_steps: int = 50) -> Dict[str, float]:
        return self._trainer.train(max_steps=max_steps) if hasattr(self._trainer, "train") else self._trainer.train()


class SftModelUseCase:
    """Executes Supervised Fine-Tuning through an injected trainer."""

    def __init__(self, trainer: ITrainer) -> None:
        self._trainer = trainer

    def execute(self, max_steps: int = 40) -> Dict[str, float]:
        return self._trainer.train(max_steps=max_steps) if hasattr(self._trainer, "train") else self._trainer.train()


class DpoModelUseCase:
    """Executes Direct Preference Optimization through an injected trainer."""

    def __init__(self, trainer: ITrainer) -> None:
        self._trainer = trainer

    def execute(self, max_steps: int = 40) -> Dict[str, float]:
        return self._trainer.train(max_steps=max_steps) if hasattr(self._trainer, "train") else self._trainer.train()


class EvaluateModelUseCase:
    """Executes benchmark evaluations across registered evaluators."""

    def __init__(self, evaluators: List[IEvaluator]) -> None:
        self._evaluators = evaluators

    def execute(self) -> Dict[str, EvaluationReport]:
        reports: Dict[str, EvaluationReport] = {}
        for evaluator in self._evaluators:
            name = getattr(evaluator, "name", evaluator.__class__.__name__)
            reports[name] = evaluator.evaluate()
        return reports


class ProfileKVCacheUseCase:
    """Profiles latency and throughput scaling across context lengths with and without KV Cache."""

    def __init__(self, model: Any, tokenizer: ITokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def execute(
        self,
        context_lengths: Tuple[int, ...] = (128, 512, 1024, 2048),
        decode_steps: int = 16,
    ) -> List[Dict[str, Any]]:
        from alignforge.infrastructure.generation.generator import generate

        device = next(self._model.parameters()).device if hasattr(self._model, "parameters") else torch.device("cpu")
        results = []

        for L in context_lengths:
            dummy_tokens = [10] * L
            input_ids = torch.tensor([dummy_tokens], dtype=torch.long, device=device)

            # Profile cached
            _, cached_metrics = generate(
                self._model, input_ids, max_new_tokens=decode_steps, use_cache=True
            )
            # Profile uncached
            _, uncached_metrics = generate(
                self._model, input_ids, max_new_tokens=decode_steps, use_cache=False
            )

            speedup = uncached_metrics["total_latency_seconds"] / max(
                cached_metrics["total_latency_seconds"], 1e-6
            )

            results.append({
                "context_length": L,
                "cached_latency_ms": cached_metrics["total_latency_seconds"] * 1000.0,
                "uncached_latency_ms": uncached_metrics["total_latency_seconds"] * 1000.0,
                "speedup_factor": speedup,
                "ttft_ms": cached_metrics["ttft_seconds"] * 1000.0,
                "tpot_ms": cached_metrics["tpot_seconds"] * 1000.0,
            })

        return results
