"""Deterministic Safety Guardrails Infrastructure implementation.

Upholds OOP Principles:
- Abstraction: ISafetyRule and ISafetyGuard domain interfaces.
- Inheritance: BaseSafetyRule subclasses for discrete checks.
- Polymorphism: Uniform evaluate() signature across all rules.
- Open/Closed Principle (OCP): Composite guard allows attaching new rules without modifying engine.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from alignforge.domain.entities import SafetyEvaluationResult
from alignforge.domain.interfaces import ISafetyGuard, ISafetyRule

# Prompt injection heuristic patterns
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(an?|in)\b", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(rules|instructions)", re.IGNORECASE),
    re.compile(r"bypass\s+(safety|filter)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(an?|a)\s+unfiltered", re.IGNORECASE),
]


class GuardrailResult:
    """Standardized guardrail outcome container for backward compatibility."""

    def __init__(
        self,
        is_valid: bool,
        status: str,
        sanitized_text: str,
        reason: Optional[str] = None,
    ) -> None:
        self.is_valid = is_valid
        self.status = status
        self.sanitized_text = sanitized_text
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "sanitized_text": self.sanitized_text,
            "reason": self.reason,
        }


class BaseSafetyRule(ISafetyRule):
    """Abstract base class for all discrete safety evaluation rules."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def rule_name(self) -> str:
        return self._name


class PromptInjectionRule(BaseSafetyRule):
    """Evaluates text against known adversarial prompt injection patterns."""

    def __init__(self) -> None:
        super().__init__("PromptInjectionRule")

    def evaluate(self, text: str, context: Optional[str] = None) -> SafetyEvaluationResult:
        for pattern in INJECTION_PATTERNS:
            if pattern.search(text):
                return SafetyEvaluationResult(
                    is_safe=False,
                    flagged_rule=self.rule_name,
                    risk_score=1.0,
                    message="Potential prompt injection instruction detected",
                )
        return SafetyEvaluationResult(is_safe=True)


class InputLengthRule(BaseSafetyRule):
    """Evaluates input length and control characters."""

    def __init__(self, max_chars: int = 500, max_words: int = 64) -> None:
        super().__init__("InputLengthRule")
        self._max_chars = max_chars
        self._max_words = max_words

    def evaluate(self, text: str, context: Optional[str] = None) -> SafetyEvaluationResult:
        cleaned = text.strip()
        if not cleaned:
            return SafetyEvaluationResult(
                is_safe=False,
                flagged_rule=self.rule_name,
                risk_score=0.9,
                message="Input text is empty",
            )
        if len(cleaned) > self._max_chars or len(cleaned.split()) > self._max_words:
            return SafetyEvaluationResult(
                is_safe=False,
                flagged_rule=self.rule_name,
                risk_score=0.8,
                message=f"Input exceeds maximum word limit ({self._max_words} words)",
            )
        return SafetyEvaluationResult(is_safe=True)


class RepetitionRule(BaseSafetyRule):
    """Evaluates output for degenerate autoregressive repetitive loops."""

    def __init__(self, ngram_size: int = 3, threshold: int = 3) -> None:
        super().__init__("RepetitionRule")
        self._ngram_size = ngram_size
        self._threshold = threshold

    def evaluate(self, text: str, context: Optional[str] = None) -> SafetyEvaluationResult:
        words = text.split()
        if len(words) < self._ngram_size * self._threshold:
            return SafetyEvaluationResult(is_safe=True)

        ngram_counts: Dict[Tuple[str, ...], int] = {}
        for i in range(len(words) - self._ngram_size + 1):
            ngram = tuple(words[i : i + self._ngram_size])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            if ngram_counts[ngram] >= self._threshold:
                return SafetyEvaluationResult(
                    is_safe=False,
                    flagged_rule=self.rule_name,
                    risk_score=0.9,
                    message="Degenerate repetitive text loop detected",
                )
        return SafetyEvaluationResult(is_safe=True)


class LengthAnomalyRule(BaseSafetyRule):
    """Heuristic rule flagging extreme length divergence between prompt and output."""

    def __init__(self, min_ratio: float = 0.25, max_ratio: float = 3.5) -> None:
        super().__init__("LengthAnomalyRule")
        self._min_ratio = min_ratio
        self._max_ratio = max_ratio

    def evaluate(self, text: str, context: Optional[str] = None) -> SafetyEvaluationResult:
        if not context:
            return SafetyEvaluationResult(is_safe=True)

        orig_words = len(context.split())
        cand_words = len(text.split())

        if orig_words > 0 and (
            cand_words < self._min_ratio * orig_words or cand_words > self._max_ratio * orig_words
        ):
            return SafetyEvaluationResult(
                is_safe=True,  # Still valid, but flagged for human review
                flagged_rule=self.rule_name,
                risk_score=0.5,
                message="Candidate length deviates significantly from source sentence",
            )
        return SafetyEvaluationResult(is_safe=True)


class CompositeSafetyGuard(ISafetyGuard):
    """Composite safety guard applying injected lists of rules (OCP & DIP)."""

    def __init__(
        self,
        input_rules: Optional[List[ISafetyRule]] = None,
        output_rules: Optional[List[ISafetyRule]] = None,
    ) -> None:
        self._input_rules = input_rules if input_rules is not None else [
            InputLengthRule(),
            PromptInjectionRule(),
        ]
        self._output_rules = output_rules if output_rules is not None else [
            RepetitionRule(),
            LengthAnomalyRule(),
        ]

    def register_input_rule(self, rule: ISafetyRule) -> None:
        self._input_rules.append(rule)

    def register_output_rule(self, rule: ISafetyRule) -> None:
        self._output_rules.append(rule)

    def inspect_input(self, text: str) -> SafetyEvaluationResult:
        for rule in self._input_rules:
            res = rule.evaluate(text)
            if not res.is_safe:
                return res
        return SafetyEvaluationResult(is_safe=True)

    def inspect_output(self, prompt: str, generated_text: str) -> SafetyEvaluationResult:
        if not generated_text.strip():
            return SafetyEvaluationResult(
                is_safe=False,
                flagged_rule="EmptyOutputRule",
                risk_score=1.0,
                message="Model produced empty output",
            )
        for rule in self._output_rules:
            res = rule.evaluate(generated_text, context=prompt)
            if not res.is_safe:
                return res
            if res.flagged_rule:
                return res  # Flagged warning like length anomaly
        return SafetyEvaluationResult(is_safe=True)


# Module-level functions for backward compatibility
_DEFAULT_GUARD = CompositeSafetyGuard()


def validate_input(
    text: str,
    max_char_len: int = 500,
    max_words: int = 64,
) -> GuardrailResult:
    cleaned = text.strip()
    if not cleaned:
        return GuardrailResult(False, "input_rejected", "", "Input text is empty")

    if len(cleaned) > max_char_len or len(cleaned.split()) > max_words:
        return GuardrailResult(
            False, "input_rejected", "", f"Input exceeds maximum word limit ({max_words} words)"
        )

    sanitized = "".join(ch for ch in cleaned if ch.isprintable() or ch in ("\n", "\t"))
    for pattern in INJECTION_PATTERNS:
        if pattern.search(sanitized):
            return GuardrailResult(
                False, "input_rejected", sanitized, "Potential prompt injection instruction detected"
            )

    return GuardrailResult(True, "input_accepted", sanitized)


def detect_repetitive_loop(text: str, ngram_size: int = 3, threshold: int = 3) -> bool:
    words = text.split()
    if len(words) < ngram_size * threshold:
        return False

    ngram_counts: Dict[Tuple[str, ...], int] = {}
    for i in range(len(words) - ngram_size + 1):
        ngram = tuple(words[i : i + ngram_size])
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        if ngram_counts[ngram] >= threshold:
            return True
    return False


def validate_output(
    candidate: str,
    original_input: str,
) -> GuardrailResult:
    cleaned = candidate.strip()
    if not cleaned:
        return GuardrailResult(False, "invalid_generation", "", "Model produced empty output")

    if detect_repetitive_loop(cleaned):
        return GuardrailResult(False, "invalid_generation", cleaned, "Degenerate repetitive text loop detected")

    orig_words = len(original_input.split())
    cand_words = len(cleaned.split())

    if orig_words > 0 and (cand_words < 0.25 * orig_words or cand_words > 3.5 * orig_words):
        return GuardrailResult(
            True, "review_required", cleaned, "Candidate length deviates significantly from source sentence"
        )

    return GuardrailResult(True, "candidate_unverified", cleaned)
