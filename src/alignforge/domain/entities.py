"""Domain Entities and Value Objects for AlignForge.

Upholds OOP Encapsulation via immutable value objects and strictly validated models.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelArchitectureConfig:
    """Immutable architectural configuration value object for causal Transformer."""
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    vocab_size: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.1
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class GenerationRequest:
    """Immutable value object encapsulating text generation parameters."""
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_k: int = 40
    use_cache: bool = True
    stop_tokens: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature cannot be negative, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k cannot be negative, got {self.top_k}")


@dataclass(frozen=True)
class GenerationResult:
    """Immutable value object holding the output of a generation request."""
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float
    cached: bool
    risk_score: float = 0.0
    safe: bool = True
    flagged_reason: Optional[str] = None


@dataclass(frozen=True)
class SafetyEvaluationResult:
    """Immutable value object representing the outcome of a safety check."""
    is_safe: bool
    flagged_rule: Optional[str] = None
    risk_score: float = 0.0
    message: str = "Input and output verified safe."


@dataclass(frozen=True)
class TextCorrectionPair:
    """Domain entity representing a Grammar Error Correction (GEC) pair."""
    source_text: str
    target_text: str

    def __post_init__(self) -> None:
        if not self.source_text:
            raise ValueError("source_text cannot be empty.")


@dataclass(frozen=True)
class PreferencePair:
    """Domain entity representing a DPO preference alignment pair."""
    prompt: str
    chosen: str
    rejected: str

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt cannot be empty.")
        if not self.chosen:
            raise ValueError("chosen text cannot be empty.")
        if not self.rejected:
            raise ValueError("rejected text cannot be empty.")


@dataclass(frozen=True)
class EvaluationReport:
    """Domain entity encapsulating formal evaluation metrics."""
    gleu_score: float
    preference_win_rate: float
    total_samples: int
    latency_p50_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenizedBatch:
    """Domain value object holding a tokenized tensor batch representation."""
    input_ids: Any
    attention_mask: Optional[Any] = None
    labels: Optional[Any] = None
