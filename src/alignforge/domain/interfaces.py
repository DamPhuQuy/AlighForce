"""Domain Interface Contracts (Ports) for AlignForge.

Enforces OOP Abstraction, Interface Segregation Principle (ISP),
and Dependency Inversion Principle (DIP).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from alignforge.domain.entities import (
    EvaluationReport,
    GenerationRequest,
    GenerationResult,
    ModelArchitectureConfig,
    PreferencePair,
    SafetyEvaluationResult,
    TextCorrectionPair,
)


class ITokenizer(ABC):
    """Abstract contract for tokenization encoding and decoding."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode textual string into token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into string."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return total vocabulary size."""
        pass

    @property
    @abstractmethod
    def pad_id(self) -> int:
        """Padding token ID."""
        pass

    @property
    @abstractmethod
    def eos_id(self) -> int:
        """End of sequence token ID."""
        pass

    @property
    @abstractmethod
    def bos_id(self) -> int:
        """Beginning of sequence token ID."""
        pass


class IKVCache(ABC):
    """Abstract contract for key-value tensor caching during autoregressive decoding."""

    @abstractmethod
    def update(self, layer_idx: int, k_val: Any, v_val: Any) -> Tuple[Any, Any]:
        """Append and return full past key and value representations for the layer."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset sequence length pointer to zero."""
        pass

    @property
    @abstractmethod
    def current_seq_len(self) -> int:
        """Current sequence length cached."""
        pass

    @property
    @abstractmethod
    def max_seq_len(self) -> int:
        """Maximum sequence capacity."""
        pass


class ICausalLM(ABC):
    """Abstract contract for causal language model architectures."""

    @property
    @abstractmethod
    def config(self) -> ModelArchitectureConfig:
        """Model architectural configuration."""
        pass

    @abstractmethod
    def forward(self, input_ids: Any, kv_cache: Optional[IKVCache] = None) -> Any:
        """Compute next-token logits given input IDs and optional KV cache."""
        pass

    @abstractmethod
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return count of model parameters."""
        pass

    @abstractmethod
    def save_weights(self, path: Path) -> None:
        """Save model weights to disk."""
        pass

    @abstractmethod
    def load_weights(self, path: Path) -> None:
        """Load model weights from disk."""
        pass


class ITextGenerator(ABC):
    """Abstract contract for autoregressive token generation."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute text generation according to request parameters."""
        pass


class ISafetyRule(ABC):
    """Abstract contract for an individual safety inspection rule."""

    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Name of the safety rule."""
        pass

    @abstractmethod
    def evaluate(self, text: str, context: Optional[str] = None) -> SafetyEvaluationResult:
        """Evaluate input or output text against the specific rule."""
        pass


class ISafetyGuard(ABC):
    """Abstract contract for comprehensive input and output guardrails."""

    @abstractmethod
    def inspect_input(self, text: str) -> SafetyEvaluationResult:
        """Inspect user prompt before inference."""
        pass

    @abstractmethod
    def inspect_output(self, prompt: str, generated_text: str) -> SafetyEvaluationResult:
        """Inspect generated text before presentation."""
        pass


class ITrainer(ABC):
    """Abstract contract for model training procedures."""

    @abstractmethod
    def train(self) -> Dict[str, float]:
        """Execute training loop and return final loss metrics."""
        pass

    @abstractmethod
    def train_step(self, batch: Any) -> float:
        """Execute single gradient step."""
        pass


class IEvaluator(ABC):
    """Abstract contract for evaluation benchmark executors."""

    @abstractmethod
    def evaluate(self) -> EvaluationReport:
        """Execute evaluation and return formal report."""
        pass


class IDatasetRepository(ABC):
    """Abstract contract for accessing training and evaluation data."""

    @abstractmethod
    def get_pretrain_data(self, limit: Optional[int] = None) -> List[str]:
        """Retrieve pretraining text samples."""
        pass

    @abstractmethod
    def get_sft_data(self, limit: Optional[int] = None) -> List[TextCorrectionPair]:
        """Retrieve supervised fine-tuning pairs."""
        pass

    @abstractmethod
    def get_dpo_data(self, limit: Optional[int] = None) -> List[PreferencePair]:
        """Retrieve preference alignment pairs."""
        pass
