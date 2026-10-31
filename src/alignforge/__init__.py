"""AlignForge English: Domain-Driven Clean Architecture LM Package."""

__version__ = "0.1.0"

from alignforge.domain import (
    DPOLossService,
    EvaluationReport,
    GLEUScoringService,
    GenerationRequest,
    GenerationResult,
    ICausalLM,
    IDatasetRepository,
    IEvaluator,
    IKVCache,
    ISafetyGuard,
    ISafetyRule,
    ITextGenerator,
    ITokenizer,
    ITrainer,
    ModelArchitectureConfig,
    PreferencePair,
    RepetitionAnalysisService,
    SafetyEvaluationResult,
    TextCorrectionPair,
    TokenizedBatch,
)
from alignforge.application import (
    CorrectGrammarUseCase,
    DpoModelUseCase,
    EvaluateModelUseCase,
    ParaphrasePreferenceUseCase,
    PretrainModelUseCase,
    ProfileKVCacheUseCase,
    SftModelUseCase,
)
from alignforge.infrastructure.models import (
    AlignForgeModel,
    CausalTransformerModel,
    LoRALinear,
    LoRAModelAdapter,
)
from alignforge.infrastructure.tokenization import ByteLevelBPETokenizer
from alignforge.infrastructure.cache import TensorKVCache
from alignforge.infrastructure.generation import AutoregressiveGenerator
from alignforge.infrastructure.safety import CompositeSafetyGuard

__all__ = [
    # Domain
    "ModelArchitectureConfig",
    "GenerationRequest",
    "GenerationResult",
    "SafetyEvaluationResult",
    "TextCorrectionPair",
    "PreferencePair",
    "EvaluationReport",
    "TokenizedBatch",
    "ITokenizer",
    "IKVCache",
    "ICausalLM",
    "ITextGenerator",
    "ISafetyRule",
    "ISafetyGuard",
    "ITrainer",
    "IEvaluator",
    "IDatasetRepository",
    "DPOLossService",
    "GLEUScoringService",
    "RepetitionAnalysisService",
    # Application
    "CorrectGrammarUseCase",
    "ParaphrasePreferenceUseCase",
    "PretrainModelUseCase",
    "SftModelUseCase",
    "DpoModelUseCase",
    "EvaluateModelUseCase",
    "ProfileKVCacheUseCase",
    # Infrastructure
    "CausalTransformerModel",
    "AlignForgeModel",
    "LoRALinear",
    "LoRAModelAdapter",
    "ByteLevelBPETokenizer",
    "TensorKVCache",
    "AutoregressiveGenerator",
    "CompositeSafetyGuard",
]
