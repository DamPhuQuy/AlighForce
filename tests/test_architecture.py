"""Unit Test Suite verifying Domain-Driven Clean Architecture, 4 OOP Pillars, and SOLID Principles."""

import dataclasses
import pytest
import torch

from alignforge.application.use_cases import (
    CorrectGrammarUseCase,
    ParaphrasePreferenceUseCase,
)
from alignforge.domain.entities import (
    GenerationRequest,
    GenerationResult,
    ModelArchitectureConfig,
    PreferencePair,
    SafetyEvaluationResult,
    TextCorrectionPair,
)
from alignforge.domain.interfaces import (
    ICausalLM,
    IDatasetRepository,
    IEvaluator,
    IKVCache,
    ISafetyGuard,
    ISafetyRule,
    ITextGenerator,
    ITokenizer,
    ITrainer,
)
from alignforge.domain.services import (
    DPOLossService,
    GLEUScoringService,
    RepetitionAnalysisService,
)
from alignforge.infrastructure.cache.kv_cache import TensorKVCache
from alignforge.infrastructure.generation.generator import AutoregressiveGenerator
from alignforge.infrastructure.models.layers import RMSNorm
from alignforge.infrastructure.models.lora import LoRALinear, LoRAModelAdapter
from alignforge.infrastructure.models.transformer import CausalTransformerModel
from alignforge.infrastructure.safety.guardrails import (
    BaseSafetyRule,
    CompositeSafetyGuard,
    InputLengthRule,
    LengthAnomalyRule,
    PromptInjectionRule,
    RepetitionRule,
)
from alignforge.infrastructure.tokenization.bpe_tokenizer import ByteLevelBPETokenizer
from alignforge.infrastructure.training.base_trainer import BaseTrainer


# ==============================================================================
# 1. DOMAIN LAYER & VALUE OBJECT IMMUTABILITY TESTS
# ==============================================================================

def test_domain_value_objects_immutability():
    """Verify that domain value objects are frozen and cannot be mutated externally (Encapsulation)."""
    cfg = ModelArchitectureConfig(d_model=256, n_heads=4, n_layers=4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.d_model = 512

    req = GenerationRequest(prompt="Hello", max_new_tokens=16)
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.temperature = 1.0

    pair = TextCorrectionPair(source_text="He go", target_text="He goes")
    with pytest.raises(dataclasses.FrozenInstanceError):
        pair.source_text = "He went"


def test_domain_config_validation():
    """Verify business invariants are guarded inside domain entities."""
    with pytest.raises(ValueError, match="divisible by n_heads"):
        ModelArchitectureConfig(d_model=255, n_heads=4)

    with pytest.raises(ValueError, match="prompt cannot be empty"):
        PreferencePair(prompt="", chosen="Yes", rejected="No")


def test_domain_services():
    """Verify domain services compute pure business algorithms independent of frameworks."""
    dpo_svc = DPOLossService(beta=0.1)
    # When policy == ref, chosen_ratio = 0, rejected_ratio = 0 -> loss = log(2) ~= 0.6931
    loss, win_p = dpo_svc.compute_log_ratio(-2.0, -2.0, -4.0, -4.0)
    assert abs(loss - 0.6931) < 1e-3
    assert abs(win_p - 0.5) < 1e-3

    gleu_svc = GLEUScoringService()
    score = gleu_svc.score_sentence("She goes to school.", ["She goes to school."])
    assert abs(score - 1.0) < 1e-4

    rep_svc = RepetitionAnalysisService(max_allowed_ngram_repeat=3, ngram_size=2)
    is_rep, count = rep_svc.check_repetition(["go", "home", "go", "home", "go", "home"])
    assert is_rep is True
    assert count >= 3


# ==============================================================================
# 2. OOP 4 PILLARS TESTS
# ==============================================================================

def test_oop_abstraction_cannot_instantiate_interfaces():
    """Verify that domain ABC interfaces cannot be instantiated directly (Abstraction)."""
    with pytest.raises(TypeError):
        ITokenizer()

    with pytest.raises(TypeError):
        ICausalLM()

    with pytest.raises(TypeError):
        IKVCache()

    with pytest.raises(TypeError):
        ISafetyRule()

    with pytest.raises(TypeError):
        ITrainer()

    with pytest.raises(TypeError):
        IEvaluator()


def test_oop_encapsulation_internal_state():
    """Verify state protection and property encapsulation."""
    cache = TensorKVCache(n_layers=2, n_heads=2, head_dim=16, max_seq_len=64)
    assert cache.current_seq_len == 0
    assert cache.max_seq_len == 64

    # Internal buffers are encapsulated with leading underscores
    assert hasattr(cache, "_k_cache")
    assert hasattr(cache, "_v_cache")

    tok = ByteLevelBPETokenizer()
    assert tok.vocab_size >= 260
    assert hasattr(tok, "_vocab")
    assert hasattr(tok, "_merges")


def test_oop_inheritance():
    """Verify class hierarchies and base class contract adherence (Inheritance)."""
    # Safety rule hierarchy
    rule1 = PromptInjectionRule()
    rule2 = RepetitionRule()
    assert isinstance(rule1, BaseSafetyRule)
    assert isinstance(rule1, ISafetyRule)
    assert isinstance(rule2, BaseSafetyRule)
    assert isinstance(rule2, ISafetyRule)

    # Model hierarchy
    model = CausalTransformerModel(vocab_size=128, d_model=64, n_heads=2, n_layers=2)
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, ICausalLM)

    # Tokenizer hierarchy
    tokenizer = ByteLevelBPETokenizer()
    assert isinstance(tokenizer, ITokenizer)


def test_oop_polymorphism():
    """Verify polymorphic behavior across interchangeable implementations."""
    rules: list[ISafetyRule] = [
        PromptInjectionRule(),
        RepetitionRule(),
        InputLengthRule(),
    ]
    # Polymorphic evaluation call
    for rule in rules:
        res = rule.evaluate("Normal innocent sentence.")
        assert isinstance(res, SafetyEvaluationResult)
        assert res.is_safe is True


# ==============================================================================
# 3. SOLID PRINCIPLES TESTS
# ==============================================================================

def test_solid_srp_single_responsibility():
    """Verify each class has a bounded, single responsibility."""
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 4, 64)
    out = norm(x)
    assert out.shape == x.shape

    # Tokenizer only tokenizes
    tok = ByteLevelBPETokenizer()
    tokens = tok.encode("Test string")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)


def test_solid_ocp_open_closed_principle():
    """Verify system is open for extension without modifying composite safety guard."""
    class CustomProfanityRule(BaseSafetyRule):
        def __init__(self, blocked_word: str = "badword"):
            super().__init__("CustomProfanityRule")
            self.blocked_word = blocked_word

        def evaluate(self, text: str, context=None) -> SafetyEvaluationResult:
            if self.blocked_word in text.lower():
                return SafetyEvaluationResult(
                    is_safe=False,
                    flagged_rule=self.rule_name,
                    risk_score=1.0,
                    message="Profanity detected",
                )
            return SafetyEvaluationResult(is_safe=True)

    guard = CompositeSafetyGuard()
    custom_rule = CustomProfanityRule("forbidden")
    guard.register_input_rule(custom_rule)

    # Original rules work
    res_clean = guard.inspect_input("Hello there.")
    assert res_clean.is_safe is True

    # Newly registered rule intercepts without modifying CompositeSafetyGuard source code
    res_dirty = guard.inspect_input("This has forbidden content.")
    assert res_dirty.is_safe is False
    assert res_dirty.flagged_rule == "CustomProfanityRule"


def test_solid_lsp_liskov_substitution():
    """Verify LoRAModelAdapter can substitute ICausalLM transparently."""
    base_model = CausalTransformerModel(vocab_size=128, d_model=64, n_heads=2, n_layers=2)
    lora_adapter = LoRAModelAdapter(base_model, r=4, alpha=8.0)

    # Both must honor ICausalLM contract
    def execute_forward_contract(model: ICausalLM, ids: torch.Tensor):
        assert isinstance(model.config, ModelArchitectureConfig)
        logits, _ = model.forward(ids)
        assert logits.shape == (ids.shape[0], ids.shape[1], 128)
        assert model.get_num_params() > 0

    input_ids = torch.randint(0, 128, (2, 8))
    execute_forward_contract(base_model, input_ids)
    execute_forward_contract(lora_adapter, input_ids)


def test_solid_dip_dependency_inversion_in_use_cases():
    """Verify application use cases depend exclusively on abstractions injected via IoC."""
    # Create mock implementations of domain interfaces
    class MockTokenizer(ITokenizer):
        def encode(self, text: str):
            return [1, 2, 3]

        def decode(self, token_ids):
            return "Mock decoded text"

        @property
        def vocab_size(self):
            return 100

        @property
        def pad_id(self):
            return 0

        @property
        def eos_id(self):
            return 3

        @property
        def bos_id(self):
            return 2

    class MockModel(ICausalLM):
        @property
        def config(self):
            return ModelArchitectureConfig(d_model=64, n_heads=2, n_layers=2, vocab_size=100)

        def forward(self, input_ids, kv_cache=None):
            return torch.zeros(input_ids.shape[0], input_ids.shape[1], 100), None

        def get_num_params(self, non_embedding=False):
            return 1000

        def save_weights(self, path):
            pass

        def load_weights(self, path):
            pass

    class MockGenerator(ITextGenerator):
        def generate(self, request: GenerationRequest) -> GenerationResult:
            return GenerationResult(
                generated_text="Correction: She goes to school.",
                prompt_tokens=5,
                generated_tokens=6,
                latency_ms=10.0,
                cached=request.use_cache,
            )

    class MockSafetyGuard(ISafetyGuard):
        def inspect_input(self, text: str) -> SafetyEvaluationResult:
            if "danger" in text:
                return SafetyEvaluationResult(is_safe=False, message="Danger detected")
            return SafetyEvaluationResult(is_safe=True)

        def inspect_output(self, prompt: str, generated_text: str) -> SafetyEvaluationResult:
            return SafetyEvaluationResult(is_safe=True)

    # Inject domain mock interfaces into Application Use Case
    use_case = CorrectGrammarUseCase(
        model=MockModel(),
        tokenizer=MockTokenizer(),
        generator=MockGenerator(),
        safety_guard=MockSafetyGuard(),
    )

    # Normal execution
    res = use_case.execute("She go to school.")
    assert res.safe is True
    assert res.generated_text == "She goes to school."

    # Guardrail rejection
    res_danger = use_case.execute("danger input")
    assert res_danger.safe is False
    assert "Danger detected" in res_danger.flagged_reason
