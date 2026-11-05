# Task TASK-CLEAN-ARCH-OOP: Domain-Driven Clean Architecture and Strict OOP/SOLID Refactor

## Intent and acceptance

User outcome and observable success:
- Reorganize `alignforge` into concentric Domain-Driven Clean Architecture layers (`domain/`, `application/`, `infrastructure/`, `presentation/`).
- Enforce the 4 OOP Pillars strictly: Encapsulation (protected/private states, property invariants, immutable value objects), Abstraction (pure ABC domain interfaces), Inheritance (clean polymorphic class hierarchies), Polymorphism (interchangeable models, trainers, evaluators, and safety rules).
- Enforce SOLID principles: Single Responsibility, Open/Closed (extensible without modifying core), Liskov Substitution, Interface Segregation, Dependency Inversion (high-level use cases depend only on domain ports; concrete implementations injected via DI).
- Maintain 100% backward compatibility: existing imports, tests, and CLI entry points continue functioning without regression.
- Observable check: `uv run pytest tests/test_clean_arch_oop.py -v && uv run pytest` passes 100% with all 30 baseline tests plus new clean architecture / OOP verification tests.

## Scope

<allowed_files>
  <file>.agents/TASK.md</file>
  <file>README.md</file>
  <file>process/features/active/clean-architecture-oop-refactor/spec.json</file>
  <file>process/features/active/clean-architecture-oop-refactor/task.json</file>
  <file>process/features/active/clean-architecture-oop-refactor/task.md</file>
  <file>src/alignforge/__init__.py</file>
  <file>src/alignforge/cli.py</file>
  <file>src/alignforge/data.py</file>
  <file>src/alignforge/evaluate.py</file>
  <file>src/alignforge/generate.py</file>
  <file>src/alignforge/kv_cache.py</file>
  <file>src/alignforge/lora.py</file>
  <file>src/alignforge/model.py</file>
  <file>src/alignforge/profile_cache.py</file>
  <file>src/alignforge/safety.py</file>
  <file>src/alignforge/tokenizer.py</file>
  <file>src/alignforge/train_dpo.py</file>
  <file>src/alignforge/train_pretrain.py</file>
  <file>src/alignforge/train_sft.py</file>
  <file>src/alignforge/domain/__init__.py</file>
  <file>src/alignforge/domain/entities.py</file>
  <file>src/alignforge/domain/interfaces.py</file>
  <file>src/alignforge/domain/services.py</file>
  <file>src/alignforge/application/__init__.py</file>
  <file>src/alignforge/application/use_cases.py</file>
  <file>src/alignforge/infrastructure/__init__.py</file>
  <file>src/alignforge/infrastructure/models/__init__.py</file>
  <file>src/alignforge/infrastructure/models/transformer.py</file>
  <file>src/alignforge/infrastructure/models/layers.py</file>
  <file>src/alignforge/infrastructure/models/lora.py</file>
  <file>src/alignforge/infrastructure/cache/__init__.py</file>
  <file>src/alignforge/infrastructure/cache/kv_cache.py</file>
  <file>src/alignforge/infrastructure/tokenization/__init__.py</file>
  <file>src/alignforge/infrastructure/tokenization/bpe_tokenizer.py</file>
  <file>src/alignforge/infrastructure/data/__init__.py</file>
  <file>src/alignforge/infrastructure/data/dataset_repository.py</file>
  <file>src/alignforge/infrastructure/training/__init__.py</file>
  <file>src/alignforge/infrastructure/training/base_trainer.py</file>
  <file>src/alignforge/infrastructure/training/pretrainer.py</file>
  <file>src/alignforge/infrastructure/training/sft_trainer.py</file>
  <file>src/alignforge/infrastructure/training/dpo_trainer.py</file>
  <file>src/alignforge/infrastructure/safety/__init__.py</file>
  <file>src/alignforge/infrastructure/safety/guardrails.py</file>
  <file>src/alignforge/infrastructure/evaluation/__init__.py</file>
  <file>src/alignforge/infrastructure/evaluation/evaluators.py</file>
  <file>src/alignforge/infrastructure/generation/__init__.py</file>
  <file>src/alignforge/infrastructure/generation/generator.py</file>
  <file>src/alignforge/presentation/__init__.py</file>
  <file>src/alignforge/presentation/cli.py</file>
  <file>tests/test_clean_arch_oop.py</file>
</allowed_files>

## Machine contracts and research artifacts

- Machine spec & task DAG: `spec.json` and `task.json`.
- Validation: `python3 .agents/harness.py validate --spec process/features/active/clean-architecture-oop-refactor/spec.json --task process/features/active/clean-architecture-oop-refactor/task.json`.

## Decisions and Architecture

1. **Clean Architecture Layering**:
   - `domain`: Pure business rules, entities (`ModelArchitectureConfig`, `GenerationRequest`, `GenerationResult`, `SafetyEvaluationResult`, `PreferencePair`), and interface contracts (`ITokenizer`, `ICausalLM`, `IKVCache`, `ISafetyGuard`, `ITrainer`, `IEvaluator`, `IDatasetRepository`).
   - `application`: Orchestration use cases (`CorrectGrammarUseCase`, `PretrainModelUseCase`, `SftModelUseCase`, `DpoModelUseCase`, `EvaluateModelUseCase`, `ProfileKVCacheUseCase`) depending only on domain interfaces.
   - `infrastructure`: Concrete PyTorch modules, BPE tokenization, Tensor KV Cache, datasets, trainers, rule-based guardrails, and evaluators.
   - `presentation`: CLI and external entrypoints.
2. **Strict OOP 4 Pillars**:
   - **Encapsulation**: Enforce private/protected attributes (`_weights`, `_k_cache`, `_merges`), property getters, immutable value objects (`@dataclass(frozen=True)`).
   - **Abstraction**: Abstract Base Classes with `@abstractmethod` defining explicit contracts.
   - **Inheritance**: Base classes `BaseTrainer`, `BaseSafetyRule`, `BaseEvaluator`, `nn.Module`.
   - **Polymorphism**: Interchangeable implementations via common interface protocols.
3. **SOLID Principles**:
   - **SRP**: Single responsibility per class.
   - **OCP**: Open for extension (e.g., adding rules/evaluators without touching caller).
   - **LSP**: Subtypes can substitute base types transparently.
   - **ISP**: Segmented granular interfaces.
   - **DIP**: Inversion of control via dependency injection in application use cases.
4. **Backward Compatibility**:
   - Top-level facade modules (`src/alignforge/model.py`, `tokenizer.py`, etc.) re-export or inherit from new clean architecture implementations so no existing scripts break.
