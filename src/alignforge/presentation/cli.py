"""Presentation Layer CLI implementation.

Adapts user terminal input to application use cases via Dependency Injection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict
import torch

from alignforge.application.use_cases import (
    CorrectGrammarUseCase,
    ParaphrasePreferenceUseCase,
)
from alignforge.infrastructure.generation.generator import AutoregressiveGenerator, generate
from alignforge.infrastructure.models.lora import apply_lora_to_model, load_lora_checkpoint
from alignforge.infrastructure.models.transformer import CausalTransformerModel
from alignforge.infrastructure.safety.guardrails import CompositeSafetyGuard, validate_input, validate_output
from alignforge.infrastructure.tokenization.bpe_tokenizer import ByteLevelBPETokenizer


def load_model_and_tokenizer(model_choice: str) -> tuple[CausalTransformerModel, ByteLevelBPETokenizer]:
    """Load tokenizer and model with requested checkpoint."""
    tok_dir = Path("artifacts/tokenizer")
    if (tok_dir / "tokenizer.json").exists():
        tokenizer = ByteLevelBPETokenizer.load(tok_dir)
    else:
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(["She goes to school every day."], target_vocab_size=2048)

    model = CausalTransformerModel(
        vocab_size=2048,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        dropout=0.0,
    )

    ckpt_base = Path("artifacts/checkpoints/c0_base.pt")
    if ckpt_base.exists():
        sd = torch.load(ckpt_base, map_location="cpu")
        model.load_state_dict(sd["model_state_dict"], strict=False)

    if model_choice in ("lora_sft", "dpo"):
        apply_lora_to_model(model, r=8, alpha=16.0)
        target_ckpt = (
            Path("artifacts/checkpoints/c2_dpo.pt")
            if model_choice == "dpo"
            else Path("artifacts/checkpoints/c1_lora.pt")
        )
        if target_ckpt.exists():
            load_lora_checkpoint(model, target_ckpt)
    elif model_choice == "full_sft":
        target_ckpt = Path("artifacts/checkpoints/c1_full.pt")
        if target_ckpt.exists():
            sd = torch.load(target_ckpt, map_location="cpu")
            model.load_state_dict(sd["model_state_dict"], strict=False)

    model.eval()
    return model, tokenizer


def cmd_edit(args: argparse.Namespace) -> int:
    """Handle alignforge edit command through Clean Architecture Use Case."""
    text = args.text
    task = args.task
    model_choice = args.model

    # 1. Input Guardrail
    in_guard = validate_input(text)
    if not in_guard.is_valid:
        res = {
            "task": task,
            "input": text,
            "candidate": "",
            "status": in_guard.status,
            "reason": in_guard.reason,
            "model_version": f"{model_choice}-v1",
            "latency_ms": 0.0,
        }
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"Task:      {task}")
            print(f"Model:     {model_choice}-v1")
            print(f"Status:    {in_guard.status} ({in_guard.reason})")
        return 1

    # 2. Dependency Injection & Use Case Execution
    try:
        model, tokenizer = load_model_and_tokenizer(model_choice)
    except Exception as exc:
        print(f"Error loading model: {exc}", file=sys.stderr)
        return 1

    generator = AutoregressiveGenerator(model, tokenizer)
    guard = CompositeSafetyGuard()

    if task == "fix_grammar":
        use_case = CorrectGrammarUseCase(
            model=model,
            tokenizer=tokenizer,
            generator=generator,
            safety_guard=guard,
        )
        result = use_case.execute(in_guard.sanitized_text, max_new_tokens=24, temperature=0.0)
    else:
        use_case = ParaphrasePreferenceUseCase(
            model=model,
            tokenizer=tokenizer,
            generator=generator,
            safety_guard=guard,
        )
        result = use_case.execute(in_guard.sanitized_text, max_new_tokens=24, temperature=0.7)

    # 3. Output Guardrail status formatting
    out_guard = validate_output(result.generated_text, in_guard.sanitized_text)

    response = {
        "task": task,
        "input": text,
        "candidate": result.generated_text,
        "status": out_guard.status,
        "model_version": f"{model_choice}-v1",
        "latency_ms": round(result.latency_ms, 2),
    }

    if args.json:
        print(json.dumps(response, indent=2))
    else:
        print(f"Task:         {task}")
        print(f"Model:        {model_choice}-v1")
        print(f"Candidate:    {result.generated_text}")
        print(f"Status:       {out_guard.status}")
        print(f"Latency:      {result.latency_ms:.1f} ms")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from alignforge.infrastructure.evaluation.evaluators import run_full_evaluation_pipeline
    run_full_evaluation_pipeline()
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    from alignforge.infrastructure.evaluation.profiling import run_kv_cache_profiling
    run_kv_cache_profiling()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alignforge",
        description="AlignForge English: Lightweight English Writing Coach CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_edit = sub.add_parser("edit", help="Propose grammar correction or paraphrase")
    p_edit.add_argument("--task", choices=["fix_grammar", "paraphrase"], default="fix_grammar")
    p_edit.add_argument(
        "--model",
        choices=["base", "full_sft", "lora_sft", "dpo"],
        default="lora_sft",
    )
    p_edit.add_argument("--text", required=True, help="Input short English sentence")
    p_edit.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    sub.add_parser("benchmark", help="Run automated evaluation benchmarks")
    sub.add_parser("profile", help="Run KV Cache multi-length profiling (128 to 2048)")

    args = parser.parse_args(argv)

    if args.command == "edit":
        return cmd_edit(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "profile":
        return cmd_profile(args)
    return 0
