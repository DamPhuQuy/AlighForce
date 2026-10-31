import json
from pathlib import Path
from alignforge.infrastructure.tokenization import BPETokenizer
from alignforge.infrastructure.data import (
    format_instruction_prompt,
    load_coedit_gec_samples,
    load_paraphrase_dpo_samples,
    load_jfleg_evaluation_samples,
    audit_token_length_distribution,
    generate_dataset_manifest,
)


def test_format_instruction_prompt():
    prompt_gec = format_instruction_prompt("fix_grammar", "She go to school.")
    assert "Fix the grammar in this sentence:" in prompt_gec
    assert "Correction: " in prompt_gec

    prompt_para = format_instruction_prompt("paraphrase", "I cannot attend.")
    assert "Paraphrase this sentence to make it more natural:" in prompt_para
    assert "Paraphrase: " in prompt_para


def test_audit_token_length_distribution(tmp_path):
    tokenizer = BPETokenizer()
    tokenizer.train(["Short test sentence for training tokenizer."], target_vocab_size=270)

    samples = [
        {
            "prompt": "Fix the grammar: She go.",
            "completion": "She goes.",
        },
        {
            "prompt": "Paraphrase: I cannot attend today.",
            "chosen": "I won't be able to make it.",
            "rejected": "Cannot attend.",
        },
    ]

    stats = audit_token_length_distribution(samples, tokenizer, context_lengths=(128, 256))
    assert stats["total_samples_analyzed"] == 2
    assert stats["retention_by_context_length"]["128"]["retained_count"] == 2
    assert stats["retention_by_context_length"]["128"]["retention_rate_pct"] == 100.0


def test_generate_dataset_manifest(tmp_path):
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = generate_dataset_manifest(manifest_path)

    assert manifest["manifest_version"] == "1.0.0"
    assert "tinystories" in manifest["datasets"]
    assert "coedit_gec" in manifest["datasets"]
    assert "paraphrase_dpo" in manifest["datasets"]
    assert "jfleg" in manifest["datasets"]
    assert manifest_path.exists()
