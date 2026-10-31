import tempfile
from pathlib import Path
import pytest
from alignforge.infrastructure.tokenization import BPETokenizer


def test_base_vocab_initialization():
    tokenizer = BPETokenizer()
    # 4 special tokens + 256 byte tokens = 260 tokens initially
    assert tokenizer.vocab_size == 260
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3


def test_tokenizer_training_and_roundtrip():
    training_corpus = [
        "She goes to school every day.",
        "They are playing soccer in the park.",
        "I cannot attend the meeting today.",
        "Could you please explain this concept to me?",
        "He does not know the answer to the question.",
    ]

    tokenizer = BPETokenizer()
    target_vocab = 300  # 260 base + 40 merges
    tokenizer.train(training_corpus, target_vocab_size=target_vocab)

    assert tokenizer.vocab_size == target_vocab
    assert len(tokenizer.merges) == 40

    test_sentences = [
        "She goes to school every day.",
        "I cannot attend the meeting today.",
        "Hello, world! 123.",
        "Unseen text with punctuation, numbers: $45.67!",
    ]

    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)
        assert decoded == sentence, f"Failed roundtrip for: {sentence}"


def test_bos_and_eos_tokens():
    tokenizer = BPETokenizer()
    tokenizer.train(["Simple training text."], target_vocab_size=270)

    text = "Hello world."
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert encoded[0] == tokenizer.bos_token_id
    assert encoded[-1] == tokenizer.eos_token_id

    # Decoding with skip_special_tokens=True restores original text
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert decoded == text


def test_save_and_load(tmp_path):
    tokenizer = BPETokenizer()
    tokenizer.train(
        ["Once upon a time there was a little girl who lived in a village."],
        target_vocab_size=280,
    )

    save_dir = tmp_path / "tokenizer_ckpt"
    tokenizer.save(save_dir)

    loaded = BPETokenizer.load(save_dir)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert len(loaded.merges) == len(tokenizer.merges)

    test_str = "Once upon a time, a girl lived."
    assert loaded.encode(test_str) == tokenizer.encode(test_str)
    assert loaded.decode(loaded.encode(test_str)) == test_str
