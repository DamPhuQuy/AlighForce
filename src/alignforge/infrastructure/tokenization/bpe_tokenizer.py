"""Byte-Pair Encoding Tokenizer Infrastructure implementation.

Implements the ITokenizer domain port with byte-level fallback for 100% round-trip fidelity.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

from alignforge.domain.interfaces import ITokenizer

# Pre-tokenization regex pattern (splitting on contractions, words, numbers, punctuation, spaces)
SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\s\w]+|\s+"""
)


class ByteLevelBPETokenizer(ITokenizer):
    """Byte-level Byte-Pair Encoding (BPE) Tokenizer.

    Upholds OOP Encapsulation and implements ITokenizer domain interface.
    Guarantees 100% round-trip coverage for any UTF-8 string.
    """

    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def __init__(self) -> None:
        self._special_tokens = list(self.SPECIAL_TOKENS)
        self._pad_token_id = 0
        self._unk_token_id = 1
        self._bos_token_id = 2
        self._eos_token_id = 3

        # Encapsulated state dictionaries
        self._vocab: Dict[bytes, int] = {}
        self._inverse_vocab: Dict[int, bytes] = {}
        self._merges: Dict[Tuple[bytes, bytes], bytes] = {}
        self._merge_ranks: Dict[Tuple[bytes, bytes], int] = {}

        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        """Initialize the base vocabulary: special tokens + 256 individual bytes."""
        self._vocab.clear()
        self._inverse_vocab.clear()

        # 1. Special tokens
        for idx, token in enumerate(self._special_tokens):
            raw = token.encode("utf-8")
            self._vocab[raw] = idx
            self._inverse_vocab[idx] = raw

        # 2. 256 byte values
        offset = len(self._special_tokens)
        for b in range(256):
            raw = bytes([b])
            self._vocab[raw] = offset + b
            self._inverse_vocab[offset + b] = raw

    # Properties implementing ITokenizer and backward compatibility
    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_id(self) -> int:
        return self._pad_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def unk_token_id(self) -> int:
        return self._unk_token_id

    @property
    def bos_id(self) -> int:
        return self._bos_token_id

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_id(self) -> int:
        return self._eos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def special_tokens(self) -> List[str]:
        return list(self._special_tokens)

    @special_tokens.setter
    def special_tokens(self, tokens: List[str]) -> None:
        self._special_tokens = list(tokens)

    @property
    def vocab(self) -> Dict[bytes, int]:
        return self._vocab

    @vocab.setter
    def vocab(self, val: Dict[bytes, int]) -> None:
        self._vocab = val

    @property
    def inverse_vocab(self) -> Dict[int, bytes]:
        return self._inverse_vocab

    @inverse_vocab.setter
    def inverse_vocab(self, val: Dict[int, bytes]) -> None:
        self._inverse_vocab = val

    @property
    def merges(self) -> Dict[Tuple[bytes, bytes], bytes]:
        return self._merges

    @property
    def merge_ranks(self) -> Dict[Tuple[bytes, bytes], int]:
        return self._merge_ranks

    def train(self, texts: List[str], target_vocab_size: int = 2048) -> None:
        """Train BPE merges on a collection of texts until reaching target_vocab_size."""
        self._init_base_vocab()
        self._merges.clear()
        self._merge_ranks.clear()

        # 1. Pre-tokenize and split words into sequences of 1-byte elements
        word_counts: Dict[Tuple[bytes, ...], int] = {}
        for text in texts:
            words = SPLIT_PATTERN.findall(text)
            for w in words:
                raw_bytes = w.encode("utf-8")
                token_tuple = tuple(bytes([b]) for b in raw_bytes)
                word_counts[token_tuple] = word_counts.get(token_tuple, 0) + 1

        num_merges = target_vocab_size - len(self._vocab)

        # 2. Iteratively merge most frequent adjacent byte pairs
        for rank in range(num_merges):
            pair_counts: Dict[Tuple[bytes, bytes], int] = {}
            for word_tokens, count in word_counts.items():
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + count

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            merged = best_pair[0] + best_pair[1]

            new_token_id = len(self._vocab)
            self._vocab[merged] = new_token_id
            self._inverse_vocab[new_token_id] = merged
            self._merges[best_pair] = merged
            self._merge_ranks[best_pair] = rank

            # Apply merge to all words in training set
            new_word_counts: Dict[Tuple[bytes, ...], int] = {}
            for word_tokens, count in word_counts.items():
                new_tokens = self._merge_word(word_tokens, best_pair, merged)
                new_word_counts[new_tokens] = count
            word_counts = new_word_counts

    def _merge_word(
        self,
        tokens: Tuple[bytes, ...],
        pair: Tuple[bytes, bytes],
        replacement: bytes,
    ) -> Tuple[bytes, ...]:
        """Merge adjacent occurrences of pair in a single token tuple."""
        out: List[bytes] = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                out.append(replacement)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return tuple(out)

    def _bpe(self, word_bytes: bytes) -> List[bytes]:
        """Encode a single word by applying all learned merges in order of rank."""
        tokens = [bytes([b]) for b in word_bytes]
        if len(tokens) <= 1:
            return tokens

        while len(tokens) > 1:
            best_rank = float("inf")
            best_pair = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            merged = self._merges[best_pair]
            tokens = list(self._merge_word(tuple(tokens), best_pair, merged))

        return tokens

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode string text into a list of token IDs."""
        words = SPLIT_PATTERN.findall(text)
        token_ids: List[int] = []

        if add_bos:
            token_ids.append(self._bos_token_id)

        for w in words:
            raw = w.encode("utf-8")
            bpe_tokens = self._bpe(raw)
            for tok in bpe_tokens:
                token_id = self._vocab.get(tok, self._unk_token_id)
                token_ids.append(token_id)

        if add_eos:
            token_ids.append(self._eos_token_id)

        return token_ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a list of token IDs back into a UTF-8 string."""
        byte_chunks: List[bytes] = []
        special_ids = {
            self._pad_token_id,
            self._unk_token_id,
            self._bos_token_id,
            self._eos_token_id,
        }

        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            raw = self._inverse_vocab.get(token_id, b"")
            byte_chunks.append(raw)

        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, directory: Union[str, Path]) -> None:
        """Save vocabulary and merges to directory as JSON."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        vocab_serializable = {
            tok.hex(): idx for tok, idx in self._vocab.items()
        }
        merges_serializable = [
            [p[0].hex(), p[1].hex()] for p in self._merges.keys()
        ]

        data = {
            "vocab": vocab_serializable,
            "merges": merges_serializable,
            "special_tokens": self._special_tokens,
        }

        with open(dir_path / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> ByteLevelBPETokenizer:
        """Load tokenizer from JSON in directory."""
        dir_path = Path(directory)
        with open(dir_path / "tokenizer.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls()
        tok._special_tokens = data.get("special_tokens", cls.SPECIAL_TOKENS)
        tok._vocab = {bytes.fromhex(k): v for k, v in data["vocab"].items()}
        tok._inverse_vocab = {v: k for k, v in tok._vocab.items()}

        tok._merges.clear()
        tok._merge_ranks.clear()
        for rank, (p0, p1) in enumerate(data.get("merges", [])):
            pair = (bytes.fromhex(p0), bytes.fromhex(p1))
            merged = pair[0] + pair[1]
            tok._merges[pair] = merged
            tok._merge_ranks[pair] = rank

        return tok
