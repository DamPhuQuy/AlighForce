"""Domain Services for AlignForge.

Encapsulates mathematical invariants, loss objectives, and evaluation algorithms
independent of infrastructure frameworks.
"""

from collections import Counter
from typing import List, Tuple
import math


class DPOLossService:
    """Encapsulates mathematical Direct Preference Optimization formulation:

    L_DPO = -log(sigmoid(beta * (log(pi_theta(y_w|x)/pi_ref(y_w|x)) - log(pi_theta(y_l|x)/pi_ref(y_l|x)))))
    """

    def __init__(self, beta: float = 0.1) -> None:
        if beta <= 0.0:
            raise ValueError(f"beta must be strictly positive, got {beta}")
        self._beta = beta

    @property
    def beta(self) -> float:
        return self._beta

    def compute_log_ratio(
        self,
        policy_chosen_logp: float,
        reference_chosen_logp: float,
        policy_rejected_logp: float,
        reference_rejected_logp: float,
    ) -> Tuple[float, float]:
        """Compute scalar DPO loss and chosen win probability."""
        chosen_ratio = policy_chosen_logp - reference_chosen_logp
        rejected_ratio = policy_rejected_logp - reference_rejected_logp
        logits = self._beta * (chosen_ratio - rejected_ratio)

        # Numerically stable log-sigmoid
        if logits >= 0:
            loss = math.log(1.0 + math.exp(-logits))
        else:
            loss = -logits + math.log(1.0 + math.exp(logits))

        chosen_win_prob = 1.0 / (1.0 + math.exp(-logits))
        return loss, chosen_win_prob


class GLEUScoringService:
    """Encapsulates GLEU n-gram precision/recall evaluation for Grammar Error Correction."""

    def __init__(self, min_n: int = 1, max_n: int = 4) -> None:
        self._min_n = min_n
        self._max_n = max_n

    def score_sentence(self, hypothesis: str, references: List[str]) -> float:
        """Compute sentence-level GLEU score bounded in [0.0, 1.0]."""
        hyp_tokens = hypothesis.strip().lower().split()
        if not hyp_tokens:
            return 0.0

        n_scores: List[float] = []
        for n in range(self._min_n, self._max_n + 1):
            hyp_ngrams = Counter(
                [tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1)]
            )
            if not hyp_ngrams:
                continue

            max_ref_counts: Counter = Counter()
            ref_lens: List[int] = []
            for ref in references:
                ref_tokens = ref.strip().lower().split()
                ref_lens.append(len(ref_tokens))
                ref_ngrams = Counter(
                    [tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)]
                )
                for ng, count in ref_ngrams.items():
                    max_ref_counts[ng] = max(max_ref_counts[ng], count)

            matches = sum(min(count, max_ref_counts[ng]) for ng, count in hyp_ngrams.items())
            total_hyp = sum(hyp_ngrams.values())
            precision = matches / max(total_hyp, 1)

            # Brevity penalty
            min_ref_len = min(ref_lens) if ref_lens else len(hyp_tokens)
            bp = min(1.0, math.exp(1.0 - min_ref_len / max(len(hyp_tokens), 1)))

            n_scores.append(precision * bp)

        if not n_scores:
            return 0.0
        return float(sum(n_scores) / len(n_scores))


class RepetitionAnalysisService:
    """Domain service analyzing repetitive n-gram degradation in generated sequences."""

    def __init__(self, max_allowed_ngram_repeat: int = 4, ngram_size: int = 3) -> None:
        self._max_repeat = max_allowed_ngram_repeat
        self._ngram_size = ngram_size

    def check_repetition(self, tokens: List[str]) -> Tuple[bool, int]:
        """Detect if any n-gram repeats more than max_allowed_ngram_repeat consecutive times."""
        if len(tokens) < self._ngram_size * 2:
            return False, 0

        ngrams = [
            tuple(tokens[i : i + self._ngram_size])
            for i in range(len(tokens) - self._ngram_size + 1)
        ]
        counts = Counter(ngrams)
        most_common = counts.most_common(1)
        if not most_common:
            return False, 0

        max_count = most_common[0][1]
        is_repetitive = max_count >= self._max_repeat
        return is_repetitive, max_count
