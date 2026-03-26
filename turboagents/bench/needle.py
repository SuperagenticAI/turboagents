"""Minimal long-context Needle-style evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Protocol

from turboagents.engines import mlx
from turboagents.quant.config import SUPPORTED_BITS

DEFAULT_NEEDLE = "turboagents-needle-1729"
DEFAULT_FILLER_SENTENCE = (
    "TurboAgents studies long-context compression tradeoffs for practical inference systems. "
)


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


@dataclass(frozen=True, slots=True)
class NeedleCase:
    context_tokens: int
    insertion_fraction: float
    needle: str = DEFAULT_NEEDLE


def _token_count(tokenizer: TokenizerLike, text: str) -> int:
    return int(len(tokenizer.encode(text)))


def _repeat_to_token_budget(tokenizer: TokenizerLike, chunk: str, budget: int) -> str:
    if budget <= 0:
        return ""
    pieces: list[str] = []
    total = 0
    chunk_tokens = max(1, _token_count(tokenizer, chunk))
    repeats = max(1, budget // chunk_tokens)
    for _ in range(repeats * 2 + 8):
        if total >= budget:
            break
        pieces.append(chunk)
        total = _token_count(tokenizer, "".join(pieces))
    text = "".join(pieces)
    while text and _token_count(tokenizer, text) > budget:
        text = text[:-1]
    return text


def build_needle_prompt(
    tokenizer: TokenizerLike,
    case: NeedleCase,
    *,
    filler_sentence: str = DEFAULT_FILLER_SENTENCE,
) -> tuple[str, dict[str, Any]]:
    prefix = (
        "You will read a long context. A secret code appears exactly once inside it. "
        "Answer with only that secret code.\n\nContext:\n"
    )
    suffix = "\n\nQuestion: What is the secret code? Answer with only the code."
    needle_line = f"\nSECRET CODE: {case.needle}\n"

    fixed_tokens = _token_count(tokenizer, prefix + needle_line + suffix)
    body_budget = max(0, case.context_tokens - fixed_tokens)
    before_budget = int(body_budget * case.insertion_fraction)
    after_budget = max(0, body_budget - before_budget)

    before_text = _repeat_to_token_budget(tokenizer, filler_sentence, before_budget)
    after_text = _repeat_to_token_budget(tokenizer, filler_sentence, after_budget)
    prompt = prefix + before_text + needle_line + after_text + suffix

    return prompt, {
        "requested_context_tokens": case.context_tokens,
        "actual_prompt_tokens": _token_count(tokenizer, prompt),
        "needle": case.needle,
        "insertion_fraction": case.insertion_fraction,
    }


def score_needle_response(response_text: str, needle: str) -> dict[str, Any]:
    normalized = response_text.strip()
    compact = re.sub(r"\s+", " ", normalized)
    canonical = compact.strip(" \t\r\n`'\".,:;!?()[]{}")
    exact_match = canonical == needle
    contains_needle = needle in canonical or needle in compact
    return {
        "exact_match": exact_match,
        "contains_needle": contains_needle,
        "response_preview": compact[:200],
    }


def run_needle_benchmark(
    *,
    model_path: str,
    context_tokens: list[int],
    insertion_fractions: list[float],
    bits_list: list[float] | None = None,
    max_tokens: int = 32,
    needle: str = DEFAULT_NEEDLE,
) -> dict[str, Any]:
    bits_values = bits_list or [bit for bit in SUPPORTED_BITS if bit >= 3.0]
    model, tokenizer = mlx.load(model_path, lazy=False)

    runs: list[dict[str, Any]] = []
    for bits in bits_values:
        for target_tokens in context_tokens:
            for insertion_fraction in insertion_fractions:
                case = NeedleCase(
                    context_tokens=target_tokens,
                    insertion_fraction=insertion_fraction,
                    needle=needle,
                )
                prompt, metadata = build_needle_prompt(tokenizer, case)
                started = time.perf_counter()
                response = mlx.generate(
                    model,
                    tokenizer,
                    prompt,
                    bits=bits,
                    max_tokens=max_tokens,
                    temp=0.0,
                )
                elapsed = time.perf_counter() - started
                response_text = response if isinstance(response, str) else str(response)
                scored = score_needle_response(response_text, needle)
                runs.append(
                    {
                        "bits": bits,
                        "context_tokens": target_tokens,
                        "insertion_fraction": insertion_fraction,
                        "elapsed_seconds": round(elapsed, 4),
                        **metadata,
                        **scored,
                    }
                )

    return {
        "benchmark": "needle",
        "model": model_path,
        "needle": needle,
        "max_tokens": max_tokens,
        "runs": runs,
    }
