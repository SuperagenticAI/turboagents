from __future__ import annotations

from turboagents.bench.needle import NeedleCase, build_needle_prompt, score_needle_response


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return text.split()


def test_build_needle_prompt_includes_needle_and_target_metadata() -> None:
    tokenizer = FakeTokenizer()
    prompt, metadata = build_needle_prompt(
        tokenizer,
        NeedleCase(context_tokens=64, insertion_fraction=0.5, needle="secret-42"),
    )

    assert "SECRET CODE: secret-42" in prompt
    assert metadata["needle"] == "secret-42"
    assert metadata["actual_prompt_tokens"] >= 1


def test_score_needle_response_reports_exact_match_and_contains() -> None:
    exact = score_needle_response("secret-42", "secret-42")
    contains = score_needle_response("The answer is secret-42.", "secret-42")
    punctuated = score_needle_response(" secret-42. ", "secret-42")

    assert exact["exact_match"] is True
    assert exact["contains_needle"] is True
    assert contains["exact_match"] is False
    assert contains["contains_needle"] is True
    assert punctuated["exact_match"] is True
    assert punctuated["contains_needle"] is True
