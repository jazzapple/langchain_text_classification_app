import math
from unittest.mock import MagicMock, patch

import pytest

from src.schema import CategoryProbabilities, ClassificationResult
from src.tools import _compute_entropy


def _make_probs(world, sports, business, sci_tech):
    return CategoryProbabilities(world=world, sports=sports, business=business, sci_tech=sci_tech)


# --- _compute_entropy ---

def test_compute_entropy_uniform_is_max():
    probs = _make_probs(0.25, 0.25, 0.25, 0.25)
    assert _compute_entropy(probs) == pytest.approx(2.0, rel=1e-6)


def test_compute_entropy_certain_is_low():
    probs = _make_probs(0.97, 0.01, 0.01, 0.01)
    assert _compute_entropy(probs) < 0.3


def test_compute_entropy_ignores_zero_probs():
    # Zero probabilities are skipped (log2(0) is undefined)
    probs = _make_probs(0.5, 0.5, 0.0, 0.0)
    expected = -2 * (0.5 * math.log2(0.5))
    assert _compute_entropy(probs) == pytest.approx(expected, rel=1e-6)


# --- check_entropy ---

@patch("src.tools._build_entropy_chain")
def test_check_entropy_returns_correct_entropy(mock_build_chain):
    probs = _make_probs(0.9, 0.05, 0.03, 0.02)
    mock_build_chain.return_value.invoke.return_value = probs

    from src.tools import check_entropy
    result = check_entropy.invoke({"text": "Federer wins Wimbledon."})

    assert result == pytest.approx(_compute_entropy(probs), rel=1e-6)


@patch("src.tools._build_entropy_chain")
def test_check_entropy_uniform_returns_two_bits(mock_build_chain):
    probs = _make_probs(0.25, 0.25, 0.25, 0.25)
    mock_build_chain.return_value.invoke.return_value = probs

    from src.tools import check_entropy
    result = check_entropy.invoke({"text": "An ambiguous article."})

    assert result == pytest.approx(2.0, rel=1e-6)


# --- human_review ---

def test_human_review_valid_input(monkeypatch):
    inputs = iter(["2", "Clearly a sports article."])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    from src.tools import human_review
    result = human_review.invoke({
        "article_text": "Federer wins Wimbledon.",
        "llm_category": "World",
        "llm_reasoning": "Discusses international events.",
    })

    assert result.category == "Sports"
    assert result.reasoning == "Clearly a sports article."


def test_human_review_invalid_then_valid(monkeypatch):
    inputs = iter(["5", "abc", "3", "Business article."])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    from src.tools import human_review
    result = human_review.invoke({
        "article_text": "Some article.",
        "llm_category": "World",
        "llm_reasoning": "Some reason.",
    })

    assert result.category == "Business"


def test_human_review_empty_reason_uses_default(monkeypatch):
    inputs = iter(["1", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    from src.tools import human_review
    result = human_review.invoke({
        "article_text": "Some article.",
        "llm_category": "Sports",
        "llm_reasoning": "Some reason.",
    })

    assert result.category == "World"
    assert result.reasoning == "Human reviewer provided no reason."
