from unittest.mock import patch

from src.schema import ClassificationResult


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
    assert result.confidence == "high"


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
