from unittest.mock import MagicMock, patch

from src.schema import ClassificationResult


LLM_RESULT_HIGH = ClassificationResult(category="Business", reasoning="Discusses stock market movements.", confidence="high")
LLM_RESULT_LOW = ClassificationResult(category="Business", reasoning="Could be Business or World.", confidence="low")
HUMAN_RESULT = ClassificationResult(category="World", reasoning="Human override.", confidence="high")


@patch("main.human_review")
@patch("main.build_chain")
def test_classify_high_confidence_returns_llm_result(mock_build_chain, mock_human_review, capsys):
    mock_build_chain.return_value.invoke.return_value = LLM_RESULT_HIGH

    from main import classify
    classify("Some article text.")

    mock_human_review.invoke.assert_not_called()
    out = capsys.readouterr().out
    assert "Business" in out
    assert "high" in out


@patch("main.human_review")
@patch("main.build_chain")
def test_classify_low_confidence_triggers_human_review(mock_build_chain, mock_human_review, capsys):
    mock_build_chain.return_value.invoke.return_value = LLM_RESULT_LOW
    mock_human_review.invoke.return_value = HUMAN_RESULT

    from main import classify
    classify("Some ambiguous article text.")

    mock_human_review.invoke.assert_called_once_with({
        "article_text": "Some ambiguous article text.",
        "llm_category": "Business",
        "llm_reasoning": "Could be Business or World.",
    })
    out = capsys.readouterr().out
    assert "World" in out
