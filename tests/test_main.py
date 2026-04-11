from unittest.mock import MagicMock, patch

from src.schema import ClassificationResult


LLM_RESULT = ClassificationResult(category="Business", reasoning="Discusses stock market movements.")
HUMAN_RESULT = ClassificationResult(category="World", reasoning="Human override.")


@patch("main.human_review")
@patch("main.check_entropy")
@patch("main.build_chain")
def test_classify_low_entropy_returns_llm_result(mock_build_chain, mock_check_entropy, mock_human_review, capsys):
    mock_build_chain.return_value.invoke.return_value = LLM_RESULT
    mock_check_entropy.invoke.return_value = 0.5  # below threshold

    from main import classify
    classify("Some article text.")

    mock_human_review.invoke.assert_not_called()
    out = capsys.readouterr().out
    assert "Business" in out
    assert "0.500" in out


@patch("main.human_review")
@patch("main.check_entropy")
@patch("main.build_chain")
def test_classify_high_entropy_triggers_human_review(mock_build_chain, mock_check_entropy, mock_human_review, capsys):
    mock_build_chain.return_value.invoke.return_value = LLM_RESULT
    mock_check_entropy.invoke.return_value = 1.9  # above threshold
    mock_human_review.invoke.return_value = HUMAN_RESULT

    from main import classify
    classify("Some ambiguous article text.")

    mock_human_review.invoke.assert_called_once_with({
        "article_text": "Some ambiguous article text.",
        "llm_category": "Business",
        "llm_reasoning": "Discusses stock market movements.",
    })
    out = capsys.readouterr().out
    assert "World" in out  # human override category printed
