from unittest.mock import MagicMock

from src.evaluate import run_evaluation
from src.schema import ClassificationResult


def _make_dataset(*items):
    """Each item is (text, label_int)."""
    return [{"text": text, "label": label} for text, label in items]


LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def test_run_evaluation_all_correct(capsys):
    dataset = _make_dataset(("Federer wins.", 1), ("Stocks rise.", 2))
    chain = MagicMock()
    chain.invoke.side_effect = [
        ClassificationResult(category="Sports", reasoning="Sports result."),
        ClassificationResult(category="Business", reasoning="Market news."),
    ]

    run_evaluation(chain, dataset, LABEL_MAP)

    out = capsys.readouterr().out
    # No incorrect predictions — article text should not appear
    assert "Federer wins." not in out
    assert "100.0%" in out


def test_run_evaluation_prints_incorrect_only(capsys):
    dataset = _make_dataset(("Federer wins.", 1), ("Stocks rise.", 2))
    chain = MagicMock()
    chain.invoke.side_effect = [
        ClassificationResult(category="World", reasoning="Wrong."),   # incorrect
        ClassificationResult(category="Business", reasoning="Right."),  # correct
    ]

    run_evaluation(chain, dataset, LABEL_MAP)

    out = capsys.readouterr().out
    assert "Federer wins." in out   # incorrect article printed
    assert "Stocks rise." not in out  # correct article not printed


def test_run_evaluation_accuracy(capsys):
    dataset = _make_dataset(("A", 0), ("B", 1), ("C", 2), ("D", 3))
    chain = MagicMock()
    chain.invoke.side_effect = [
        ClassificationResult(category="World", reasoning="r"),    # correct
        ClassificationResult(category="World", reasoning="r"),    # wrong
        ClassificationResult(category="Business", reasoning="r"), # correct
        ClassificationResult(category="World", reasoning="r"),    # wrong
    ]

    run_evaluation(chain, dataset, LABEL_MAP)

    out = capsys.readouterr().out
    assert "50.0%" in out
