from unittest.mock import MagicMock

from src.evaluate import run_evaluation, _print_per_confidence_metrics
from src.schema import ClassificationResult


def _make_dataset(*items):
    """Each item is (text, label_int)."""
    return [{"text": text, "label": label} for text, label in items]


LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def test_run_evaluation_all_correct(capsys):
    dataset = _make_dataset(("Federer wins.", 1), ("Stocks rise.", 2))
    chain = MagicMock()
    chain.invoke.side_effect = [
        ClassificationResult(category="Sports", reasoning="Sports result.", confidence="high"),
        ClassificationResult(category="Business", reasoning="Market news.", confidence="high"),
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
        ClassificationResult(category="World", reasoning="Wrong.", confidence="low"),   # incorrect
        ClassificationResult(category="Business", reasoning="Right.", confidence="high"),  # correct
    ]

    run_evaluation(chain, dataset, LABEL_MAP)

    out = capsys.readouterr().out
    assert "Federer wins." in out   # incorrect article printed
    assert "Stocks rise." not in out  # correct article not printed


def test_run_evaluation_accuracy(capsys):
    dataset = _make_dataset(("A", 0), ("B", 1), ("C", 2), ("D", 3))
    chain = MagicMock()
    chain.invoke.side_effect = [
        ClassificationResult(category="World", reasoning="r", confidence="high"),    # correct
        ClassificationResult(category="World", reasoning="r", confidence="high"),    # wrong
        ClassificationResult(category="Business", reasoning="r", confidence="high"), # correct
        ClassificationResult(category="World", reasoning="r", confidence="high"),    # wrong
    ]

    run_evaluation(chain, dataset, LABEL_MAP)

    out = capsys.readouterr().out
    assert "50.0%" in out


def test_per_confidence_accuracy(capsys):
    true_labels  = ["World",    "Sports",   "Business", "World"]
    predictions  = ["World",    "World",    "Business", "Sports"]
    confidences  = ["high",     "high",     "low",      "low"]
    # high: 1/2 correct (50%), low: 1/2 correct (50%)

    _print_per_confidence_metrics(true_labels, predictions, confidences)

    out = capsys.readouterr().out
    assert out.count("50.0%") == 2


def test_per_confidence_missing_level(capsys):
    # All high confidence — low row should show "—"
    _print_per_confidence_metrics(["World"], ["World"], ["high"])
    out = capsys.readouterr().out
    assert "—" in out
