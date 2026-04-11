import math

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from .config import MODEL
from .prompts import entropy_prompt
from .schema import CategoryProbabilities, ClassificationResult

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]


def _compute_entropy(probs: CategoryProbabilities) -> float:
    """Compute Shannon entropy in bits from a CategoryProbabilities distribution."""
    p = [probs.world, probs.sports, probs.business, probs.sci_tech]
    return -sum(pi * math.log2(pi) for pi in p if pi > 0)


def _build_entropy_chain():
    llm = ChatOllama(model=MODEL)
    structured_llm = llm.with_structured_output(CategoryProbabilities)
    return entropy_prompt | structured_llm


@tool
def check_entropy(text: str) -> float:
    """
    Ask the LLM to score its confidence across all 4 news categories for the
    given article, then compute Shannon entropy from those scores.
    Higher entropy = more uncertainty. Max entropy is 2.0 bits (uniform distribution).

    Note: this is a second, independent LLM call — it does not access the token-level
    probabilities from the original classification. The entropy reflects a fresh
    evaluation using the same persona as the classifier, maximising alignment but
    not guaranteeing identical reasoning.
    """
    probs = _build_entropy_chain().invoke({"text": text})
    return _compute_entropy(probs)


@tool
def human_review(article_text: str, llm_category: str, llm_reasoning: str) -> ClassificationResult:
    """
    Surface a high-entropy article to a human reviewer via the terminal.
    Presents the LLM's prediction and asks the human to confirm or override.
    Returns a ClassificationResult with the human's chosen category and reason.
    """
    print("\n--- High uncertainty detected, human review required ---")
    print(f"\nArticle: \"{article_text[:120]}...\"")
    print(f"LLM predicted: {llm_category}")
    print(f"Reason:        {llm_reasoning}\n")

    category_menu = "  ".join(f"{i}. {cat}" for i, cat in enumerate(CATEGORIES, 1))
    print(category_menu)

    choice = None
    while choice is None:
        raw = input(f"\nYour choice (1-4)  [{category_menu}]: ").strip()
        if raw in {"1", "2", "3", "4"}:
            choice = CATEGORIES[int(raw) - 1]
        else:
            print(f"  Invalid input. {category_menu}")

    reason = input("Your reason (press Enter to skip): ").strip()
    if not reason:
        reason = "Human reviewer provided no reason."

    return ClassificationResult(category=choice, reasoning=reason)
