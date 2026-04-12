# UNCERTAINTY ESTIMATION — APPROACHES EXPLORED
#
# The goal is to detect when the model is uncertain so we can surface the article for human review.
# Three approaches were considered:
#
# 1. SELF-REPORTED CONFIDENCE (current approach)
#    Add confidence: Literal["high", "medium", "low"] to ClassificationResult.
#    The model expresses its uncertainty as part of the same structured output call.
#    These are not calibrated probabilities — the model generates "low" the same way
#    it generates any other token, based on learned patterns. But as an ordinal signal
#    for flagging ambiguous articles, it is sufficient and honest about what it is.
#
# 2. SECOND LLM CALL WITH SELF-REPORTED FLOAT PROBABILITIES (explored, discarded)
#    A second chain call using CategoryProbabilities schema asked the model to output
#    a probability distribution across all 4 categories. Shannon entropy was computed
#    from those floats. This was discarded because:
#    - The float values carry false precision — 0.8 and 0.79 are not meaningfully different
#    - The probabilities are still self-reported text tokens, no more calibrated than
#      asking for high/medium/low
#    - Requires a second LLM call, doubling inference time
#
# 3. OLLAMA NATIVE API WITH TOKEN LOGPROBS (explored, discarded)
#    Called Ollama's /api/generate with logprobs=True and a GBNF grammar constraint
#    to force single-token output. This would give true softmax probabilities rather
#    than self-reported values. Discarded because:
#    - Ollama only returns the logprob of the token it generated, not a full distribution
#      over all candidate tokens (no top_logprobs support)
#    - The grammar constraint did not reliably constrain output in testing
#    - Even if it worked, LLM token probabilities are not well calibrated for
#      downstream classification accuracy without empirical calibration analysis

from langchain_core.tools import tool

from .schema import ClassificationResult

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]


@tool
def human_review(article_text: str, llm_category: str, llm_reasoning: str) -> ClassificationResult:
    """
    Surface a low-confidence article to a human reviewer via the terminal.
    Presents the LLM's prediction and asks the human to confirm or override.
    Returns a ClassificationResult with the human's chosen category and reason.
    """
    print("\n--- Low confidence classification detected, human review required ---")
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

    return ClassificationResult(category=choice, reasoning=reason, confidence="high")
