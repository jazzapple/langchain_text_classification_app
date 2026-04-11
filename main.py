import argparse

from src.classifier import build_chain
from src.data import LABEL_MAP, load_sample
from src.evaluate import run_evaluation
from src.config import ENTROPY_THRESHOLD, EVAL_SAMPLE_SIZE, MODEL
from src.tools import check_entropy, human_review


def classify(text: str) -> None:
    chain = build_chain(model=MODEL)
    result = chain.invoke({"text": text})

    # Second tool call: ask the LLM to score confidence across all 4 categories
    # and compute Shannon entropy. Higher entropy = more uncertain.
    # Called via .invoke() — the standard interface for all LangChain tools and runnables.
    entropy = check_entropy.invoke({"text": text})

    if entropy > ENTROPY_THRESHOLD:
        result = human_review.invoke({
            "article_text": text,
            "llm_category": result.category,
            "llm_reasoning": result.reasoning,
        })

    print(f"Category:  {result.category}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Entropy:   {entropy:.3f} bits")


def evaluate() -> None:
    chain = build_chain(model=MODEL)
    dataset = load_sample(split="test", n=EVAL_SAMPLE_SIZE)
    run_evaluation(chain, dataset, LABEL_MAP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain text classification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("evaluate", help="Run evaluation on 100 AG News test samples")

    infer_parser = subparsers.add_parser("infer", help="Classify a single article")
    infer_parser.add_argument("text", help="Article text to classify")

    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate()
    elif args.command == "infer":
        classify(args.text)
