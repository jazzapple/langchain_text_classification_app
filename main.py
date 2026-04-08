import argparse

from src.classifier import build_chain
from src.data import LABEL_MAP, load_sample
from src.evaluate import run_evaluation

MODEL = "qwen2.5:3b"


def classify(text: str) -> None:
    chain = build_chain(model=MODEL)
    result = chain.invoke({"text": text})
    print(f"Category:  {result.category}")
    print(f"Reasoning: {result.reasoning}")


def evaluate() -> None:
    chain = build_chain(model=MODEL)
    dataset = load_sample(split="test", n=100)
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
