from src.classifier import build_chain
from src.data import LABEL_MAP, load_sample
from src.evaluate import run_evaluation

chain = build_chain(model="qwen2.5:3b")
dataset = load_sample(split="test", n=100)
run_evaluation(chain, dataset, LABEL_MAP)
