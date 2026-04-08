from datasets import load_dataset

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def load_sample(split: str = "test", n: int = 100):
    ds = load_dataset("fancyzhx/ag_news")
    return ds[split].select(range(n))
