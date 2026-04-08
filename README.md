# LangChain Text Classification App

News article classification using LangChain, Pydantic, and a local LLM via Ollama.

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- [uv](https://docs.astral.sh/uv/) installed

## Setup

```bash
# Pull the model
ollama pull qwen2.5:3b

# Install dependencies
uv sync
```

## Run

```bash
uv run python main.py
```

Classifies 100 articles from the [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) test set and prints accuracy + confusion matrix.
