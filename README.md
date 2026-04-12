# LangChain Text Classification App

News article classification using LangChain, Pydantic, and a local LLM via Ollama.

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- [uv](https://docs.astral.sh/uv/) installed

## Setup

```bash
# Pull the model (see src/config.py to change)
ollama pull gemma3:4b

# Install dependencies
uv sync
```

## Run

### Evaluate

Classifies 100 articles from the [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) test set. Prints incorrect predictions (with article text and reasoning), a confusion matrix, per-class metrics, overall accuracy, and accuracy broken down by confidence level.

```bash
uv run python main.py evaluate
```

### Infer

Classifies a single article. The model self-reports its confidence (`high` or `low`) as part of the classification output. If confidence is `low`, the article is surfaced for human review in the terminal.

```bash
uv run python main.py infer "Apple reports record quarterly earnings driven by iPhone sales."
```

**Human review prompt** (triggered when confidence is low):

```
--- Low confidence classification detected, human review required ---

Article: "..."
LLM predicted: Business
Reason: ...

1. World  2. Sports  3. Business  4. Sci/Tech

Your choice (1-4)  [1. World  2. Sports  3. Business  4. Sci/Tech]:
Your reason (press Enter to skip):
```

## Tests

Unit tests mock LLM calls — no Ollama instance required.

```bash
uv run pytest tests/ -v
```

## Configuration

| Setting | File | Description |
|---|---|---|
| Model | `src/config.py` | Ollama model name |
| Confidence threshold | `src/config.py` | Confidence level that triggers human review (`"low"`) |
| Sample size | `src/config.py` | Number of articles used in evaluation |

## Key Concepts

- **`ChatPromptTemplate`** — separates prompt definition from chain execution
- **`with_structured_output(Schema)`** — forces the LLM to return JSON matching a Pydantic schema; not deterministically enforced — relies on tool calling or JSON mode depending on model support
- **LangChain pipe operator (`|`)** — composes `prompt | llm` into an invokable chain
- **`@tool`** — wraps a Python function as a LangChain `StructuredTool`, auto-generating a schema from type annotations and docstring; called via `.invoke()` like any other LangChain Runnable
- **Self-reported confidence** — the model outputs `confidence: Literal["high", "low"]` as part of the structured schema; this is a qualitative signal generated as text, not a calibrated probability. See `src/tools.py` for alternative approaches explored (Shannon entropy via second LLM call, Ollama native logprobs) and why they were discarded.
