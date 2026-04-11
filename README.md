# LangChain Text Classification App

News article classification using LangChain, Pydantic, and a local LLM via Ollama.

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- [uv](https://docs.astral.sh/uv/) installed

## Setup

```bash
# Pull the model (see src/config.py to change)
ollama pull phi3:mini

# Install dependencies
uv sync
```

## Run

### Evaluate

Classifies 100 articles from the [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) test set. Prints incorrect predictions (with article text and reasoning), a confusion matrix, per-class metrics, and overall accuracy.

```bash
uv run python main.py evaluate
```

### Infer

Classifies a single article. A second LLM call estimates a probability distribution across all 4 categories and computes Shannon entropy. If entropy exceeds the threshold (`ENTROPY_THRESHOLD` in `main.py`), the article is surfaced for human review in the terminal.

```bash
uv run python main.py infer "Apple reports record quarterly earnings driven by iPhone sales."
```

**Human review prompt** (triggered when entropy > threshold):

```
--- High uncertainty detected, human review required ---

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
| Entropy threshold | `src/config.py` | Bits above which human review is triggered (0.0–2.0) |
| Sample size | `src/config.py` | Number of articles used in evaluation |

## Key Concepts

- **`ChatPromptTemplate`** — separates prompt definition from chain execution
- **`with_structured_output(Schema)`** — forces the LLM to return JSON matching a Pydantic schema
- **LangChain pipe operator (`|`)** — composes `prompt | llm` into an invokable chain
- **`@tool`** — wraps a Python function as a LangChain `StructuredTool`, auto-generating a schema from type annotations and docstring; called via `.invoke()` like any other LangChain Runnable
- **Sequential tool pipeline** — explicit orchestration of multiple tool calls, as a deliberate contrast to a ReAct agent where the LLM decides when to call tools
- **Shannon entropy** — measures uncertainty across the category probability distribution (0.0 = certain, 2.0 = uniform); computed from a second, independent LLM call
