# langchain-text-classification-app

A learning project exploring LangChain's core primitives (prompt templates, structured output, chain composition) applied to news article classification.

## Stack
- **LLM**: `qwen2.5:3b` via [Ollama](https://ollama.com) — local, no API key needed
- **Data**: AG News dataset (`fancyzhx/ag_news`) loaded via HuggingFace `datasets`
- **Schema validation**: Pydantic v2 with `Literal` type for category constraints
- **Orchestration**: LangChain (`langchain`, `langchain-ollama`, `langchain-core`)
- **Evaluation**: scikit-learn (accuracy + confusion matrix), rich (terminal output)
- **Package manager**: `uv`

## Project Structure
```
src/
  schema.py      # Pydantic ClassificationResult model
  data.py        # AG News loading and sampling
  prompts.py     # LangChain ChatPromptTemplate
  classifier.py  # Chain assembly: prompt | structured LLM
  evaluate.py    # Accuracy scoring and confusion matrix display
main.py          # Entry point
```

## Running

### Prerequisites
```bash
# Install Ollama: https://ollama.com
ollama pull qwen2.5:3b
ollama serve      # if not already running as a daemon
```

### Run
```bash
uv run python main.py
```

Classifies 100 test articles from AG News and prints per-item results, a confusion matrix, and overall accuracy.

## Key Concepts
- **`with_structured_output(ClassificationResult)`** — forces the LLM to return JSON matching the Pydantic schema
- **`ChatPromptTemplate`** — separates prompt definition from chain execution
- **LangChain pipe operator (`|`)** — composes `prompt | llm` into an invokable chain
