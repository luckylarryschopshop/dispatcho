# Dispatcho Soup

Intent classifier for tool operations. Classifies natural language into file, git, and dev tool commands in under 5ms. No LLM needed.

## Install

```bash
pip install -e .
```

## Usage

```python
from dispatcho import classify

intent, confidence = classify("list the files")
# ("list_files", 0.99)

intent, confidence = classify("git status")
# ("git_status", 0.99)

intent, confidence = classify("hello")
# (None, 0.0)  — out of domain, not a tool command
```

## What It Classifies

27 intents across 4 categories:

**Filesystem:** list_files, create_file, read_file, edit_file, delete_file, create_dir, move_file, copy_file, search_files, change_dir, current_dir

**Git:** git_status, git_log, git_commit, git_add, git_diff, git_branch, git_push, git_pull, git_clone, git_init, git_merge

**Dev tools:** run_tests, install_deps, run_server, format_code, open_url

**Workflows:** stage_and_commit, commit_and_push, create_and_commit, create_project

## How It Works

1. MiniLM-L6 sentence embeddings (384-dim, via fastembed)
2. LogisticRegression classifier (28 classes: 27 intents + out-of-domain)
3. Returns `(intent_name, confidence)` — that's it

The classifier does **not** extract parameters. Use `extract_params()` separately if you need to parse params from free-text user input.

## Parameter Extraction (Optional)

```python
from dispatcho import classify, extract_params, build_frame_vocab, INTENTS

# Build frame vocab once
frame_vocab = build_frame_vocab(INTENTS)

# Classify
intent, confidence = classify("create app.py with content 'print(hello)'")

# Extract params for the classified intent
slot_names = INTENTS[intent]["slots"]
params = extract_params(intent, "create app.py with content 'print(hello)'", slot_names, frame_vocab)
# {"path": "app.py", "content": "print(hello)"}
```

## Router (Optional)

Execute classified intents against the filesystem and git:

```python
from dispatcho import Router

router = Router(working_dir="/tmp/my-project")
result = await router.execute("create_file", {"path": "app.py", "content": "print('hello')"})
# {"success": True, "result": "Successfully wrote to /tmp/my-project/app.py", "method": "local"}

result = await router.execute("git_status", {})
# {"success": True, "result": "M app.py", "method": "bash"}
```

## Training

To retrain the classifier (e.g., after adding new training examples):

```bash
python dispatcho/train.py
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## Architecture

```
User text → classify() → (intent, confidence)
                              ↓
                    High confidence? → router.execute(intent, params)
                    Low confidence?  → pass to LLM or ask for clarification
```

The classifier is the fast path. Simple commands execute in <5ms. Complex or ambiguous requests fall through to whatever LLM or UI you wire up.

## Contract

```python
def classify(message: str) -> tuple[str | None, float]:
    """
    Returns (intent_name, confidence) where:
    - intent_name: str (one of 27 intents), "ambiguous", "workflow:*", or None
    - confidence: float 0.0 to 1.0
    """
```

No side effects. No network calls. No file writes. Just classification.
