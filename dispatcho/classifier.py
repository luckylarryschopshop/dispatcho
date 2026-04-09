"""Intent classifier — MiniLM embeddings + trained LogisticRegression.

Classifies natural language text into one of 28 tool intents.
Returns (intent, confidence). Does NOT extract parameters.

Contract:
    classify(text) → (intent_name: str | None, confidence: float)
    - intent_name: one of the trained intents, "ambiguous", or None (out-of-domain)
    - confidence: 0.0 to 1.0

Dependencies: fastembed, numpy, scikit-learn (via pickle model)
No web frameworks. No subprocess. No regex. Portable.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

_DIR = Path(__file__).parent
_DATA_PATH = _DIR / "training_data.json"
_MODEL_PATH = _DIR / "classifier_model.pkl"

# Load training data (for intent names, workflows, exact-match lookup)
with open(_DATA_PATH, encoding="utf-8") as f:
    _DATA = json.load(f)

INTENTS = _DATA["intents"]
WORKFLOWS = _DATA.get("workflows", {})

# Load embedding model + trained classifier
_EMBED_MODEL = TextEmbedding("BAAI/bge-small-en-v1.5")
with open(_MODEL_PATH, "rb") as f:
    _MODEL_DATA = pickle.load(f)
_CLASSIFIER = _MODEL_DATA["classifier"]
_CLASSES = _MODEL_DATA["classes"]

# Exact-match lookup: resolved training phrases → intent
_PHRASE_TO_INTENT: dict[str, str] = {}
for _name, _data in INTENTS.items():
    for _ex in _data["examples"]:
        _resolved = _ex["text"]
        for _sn, _sv in _ex.get("slots", {}).items():
            _resolved = _resolved.replace("{" + _sn + "}", _sv, 1)
        _PHRASE_TO_INTENT[_resolved.lower()] = _name
    _PHRASE_TO_INTENT[_name] = _name
    _PHRASE_TO_INTENT[_name.replace("_", " ")] = _name

# Thresholds
MIN_CONFIDENCE = 0.30
AMBIGUITY_MARGIN = 0.10


def classify(message: str) -> tuple[str | None, float]:
    """Classify user intent from natural language text.

    Returns:
        (intent_name, confidence) where:
        - intent_name: one of 28 known intents, "ambiguous",
          "workflow:<name>", or None (out-of-domain/low confidence)
        - confidence: 0.0 to 1.0

    The classifier does NOT extract parameters. Use
    param_extractor.extract_params() separately for free-text input.
    """
    text = message.strip()
    if not text or len(text) < 2:
        return None, 0.0
    lower = text.lower()

    # 1. Workflow detection (substring match on triggers)
    for wf_name, wf in WORKFLOWS.items():
        for trigger in wf["triggers"]:
            if trigger in lower:
                return f"workflow:{wf_name}", 0.95

    # 2. Exact match against training phrases
    if lower in _PHRASE_TO_INTENT:
        return _PHRASE_TO_INTENT[lower], 0.99

    # 3. Embed and classify via trained LR model
    query_emb = np.array(list(_EMBED_MODEL.embed([lower])))
    probs = _CLASSIFIER.predict_proba(query_emb)[0]
    sorted_idx = np.argsort(probs)[::-1]

    top_class = _CLASSES[sorted_idx[0]]
    top_prob = float(probs[sorted_idx[0]])
    second_prob = float(probs[sorted_idx[1]])
    second_class = _CLASSES[sorted_idx[1]]

    # Out of domain
    if top_class == "__out_of_domain__":
        return None, 0.0

    # Below confidence threshold
    if top_prob < MIN_CONFIDENCE:
        return None, 0.0

    # Ambiguous (top two intents too close)
    if top_prob - second_prob < AMBIGUITY_MARGIN and second_class != "__out_of_domain__":
        return "ambiguous", round(top_prob, 2)

    return top_class, round(top_prob, 2)
