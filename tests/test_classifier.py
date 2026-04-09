"""Tests for intent classification.

Contract: classify(text) → (intent: str|None, confidence: float)
The classifier does NOT extract parameters.

Run:  pytest test_classifier.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dispatcho.classifier import classify, INTENTS, WORKFLOWS

DATA_PATH = Path(__file__).parent.parent / "dispatcho" / "training_data.json"

MIN_INTENT_ACCURACY = 0.90
MIN_WORKFLOW_ACCURACY = 0.95
MIN_OOD_ACCURACY = 0.85


@pytest.fixture(scope="module")
def training_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


# --- Contract ---

class TestClassifyContract:
    """classify() returns (intent, confidence) — nothing else."""

    def test_returns_2_tuple(self):
        result = classify("list the files")
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    def test_intent_is_string_or_none(self):
        intent, _ = classify("list the files")
        assert isinstance(intent, str)
        intent, _ = classify("how do I cook pasta")
        assert intent is None or isinstance(intent, str)

    def test_confidence_is_float_in_range(self):
        _, conf = classify("list the files")
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_empty_input_returns_none(self):
        assert classify("") == (None, 0.0)
        assert classify("   ") == (None, 0.0)


# --- Accuracy ---

class TestIntentAccuracy:

    def test_above_threshold(self, training_data):
        total = correct = 0
        for intent_name, intent_data in training_data["intents"].items():
            for example in intent_data["examples"]:
                resolved = example["text"]
                for sn, sv in example.get("slots", {}).items():
                    resolved = resolved.replace("{" + sn + "}", sv, 1)
                total += 1
                result_intent, _ = classify(resolved)
                if result_intent == intent_name or result_intent == "ambiguous":
                    correct += 1
        accuracy = correct / total if total else 0
        assert accuracy >= MIN_INTENT_ACCURACY, f"{accuracy:.1%} < {MIN_INTENT_ACCURACY:.0%}"


class TestWorkflows:

    def test_triggers(self, training_data):
        total = correct = 0
        for wf_name, wf in training_data.get("workflows", {}).items():
            for trigger in wf["triggers"]:
                total += 1
                intent, conf = classify(trigger)
                if intent == f"workflow:{wf_name}" and conf >= 0.9:
                    correct += 1
        if total:
            assert correct / total >= MIN_WORKFLOW_ACCURACY


# --- OOD rejection ---

OOD_PHRASES = [
    "what's the weather like", "tell me a joke", "how do I make pasta",
    "build me an app", "explain quantum computing", "write me a REST API",
    "hello", "thanks", "deploy to kubernetes", "what time is it",
]

class TestOutOfDomain:

    @pytest.mark.parametrize("phrase", OOD_PHRASES)
    def test_rejected(self, phrase):
        intent, conf = classify(phrase)
        assert intent is None or conf < 0.30, f"False positive: '{phrase}' → {intent} ({conf:.2f})"


# --- User phrases ---

USER_PHRASES = [
    ("what directory am I in", "current_dir"),
    ("create a new folder called dog", "create_dir"),
    ("cd dog", "change_dir"),
    ("list the files", "list_files"),
    ("list", "list_files"),
    ("create a new file called poodle.md", "create_file"),
    ("mkdir test", "create_dir"),
    ("git status", "git_status"),
    ("rm -rf test", "delete_file"),
    ("push to origin main", "git_push"),
    ("create a branch called feature", "git_branch"),
    ("nuke that folder", "delete_file"),
    ("ship it", "git_push"),
    ("run pytest", "run_tests"),
    ("pip install flask", "install_deps"),
    ("start the server", "run_server"),
    ("format with black", "format_code"),
    ("git init", "git_init"),
]

class TestUserPhrases:

    @pytest.mark.parametrize("phrase,expected", USER_PHRASES,
                             ids=[p[0][:30] for p in USER_PHRASES])
    def test_phrase(self, phrase, expected):
        intent, _ = classify(phrase)
        assert intent == expected, f"'{phrase}' → {intent} (expected {expected})"


# --- Safeguards (enforced, not aspirational) ---

class TestSafeguards:

    def test_classifier_has_no_regex(self):
        assert "import re" not in Path("dispatcho/classifier.py").read_text()

    def test_classifier_has_no_param_extraction(self):
        code = Path("dispatcho/classifier.py").read_text()
        assert "import param_extractor" not in code
        assert "from param_extractor" not in code

    def test_classifier_line_budget(self):
        lines = len(Path("dispatcho/classifier.py").read_text().splitlines())
        assert lines <= 200, f"{lines} lines (max 200)"

    def test_classifier_is_portable(self):
        code = Path("dispatcho/classifier.py").read_text()
        assert "import subprocess" not in code
        assert "import fastapi" not in code.lower()
        assert "import httpx" not in code

    def test_no_shell_true_anywhere(self):
        for f in Path("dispatcho").glob("*.py"):
            if f.name.startswith("test_"):
                continue
            assert "shell=True" not in f.read_text(), f"{f.name} has shell=True"
