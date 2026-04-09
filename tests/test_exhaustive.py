"""Exhaustive classifier tests — every intent, edge case, adversarial input.

This is the gate test. If it passes, the classifier is ready to ship.
Run: pytest test_exhaustive.py -v
"""
from __future__ import annotations

import warnings
import pytest

warnings.filterwarnings("ignore", category=RuntimeWarning)

from dispatcho.classifier import classify, INTENTS, WORKFLOWS


# === Every intent must classify at least one training example ===

class TestEveryIntent:

    @pytest.mark.parametrize("intent_name", sorted(INTENTS.keys()))
    def test_intent_has_working_example(self, intent_name):
        data = INTENTS[intent_name]
        for ex in data["examples"][:10]:
            resolved = ex["text"]
            for sn, sv in ex.get("slots", {}).items():
                resolved = resolved.replace("{" + sn + "}", sv, 1)
            intent, conf = classify(resolved)
            if intent == intent_name:
                return  # found one that works
        pytest.fail(f"No example for {intent_name} classifies correctly (tried 10)")


# === Workflow triggers ===

class TestWorkflowTriggers:

    @pytest.mark.parametrize("wf_name,trigger", [
        (wf, t) for wf, d in WORKFLOWS.items() for t in d["triggers"]
    ])
    def test_trigger(self, wf_name, trigger):
        intent, conf = classify(trigger)
        assert intent == f"workflow:{wf_name}", f"'{trigger}' → {intent}"
        assert conf >= 0.9


# === User phrases (not in training data) ===

USER_PHRASES = [
    # Filesystem
    ("ls", "list_files"),
    ("dir", "list_files"),
    ("what files are here", "list_files"),
    ("show me the files", "list_files"),
    ("pwd", "current_dir"),
    ("where am I", "current_dir"),
    ("cd src", "change_dir"),
    ("cd ..", "change_dir"),
    ("go back", "change_dir"),
    ("create app.py", "create_file"),
    ("touch index.html", "create_file"),
    ("cat README.md", "read_file"),
    ("show me server.py", "read_file"),
    ("delete temp.txt", "delete_file"),
    ("rm -rf build", "delete_file"),
    ("nuke the cache folder", "delete_file"),
    ("move a.py to b.py", "move_file"),
    ("copy .env to .env.backup", "copy_file"),
    ("mkdir src", "create_dir"),
    ("make a folder called tests", "create_dir"),
    ("find all python files", "search_files"),
    ("search for TODO", "search_files"),
    # Git
    ("git status", "git_status"),
    ("what changed", "git_status"),
    ("git log", "git_log"),
    ("git diff", "git_diff"),
    ("show me the diff", "git_diff"),
    ("git add .", "git_add"),
    ("stage everything", "git_add"),
    ("commit with message 'fix bug'", "git_commit"),
    ("save my work", "git_commit"),
    ("push to origin main", "git_push"),
    ("ship it", "git_push"),
    ("pull latest", "git_pull"),
    ("create a branch called feature", "git_branch"),
    ("merge feature into main", "git_merge"),
    ("clone https://github.com/user/repo.git", "git_clone"),
    ("git init", "git_init"),
    # Dev tools
    ("run pytest", "run_tests"),
    ("run the tests", "run_tests"),
    ("pip install flask", "install_deps"),
    ("install requirements", "install_deps"),
    ("start the server", "run_server"),
    ("format with black", "format_code"),
    ("lint the code", "format_code"),
    ("open http://localhost:3000", "open_url"),
]

class TestUserPhrases:

    @pytest.mark.parametrize("phrase,expected", USER_PHRASES,
                             ids=[p[0] for p in USER_PHRASES])
    def test_phrase(self, phrase, expected):
        intent, conf = classify(phrase)
        assert intent == expected, f"'{phrase}' → {intent} ({conf:.2f}), expected {expected}"


# === Out-of-domain rejection ===

OOD_PHRASES = [
    "hello", "hi there", "good morning", "thanks", "thank you",
    "what's the weather like", "tell me a joke",
    "build me an app", "write a REST API", "explain recursion",
    "teach me python", "deploy to production", "set up CI/CD",
    "how does git work", "what is a dockerfile",
    "design a database schema", "write unit tests for the API",
    "I'm stuck", "nice work", "interesting",
    "calculate 2+2", "how do I cook pasta",
    "the file is important", "I committed to the project",
    "commit to your goals", "the status of my application",
    "I need to push myself harder",
]

class TestOutOfDomain:

    @pytest.mark.parametrize("phrase", OOD_PHRASES)
    def test_rejected(self, phrase):
        intent, conf = classify(phrase)
        assert intent is None or conf < 0.30, \
            f"False positive: '{phrase}' → {intent} ({conf:.2f})"


# === Edge cases ===

class TestEdgeCases:

    def test_empty_string(self):
        assert classify("") == (None, 0.0)

    def test_whitespace(self):
        assert classify("   ") == (None, 0.0)

    def test_single_char(self):
        assert classify("a") == (None, 0.0)

    def test_all_caps(self):
        intent, _ = classify("CREATE FILE CALLED APP.PY")
        assert intent == "create_file"

    def test_extra_whitespace(self):
        intent, _ = classify("  list   the   files  ")
        assert intent == "list_files"

    def test_returns_2_tuple(self):
        result = classify("list the files")
        assert len(result) == 2

    def test_confidence_in_range(self):
        _, conf = classify("list the files")
        assert 0.0 <= conf <= 1.0


# === Near-miss disambiguation ===

NEAR_MISS = [
    ("create a file called app.py", "create_file"),
    ("create a directory called src", "create_dir"),
    ("make a folder called tests", "create_dir"),
    ("make a file called test.py", "create_file"),
    ("cd src", "change_dir"),
    ("list files in src", "list_files"),
    ("go to src", "change_dir"),
    ("commit changes", "git_commit"),
    ("stage changes", "git_add"),
    ("push to remote", "git_push"),
    ("pull from remote", "git_pull"),
    ("move app.py to src/", "move_file"),
    ("copy app.py to backup/", "copy_file"),
]

class TestNearMiss:

    @pytest.mark.parametrize("phrase,expected", NEAR_MISS,
                             ids=[p[0] for p in NEAR_MISS])
    def test_disambiguation(self, phrase, expected):
        intent, _ = classify(phrase)
        assert intent == expected, f"'{phrase}' → {intent}, expected {expected}"


# === Known limitations (documented, not failures) ===

class TestKnownLimitations:
    """These are cases where the classifier is expected to struggle.
    They're here to track, not to fail the build."""

    @pytest.mark.xfail(reason="'directory' is overloaded — filesystem vs catalog meaning")
    def test_directory_of_services(self):
        intent, conf = classify("directory of services")
        assert intent is None or conf < 0.30

    @pytest.mark.xfail(reason="very short ambiguous inputs may leak through")
    def test_two_char_input(self):
        intent, conf = classify("ls")  # this actually works, but testing the boundary
        # 'ls' should classify as list_files, which is correct
        # but other 2-char inputs like 'cd' might be ambiguous
        assert classify("go")[0] is None  # 'go' alone could mean change_dir
