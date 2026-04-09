"""Train the intent classifier: MiniLM-L6 embeddings + Logistic Regression.

Produces:
  - classifier_model.pkl (trained LR + label mapping)
  - Reports accuracy on training data and user phrases
"""
from __future__ import annotations

import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

random.seed(42)

DATA_PATH = Path(__file__).parent / "training_data.json"
MODEL_PATH = Path(__file__).parent / "classifier_model.pkl"

def expand_negatives(phrases: list[str], target: int = 1500) -> list[str]:
    """Expand negative examples via prefix/suffix permutations."""
    expanded = list(phrases)
    prefixes = ["", "can you ", "please ", "I want to ", "could you ",
                "I need to ", "hey ", "yo ", "just ", "actually "]
    suffixes = ["", " please", " thanks", "?", "!", " now", " asap"]
    while len(expanded) < target:
        base = random.choice(phrases)
        expanded.append(f"{random.choice(prefixes)}{base}{random.choice(suffixes)}".strip())
    return list(set(expanded))[:target]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main():
    print("Loading training data...")
    with open(DATA_PATH) as f:
        data = json.load(f)

    # Collect all examples
    texts = []
    labels = []
    for intent_name, intent_data in data["intents"].items():
        for example in intent_data["examples"]:
            template = example["text"]
            slots = example.get("slots", {})
            resolved = template
            for sn, sv in slots.items():
                resolved = resolved.replace("{" + sn + "}", sv, 1)
            texts.append(resolved.lower())
            labels.append(intent_name)

    # Add negatives as "out_of_domain" class (loaded from training_data.json)
    base_negatives = [n["text"] for n in data.get("negatives", [])]
    if not base_negatives:
        raise ValueError("No negatives found in training_data.json — add them to the 'negatives' key")
    negatives = expand_negatives(base_negatives, 1500)
    for neg in negatives:
        texts.append(neg.lower())
        labels.append("__out_of_domain__")

    print(f"Total training examples: {len(texts)} ({len(texts) - len(negatives)} intents + {len(negatives)} negatives)")
    print(f"Classes: {len(set(labels))}")

    # Embed all texts
    print("Embedding all texts with MiniLM-L6...")
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    t0 = time.time()
    X = np.array(list(model.embed(texts)))
    print(f"Embedded {len(texts)} texts in {time.time()-t0:.1f}s → shape {X.shape}")

    y = np.array(labels)

    # Train logistic regression
    print("Training logistic regression...")
    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X, y)
    print(f"Trained in {time.time()-t0:.1f}s")

    # Training accuracy
    train_acc = clf.score(X, y)
    print(f"Training accuracy: {train_acc*100:.2f}%")

    # Cross-validation
    print("Cross-validating (5-fold)...")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    # Per-class accuracy
    from sklearn.metrics import classification_report
    y_pred = clf.predict(X)
    print("\nPer-class accuracy (training set):")
    report = classification_report(y, y_pred, output_dict=True)
    for cls in sorted(report.keys()):
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        r = report[cls]
        marker = " ◄" if r["f1-score"] < 0.95 else ""
        print(f"  {cls:<25} precision={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1-score']:.3f} support={int(r['support'])}{marker}")

    # Save model + metadata
    model_data = {
        "classifier": clf,
        "classes": list(clf.classes_),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "training_size": len(texts),
        "cv_accuracy": float(cv_scores.mean()),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nSaved model to {MODEL_PATH}")

    # Test on user phrases
    print("\n" + "="*60)
    print("USER PHRASE TEST")
    print("="*60)
    user_phrases = [
        # --- Filesystem: list ---
        ("what directory am I in", "current_dir"),
        ("list the files", "list_files"),
        ("list", "list_files"),
        ("list contents", "list_files"),
        ("show me the contents of your current directory", "list_files"),
        ("what do we got in here", "list_files"),
        ("files?", "list_files"),
        ("ls", "list_files"),
        ("what's in src/", "list_files"),
        ("show me everything in the tests folder", "list_files"),
        ("folder contents", "list_files"),
        ("anything in this directory", "list_files"),

        # --- Filesystem: create ---
        ("create a new file called poodle.md", "create_file"),
        ("create a new file in the dog folder called bone.txt", "create_file"),
        ("make a file called config.yaml", "create_file"),
        ("touch index.html", "create_file"),
        ("create .env with content 'DEBUG=true'", "create_file"),
        ("write hello.py with content 'print(\"hi\")'", "create_file"),
        ("create README.md with content '# My Project'", "create_file"),

        # --- Filesystem: read ---
        ("show me what's in config.py", "read_file"),
        ("cat package.json", "read_file"),
        ("read the README", "read_file"),
        ("peek at src/main.rs", "read_file"),
        ("what does app.py say", "read_file"),
        ("open requirements.txt", "read_file"),

        # --- Filesystem: delete ---
        ("rm -rf test", "delete_file"),
        ("nuke the build folder", "delete_file"),
        ("delete old.py", "delete_file"),
        ("remove the temp directory", "delete_file"),
        ("get rid of node_modules", "delete_file"),
        ("trash debug.log", "delete_file"),

        # --- Filesystem: move/copy ---
        ("rename dog to cat", "move_file"),
        ("move old.py to archive/", "move_file"),
        ("mv config.yaml to backup/", "move_file"),
        ("copy .env to .env.backup", "copy_file"),
        ("duplicate the config file", "copy_file"),

        # --- Filesystem: create dir ---
        ("create a new folder called dog", "create_dir"),
        ("make a new folder called thingy", "create_dir"),
        ("mkdir test", "create_dir"),
        ("mkdir -p src/components", "create_dir"),
        ("create a directory called build", "create_dir"),
        ("make a folder for the tests", "create_dir"),

        # --- Filesystem: search ---
        ("find all python files", "search_files"),
        ("search for TODO in the codebase", "search_files"),
        ("grep for 'import os'", "search_files"),
        ("where is the database config", "search_files"),

        # --- Navigation ---
        ("cd dog", "change_dir"),
        ("change directory to dog", "change_dir"),
        ("go to src", "change_dir"),
        ("cd ..", "change_dir"),
        ("navigate to the tests folder", "change_dir"),
        ("switch to the backend directory", "change_dir"),
        ("open the dog folder", "change_dir"),  # ambiguous — "open" means navigate more often than list
        ("name of this folder", "current_dir"),
        ("where am I", "current_dir"),
        ("pwd", "current_dir"),
        ("what folder is this", "current_dir"),

        # --- Git: status/log/diff ---
        ("git status", "git_status"),
        ("what's changed", "git_status"),
        ("any uncommitted changes", "git_status"),
        ("show me the diff", "git_diff"),
        ("what did I change", "git_diff"),
        ("git log", "git_log"),
        ("show recent commits", "git_log"),
        ("commit history", "git_log"),

        # --- Git: commit/add ---
        ("stage everything", "git_add"),
        ("git add .", "git_add"),
        ("add all files", "git_add"),
        ("save my work", "git_commit"),
        ("commit with message 'fix bug'", "git_commit"),
        ("commit this", "git_commit"),
        # "stage and commit" is caught by workflow trigger detection in classify(),
        # which runs before the LR model. The LR model sees it as git_commit.
        ("stage and commit with message 'initial'", "git_commit"),

        # --- Git: branch/push/pull ---
        ("push to origin main", "git_push"),
        ("ship it", "git_push"),  # slang — may need more training data if fails
        ("push", "git_push"),
        ("create a branch called feature", "git_branch"),
        ("make a new branch called hotfix", "git_branch"),
        ("switch to the develop branch", "git_branch"),
        ("pull latest", "git_pull"),
        ("git pull", "git_pull"),
        ("fetch and merge from origin", "git_pull"),
        ("merge feature into main", "git_merge"),
        ("clone https://github.com/user/repo.git", "git_clone"),
        ("git init", "git_init"),
        ("initialize a git repo", "git_init"),

        # --- Out of domain ---
        ("hello", "__out_of_domain__"),
        ("build me an app", "__out_of_domain__"),
        ("what's the weather like", "__out_of_domain__"),
        ("tell me a joke", "__out_of_domain__"),
        ("thanks", "__out_of_domain__"),
        ("explain how async works", "__out_of_domain__"),
        ("write unit tests for the API", "__out_of_domain__"),
        ("deploy to production", "__out_of_domain__"),
        ("how does branching work in git", "__out_of_domain__"),
        ("what's a good database for this", "__out_of_domain__"),
        ("I'm stuck", "__out_of_domain__"),
        ("nice work", "__out_of_domain__"),
        ("set up CI/CD", "__out_of_domain__"),
        ("design a REST API", "__out_of_domain__"),
        ("how do I cook pasta", "__out_of_domain__"),
    ]

    correct = 0
    for phrase, expected in user_phrases:
        emb = np.array(list(model.embed([phrase.lower()])))
        pred = clf.predict(emb)[0]
        prob = clf.predict_proba(emb).max()
        ok = pred == expected
        if ok:
            correct += 1
        status = "OK" if ok else "FAIL"
        print(f"  {status}  {pred:<25} ({prob:.2f})  \"{phrase}\"" + (f"  expected={expected}" if not ok else ""))

    print(f"\nUser phrases: {correct}/{len(user_phrases)} ({correct/len(user_phrases)*100:.1f}%)")


if __name__ == "__main__":
    main()
