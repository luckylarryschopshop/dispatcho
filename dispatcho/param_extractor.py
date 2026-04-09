"""Parameter extraction — learned from training data, zero regex.

For each intent, builds a vocabulary of "frame words" from training examples.
At extract time, subtracts frame words — what's left are parameters.

Structural detection (no regex):
- Quoted strings: split on quote characters
- URLs: startswith http:// https:// git@
- Paths: contains /
- Dotfiles: starts with . followed by letters
- Navigation: exactly ".." or "." or "~"
"""
from __future__ import annotations

from collections import Counter


def build_frame_vocab(intents: dict) -> dict[str, set[str]]:
    """Build per-intent frame vocabulary from training data."""
    frames: dict[str, set[str]] = {}

    for intent_name, intent_data in intents.items():
        word_counts = Counter()
        slot_words = set()

        for example in intent_data["examples"]:
            text = example["text"].lower()
            slots = example.get("slots", {})

            for slot_val in slots.values():
                for w in str(slot_val).lower().replace("/", " ").replace(".", " ").split():
                    if w:
                        slot_words.add(w)

            resolved = text
            for sn, sv in slots.items():
                resolved = resolved.replace("{" + sn + "}", sv, 1)

            for w in resolved.split():
                w_clean = w.lstrip("'\"").rstrip("'\".,!?")
                is_slot = False
                for sv in slots.values():
                    if w_clean in str(sv).lower() or str(sv).lower() in w_clean:
                        is_slot = True
                        break
                if not is_slot:
                    word_counts[w_clean] += 1

        frame = set()
        for word, count in word_counts.items():
            if count >= 2 and word not in slot_words and len(word) > 0:
                frame.add(word)

        frame.update({
            "a", "an", "the", "in", "at", "to", "for", "with", "from",
            "and", "or", "of", "on", "is", "it", "my", "me", "your",
            "i", "you", "we", "can", "could", "would", "please", "just",
            "hey", "yo", "go", "ahead", "want", "need", "like", "i'd",
        })

        frames[intent_name] = frame

    return frames


def _extract_quoted(text: str) -> list[str]:
    """Extract quoted strings — no regex, just split on quotes.

    Filters out contractions: "it's", "don't", "isn't" etc.
    A contraction produces a fragment starting with s/t/re/ve/ll/d.
    """
    _contraction_starts = {"s ", "t ", "re ", "ve ", "ll ", "d ", "s,", "t,", "s.", "t."}
    results = []
    for quote_char in ("'", '"'):
        parts = text.split(quote_char)
        for i in range(1, len(parts), 2):
            candidate = parts[i].strip()
            if not candidate:
                continue
            # Skip contraction fragments: it'S roadmap → "s roadmap"
            if quote_char == "'" and any(candidate.lower().startswith(c) for c in _contraction_starts):
                continue
            results.append(candidate)
    return results


def _extract_urls(words: list[str]) -> list[str]:
    """Extract URLs — startswith check, no regex."""
    return [w for w in words if w.startswith(("http://", "https://", "git@"))]


def _is_path_token(word: str) -> bool:
    """Is this word a file/directory path? No regex — structural checks only."""
    w = word.strip("'\".,!?")
    if not w:
        return False
    # Contains slash → definitely a path
    if "/" in w:
        return True
    # Dotfile: starts with . followed by letters (not just . or ..)
    if w.startswith(".") and len(w) > 2 and w[1:].replace("_", "").replace("-", "").replace(".", "").isalnum():
        return True
    # Has file extension: word.ext where ext is 1-4 chars
    if "." in w:
        parts = w.rsplit(".", 1)
        if len(parts) == 2 and 1 <= len(parts[1]) <= 4 and parts[1].isalnum():
            return True
    return False


_NAV_ALIASES = {"parent": "..", "back": "..", "up": "..", "home": "~", "root": "/"}


def _is_nav_token(word: str) -> bool:
    """Is this a navigation token (.., ., ~) or a nav alias (parent, back, up)?"""
    w = word.strip().lower()
    return w in ("..", ".", "~") or w in _NAV_ALIASES


def extract_params(
    intent: str,
    user_text: str,
    slot_names: list[str],
    frame_vocab: dict[str, set[str]],
) -> dict:
    """Extract parameters by subtracting frame vocabulary from user text."""
    if not slot_names:
        return {}

    text = user_text.strip()
    words = text.split()

    # Step 1: High-confidence structural extractions
    quoted = _extract_quoted(text)
    urls = _extract_urls(words)
    nav_tokens = [_NAV_ALIASES.get(w.strip().lower(), w.strip()) for w in words if _is_nav_token(w)]

    # For path extraction, exclude words inside quoted strings
    quoted_words = set()
    for q in quoted:
        for qw in q.split():
            quoted_words.add(qw.lower())
    path_tokens = []
    for w in words:
        cleaned = w.lstrip("'\"").rstrip("'\".,!?")
        if _is_path_token(w) and cleaned.lower() not in quoted_words:
            path_tokens.append(cleaned)

    # Step 2: Subtract frame words
    frame = frame_vocab.get(intent, set())
    remaining = []
    for w in words:
        w_clean = w.lstrip("'\"").rstrip("'\".,!?").lower()
        if _is_nav_token(w_clean):
            remaining.append(w_clean)
        elif _is_path_token(w):
            remaining.append(w.lstrip("'\"").rstrip("'\".,!?"))
        elif w_clean not in frame and len(w_clean) > 0:
            remaining.append(w_clean)

    # Step 3: Assign to slots based on intent
    params = {}

    if intent in ("list_files", "read_file", "delete_file", "create_dir", "change_dir", "edit_file"):
        if "path" in slot_names:
            if nav_tokens:
                params["path"] = nav_tokens[0]
            elif path_tokens:
                params["path"] = path_tokens[-1]
            elif remaining:
                params["path"] = remaining[-1]

    elif intent == "create_file":
        if "content" in slot_names and quoted:
            params["content"] = quoted[0]
        if "path" in slot_names:
            # For create_file, look for path BEFORE content markers
            # Split text at "with content", "with", "containing" to isolate the path part
            text_lower = text.lower()
            path_part = text
            for marker in ("with content ", " containing ", " with '", ' with "', " - "):
                if marker in text_lower:
                    idx = text_lower.index(marker)
                    path_part = text[:idx]
                    break
            # First: check for "called X" or "named X" pattern — most explicit
            path_part_lower = path_part.lower()
            for keyword in ("called ", "named "):
                if keyword in path_part_lower:
                    after = path_part[path_part_lower.index(keyword) + len(keyword):].strip()
                    first_word = after.split()[0].lstrip("'\"").rstrip("'\".,!?") if after else ""
                    if first_word and (_is_path_token(first_word) or "." in first_word):
                        params["path"] = first_word
                        break

            # Fallback: extract paths from the path part
            if "path" not in params:
                path_part_words = path_part.split()
                path_candidates = [w.lstrip("'\"").rstrip("'\".,!?") for w in path_part_words if _is_path_token(w)]
                if path_candidates:
                    params["path"] = path_candidates[-1]

            if "path" not in params:
                # Last resort: subtract frame words from path part
                path_frame = frame_vocab.get(intent, set())
                path_remaining = []
                for w in path_part.split():
                    wc = w.lstrip("'\"").rstrip("'\".,!?").lower()
                    if wc not in path_frame and len(wc) > 0:
                        path_remaining.append(w.lstrip("'\"").rstrip("'\".,!?"))
                if path_remaining:
                    # Last non-frame word before content marker is the filename
                    params["path"] = path_remaining[-1]

    elif intent in ("move_file", "copy_file"):
        if path_tokens and len(path_tokens) >= 2:
            params["source"] = path_tokens[0]
            params["destination"] = path_tokens[1]
        elif remaining:
            try:
                to_idx = [r.lower() for r in remaining].index("to")
                src_parts = remaining[:to_idx]
                dst_parts = remaining[to_idx + 1:]
                if src_parts:
                    params["source"] = src_parts[-1]
                if dst_parts:
                    params["destination"] = dst_parts[0]
            except ValueError:
                if len(remaining) >= 2:
                    params["source"] = remaining[0]
                    params["destination"] = remaining[-1]

    elif intent == "search_files":
        if "pattern" in slot_names:
            if quoted:
                params["pattern"] = quoted[0]
            elif remaining:
                params["pattern"] = " ".join(remaining)

    elif intent == "git_commit":
        if "message" in slot_names and quoted:
            params["message"] = quoted[0]

    elif intent == "git_branch":
        if "branch_name" in slot_names:
            if quoted:
                params["branch_name"] = quoted[0]
            elif remaining:
                params["branch_name"] = remaining[-1]

    elif intent == "git_push":
        if remaining:
            if len(remaining) >= 2:
                params["remote"] = remaining[0]
                params["branch"] = remaining[1]
            elif len(remaining) == 1:
                params["branch"] = remaining[0]

    elif intent == "git_pull":
        if remaining:
            if len(remaining) >= 2:
                params["remote"] = remaining[0]
                params["branch"] = remaining[1]

    elif intent == "git_clone":
        if "url" in slot_names and urls:
            params["url"] = urls[0]

    return params
