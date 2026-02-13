from __future__ import annotations

import re
from typing import Any


_STOPWORDS: set[str] = {
    # Danish function words / common standalone words that we should avoid gluing
    # together with adjacent tokens when repairing extraction artifacts.
    "den",
    "det",
    "de",
    "en",
    "et",
    "til",
    "og",
    "eller",
    "med",
    "ved",
    "fra",
    "for",
    "af",
    "på",
    "som",
    "der",
    "ikke",
    "hvis",
    "men",
    "kan",
    "skal",
    "må",
    "har",
    "have",
}


def normalize_title(value: Any) -> str | None:
    """Normalize extracted titles.

    Handles common extraction artifacts:
    - extra/multiple spaces
    - split words ("Genst and" -> "Genstand")
    - spaced-out letters ("G e n s t a n d" -> "Genstand")

    Keeps casing as-is (only merges tokens).
    """

    if value is None:
        return None

    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    tokens = text.split(" ")

    merged: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            left = tokens[i]
            right = tokens[i + 1]

            left_lower = left.lower()
            right_lower = right.lower()

            if left.isalpha() and right.isalpha() and right[:1].islower():
                left_is_stop = left_lower in _STOPWORDS and len(left_lower) >= 3
                right_is_stop = right_lower in _STOPWORDS and len(right_lower) >= 3

                # Merge when it looks like a single word broken mid-word.
                # Allow up to 10 chars on the left to cover cases like "registrer" + "ede".
                if not left_is_stop and not right_is_stop and 1 <= len(left) <= 10:
                    merged.append(left + right)
                    i += 2
                    continue

        merged.append(tokens[i])
        i += 1

    fixed: list[str] = []
    i = 0
    while i < len(merged):
        if len(merged[i]) == 1 and merged[i].isalpha():
            j = i
            letters: list[str] = []
            while j < len(merged) and len(merged[j]) == 1 and merged[j].isalpha():
                letters.append(merged[j])
                j += 1
            if len(letters) >= 4:
                fixed.append("".join(letters))
                i = j
                continue
        fixed.append(merged[i])
        i += 1

    result = " ".join(fixed).strip()
    return result or None
