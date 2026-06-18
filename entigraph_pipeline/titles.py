"""Title inference helpers."""

from __future__ import annotations

import re


def infer_title_from_text(text: str, index: int, *, max_chars: int = 80) -> str:
    """Infer a readable title from document text when no title field exists."""
    for line in text.splitlines():
        candidate = clean_title_candidate(line)
        if candidate and not line.lstrip().startswith("#"):
            candidate = first_sentence(candidate)
        if candidate:
            return truncate_title(candidate, max_chars=max_chars)

    compact = " ".join(text.split())
    if compact:
        sentence = re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)[0]
        candidate = clean_title_candidate(sentence)
        if candidate:
            return truncate_title(candidate, max_chars=max_chars)
    return f"Untitled source {index}"


def clean_title_candidate(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^\s{0,3}#{1,6}\s+", "", value)
    value = re.sub(r"^\s*[-*+]\s+", "", value)
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n-:;,.")
    return value


def truncate_title(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    clipped = value[: max_chars - 1].rsplit(" ", 1)[0].strip()
    if len(clipped) < 20:
        clipped = value[: max_chars - 1].strip()
    return clipped.rstrip(".,;:") + "..."


def first_sentence(value: str) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", value, maxsplit=1)[0]
    return sentence.rstrip(" .!?") or value
