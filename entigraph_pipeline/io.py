"""JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path, *, strict: bool = True) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if not strict:
                    continue
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                if not strict:
                    continue
                raise ValueError(f"Expected JSON object on {path}:{line_number}")
            yield value


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_completed_keys(path: str | Path, key: str) -> set[str]:
    file_path = Path(path)
    if not file_path.exists():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(file_path, strict=False):
        if row.get("error"):
            continue
        value = row.get(key)
        if isinstance(value, str):
            completed.add(value)
    return completed
