#!/usr/bin/env python3
"""Summarize overlap in the synthetic wiki fixture.

This helper is intentionally dependency-free. It can inspect the fixture's
embedded entities or a pipeline entity-cache JSONL and can also summarize one
or more generated JSONL outputs for quick offline comparisons.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any


DEFAULT_FIXTURE = Path(__file__).with_name("wiki_fixture.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare single-doc and cross-doc entity opportunities in the wiki fixture."
    )
    parser.add_argument("fixture", nargs="?", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--entity-cache",
        type=Path,
        help="Optional EntiGraph entity cache JSONL; overrides fixture entities by doc_id.",
    )
    parser.add_argument(
        "--combo-sizes",
        default="2,3",
        help="Comma-separated single-document combo sizes to count.",
    )
    parser.add_argument(
        "--generated",
        action="append",
        type=Path,
        default=[],
        help="Optional generated JSONL output to summarize. Repeat for multiple files.",
    )
    args = parser.parse_args()

    combo_sizes = parse_combo_sizes(args.combo_sizes)
    docs = load_docs(args.fixture)
    if args.entity_cache:
        overlay_entities(docs, args.entity_cache)

    print(f"fixture: {args.fixture}")
    print(f"documents: {len(docs)}")
    print()
    print_single_doc_counts(docs, combo_sizes)
    print()
    print_cross_doc_overlaps(docs)

    for path in args.generated:
        print()
        print_generated_summary(path)

    return 0


def load_docs(path: Path) -> list[dict[str, Any]]:
    docs = []
    for line_number, row in read_jsonl(path):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_number} is missing non-empty text")
        doc_id = first_string(row, "id", "doc_id", "document_id") or f"row-{line_number - 1}"
        title = first_string(row, "title") or f"Document {line_number - 1}"
        entities = normalize_entities(row.get("entities", []))
        docs.append({"id": doc_id, "title": title, "entities": entities, "row": row})
    return docs


def overlay_entities(docs: list[dict[str, Any]], entity_cache: Path) -> None:
    by_id = {doc["id"]: doc for doc in docs}
    for _, row in read_jsonl(entity_cache):
        doc_id = row.get("doc_id")
        if isinstance(doc_id, str) and doc_id in by_id:
            by_id[doc_id]["entities"] = normalize_entities(row.get("entities", []))


def print_single_doc_counts(docs: list[dict[str, Any]], combo_sizes: tuple[int, ...]) -> None:
    print("single-document candidates")
    total_by_size = Counter()
    for doc in docs:
        entities = doc["entities"]
        counts = {size: comb(len(entities), size) if len(entities) >= size else 0 for size in combo_sizes}
        for size, count in counts.items():
            total_by_size[size] += count
        count_text = ", ".join(f"{size}-way={counts[size]}" for size in combo_sizes)
        print(f"- {doc['id']}: entities={len(entities)}; {count_text}")
    total_text = ", ".join(f"{size}-way={total_by_size[size]}" for size in combo_sizes)
    print(f"total: {total_text}")


def print_cross_doc_overlaps(docs: list[dict[str, Any]]) -> None:
    print("cross-document overlap candidates")
    by_entity: dict[str, list[str]] = defaultdict(list)
    display_names: dict[str, str] = {}
    for doc in docs:
        for entity in doc["entities"]:
            key = entity.casefold()
            by_entity[key].append(doc["id"])
            display_names.setdefault(key, entity)

    shared = {
        key: sorted(set(doc_ids))
        for key, doc_ids in by_entity.items()
        if len(set(doc_ids)) > 1
    }
    if not shared:
        print("- no shared entities")
        return

    for key in sorted(shared, key=lambda value: display_names[value].casefold()):
        print(f"- {display_names[key]}: {', '.join(shared[key])}")

    doc_pair_counts = Counter()
    for doc_ids in shared.values():
        for left, right in itertools.combinations(doc_ids, 2):
            doc_pair_counts[(left, right)] += 1
    print("document pairs with shared entities:")
    for (left, right), count in sorted(doc_pair_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"- {left} <-> {right}: shared_entities={count}")


def print_generated_summary(path: Path) -> None:
    rows = [row for _, row in read_jsonl(path)]
    single_rows = [row for row in rows if row.get("generation_mode") in {None, "single_doc"} and "relation_id" in row]
    cross_rows = [row for row in rows if row.get("generation_mode") == "cross_doc" or "graph_id" in row]
    failed_rows = [row for row in rows if row.get("error")]
    by_size = Counter(row.get("combo_size", "missing") for row in single_rows)
    by_doc = Counter(str(row.get("doc_id", "missing")) for row in single_rows)
    unique_combos = {
        tuple(str(entity) for entity in row.get("entities", []))
        for row in rows
        if isinstance(row.get("entities"), list)
    }
    cross_shared = Counter(row.get("shared_entity_count", "missing") for row in cross_rows)
    cross_pairs = {
        tuple(str(doc_id) for doc_id in row.get("doc_ids", []))
        for row in cross_rows
        if isinstance(row.get("doc_ids"), list)
    }
    print(f"generated: {path}")
    print(f"- rows: {len(rows)}")
    print(f"- failed rows: {len(failed_rows)}")
    print(f"- single-doc rows: {len(single_rows)}")
    print(f"- cross-doc rows: {len(cross_rows)}")
    if single_rows:
        print(f"- single-doc combo sizes: {format_counter(by_size)}")
        print(f"- single-doc docs: {format_counter(by_doc)}")
        print(f"- unique entity combos: {len(unique_combos)}")
    if cross_rows:
        print(f"- cross-doc shared entity counts: {format_counter(cross_shared)}")
        print(f"- unique cross-doc pairs: {len(cross_pairs)}")


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append((line_number, value))
    return rows


def normalize_entities(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    entities = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        entity = " ".join(item.split()).strip(" -:;,.\t\r\n")
        key = entity.casefold()
        if entity and key not in seen:
            seen.add(key)
            entities.append(entity)
    return entities


def parse_combo_sizes(value: str) -> tuple[int, ...]:
    sizes = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        size = int(part)
        if size <= 0:
            raise argparse.ArgumentTypeError("combo sizes must be positive integers")
        sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one combo size is required")
    return tuple(sizes)


def first_string(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def format_counter(counter: Counter[Any]) -> str:
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter, key=str)) or "none"


if __name__ == "__main__":
    raise SystemExit(main())
