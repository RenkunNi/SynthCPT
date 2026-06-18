#!/usr/bin/env python3
"""Compare generated SynthCPT outputs with dependency-free quality signals.

The helper is intentionally lightweight: it only uses the Python standard
library and accepts any generated JSONL that follows the current EntiGraph
single-doc or cross-doc row schema. Pass files as either PATH or LABEL=PATH.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_FIXTURE = Path(__file__).with_name("wiki_fixture.jsonl")
DEFAULT_SAMPLE = Path(__file__).with_name("generated") / "wiki_synth_offline_sample.jsonl"

RELATION_TERMS = {
    "associated",
    "because",
    "between",
    "borrowed",
    "caused",
    "connected",
    "connects",
    "context",
    "contrasts",
    "described",
    "describes",
    "difference",
    "inspired",
    "interaction",
    "linked",
    "relationship",
    "shared",
    "similar",
    "together",
    "translated",
}

SECTION_WORDS = {
    "Article",
    "Context",
    "Cross",
    "Discussion",
    "Document",
    "Entities",
    "Interaction",
    "Integrated",
    "Relation",
    "Relations",
    "Shared",
    "Synthesis",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare generated outputs using local schema and source-grounding quality signals."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="Source JSONL fixture or corpus.")
    parser.add_argument(
        "--entity-cache",
        type=Path,
        help="Optional entity cache JSONL; overrides fixture entities by doc_id.",
    )
    parser.add_argument(
        "generated",
        nargs="*",
        help="Generated output as PATH or LABEL=PATH. Defaults to the offline sample.",
    )
    args = parser.parse_args()

    sources = load_sources(args.fixture)
    if args.entity_cache:
        overlay_entities(sources, args.entity_cache)
    specs = [parse_generated_spec(value) for value in args.generated] or [("offline-sample", DEFAULT_SAMPLE)]
    summaries = [summarize_generation(label, path, sources) for label, path in specs]

    print(render_markdown(summaries))
    return 0


def load_sources(path: Path) -> dict[str, dict[str, Any]]:
    sources = {}
    for index, row in read_jsonl(path):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        doc_id = first_string(row, "id", "doc_id", "document_id") or str(index)
        sources[doc_id] = {
            "doc_id": doc_id,
            "title": first_string(row, "title") or f"Document {index}",
            "text": text,
            "entities": normalize_entities(row.get("entities", [])),
        }
    return sources


def overlay_entities(sources: dict[str, dict[str, Any]], entity_cache: Path) -> None:
    for _, row in read_jsonl(entity_cache):
        doc_id = row.get("doc_id")
        entities = row.get("entities")
        if isinstance(doc_id, str) and doc_id in sources and isinstance(entities, list):
            sources[doc_id]["entities"] = normalize_entities(entities)


def summarize_generation(label: str, path: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [row for _, row in read_jsonl(path)]
    scored = [score_row(row, sources) for row in rows]
    rows_by_mode = Counter(row["mode"] for row in scored)
    warning_counts = Counter(warning for row in scored for warning in row["warnings"])
    duplicate_texts = duplicate_count(normalize_text(str(row.get("text", ""))) for row in rows)
    return {
        "label": label,
        "path": path,
        "rows": len(rows),
        "rows_by_mode": rows_by_mode,
        "errors": sum(1 for row in rows if row.get("error")),
        "duplicate_texts": duplicate_texts,
        "entity_support": mean(row["entity_support"] for row in scored),
        "entity_mention": mean(row["entity_mention"] for row in scored),
        "unsupported_proper_nouns": mean(row["unsupported_proper_nouns"] for row in scored),
        "source_5gram_overlap": mean(row["source_5gram_overlap"] for row in scored),
        "relation_signal": mean(row["relation_signal"] for row in scored),
        "length_tokens": mean(row["length_tokens"] for row in scored),
        "warnings": warning_counts,
    }


def score_row(row: dict[str, Any], sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mode = generation_mode(row)
    doc_ids = row_doc_ids(row)
    source_docs = [sources[doc_id] for doc_id in doc_ids if doc_id in sources]
    source_text = "\n".join(f"{doc['title']}\n{doc['text']}" for doc in source_docs)
    source_entities = sorted({entity for doc in source_docs for entity in doc["entities"]}, key=str.casefold)
    entities = row_entities(row, mode)
    text = str(row.get("text", ""))
    unsupported = unsupported_proper_nouns(text, source_text, entities, source_entities)
    warnings = []
    if row.get("error"):
        warnings.append("row has error")
    if len(source_docs) != len(doc_ids):
        warnings.append("missing source document")
    entity_support = selected_entity_support(entities, source_docs, mode)
    entity_mention = mention_rate(entities, text)
    source_overlap = max_ngram_overlap(text, [doc["text"] for doc in source_docs], n=5)
    relation_signal = relation_signal_score(text, entities)
    if entity_support < 0.95:
        warnings.append("unsupported selected entity")
    if entity_mention < 0.8:
        warnings.append("selected entity omitted")
    if unsupported:
        warnings.append("unsupported proper noun")
    if source_overlap > 0.65:
        warnings.append("high source overlap")
    if not text.strip():
        warnings.append("empty generated text")
    return {
        "mode": mode,
        "entity_support": entity_support,
        "entity_mention": entity_mention,
        "unsupported_proper_nouns": len(unsupported),
        "source_5gram_overlap": source_overlap,
        "relation_signal": relation_signal,
        "length_tokens": len(tokenize(text)),
        "warnings": warnings,
    }


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Generation Quality Comparison",
        "",
        "| method | rows | modes | errors | entity support | entity mention | unsupported nouns | source 5-gram overlap | relation signal | avg tokens | duplicate texts |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        modes = ", ".join(f"{key}:{summary['rows_by_mode'][key]}" for key in sorted(summary["rows_by_mode"])) or "none"
        lines.append(
            "| {label} | {rows} | {modes} | {errors} | {entity_support:.3f} | {entity_mention:.3f} | "
            "{unsupported_proper_nouns:.2f} | {source_5gram_overlap:.3f} | {relation_signal:.3f} | "
            "{length_tokens:.1f} | {duplicate_texts} |".format(modes=modes, **summary)
        )
    lines.extend(["", "## Warning Counts", ""])
    for summary in summaries:
        lines.append(f"### {summary['label']}")
        if summary["warnings"]:
            for warning, count in sorted(summary["warnings"].items()):
                lines.append(f"- {warning}: {count}")
        else:
            lines.append("- none")
        lines.append(f"- path: {summary['path']}")
        lines.append("")
    lines.extend(
        [
            "## Reading The Signals",
            "",
            "- Higher entity support, entity mention, and relation signal are better.",
            "- Lower unsupported nouns, source 5-gram overlap, errors, and duplicate texts are better.",
            "- Use the same command with new SoG-lite graph-path or LongFaith-style QA JSONL outputs once they exist.",
        ]
    )
    return "\n".join(lines)


def selected_entity_support(entities: list[str], source_docs: list[dict[str, Any]], mode: str) -> float:
    if not entities:
        return 0.0
    supported = 0
    for entity in entities:
        if mode == "cross_doc":
            found = bool(source_docs) and all(entity_supported(entity, doc) for doc in source_docs)
        else:
            found = any(entity_supported(entity, doc) for doc in source_docs)
        if found:
            supported += 1
    return supported / len(entities)


def entity_supported(entity: str, source_doc: dict[str, Any]) -> bool:
    key = normalize_text(entity)
    entity_keys = {normalize_text(value) for value in source_doc.get("entities", [])}
    return key in entity_keys or key in normalize_text(str(source_doc.get("text", "")))


def mention_rate(entities: list[str], text: str) -> float:
    if not entities:
        return 0.0
    normalized = normalize_text(text)
    return sum(1 for entity in entities if normalize_text(entity) in normalized) / len(entities)


def unsupported_proper_nouns(
    generated_text: str,
    source_text: str,
    selected_entities: list[str],
    source_entities: list[str],
) -> list[str]:
    allowed = {normalize_text(entity) for entity in selected_entities + source_entities}
    allowed.update({normalize_text(phrase) for phrase in extract_proper_noun_phrases(source_text)})
    unsupported = []
    seen = set()
    for phrase in extract_proper_noun_phrases(generated_text):
        key = normalize_text(phrase)
        if not key or key in seen or key in allowed or phrase in SECTION_WORDS:
            continue
        if all(part in SECTION_WORDS for part in phrase.split()):
            continue
        seen.add(key)
        unsupported.append(phrase)
    return unsupported


def extract_proper_noun_phrases(text: str) -> list[str]:
    phrases = []
    pattern = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:[ \t]+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b")
    for match in pattern.finditer(text):
        phrase = match.group(0).strip()
        if phrase in SECTION_WORDS:
            continue
        if " " not in phrase and phrase not in {"LLM", "CPT"}:
            continue
        phrases.append(phrase)
    return phrases


def max_ngram_overlap(generated_text: str, source_texts: list[str], *, n: int) -> float:
    generated = ngrams(tokenize(generated_text), n)
    if not generated:
        return 0.0
    return max((len(generated & ngrams(tokenize(source), n)) / len(generated) for source in source_texts), default=0.0)


def relation_signal_score(text: str, entities: list[str]) -> float:
    tokens = set(tokenize(text))
    term_hits = len(tokens & RELATION_TERMS)
    return min(1.0, (term_hits / 4.0) * 0.6 + mention_rate(entities, text) * 0.4)


def generation_mode(row: dict[str, Any]) -> str:
    if row.get("generation_mode") == "sog_lite" or "path_id" in row:
        return "sog_lite"
    if row.get("generation_mode") == "cross_doc" or "graph_id" in row:
        return "cross_doc"
    return "single_doc"


def row_doc_ids(row: dict[str, Any]) -> list[str]:
    doc_ids = row.get("doc_ids")
    if isinstance(doc_ids, list):
        return [str(value) for value in doc_ids]
    doc_id = row.get("doc_id")
    return [str(doc_id)] if doc_id is not None else []


def row_entities(row: dict[str, Any], mode: str) -> list[str]:
    key = "shared_entities" if mode == "cross_doc" else "entities"
    return normalize_entities(row.get(key, []))


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
        entity = re.sub(r"\s+", " ", item).strip(" \t\r\n-:;,.")
        key = normalize_text(entity)
        if entity and key not in seen:
            seen.add(key)
            entities.append(entity)
    return entities


def parse_generated_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError("generated label must not be empty")
        return label, Path(path)
    path = Path(value)
    return path.stem, path


def first_string(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def duplicate_count(values: Any) -> int:
    counter = Counter(value for value in values if value)
    return sum(count - 1 for count in counter.values() if count > 1)


def mean(values: Any) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


if __name__ == "__main__":
    raise SystemExit(main())
