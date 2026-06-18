"""Quality checks for generated EntiGraph data."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import append_jsonl, read_jsonl
from .titles import infer_title_from_text


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "was",
    "were",
    "with",
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


@dataclass(frozen=True)
class EvaluationConfig:
    input_path: Path
    generated_path: Path
    output_dir: Path
    entity_cache_path: Path | None = None
    text_key: str = "text"
    title_key: str | None = "title"
    id_key: str | None = None
    min_overall_score: float = 0.75
    max_unsupported_proper_nouns: int = 2
    max_redundancy_overlap: float = 0.65


class EntiGraphEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def run(self) -> dict[str, Any]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        sources = self.load_sources()
        generated_rows = list(read_jsonl(self.config.generated_path, strict=False))
        row_path = self.config.output_dir / "rows.jsonl"
        probe_path = self.config.output_dir / "probes.jsonl"
        for path in (row_path, probe_path):
            if path.exists():
                path.unlink()

        evaluations = []
        probes = []
        for index, row in enumerate(generated_rows):
            evaluation = self.evaluate_row(index, row, sources)
            evaluations.append(evaluation)
            probes.extend(evaluation.pop("probes"))
            append_jsonl(row_path, [evaluation])
        append_jsonl(probe_path, probes)

        summary = summarize_evaluations(evaluations)
        summary.update(
            {
                "input_path": str(self.config.input_path),
                "generated_path": str(self.config.generated_path),
                "output_dir": str(self.config.output_dir),
                "thresholds": {
                    "min_overall_score": self.config.min_overall_score,
                    "max_unsupported_proper_nouns": self.config.max_unsupported_proper_nouns,
                    "max_redundancy_overlap": self.config.max_redundancy_overlap,
                },
            }
        )
        (self.config.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.config.output_dir / "report.md").write_text(render_report(summary, evaluations), encoding="utf-8")
        return summary

    def load_sources(self) -> dict[str, dict[str, Any]]:
        sources: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(read_jsonl(self.config.input_path)):
            text = row.get(self.config.text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            doc_id = source_doc_id(row, index, text, self.config.id_key)
            title = source_title(row, index, self.config.title_key, text)
            entities = normalize_entity_list(row.get("entities", []))
            sources[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "entities": entities,
            }
        if self.config.entity_cache_path and self.config.entity_cache_path.exists():
            for row in read_jsonl(self.config.entity_cache_path, strict=False):
                doc_id = row.get("doc_id")
                entities = row.get("entities")
                if isinstance(doc_id, str) and doc_id in sources and isinstance(entities, list):
                    sources[doc_id]["entities"] = normalize_entity_list(entities)
        return sources

    def evaluate_row(
        self,
        index: int,
        row: dict[str, Any],
        sources: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        mode = generation_mode(row)
        doc_ids = row_doc_ids(row)
        source_docs = [sources[doc_id] for doc_id in doc_ids if doc_id in sources]
        source_text = "\n".join(f"{doc['title']}\n{doc['text']}" for doc in source_docs)
        source_entities = sorted({entity for doc in source_docs for entity in doc["entities"]}, key=str.casefold)
        selected_entities = row_entities(row)
        text = str(row.get("text", ""))
        warnings: list[str] = []
        if row.get("error"):
            warnings.append("generated row has error")
        if len(source_docs) != len(doc_ids):
            warnings.append("one or more source documents were not found")
        if not text.strip():
            warnings.append("generated text is empty")

        selected_support = selected_entity_support(selected_entities, source_docs, mode)
        selected_mention = mention_rate(selected_entities, text)
        unsupported = unsupported_proper_nouns(text, source_text, selected_entities, source_entities)
        ngram_overlap = max_ngram_overlap(text, [doc["text"] for doc in source_docs], n=5)
        redundancy_score = clamp01(1.0 - max(0.0, ngram_overlap - 0.25) / 0.5)
        relation_score = relation_signal_score(text, selected_entities)
        specificity = specificity_score(text, selected_entities)
        structure = structure_score(text)
        length = length_score(text)
        unsupported_score = clamp01(1.0 - len(unsupported) / 5.0)
        overall = round(
            0.22 * selected_support
            + 0.18 * selected_mention
            + 0.18 * unsupported_score
            + 0.12 * redundancy_score
            + 0.12 * relation_score
            + 0.08 * specificity
            + 0.05 * structure
            + 0.05 * length,
            4,
        )
        pass_gate = (
            overall >= self.config.min_overall_score
            and selected_support >= 0.95
            and selected_mention >= 0.8
            and len(unsupported) <= self.config.max_unsupported_proper_nouns
            and ngram_overlap <= self.config.max_redundancy_overlap
            and not row.get("error")
            and bool(text.strip())
        )
        if selected_support < 0.95:
            warnings.append("selected entities are not fully supported by source documents")
        if selected_mention < 0.8:
            warnings.append("generated text omits selected entities")
        if unsupported:
            warnings.append("generated text contains unsupported proper nouns")
        if ngram_overlap > self.config.max_redundancy_overlap:
            warnings.append("generated text is too close to source text")

        return {
            "row_index": index,
            "generated_id": str(row.get("relation_id") or row.get("graph_id") or index),
            "generation_mode": mode,
            "doc_ids": doc_ids,
            "selected_entities": selected_entities,
            "scores": {
                "overall": overall,
                "selected_entity_support": round(selected_support, 4),
                "selected_entity_mention": round(selected_mention, 4),
                "unsupported_proper_noun_score": round(unsupported_score, 4),
                "redundancy_score": round(redundancy_score, 4),
                "source_5gram_overlap": round(ngram_overlap, 4),
                "relation_signal": round(relation_score, 4),
                "specificity": round(specificity, 4),
                "structure": round(structure, 4),
                "length": round(length, 4),
            },
            "pass_gate": pass_gate,
            "warnings": warnings,
            "unsupported_proper_nouns": unsupported,
            "probes": build_probes(row, selected_entities, doc_ids, mode),
        }


def selected_entity_support(
    selected_entities: list[str],
    source_docs: list[dict[str, Any]],
    mode: str,
) -> float:
    if not selected_entities:
        return 0.0
    supported = 0
    for entity in selected_entities:
        if mode == "cross_doc":
            found = all(entity_supported(entity, doc) for doc in source_docs)
        else:
            found = any(entity_supported(entity, doc) for doc in source_docs)
        if found:
            supported += 1
    return supported / len(selected_entities)


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
    allowed.update(extract_proper_noun_keys(source_text))
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


def extract_proper_noun_keys(text: str) -> set[str]:
    return {normalize_text(phrase) for phrase in extract_proper_noun_phrases(text)}


def extract_proper_noun_phrases(text: str) -> list[str]:
    phrases = []
    pattern = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:[ \t]+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b")
    for match in pattern.finditer(text):
        phrase = match.group(0).strip()
        if phrase in SECTION_WORDS:
            continue
        # Ignore a lone sentence-initial capitalized common word; multi-word spans carry more signal.
        if " " not in phrase and phrase not in {"LLM", "CPT"}:
            continue
        phrases.append(phrase)
    return phrases


def max_ngram_overlap(generated_text: str, source_texts: list[str], *, n: int) -> float:
    generated = ngrams(tokenize(generated_text), n)
    if not generated:
        return 0.0
    max_overlap = 0.0
    for source in source_texts:
        source_ngrams = ngrams(tokenize(source), n)
        if not source_ngrams:
            continue
        max_overlap = max(max_overlap, len(generated & source_ngrams) / len(generated))
    return max_overlap


def relation_signal_score(text: str, entities: list[str]) -> float:
    tokens = set(tokenize(text))
    term_hits = len(tokens & RELATION_TERMS)
    entity_mentions = mention_rate(entities, text)
    return clamp01((term_hits / 4.0) * 0.6 + entity_mentions * 0.4)


def specificity_score(text: str, entities: list[str]) -> float:
    tokens = [token for token in tokenize(text) if token not in STOPWORDS]
    if not tokens:
        return 0.0
    entity_token_count = sum(len(tokenize(entity)) for entity in entities if normalize_text(entity) in normalize_text(text))
    density = entity_token_count / max(1, len(tokens))
    return clamp01(density * 8.0)


def structure_score(text: str) -> float:
    if not text.strip():
        return 0.0
    headings = len(re.findall(r"^###\s+", text, flags=re.MULTILINE))
    paragraphs = len([part for part in text.splitlines() if part.strip()])
    return clamp01(0.5 * min(headings, 3) / 3.0 + 0.5 * min(paragraphs, 4) / 4.0)


def length_score(text: str) -> float:
    count = len(tokenize(text))
    if count < 20:
        return count / 20.0
    if count <= 700:
        return 1.0
    return max(0.0, 1.0 - (count - 700) / 700.0)


def build_probes(
    row: dict[str, Any],
    entities: list[str],
    doc_ids: list[str],
    mode: str,
) -> list[dict[str, Any]]:
    generated_id = str(row.get("relation_id") or row.get("graph_id") or "")
    probes = []
    if mode == "cross_doc":
        for entity in entities:
            probes.append(
                {
                    "generated_id": generated_id,
                    "probe_type": "cross_doc_bridge",
                    "question": f"Which source documents are connected by the shared entity {entity}?",
                    "expected_answer": ", ".join(doc_ids),
                    "entity": entity,
                }
            )
    else:
        for entity in entities:
            probes.append(
                {
                    "generated_id": generated_id,
                    "probe_type": "single_doc_entity",
                    "question": f"Which generated passage discusses {entity} in relation to the source document?",
                    "expected_answer": str(row.get("title") or doc_ids[0] if doc_ids else ""),
                    "entity": entity,
                }
            )
    return probes


def summarize_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = Counter(row["generation_mode"] for row in evaluations)
    pass_count = sum(1 for row in evaluations if row["pass_gate"])
    score_keys = [
        "overall",
        "selected_entity_support",
        "selected_entity_mention",
        "unsupported_proper_noun_score",
        "redundancy_score",
        "source_5gram_overlap",
        "relation_signal",
        "specificity",
        "structure",
        "length",
    ]
    score_summary = {}
    for key in score_keys:
        values = [row["scores"][key] for row in evaluations]
        score_summary[key] = summarize_values(values)
    warning_counts = Counter(warning for row in evaluations for warning in row["warnings"])
    return {
        "rows": len(evaluations),
        "passed": pass_count,
        "failed": len(evaluations) - pass_count,
        "pass_rate": round(pass_count / len(evaluations), 4) if evaluations else 0.0,
        "rows_by_mode": dict(sorted(by_mode.items())),
        "score_summary": score_summary,
        "warning_counts": dict(sorted(warning_counts.items())),
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(statistics.fmean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def render_report(summary: dict[str, Any], evaluations: list[dict[str, Any]]) -> str:
    lines = [
        "# EntiGraph Evaluation Report",
        "",
        f"- rows: {summary['rows']}",
        f"- passed: {summary['passed']}",
        f"- failed: {summary['failed']}",
        f"- pass_rate: {summary['pass_rate']}",
        f"- rows_by_mode: {summary['rows_by_mode']}",
        "",
        "## Score Summary",
        "",
    ]
    for name, values in summary["score_summary"].items():
        lines.append(f"- {name}: mean={values['mean']}, min={values['min']}, max={values['max']}")
    if summary["warning_counts"]:
        lines.extend(["", "## Warnings", ""])
        for warning, count in summary["warning_counts"].items():
            lines.append(f"- {warning}: {count}")
    failed = [row for row in evaluations if not row["pass_gate"]]
    if failed:
        lines.extend(["", "## Failed Rows", ""])
        for row in failed[:25]:
            lines.append(
                f"- {row['generated_id']}: overall={row['scores']['overall']}, warnings={'; '.join(row['warnings'])}"
            )
    return "\n".join(lines) + "\n"


def generation_mode(row: dict[str, Any]) -> str:
    mode = row.get("generation_mode")
    if mode == "sog_lite" or "path_id" in row:
        return "sog_lite"
    if mode == "cross_doc" or "graph_id" in row:
        return "cross_doc"
    return "single_doc"


def row_doc_ids(row: dict[str, Any]) -> list[str]:
    doc_ids = row.get("doc_ids")
    if isinstance(doc_ids, list):
        return [str(value) for value in doc_ids]
    doc_id = row.get("doc_id")
    return [str(doc_id)] if doc_id is not None else []


def row_entities(row: dict[str, Any]) -> list[str]:
    key = "shared_entities" if generation_mode(row) == "cross_doc" else "entities"
    return normalize_entity_list(row.get(key, []))


def normalize_entity_list(value: Any) -> list[str]:
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


def source_doc_id(row: dict[str, Any], index: int, text: str, id_key: str | None) -> str:
    if id_key and row.get(id_key) is not None:
        return str(row[id_key])
    for key in ("id", "doc_id", "document_id"):
        if row.get(key) is not None:
            return str(row[key])
    return str(index)


def source_title(row: dict[str, Any], index: int, title_key: str | None, text: str) -> str:
    if title_key and isinstance(row.get(title_key), str) and row[title_key].strip():
        return row[title_key].strip()
    return infer_title_from_text(text, index)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
