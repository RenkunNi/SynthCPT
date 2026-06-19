"""Entity-level context graph construction."""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .graph_paths import SectionNode, build_section_nodes
from .io import read_jsonl
from .pipeline import EntityRecord, SourceDocument, entity_normal_key, sha256_text, stable_id
from .titles import infer_title_from_text


RELATION_CATEGORIES = (
    "created_or_designed_by",
    "authored_or_translated_by",
    "located_in",
    "part_of",
    "used_for",
    "inspired_by",
    "causes_or_enables",
    "compares_or_contrasts",
    "temporal",
    "associated_with",
    "other",
)


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        ...


@dataclass(frozen=True)
class ContextGraphConfig:
    input_path: Path
    entity_cache_path: Path
    output_path: Path
    text_key: str = "text"
    title_key: str | None = "title"
    id_key: str | None = None
    max_section_chars: int = 1800
    min_shared_contexts: int = 1
    min_edge_weight: float = 1.0
    max_evidence_per_fact: int = 4
    max_contexts_per_entity: int = 20
    include_context_text: bool = True
    graph_name: str = "entity_context_graph"
    typed_fact_mode: str = "heuristic"
    min_typed_confidence: float = 0.4
    max_typed_facts_per_context: int = 12
    typed_fact_temperature: float = 0.0
    typed_fact_max_tokens: int | None = 2048
    json_mode: bool = False


class ContextGraphBuilder:
    def __init__(self, config: ContextGraphConfig, client: ChatClient | None = None):
        self.config = config
        self.client = client

    def run(self) -> dict[str, Any]:
        if self.config.typed_fact_mode not in {"off", "heuristic", "llm"}:
            raise ValueError("typed_fact_mode must be one of: off, heuristic, llm")
        if self.config.typed_fact_mode == "llm" and self.client is None:
            raise ValueError("typed_fact_mode='llm' requires a chat client")
        docs = self.load_documents()
        entity_records = self.load_entity_records()
        nodes = build_section_nodes(
            docs,
            entity_records,
            max_section_chars=self.config.max_section_chars,
        )
        graph = self.build_graph(docs, nodes)
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.output_path.write_text(
            json.dumps(graph, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {
            "graph_id": graph["graph_id"],
            "documents": len(docs),
            "contexts": len(graph["contexts"]),
            "entities": len(graph["entities"]),
            "facts": len(graph["facts"]),
            "cooccurrence_facts": graph["summary"]["cooccurrence_facts"],
            "typed_facts": graph["summary"]["typed_facts"],
            "output_path": str(self.config.output_path),
        }

    def load_documents(self) -> list[SourceDocument]:
        docs = []
        for index, row in enumerate(read_jsonl(self.config.input_path)):
            text = row.get(self.config.text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            doc_id = source_doc_id(row, index, text, self.config.id_key)
            title = source_title(row, index, self.config.title_key, text)
            docs.append(
                SourceDocument(
                    doc_id=doc_id,
                    source_index=index,
                    title=title,
                    text=text.strip(),
                    source_sha256=sha256_text(text),
                    raw=row,
                )
            )
        return docs

    def load_entity_records(self) -> dict[str, EntityRecord]:
        records = {}
        for row in read_jsonl(self.config.entity_cache_path, strict=False):
            doc_id = row.get("doc_id")
            entities = row.get("entities")
            if not isinstance(doc_id, str) or not isinstance(entities, list):
                continue
            records[doc_id] = EntityRecord(
                doc_id=doc_id,
                source_index=int(row.get("source_index", -1)),
                title=str(row.get("title", "")),
                source_sha256=str(row.get("source_sha256", "")),
                summary=str(row.get("summary", "")),
                entities=tuple(str(entity) for entity in entities),
                raw_response=str(row.get("raw_response", "")),
                error=str(row.get("error", "")),
            )
        return records

    def build_graph(self, docs: list[SourceDocument], nodes: list[SectionNode]) -> dict[str, Any]:
        graph_id = self.graph_id(docs)
        entity_stats = build_entity_stats(nodes, max_contexts=self.config.max_contexts_per_entity)
        contexts = [context_node(node, include_text=self.config.include_context_text) for node in nodes]
        cooccurrence_facts = self.build_cooccurrence_facts(nodes)
        typed_facts = self.build_typed_facts(nodes)
        facts = cooccurrence_facts + typed_facts
        return {
            "graph_id": graph_id,
            "graph_name": self.config.graph_name,
            "schema_version": "entity-context-graph-v1",
            "build_config": {
                "max_section_chars": self.config.max_section_chars,
                "min_shared_contexts": self.config.min_shared_contexts,
                "min_edge_weight": self.config.min_edge_weight,
                "max_evidence_per_fact": self.config.max_evidence_per_fact,
                "max_contexts_per_entity": self.config.max_contexts_per_entity,
                "include_context_text": self.config.include_context_text,
                "typed_fact_mode": self.config.typed_fact_mode,
                "min_typed_confidence": self.config.min_typed_confidence,
                "max_typed_facts_per_context": self.config.max_typed_facts_per_context,
                "relation_categories": list(RELATION_CATEGORIES),
            },
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "source_index": doc.source_index,
                    "title": doc.title,
                    "source_sha256": doc.source_sha256,
                }
                for doc in docs
            ],
            "entities": entity_stats,
            "contexts": contexts,
            "facts": facts,
            "summary": {
                "documents": len(docs),
                "entities": len(entity_stats),
                "contexts": len(contexts),
                "facts": len(facts),
                "cooccurrence_facts": len(cooccurrence_facts),
                "typed_facts": len(typed_facts),
            },
        }

    def graph_id(self, docs: list[SourceDocument]) -> str:
        source_key = "|".join(f"{doc.doc_id}:{doc.source_sha256}" for doc in docs)
        config_key = json.dumps(
            {
                "max_section_chars": self.config.max_section_chars,
                "min_shared_contexts": self.config.min_shared_contexts,
                "min_edge_weight": self.config.min_edge_weight,
                "max_evidence_per_fact": self.config.max_evidence_per_fact,
                "include_context_text": self.config.include_context_text,
                "typed_fact_mode": self.config.typed_fact_mode,
                "min_typed_confidence": self.config.min_typed_confidence,
                "max_typed_facts_per_context": self.config.max_typed_facts_per_context,
            },
            sort_keys=True,
        )
        return stable_id("context-graph", source_key, config_key)

    def build_cooccurrence_facts(self, nodes: list[SectionNode]) -> list[dict[str, Any]]:
        evidence_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
        surface_by_key: dict[str, str] = {}
        for node in nodes:
            entity_keys = []
            for entity in node.entities:
                key = entity_normal_key(entity)
                if not key:
                    continue
                entity_keys.append(key)
                surface_by_key[key] = choose_surface(surface_by_key.get(key), entity)
            for left, right in itertools.combinations(sorted(set(entity_keys)), 2):
                snippet = evidence_snippet(node.text, surface_by_key[left], surface_by_key[right])
                evidence_by_pair.setdefault((left, right), []).append(
                    {
                        "context_id": node.section_id,
                        "doc_id": node.doc_id,
                        "source_sha256": node.source_sha256,
                        "section_title": node.section_title,
                        "evidence": snippet,
                    }
                )

        facts = []
        for (left_key, right_key), evidence in sorted(evidence_by_pair.items()):
            context_ids = sorted({item["context_id"] for item in evidence})
            doc_ids = sorted({item["doc_id"] for item in evidence})
            if len(context_ids) < self.config.min_shared_contexts:
                continue
            weight = fact_weight(context_count=len(context_ids), doc_count=len(doc_ids))
            if weight < self.config.min_edge_weight:
                continue
            head = surface_by_key[left_key]
            tail = surface_by_key[right_key]
            facts.append(
                {
                    "fact_id": stable_id("context-fact", left_key, "co_occurs_with", right_key, "|".join(context_ids)),
                    "head": head,
                    "head_id": entity_id(head),
                    "relation": "co_occurs_with",
                    "tail": tail,
                    "tail_id": entity_id(tail),
                    "edge_type": "contextual_cooccurrence",
                    "weight": round(weight, 4),
                    "context_count": len(context_ids),
                    "doc_count": len(doc_ids),
                    "context_ids": context_ids,
                    "doc_ids": doc_ids,
                    "evidence": evidence[: self.config.max_evidence_per_fact],
                }
            )
        return facts

    def build_typed_facts(self, nodes: list[SectionNode]) -> list[dict[str, Any]]:
        if self.config.typed_fact_mode == "off":
            return []
        raw_facts: list[dict[str, Any]] = []
        for node in nodes:
            if self.config.typed_fact_mode == "llm":
                raw_facts.extend(self.extract_llm_typed_facts(node))
            else:
                raw_facts.extend(extract_heuristic_typed_facts(node, self.config.max_typed_facts_per_context))
        return merge_typed_facts(
            raw_facts,
            max_evidence_per_fact=self.config.max_evidence_per_fact,
            min_confidence=self.config.min_typed_confidence,
        )

    def extract_llm_typed_facts(self, node: SectionNode) -> list[dict[str, Any]]:
        assert self.client is not None
        messages = [
            {"role": "system", "content": typed_fact_system_prompt()},
            {"role": "user", "content": typed_fact_user_prompt(node, self.config.max_typed_facts_per_context)},
        ]
        raw = self.client.chat(
            messages,
            temperature=self.config.typed_fact_temperature,
            max_tokens=self.config.typed_fact_max_tokens,
            response_format={"type": "json_object"} if self.config.json_mode else None,
        )
        parsed = parse_json_object(raw)
        facts = parsed.get("facts", [])
        if not isinstance(facts, list):
            return []
        valid_entities = {entity_normal_key(entity): entity for entity in node.entities}
        rows = []
        for fact in facts[: self.config.max_typed_facts_per_context]:
            if not isinstance(fact, dict):
                continue
            head = str(fact.get("head", "")).strip()
            tail = str(fact.get("tail", "")).strip()
            head_key = entity_normal_key(head)
            tail_key = entity_normal_key(tail)
            if not head_key or not tail_key or head_key == tail_key:
                continue
            if head_key not in valid_entities or tail_key not in valid_entities:
                continue
            relation = normalize_relation(str(fact.get("relation", "")))
            category = normalize_category(str(fact.get("relation_category", "")))
            evidence = str(fact.get("evidence", "")).strip() or evidence_snippet(node.text, head, tail)
            confidence = safe_float(fact.get("confidence"), default=0.6)
            rows.append(
                typed_fact_candidate(
                    node=node,
                    head=valid_entities[head_key],
                    relation=relation,
                    relation_category=category,
                    tail=valid_entities[tail_key],
                    evidence=evidence,
                    confidence=confidence,
                    extraction_method="llm",
                )
            )
        return rows


def build_entity_stats(nodes: list[SectionNode], *, max_contexts: int) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for node in nodes:
        seen_in_context = set()
        for entity in node.entities:
            key = entity_normal_key(entity)
            if not key:
                continue
            item = stats.setdefault(
                key,
                {
                    "entity": entity,
                    "entity_id": entity_id(entity),
                    "aliases": set(),
                    "doc_ids": set(),
                    "context_ids": [],
                    "context_frequency": 0,
                },
            )
            item["entity"] = choose_surface(item["entity"], entity)
            item["entity_id"] = entity_id(item["entity"])
            item["aliases"].add(entity)
            item["doc_ids"].add(node.doc_id)
            if key not in seen_in_context:
                item["context_frequency"] += 1
                seen_in_context.add(key)
            if node.section_id not in item["context_ids"] and len(item["context_ids"]) < max_contexts:
                item["context_ids"].append(node.section_id)

    rows = []
    for item in stats.values():
        doc_ids = sorted(item["doc_ids"])
        rows.append(
            {
                "entity": item["entity"],
                "entity_id": item["entity_id"],
                "aliases": sorted(item["aliases"], key=str.casefold),
                "document_frequency": len(doc_ids),
                "context_frequency": item["context_frequency"],
                "doc_ids": doc_ids,
                "context_ids": item["context_ids"],
            }
        )
    return sorted(rows, key=lambda row: (-row["document_frequency"], -row["context_frequency"], row["entity"].casefold()))


def context_node(node: SectionNode, *, include_text: bool) -> dict[str, Any]:
    row = {
        "context_id": node.section_id,
        "doc_id": node.doc_id,
        "source_index": node.source_index,
        "source_sha256": node.source_sha256,
        "doc_title": node.doc_title,
        "section_title": node.section_title,
        "entities": list(node.entities),
        "char_length": len(node.text),
    }
    if include_text:
        row["text"] = node.text
    return row


def extract_heuristic_typed_facts(node: SectionNode, max_facts: int) -> list[dict[str, Any]]:
    facts = []
    entity_surfaces = {entity_normal_key(entity): entity for entity in node.entities if entity_normal_key(entity)}
    for sentence in split_sentences(node.text):
        for fact in extract_pattern_typed_facts(node, sentence, entity_surfaces):
            facts.append(fact)
            if len(facts) >= max_facts:
                return facts
    return facts


def extract_pattern_typed_facts(
    node: SectionNode,
    sentence: str,
    entity_surfaces: dict[str, str],
) -> list[dict[str, Any]]:
    patterns = (
        ("designed by", "designed_by", "created_or_designed_by"),
        ("designed", "designed", "created_or_designed_by"),
        ("translated", "translated", "authored_or_translated_by"),
        ("published", "published_account_of", "authored_or_translated_by"),
        ("expanded", "expanded_with", "authored_or_translated_by"),
        ("associated with", "associated_with", "associated_with"),
        ("built to", "built_to", "used_for"),
    )
    mentions = entity_mentions(sentence, entity_surfaces)
    if len(mentions) < 2:
        return []
    rows = []
    lowered = sentence.casefold()
    used_pairs = set()
    for marker, relation, category in patterns:
        start = lowered.find(marker)
        if start < 0:
            continue
        end = start + len(marker)
        head = subject_before_marker(mentions, start)
        tail = object_after_marker(mentions, end)
        if head is None or tail is None or head["surface"] == tail["surface"]:
            continue
        pair_key = (head["surface"], relation, tail["surface"])
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        rows.append(
            typed_fact_candidate(
                node=node,
                head=head["surface"],
                relation=relation,
                relation_category=category,
                tail=tail["surface"],
                evidence=sentence,
                confidence=0.82,
                extraction_method="heuristic",
            )
        )
    return rows


def entity_mentions(sentence: str, entity_surfaces: dict[str, str]) -> list[dict[str, Any]]:
    mentions = []
    for surface in entity_surfaces.values():
        span = find_entity_span(sentence, surface)
        if span is None:
            continue
        mentions.append({"surface": surface, "start": span[0], "end": span[1]})
    return sorted(mentions, key=lambda item: (item["start"], item["end"]))


def subject_before_marker(mentions: list[dict[str, Any]], marker_start: int) -> dict[str, Any] | None:
    before = [mention for mention in mentions if mention["end"] <= marker_start]
    if not before:
        return None
    return before[0]


def object_after_marker(mentions: list[dict[str, Any]], marker_end: int) -> dict[str, Any] | None:
    after = [mention for mention in mentions if mention["start"] >= marker_end]
    if not after:
        return None
    return after[0]


def entities_in_sentence(sentence: str, entity_surfaces: dict[str, str]) -> list[str]:
    sentence_key = entity_normal_key(sentence)
    mentions = []
    for key, surface in entity_surfaces.items():
        if key and key in sentence_key:
            mentions.append(surface)
    return sorted(mentions, key=lambda entity: sentence_key.find(entity_normal_key(entity)))


def relation_between(sentence: str, head: str, tail: str, mentions: list[str]) -> str:
    head_match = find_entity_span(sentence, head)
    tail_match = find_entity_span(sentence, tail)
    if head_match is None or tail_match is None or head_match[0] >= tail_match[0]:
        return ""
    between = sentence[head_match[1] : tail_match[0]]
    for entity in mentions:
        if entity in {head, tail}:
            continue
        if re.search(re.escape(entity), between, flags=re.IGNORECASE):
            return ""
    return normalize_relation(between)


def find_entity_span(text: str, entity: str) -> tuple[int, int] | None:
    match = re.search(re.escape(entity), text, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    compact_text = entity_normal_key(text)
    compact_entity = entity_normal_key(entity)
    start = compact_text.find(compact_entity)
    if start < 0:
        return None
    return start, start + len(compact_entity)


def normalize_relation(value: str) -> str:
    relation = re.sub(r"[^A-Za-z0-9_ /'-]+", " ", value)
    relation = re.sub(r"\s+", " ", relation).strip(" -_/")
    relation = trim_relation_edges(relation)
    if not relation:
        return "related_to"
    lowered = relation.casefold()
    if "designed by" in lowered:
        return "designed_by"
    if "designed" in lowered:
        return "designed"
    if "translated" in lowered:
        return "translated"
    if "published" in lowered:
        return "published"
    if "inspired" in lowered:
        return "inspired_by"
    if "associated with" in lowered:
        return "associated_with"
    if "built to" in lowered:
        return "built_to"
    if "used to" in lowered or "used for" in lowered:
        return "used_for"
    tokens = relation.split()
    if len(tokens) > 6:
        tokens = tokens[:6]
    return "_".join(token.casefold().strip("'") for token in tokens if token)


def trim_relation_edges(value: str) -> str:
    stop_edges = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "for",
        "in",
        "of",
        "the",
        "to",
        "with",
    }
    tokens = value.split()
    while tokens and tokens[0].casefold() in stop_edges:
        tokens.pop(0)
    while tokens and tokens[-1].casefold() in stop_edges:
        tokens.pop()
    return " ".join(tokens)


def classify_relation(relation: str, sentence: str) -> str:
    relation_text = relation.casefold()
    text = f"{relation} {sentence}".casefold()
    if any(term in relation_text for term in ("after", "before", "later", "during")):
        return "temporal"
    if any(term in text for term in ("designed", "built", "created", "developed", "proposed")):
        return "created_or_designed_by"
    if any(term in text for term in ("translated", "published", "wrote", "notes", "article", "account")):
        return "authored_or_translated_by"
    if any(term in text for term in ("in london", "in turin", "located", "based in", "from ")):
        return "located_in"
    if any(term in text for term in ("part of", "component", "section", "included")):
        return "part_of"
    if any(term in text for term in ("used for", "used to", "built to", "automate", "direct")):
        return "used_for"
    if any(term in text for term in ("inspired by", "inspired", "influence")):
        return "inspired_by"
    if any(term in text for term in ("helped", "caused", "enabled", "allowed", "made")):
        return "causes_or_enables"
    if any(term in text for term in ("similar", "different", "contrast", "compared")):
        return "compares_or_contrasts"
    if any(term in text for term in ("associated", "connected", "linked", "related")):
        return "associated_with"
    return "other"


def normalize_category(value: str) -> str:
    key = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
    return key if key in RELATION_CATEGORIES else "other"


def typed_fact_candidate(
    *,
    node: SectionNode,
    head: str,
    relation: str,
    relation_category: str,
    tail: str,
    evidence: str,
    confidence: float,
    extraction_method: str,
) -> dict[str, Any]:
    return {
        "head": head,
        "head_id": entity_id(head),
        "relation": relation,
        "relation_category": normalize_category(relation_category),
        "tail": tail,
        "tail_id": entity_id(tail),
        "edge_type": "typed_contextual_fact",
        "confidence": clamp(confidence, 0.0, 1.0),
        "context_id": node.section_id,
        "doc_id": node.doc_id,
        "source_sha256": node.source_sha256,
        "section_title": node.section_title,
        "evidence": truncate(evidence, 420),
        "extraction_method": extraction_method,
    }


def merge_typed_facts(
    raw_facts: list[dict[str, Any]],
    *,
    max_evidence_per_fact: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for fact in raw_facts:
        if fact["confidence"] < min_confidence:
            continue
        head_key = entity_normal_key(fact["head"])
        tail_key = entity_normal_key(fact["tail"])
        if not head_key or not tail_key or head_key == tail_key:
            continue
        relation_key = normalize_relation(fact["relation"])
        key = (head_key, relation_key, tail_key, fact["relation_category"])
        grouped.setdefault(key, []).append(fact)

    merged = []
    for (head_key, relation, tail_key, category), facts in sorted(grouped.items()):
        context_ids = sorted({fact["context_id"] for fact in facts})
        doc_ids = sorted({fact["doc_id"] for fact in facts})
        confidence = sum(fact["confidence"] for fact in facts) / len(facts)
        head = choose_surface(None, facts[0]["head"])
        tail = choose_surface(None, facts[0]["tail"])
        evidence = [
            {
                "context_id": fact["context_id"],
                "doc_id": fact["doc_id"],
                "source_sha256": fact["source_sha256"],
                "section_title": fact["section_title"],
                "evidence": fact["evidence"],
                "extraction_method": fact["extraction_method"],
            }
            for fact in facts[:max_evidence_per_fact]
        ]
        merged.append(
            {
                "fact_id": stable_id("typed-context-fact", head_key, relation, tail_key, "|".join(context_ids)),
                "head": head,
                "head_id": entity_id(head),
                "relation": relation,
                "relation_category": category,
                "tail": tail,
                "tail_id": entity_id(tail),
                "edge_type": "typed_contextual_fact",
                "confidence": round(confidence, 4),
                "weight": round(fact_weight(context_count=len(context_ids), doc_count=len(doc_ids)) * confidence, 4),
                "context_count": len(context_ids),
                "doc_count": len(doc_ids),
                "context_ids": context_ids,
                "doc_ids": doc_ids,
                "evidence": evidence,
            }
        )
    return merged


def typed_fact_system_prompt() -> str:
    categories = ", ".join(RELATION_CATEGORIES)
    return f"""Extract typed entity facts from a source context.

Requirements:
1. Use only entities from the provided entity list.
2. Every fact must be directly supported by the context text.
3. Prefer salient semantic facts over vague co-occurrence.
4. Use a concise relation phrase, such as designed_by, translated, inspired_by, used_for, located_in.
5. Choose one relation_category from: {categories}.
6. Include a short evidence sentence copied from or closely matching the context.
7. Do not invent facts or outside knowledge.

Return valid JSON:
{{
  "facts": [
    {{
      "head": "entity from list",
      "relation": "short_relation_phrase",
      "relation_category": "one category",
      "tail": "entity from list",
      "evidence": "supporting sentence",
      "confidence": 0.0
    }}
  ]
}}"""


def typed_fact_user_prompt(node: SectionNode, max_facts: int) -> str:
    entity_lines = "\n".join(f"- {entity}" for entity in node.entities)
    return f"""Document: {node.doc_title}
Section: {node.section_title}
Context ID: {node.section_id}

Entities:
{entity_lines}

Context text:
{node.text}

Extract at most {max_facts} typed facts."""


def parse_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Could not parse typed fact JSON: {raw_response[:500]}")


def safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def source_doc_id(row: dict[str, Any], index: int, text: str, id_key: str | None) -> str:
    if id_key and row.get(id_key) is not None:
        return str(row[id_key])
    for key in ("id", "doc_id", "document_id"):
        if row.get(key) is not None:
            return str(row[key])
    return stable_id(str(index), sha256_text(text))


def source_title(row: dict[str, Any], index: int, title_key: str | None, text: str) -> str:
    if title_key and isinstance(row.get(title_key), str) and row[title_key].strip():
        return row[title_key].strip()
    return infer_title_from_text(text, index)


def choose_surface(current: str | None, candidate: str) -> str:
    if not current:
        return candidate
    if len(candidate) > len(current):
        return candidate
    return current


def entity_id(entity: str) -> str:
    return stable_id("entity", entity_normal_key(entity))


def fact_weight(*, context_count: int, doc_count: int) -> float:
    return context_count + max(0, doc_count - 1) * 0.5


def evidence_snippet(text: str, head: str, tail: str, *, max_chars: int = 420) -> str:
    sentences = split_sentences(text)
    head_key = entity_normal_key(head)
    tail_key = entity_normal_key(tail)
    for sentence in sentences:
        sentence_key = entity_normal_key(sentence)
        if head_key in sentence_key and tail_key in sentence_key:
            return truncate(sentence, max_chars)
    for sentence in sentences:
        sentence_key = entity_normal_key(sentence)
        if head_key in sentence_key or tail_key in sentence_key:
            return truncate(sentence, max_chars)
    return truncate(text, max_chars)


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", compact) if part.strip()]


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 1].rsplit(" ", 1)[0].strip()
    if len(clipped) < max_chars // 2:
        clipped = text[: max_chars - 1].strip()
    return clipped.rstrip() + "..."
