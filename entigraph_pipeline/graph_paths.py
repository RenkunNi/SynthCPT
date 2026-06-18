"""Section-level graph paths for SoG-lite generation."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SectionNode:
    section_id: str
    doc_id: str
    source_index: int
    doc_title: str
    section_title: str
    text: str
    entities: tuple[str, ...]


def split_markdown_sections(
    *,
    doc_id: str,
    source_index: int,
    doc_title: str,
    text: str,
    entities: tuple[str, ...],
    max_section_chars: int,
) -> list[SectionNode]:
    raw_sections = markdown_sections(text, fallback_title=doc_title)
    nodes = []
    for index, (title, body) in enumerate(raw_sections):
        section_text = body.strip()
        if not section_text:
            continue
        section_entities = tuple(entity for entity in entities if entity_in_text(entity, section_text))
        if not section_entities:
            continue
        nodes.append(
            SectionNode(
                section_id=f"{doc_id}#s{index}",
                doc_id=doc_id,
                source_index=source_index,
                doc_title=doc_title,
                section_title=title,
                text=truncate_text(section_text, max_section_chars),
                entities=section_entities,
            )
        )
    if nodes:
        return nodes

    fallback_text = truncate_text(text.strip(), max_section_chars)
    fallback_entities = tuple(entity for entity in entities if entity_in_text(entity, fallback_text))
    if fallback_text and fallback_entities:
        return [
            SectionNode(
                section_id=f"{doc_id}#s0",
                doc_id=doc_id,
                source_index=source_index,
                doc_title=doc_title,
                section_title=doc_title,
                text=fallback_text,
                entities=fallback_entities,
            )
        ]
    return []


def markdown_sections(text: str, *, fallback_title: str) -> list[tuple[str, str]]:
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$", flags=re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
    if not matches:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        return [(first_sentence(part, fallback_title), part) for part in paragraphs]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        prefix = text[: matches[0].start()].strip()
        if prefix:
            sections.append((fallback_title, prefix))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = clean_heading(match.group(2))
        body = text[start:end].strip()
        if body:
            sections.append((title, f"{title}\n{body}"))
    return sections


def build_section_nodes(
    docs: Iterable[object],
    entity_records: dict[str, object],
    *,
    max_section_chars: int,
) -> list[SectionNode]:
    nodes: list[SectionNode] = []
    for doc in docs:
        record = entity_records.get(getattr(doc, "doc_id"))
        if record is None or getattr(record, "error", ""):
            continue
        nodes.extend(
            split_markdown_sections(
                doc_id=getattr(doc, "doc_id"),
                source_index=getattr(doc, "source_index"),
                doc_title=getattr(doc, "title"),
                text=getattr(doc, "text"),
                entities=tuple(getattr(record, "entities", ())),
                max_section_chars=max_section_chars,
            )
        )
    return nodes


def sample_graph_paths(
    nodes: list[SectionNode],
    *,
    path_length: int,
    max_paths: int | None,
) -> list[tuple[SectionNode, ...]]:
    if path_length <= 1:
        return [(node,) for node in nodes[:max_paths]]
    neighbors = build_neighbors(nodes)
    paths: list[tuple[SectionNode, ...]] = []
    for start in sorted(nodes, key=lambda node: node.section_id):
        extend_paths((start,), neighbors, paths, path_length)
        if max_paths is not None and len(paths) >= max_paths:
            return paths[:max_paths]
    return paths


def extend_paths(
    path: tuple[SectionNode, ...],
    neighbors: dict[str, list[SectionNode]],
    paths: list[tuple[SectionNode, ...]],
    target_length: int,
) -> None:
    if len(path) >= target_length:
        paths.append(path)
        return
    seen_sections = {node.section_id for node in path}
    seen_docs = {node.doc_id for node in path}
    for neighbor in neighbors.get(path[-1].section_id, []):
        if neighbor.section_id in seen_sections:
            continue
        # Prefer paths that expand context rather than staying inside one page.
        if len(seen_docs) == 1 and neighbor.doc_id in seen_docs and len(neighbors[path[-1].section_id]) > 1:
            continue
        extend_paths(path + (neighbor,), neighbors, paths, target_length)


def build_neighbors(nodes: list[SectionNode]) -> dict[str, list[SectionNode]]:
    by_entity: dict[str, list[SectionNode]] = {}
    for node in nodes:
        for entity in node.entities:
            by_entity.setdefault(entity_key(entity), []).append(node)
    neighbors: dict[str, dict[str, SectionNode]] = {node.section_id: {} for node in nodes}
    for entity_nodes in by_entity.values():
        for left, right in itertools.combinations(entity_nodes, 2):
            if left.section_id == right.section_id:
                continue
            neighbors[left.section_id][right.section_id] = right
            neighbors[right.section_id][left.section_id] = left
    return {
        section_id: sorted(values.values(), key=lambda node: (node.doc_id, node.section_id))
        for section_id, values in neighbors.items()
    }


def path_entities(path: tuple[SectionNode, ...]) -> tuple[str, ...]:
    entities = []
    seen = set()
    for node in path:
        for entity in node.entities:
            key = entity_key(entity)
            if key in seen:
                continue
            seen.add(key)
            entities.append(entity)
    return tuple(entities)


def shared_path_entities(path: tuple[SectionNode, ...]) -> tuple[str, ...]:
    counts: dict[str, tuple[str, int]] = {}
    for node in path:
        for entity in node.entities:
            key = entity_key(entity)
            surface, count = counts.get(key, (entity, 0))
            counts[key] = (surface if len(surface) >= len(entity) else entity, count + 1)
    return tuple(surface for surface, count in counts.values() if count > 1)


def entity_in_text(entity: str, text: str) -> bool:
    return entity_key(entity) in entity_key(text)


def entity_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def clean_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n#-:;,.")


def first_sentence(value: str, fallback: str) -> str:
    compact = " ".join(value.split())
    if not compact:
        return fallback
    return re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)[0].strip(" .!?") or fallback


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 1].rsplit(" ", 1)[0].strip()
    if len(clipped) < max_chars // 2:
        clipped = text[: max_chars - 1].strip()
    return clipped.rstrip() + "..."
