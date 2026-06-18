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
    source_sha256: str
    doc_title: str
    section_title: str
    text: str
    entities: tuple[str, ...]


def split_markdown_sections(
    *,
    doc_id: str,
    source_index: int,
    source_sha256: str,
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
                source_sha256=source_sha256,
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
                source_sha256=source_sha256,
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
                source_sha256=getattr(doc, "source_sha256"),
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
    strategy: str = "dfs",
    candidate_multiplier: int = 8,
) -> list[tuple[SectionNode, ...]]:
    if path_length <= 1:
        return [(node,) for node in nodes[:max_paths]]
    if strategy not in {"dfs", "bridge", "coverage"}:
        raise ValueError("graph path strategy must be one of: dfs, bridge, coverage")
    neighbors = build_neighbors(nodes)
    paths: list[tuple[SectionNode, ...]] = []
    candidate_limit = max_paths
    if strategy != "dfs" and max_paths is not None:
        candidate_limit = max(max_paths, max_paths * max(1, candidate_multiplier))
    for start in sorted(nodes, key=lambda node: node.section_id):
        extend_paths((start,), neighbors, paths, path_length, max_paths=candidate_limit)
        if candidate_limit is not None and len(paths) >= candidate_limit:
            break
    if strategy == "dfs":
        return paths[:max_paths] if max_paths is not None else paths
    ranked = rank_graph_paths(paths, strategy=strategy)
    return ranked[:max_paths] if max_paths is not None else ranked


def extend_paths(
    path: tuple[SectionNode, ...],
    neighbors: dict[str, list[SectionNode]],
    paths: list[tuple[SectionNode, ...]],
    target_length: int,
    max_paths: int | None = None,
) -> None:
    if max_paths is not None and len(paths) >= max_paths:
        return
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
        extend_paths(path + (neighbor,), neighbors, paths, target_length, max_paths=max_paths)


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


def rank_graph_paths(
    paths: list[tuple[SectionNode, ...]],
    *,
    strategy: str,
) -> list[tuple[SectionNode, ...]]:
    if strategy == "bridge":
        return sorted(paths, key=lambda path: (-score_graph_path(path), path_sort_key(path)))
    if strategy == "coverage":
        return sorted(paths, key=lambda path: (-coverage_score(path), path_sort_key(path)))
    return paths


def score_graph_path(path: tuple[SectionNode, ...]) -> float:
    metrics = graph_path_metrics(path)
    return (
        metrics["shared_entity_count"] * 3.0
        + metrics["doc_count"] * 1.5
        + min(metrics["entity_count"], 12) * 0.25
        + metrics["cross_doc_edges"] * 0.75
    )


def coverage_score(path: tuple[SectionNode, ...]) -> float:
    metrics = graph_path_metrics(path)
    return (
        metrics["doc_count"] * 3.0
        + metrics["entity_count"] * 0.7
        + metrics["shared_entity_count"] * 1.5
        + metrics["section_count"] * 0.25
    )


def graph_path_metrics(path: tuple[SectionNode, ...]) -> dict[str, int | float]:
    entities = path_entities(path)
    shared = shared_path_entities(path)
    doc_ids = [node.doc_id for node in path]
    cross_doc_edges = sum(
        1
        for left, right in zip(path, path[1:])
        if left.doc_id != right.doc_id
    )
    return {
        "section_count": len(path),
        "doc_count": len(set(doc_ids)),
        "entity_count": len(entities),
        "shared_entity_count": len(shared),
        "cross_doc_edges": cross_doc_edges,
        "bridge_score": round(score_graph_path_without_metrics(path, entities, shared, cross_doc_edges), 4),
        "coverage_score": round(
            len(set(doc_ids)) * 3.0 + len(entities) * 0.7 + len(shared) * 1.5 + len(path) * 0.25,
            4,
        ),
    }


def score_graph_path_without_metrics(
    path: tuple[SectionNode, ...],
    entities: tuple[str, ...],
    shared: tuple[str, ...],
    cross_doc_edges: int,
) -> float:
    return len(shared) * 3.0 + len({node.doc_id for node in path}) * 1.5 + min(len(entities), 12) * 0.25 + cross_doc_edges * 0.75


def path_sort_key(path: tuple[SectionNode, ...]) -> tuple[str, ...]:
    return tuple(node.section_id for node in path)


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
