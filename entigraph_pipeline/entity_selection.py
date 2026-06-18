"""Entity ranking and selection strategies."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


ENTITY_SELECTION_STRATEGIES = {"llm-order", "importance", "rarity", "hybrid"}


@dataclass(frozen=True)
class EntitySelectionStats:
    extracted_total: int
    selected_total: int
    strategy: str


def select_entities_for_records(
    records: dict[str, object],
    docs_by_id: dict[str, object],
    *,
    strategy: str,
    max_entities: int,
) -> tuple[dict[str, tuple[str, ...]], EntitySelectionStats]:
    if strategy not in ENTITY_SELECTION_STRATEGIES:
        raise ValueError(f"entity_selection_strategy must be one of: {sorted(ENTITY_SELECTION_STRATEGIES)}")
    extracted_total = sum(len(tuple(getattr(record, "entities", ()))) for record in records.values())
    doc_frequency = entity_document_frequency(record.entities for record in records.values())
    selected: dict[str, tuple[str, ...]] = {}
    for doc_id, record in records.items():
        entities = tuple(getattr(record, "entities", ()))
        doc = docs_by_id.get(doc_id)
        text = str(getattr(doc, "text", "")) if doc is not None else ""
        selected[doc_id] = select_entities(
            entities,
            text,
            doc_frequency=doc_frequency,
            total_docs=max(1, len(docs_by_id)),
            strategy=strategy,
            max_entities=max_entities,
        )
    selected_total = sum(len(entities) for entities in selected.values())
    return selected, EntitySelectionStats(extracted_total, selected_total, strategy)


def select_entities(
    entities: tuple[str, ...],
    text: str,
    *,
    doc_frequency: Counter[str],
    total_docs: int,
    strategy: str,
    max_entities: int,
) -> tuple[str, ...]:
    if max_entities <= 0:
        return ()
    if len(entities) <= max_entities:
        return entities
    if strategy == "llm-order":
        return entities[:max_entities]
    scored = []
    for index, entity in enumerate(entities):
        importance = entity_importance(entity, text, index)
        rarity = entity_rarity(entity, doc_frequency, total_docs)
        if strategy == "importance":
            score = importance
        elif strategy == "rarity":
            score = rarity
        elif strategy == "hybrid":
            score = 0.65 * importance + 0.35 * rarity
        else:
            raise ValueError(f"Unknown entity selection strategy: {strategy}")
        scored.append((score, -index, entity))
    chosen = sorted(scored, reverse=True)[:max_entities]
    # Preserve original local order in prompts after ranking selection.
    chosen_keys = {entity_normal_key(entity) for _, _, entity in chosen}
    return tuple(entity for entity in entities if entity_normal_key(entity) in chosen_keys)


def entity_importance(entity: str, text: str, index: int) -> float:
    normalized_text = text.casefold()
    normalized_entity = entity.casefold()
    mentions = count_entity_mentions(normalized_entity, normalized_text)
    if mentions == 0:
        mention_score = 0.0
        early_score = 0.0
    else:
        mention_score = min(1.0, math.log1p(mentions) / math.log(6))
        first = normalized_text.find(normalized_entity)
        early_score = 1.0 - min(1.0, first / max(1, len(normalized_text))) if first >= 0 else 0.0
    llm_order_score = 1.0 / (1.0 + index)
    return 0.7 * mention_score + 0.2 * early_score + 0.1 * llm_order_score


def entity_rarity(entity: str, doc_frequency: Counter[str], total_docs: int) -> float:
    frequency = max(1, doc_frequency.get(entity_normal_key(entity), 1))
    return math.log((total_docs + 1) / frequency) / math.log(total_docs + 1)


def entity_document_frequency(entity_lists: Iterable[tuple[str, ...]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entities in entity_lists:
        seen = {entity_normal_key(entity) for entity in entities if entity_normal_key(entity)}
        counts.update(seen)
    return counts


def count_entity_mentions(entity: str, text: str) -> int:
    if not entity:
        return 0
    pattern = re.compile(rf"(?<!\w){re.escape(entity)}(?!\w)", flags=re.IGNORECASE)
    return len(pattern.findall(text))


def entity_normal_key(entity: str) -> str:
    key = re.sub(r"\s+", " ", entity).strip().casefold()
    return re.sub(r"^[^\w]+|[^\w]+$", "", key)
