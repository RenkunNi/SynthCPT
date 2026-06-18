"""EntiGraph data generation pipeline."""

from __future__ import annotations

import concurrent.futures
import hashlib
import itertools
import json
import math
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .io import append_jsonl, read_completed_keys, read_jsonl
from .llm import OpenAICompatibleClient
from .progress import ProgressBar
from .prompts import (
    CROSS_DOCUMENT_SYSTEM_PROMPT,
    ENTITY_SYSTEM_PROMPT,
    cross_document_user_prompt,
    document_user_prompt,
    relation_system_prompt,
    relation_user_prompt,
)
from .titles import infer_title_from_text


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
class EntiGraphConfig:
    input_path: Path
    output_path: Path
    entity_cache_path: Path
    text_key: str = "text"
    title_key: str | None = "title"
    id_key: str | None = None
    combo_sizes: tuple[int, ...] = (2, 3)
    mode: str = "single-doc"
    max_docs: int | None = None
    max_entities: int = 60
    min_entity_chars: int = 2
    max_combos_per_doc: int | None = None
    sample_combos: bool = False
    cross_doc_min_shared_entities: int = 1
    cross_doc_max_shared_entities: int = 12
    cross_doc_max_pairs: int | None = None
    cross_doc_sample_pairs: bool = False
    random_seed: int = 13
    max_workers: int = 8
    entity_temperature: float = 0.0
    relation_temperature: float = 1.0
    entity_max_tokens: int | None = 2048
    relation_max_tokens: int | None = 2048
    json_mode: bool = False
    resume: bool = True
    max_in_flight: int | None = None
    include_source_text: bool = False
    show_progress: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    source_index: int
    title: str
    text: str
    source_sha256: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class EntityRecord:
    doc_id: str
    source_index: int
    title: str
    source_sha256: str
    summary: str
    entities: tuple[str, ...]
    raw_response: str
    error: str = ""


@dataclass(frozen=True)
class RelationTask:
    doc: SourceDocument
    entities: tuple[str, ...]

    @property
    def relation_id(self) -> str:
        combo = "|".join(self.entities)
        return stable_id("single-doc", self.doc.doc_id, self.doc.source_sha256, str(len(self.entities)), combo)


@dataclass(frozen=True)
class CrossDocumentTask:
    doc_a: SourceDocument
    doc_b: SourceDocument
    shared_entities: tuple[str, ...]

    @property
    def graph_id(self) -> str:
        doc_ids = sorted((self.doc_a.doc_id, self.doc_b.doc_id))
        hashes = {
            self.doc_a.doc_id: self.doc_a.source_sha256,
            self.doc_b.doc_id: self.doc_b.source_sha256,
        }
        entity_key = "|".join(entity_normal_key(entity) for entity in self.shared_entities)
        return stable_id("cross-doc", doc_ids[0], hashes[doc_ids[0]], doc_ids[1], hashes[doc_ids[1]], entity_key)


class EntiGraphPipeline:
    def __init__(self, config: EntiGraphConfig, client: ChatClient):
        self.config = config
        self.client = client
        self._entity_lock = threading.Lock()
        self._output_lock = threading.Lock()

    def run(self) -> dict[str, int]:
        if self.config.mode not in {"single-doc", "cross-doc", "both"}:
            raise ValueError("mode must be one of: single-doc, cross-doc, both")
        docs = list(self.iter_documents())
        entity_records = self.ensure_entities(docs)
        relation_stats = empty_generation_stats("relations")
        cross_doc_stats = empty_generation_stats("cross_doc")
        if self.config.mode in {"single-doc", "both"}:
            relation_total = self.count_relation_tasks(docs, entity_records) if self.config.show_progress else None
            relation_stats = self._run_generation_tasks(
                self.iter_relation_tasks(docs, entity_records),
                completed_key="relation_id",
                task_key=lambda task: task.relation_id,
                generate=lambda task: self.generate_relation(task),
                prefix="relations",
                total=relation_total,
            )
        if self.config.mode in {"cross-doc", "both"}:
            cross_doc_total = self.count_cross_document_tasks(docs, entity_records) if self.config.show_progress else None
            cross_doc_stats = self._run_generation_tasks(
                self.iter_cross_document_tasks(docs, entity_records),
                completed_key="graph_id",
                task_key=lambda task: task.graph_id,
                generate=lambda task: self.generate_cross_document(task),
                prefix="cross_doc",
                total=cross_doc_total,
            )
        return {
            "documents": len(docs),
            "entity_records": len(entity_records),
            "entity_errors": sum(1 for record in entity_records.values() if record.error),
            **relation_stats,
            **cross_doc_stats,
        }

    def _run_generation_tasks(
        self,
        tasks: Iterable[Any],
        *,
        completed_key: str,
        task_key: Callable[[Any], str],
        generate: Callable[[Any], dict[str, Any]],
        prefix: str,
        total: int | None = None,
    ) -> dict[str, int]:
        completed = read_completed_keys(self.config.output_path, completed_key) if self.config.resume else set()
        max_in_flight = self.config.max_in_flight or max(1, self.config.max_workers * 4)
        label = "generate cross-doc" if prefix == "cross_doc" else "generate relations"
        with ProgressBar(label, total, enabled=self.config.show_progress) as progress:
            stats = self._run_generation_tasks_with_progress(
                tasks,
                completed,
                task_key,
                generate,
                prefix,
                progress,
                max_in_flight,
            )
        return stats

    def _run_generation_tasks_with_progress(
        self,
        tasks: Iterable[Any],
        completed: set[str],
        task_key: Callable[[Any], str],
        generate: Callable[[Any], dict[str, Any]],
        prefix: str,
        progress: ProgressBar,
        max_in_flight: int,
    ) -> dict[str, int]:
        skipped = 0
        considered = 0
        submitted = 0
        generated = 0
        failed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            pending: set[concurrent.futures.Future[dict[str, Any]]] = set()

            def drain_one() -> None:
                nonlocal generated, failed
                done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    pending.remove(future)
                    row = future.result()
                    if row.get("error"):
                        failed += 1
                    else:
                        generated += 1
                    with self._output_lock:
                        append_jsonl(self.config.output_path, [row])
                    progress.update()

            for task in tasks:
                considered += 1
                if task_key(task) in completed:
                    skipped += 1
                    progress.update()
                    continue
                while len(pending) >= max_in_flight:
                    drain_one()
                pending.add(pool.submit(generate, task))
                submitted += 1

            while pending:
                drain_one()

        return {
            f"{prefix}_considered": considered,
            f"{prefix}_tasks": submitted,
            f"{prefix}_skipped": skipped,
            f"{prefix}_generated": generated,
            f"{prefix}_failed": failed,
        }

    def iter_documents(self) -> Iterable[SourceDocument]:
        count = 0
        for index, row in enumerate(read_jsonl(self.config.input_path)):
            if self.config.max_docs is not None and count >= self.config.max_docs:
                break
            text = row.get(self.config.text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            doc_id = self._doc_id(row, index, text)
            title = self._title(row, index, text)
            yield SourceDocument(
                doc_id=doc_id,
                source_index=index,
                title=title,
                text=text.strip(),
                source_sha256=sha256_text(text),
                raw=row,
            )
            count += 1

    def ensure_entities(self, docs: list[SourceDocument]) -> dict[str, EntityRecord]:
        records = self._load_entity_cache()
        missing = [
            doc
            for doc in docs
            if doc.doc_id not in records
            or records[doc.doc_id].source_sha256 != doc.source_sha256
            or bool(records[doc.doc_id].error)
        ]
        if not missing:
            return records

        with ProgressBar("extract entities", len(missing), enabled=self.config.show_progress) as progress:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                futures = {pool.submit(self.extract_entities, doc): doc for doc in missing}
                for future in concurrent.futures.as_completed(futures):
                    doc = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = EntityRecord(
                            doc_id=doc.doc_id,
                            source_index=doc.source_index,
                            title=doc.title,
                            source_sha256=doc.source_sha256,
                            summary="",
                            entities=(),
                            raw_response="",
                            error=str(exc),
                        )
                    records[record.doc_id] = record
                    with self._entity_lock:
                        append_jsonl(self.config.entity_cache_path, [entity_record_to_row(record)])
                    progress.update()
        return records

    def extract_entities(self, doc: SourceDocument) -> EntityRecord:
        response_format = {"type": "json_object"} if self.config.json_mode else None
        messages = [
            {"role": "system", "content": ENTITY_SYSTEM_PROMPT},
            {"role": "user", "content": document_user_prompt(doc.text, doc.title)},
        ]
        raw_response = self.client.chat(
            messages,
            temperature=self.config.entity_temperature,
            max_tokens=self.config.entity_max_tokens,
            response_format=response_format,
        )
        try:
            parsed = parse_entity_response(raw_response)
        except ValueError:
            repair_messages = messages + [
                {"role": "assistant", "content": raw_response},
                {
                    "role": "user",
                    "content": 'Return only valid JSON with keys "summary" and "entities". Do not include markdown.',
                },
            ]
            raw_response = self.client.chat(
                repair_messages,
                temperature=0.0,
                max_tokens=self.config.entity_max_tokens,
                response_format=response_format,
            )
            parsed = parse_entity_response(raw_response)
        entities = normalize_entities(
            parsed.get("entities", []),
            max_entities=self.config.max_entities,
            min_chars=self.config.min_entity_chars,
        )
        return EntityRecord(
            doc_id=doc.doc_id,
            source_index=doc.source_index,
            title=doc.title,
            source_sha256=doc.source_sha256,
            summary=str(parsed.get("summary", "")).strip(),
            entities=tuple(entities),
            raw_response=raw_response,
        )

    def iter_relation_tasks(
        self,
        docs: list[SourceDocument],
        entity_records: dict[str, EntityRecord],
    ) -> Iterable[RelationTask]:
        rng = random.Random(self.config.random_seed)
        docs_by_id = {doc.doc_id: doc for doc in docs}
        for doc in docs:
            record = entity_records.get(doc.doc_id)
            if record is None:
                continue
            doc_id = record.doc_id
            doc = docs_by_id.get(doc_id)
            if doc is None:
                continue
            combo_iter = iter_entity_combos(record.entities, self.config.combo_sizes)
            if self.config.max_combos_per_doc is not None:
                if self.config.sample_combos:
                    combo_iter = iter(
                        reservoir_sample(combo_iter, self.config.max_combos_per_doc, rng)
                    )
                else:
                    combo_iter = itertools.islice(combo_iter, self.config.max_combos_per_doc)
            for combo in combo_iter:
                yield RelationTask(doc=doc, entities=combo)

    def count_relation_tasks(
        self,
        docs: list[SourceDocument],
        entity_records: dict[str, EntityRecord],
    ) -> int:
        total = 0
        for doc in docs:
            record = entity_records.get(doc.doc_id)
            if record is None:
                continue
            count = 0
            for size in self.config.combo_sizes:
                if size <= 0 or len(record.entities) < size:
                    continue
                count += math.comb(len(record.entities), size)
            if self.config.max_combos_per_doc is not None:
                count = min(count, self.config.max_combos_per_doc)
            total += count
        return total

    def iter_cross_document_tasks(
        self,
        docs: list[SourceDocument],
        entity_records: dict[str, EntityRecord],
    ) -> Iterable[CrossDocumentTask]:
        docs_by_id = {doc.doc_id: doc for doc in docs}
        shared_by_pair: dict[tuple[str, str], dict[str, str]] = {}
        entity_index: dict[str, list[tuple[str, str]]] = {}
        for record in entity_records.values():
            if record.error or record.doc_id not in docs_by_id:
                continue
            seen_in_doc: set[str] = set()
            for entity in record.entities:
                key = entity_normal_key(entity)
                if not key or key in seen_in_doc:
                    continue
                seen_in_doc.add(key)
                entity_index.setdefault(key, []).append((record.doc_id, entity))

        for key, mentions in entity_index.items():
            if len(mentions) < 2:
                continue
            for left, right in itertools.combinations(mentions, 2):
                doc_a, surface_a = left
                doc_b, surface_b = right
                if doc_a == doc_b:
                    continue
                pair_key = tuple(sorted((doc_a, doc_b)))
                shared = shared_by_pair.setdefault(pair_key, {})
                shared.setdefault(key, choose_entity_surface(surface_a, surface_b))

        tasks = (
            CrossDocumentTask(
                doc_a=docs_by_id[pair_key[0]],
                doc_b=docs_by_id[pair_key[1]],
                shared_entities=tuple(sorted(shared.values(), key=str.casefold))[
                    : self.config.cross_doc_max_shared_entities
                ],
            )
            for pair_key, shared in sorted(shared_by_pair.items())
            if len(shared) >= self.config.cross_doc_min_shared_entities
        )
        if self.config.cross_doc_max_pairs is None:
            yield from tasks
        elif self.config.cross_doc_sample_pairs:
            rng = random.Random(self.config.random_seed)
            yield from reservoir_sample(tasks, self.config.cross_doc_max_pairs, rng)
        else:
            yield from itertools.islice(tasks, self.config.cross_doc_max_pairs)

    def count_cross_document_tasks(
        self,
        docs: list[SourceDocument],
        entity_records: dict[str, EntityRecord],
    ) -> int:
        return sum(1 for _ in self.iter_cross_document_tasks(docs, entity_records))

    def generate_relation(self, task: RelationTask) -> dict[str, Any]:
        system_prompt = relation_system_prompt(len(task.entities))
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": relation_user_prompt(task.doc.text, task.doc.title, task.entities),
            },
        ]
        base = {
            "relation_id": task.relation_id,
            "doc_id": task.doc.doc_id,
            "source_index": task.doc.source_index,
            "source_sha256": task.doc.source_sha256,
            "title": task.doc.title,
            "combo_size": len(task.entities),
            "entities": list(task.entities),
            "prompt_name": f"entigraph_relation_{len(task.entities)}",
            "generation_mode": "single_doc",
            "task_type": "entity_relation",
            **self.config.metadata,
        }
        if self.config.include_source_text:
            base["source_text"] = task.doc.text
        try:
            text = self.client.chat(
                messages,
                temperature=self.config.relation_temperature,
                max_tokens=self.config.relation_max_tokens,
            )
            return {**base, "text": text}
        except Exception as exc:
            return {**base, "text": "", "error": str(exc)}

    def generate_cross_document(self, task: CrossDocumentTask) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": CROSS_DOCUMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": cross_document_user_prompt(
                    task.doc_a.title,
                    task.doc_a.text,
                    task.doc_b.title,
                    task.doc_b.text,
                    task.shared_entities,
                ),
            },
        ]
        base = {
            "graph_id": task.graph_id,
            "doc_ids": [task.doc_a.doc_id, task.doc_b.doc_id],
            "source_indices": [task.doc_a.source_index, task.doc_b.source_index],
            "source_sha256": [task.doc_a.source_sha256, task.doc_b.source_sha256],
            "titles": [task.doc_a.title, task.doc_b.title],
            "shared_entities": list(task.shared_entities),
            "shared_entity_count": len(task.shared_entities),
            "prompt_name": "cross_document_graph_pair",
            "generation_mode": "cross_doc",
            "task_type": "cross_document_graph",
            **self.config.metadata,
        }
        if self.config.include_source_text:
            base["source_texts"] = [task.doc_a.text, task.doc_b.text]
        try:
            text = self.client.chat(
                messages,
                temperature=self.config.relation_temperature,
                max_tokens=self.config.relation_max_tokens,
            )
            return {**base, "text": text}
        except Exception as exc:
            return {**base, "text": "", "error": str(exc)}

    def _load_entity_cache(self) -> dict[str, EntityRecord]:
        path = self.config.entity_cache_path
        if not self.config.resume or not path.exists():
            return {}
        records: dict[str, EntityRecord] = {}
        for row in read_jsonl(path, strict=False):
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

    def _doc_id(self, row: dict[str, Any], index: int, text: str) -> str:
        if self.config.id_key:
            value = row.get(self.config.id_key)
            if value is not None:
                return str(value)
        for key in ("id", "doc_id", "document_id"):
            value = row.get(key)
            if value is not None:
                return str(value)
        return stable_id(str(index), sha256_text(text))

    def _title(self, row: dict[str, Any], index: int, text: str) -> str:
        if self.config.title_key:
            value = row.get(self.config.title_key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return infer_title_from_text(text, index)


def entity_record_to_row(record: EntityRecord) -> dict[str, Any]:
    row = {
        "doc_id": record.doc_id,
        "source_index": record.source_index,
        "source_sha256": record.source_sha256,
        "title": record.title,
        "summary": record.summary,
        "entities": list(record.entities),
        "raw_response": record.raw_response,
    }
    if record.error:
        row["error"] = record.error
    return row


def parse_entity_response(raw_response: str) -> dict[str, Any]:
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
    raise ValueError(f"Could not parse entity extraction JSON: {raw_response[:500]}")


def normalize_entities(values: Any, *, max_entities: int, min_chars: int) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        entity = re.sub(r"\s+", " ", value).strip(" \t\r\n-:;,.")
        if len(entity) < min_chars:
            continue
        key = entity.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(entity)
        if len(normalized) >= max_entities:
            break
    return normalized


def iter_entity_combos(
    entities: tuple[str, ...],
    combo_sizes: tuple[int, ...],
) -> Iterable[tuple[str, ...]]:
    for size in combo_sizes:
        if size <= 0 or len(entities) < size:
            continue
        yield from itertools.combinations(entities, size)


def reservoir_sample(
    values: Iterable[Any],
    sample_size: int,
    rng: random.Random,
) -> list[Any]:
    if sample_size <= 0:
        return []
    sample: list[Any] = []
    for index, value in enumerate(values):
        if index < sample_size:
            sample.append(value)
            continue
        replacement = rng.randint(0, index)
        if replacement < sample_size:
            sample[replacement] = value
    return sample


def entity_normal_key(entity: str) -> str:
    key = re.sub(r"\s+", " ", entity).strip().casefold()
    return re.sub(r"^[^\w]+|[^\w]+$", "", key)


def choose_entity_surface(left: str, right: str) -> str:
    if len(left) >= len(right):
        return left
    return right


def empty_generation_stats(prefix: str) -> dict[str, int]:
    return {
        f"{prefix}_considered": 0,
        f"{prefix}_tasks": 0,
        f"{prefix}_skipped": 0,
        f"{prefix}_generated": 0,
        f"{prefix}_failed": 0,
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: str) -> str:
    joined = "\x1f".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def build_default_client(provider_config: Any) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(provider_config)
