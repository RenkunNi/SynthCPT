#!/usr/bin/env python3
"""Run the EntiGraph pipeline on the wiki fixture without calling an LLM API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entigraph_pipeline.pipeline import EntiGraphConfig, EntiGraphPipeline


DEFAULT_FIXTURE = Path(__file__).with_name("wiki_fixture.jsonl")


class OfflineWikiClient:
    def __init__(self, fixture: Path):
        self.by_title = {}
        for row in read_jsonl(fixture):
            self.by_title[row["title"]] = row

    def chat(self, messages, *, temperature, max_tokens=None, response_format=None):
        system = messages[0]["content"]
        user = messages[1]["content"]
        if "Extract Entities" in system:
            title = parse_title(user)
            row = self.by_title[title]
            return json.dumps(
                {
                    "summary": f"{title} discusses {', '.join(row['entities'][:3])}.",
                    "entities": row["entities"],
                }
            )
        if "cross-document" in system.lower():
            shared = parse_bullet_block(user, "Shared entities:")
            titles = parse_article_titles(user)
            return (
                "### Cross-document context\n"
                f"{titles[0]} and {titles[1]} are linked by shared early-computing topics.\n"
                "### Shared entities\n"
                + ", ".join(shared)
                + "\n### Cross-document relations\n"
                "The shared entities connect people, machines, and technical ideas across both pages.\n"
                "### Integrated synthesis\n"
                "Together, the pages describe how mechanical calculation, programmable control, and interpretive notes formed a connected historical graph."
            )
        if "faithful long-context qa" in system.lower():
            entities = parse_bullet_block(user, "Path entities:")
            return json.dumps(
                {
                    "question": f"How do {entities[0]} and {entities[1]} connect across the cited chunks?",
                    "answer": f"{entities[0]} and {entities[1]} are connected through the supplied early-computing context.",
                    "reasoning": f"According to [1], {entities[0]} appears in the graph path. According to [2], {entities[1]} provides a related chunk in the same path.",
                    "support_ids": ["1", "2"],
                }
            )
        if "graph-path" in system.lower():
            entities = parse_bullet_block(user, "Path entities:")
            return (
                "### Graph path context\n"
                "The selected chunks form a multi-section path through early-computing entities.\n"
                "### Path entities\n"
                + ", ".join(entities)
                + "\n### Grounded synthesis\n"
                "Across the path, the entities connect people, machines, source texts, and control mechanisms using only the supplied chunks."
            )
        title = parse_title(user)
        entities = parse_bullet_block(user, "Entities:")
        return (
            f"### Offline relation synthesis for {title}\n"
            f"The selected entities are {', '.join(entities)}.\n"
            "They are discussed together in the source article and form one single-document EntiGraph relation."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline smoke run for the wiki EntiGraph fixture.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=Path("/tmp/wiki_synth_offline.jsonl"))
    parser.add_argument("--entity-cache", type=Path, default=Path("/tmp/wiki_entities_offline.jsonl"))
    parser.add_argument("--mode", choices=["single-doc", "cross-doc", "sog-lite", "longfaith-qa", "both", "all"], default="all")
    parser.add_argument("--combo-sizes", default="2")
    parser.add_argument("--max-combos-per-doc", type=int, default=2)
    parser.add_argument("--cross-doc-max-pairs", type=int, default=4)
    parser.add_argument("--sog-max-paths", type=int, default=4)
    parser.add_argument("--sog-path-length", type=int, default=3)
    args = parser.parse_args()

    for path in (args.output, args.entity_cache):
        if path.exists():
            path.unlink()

    config = EntiGraphConfig(
        input_path=args.fixture,
        output_path=args.output,
        entity_cache_path=args.entity_cache,
        mode=args.mode,
        combo_sizes=parse_combo_sizes(args.combo_sizes),
        max_combos_per_doc=args.max_combos_per_doc,
        cross_doc_max_pairs=args.cross_doc_max_pairs,
        sog_max_paths=args.sog_max_paths,
        sog_path_length=args.sog_path_length,
        max_workers=2,
        max_in_flight=4,
        metadata={"provider": "offline", "model": "fixture", "generator": "offline-wiki-example"},
    )
    stats = EntiGraphPipeline(config, OfflineWikiClient(args.fixture)).run()
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"output: {args.output}")
    print(f"entity_cache: {args.entity_cache}")
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_title(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Title: "):
            return line.removeprefix("Title: ").strip()
    raise ValueError("prompt did not include a Title line")


def parse_article_titles(prompt: str) -> tuple[str, str]:
    titles = []
    for line in prompt.splitlines():
        if line.startswith("Article A Title: ") or line.startswith("Article B Title: "):
            titles.append(line.split(": ", 1)[1].strip())
    if len(titles) != 2:
        raise ValueError("prompt did not include two article titles")
    return titles[0], titles[1]


def parse_bullet_block(prompt: str, marker: str) -> list[str]:
    _, _, rest = prompt.partition(marker)
    values = []
    for line in rest.splitlines():
        line = line.strip()
        if line.startswith("- "):
            values.append(line[2:])
    return values


def parse_combo_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


if __name__ == "__main__":
    raise SystemExit(main())
