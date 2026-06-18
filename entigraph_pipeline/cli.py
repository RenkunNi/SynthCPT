"""Command-line interface for EntiGraph generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .entity_selection import ENTITY_SELECTION_STRATEGIES
from .evaluator import EntiGraphEvaluator, EvaluationConfig
from .llm import LLMConfig, OpenAICompatibleClient
from .pipeline import EntiGraphConfig, EntiGraphPipeline


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return run_generate(args)
    if args.command == "evaluate":
        return run_evaluate(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="entigraph",
        description="Generate EntiGraph-style synthetic CPT JSONL from a source JSONL corpus.",
    )
    subparsers = parser.add_subparsers(dest="command")
    generate = subparsers.add_parser("generate", help="Run entity extraction and relation generation.")
    generate.add_argument("--input", required=True, type=Path, help="Input JSONL; each row must contain text_key.")
    generate.add_argument("--output", required=True, type=Path, help="Output synthetic JSONL path.")
    generate.add_argument("--entity-cache", required=True, type=Path, help="Entity extraction cache JSONL path.")
    generate.add_argument("--text-key", default="text")
    generate.add_argument("--title-key", default="title")
    generate.add_argument("--id-key", default=None)
    generate.add_argument("--provider", choices=["local", "openai"], default="local")
    generate.add_argument("--model", required=True)
    generate.add_argument("--base-url", default=None)
    generate.add_argument("--api-key", default=None)
    generate.add_argument("--api-key-env", default=None)
    generate.add_argument("--timeout", type=float, default=120.0)
    generate.add_argument("--max-retries", type=int, default=4)
    generate.add_argument("--json-mode", action="store_true", help="Ask API for JSON objects during entity extraction.")
    generate.add_argument(
        "--mode",
        choices=["single-doc", "cross-doc", "sog-lite", "both", "all"],
        default="single-doc",
        help="Generate single-doc, cross-doc, SoG-lite graph paths, both single/cross, or all modes.",
    )
    generate.add_argument("--combo-sizes", default="2,3", help="Comma-separated entity combination sizes.")
    generate.add_argument("--max-docs", type=int, default=None)
    generate.add_argument("--max-entities", type=int, default=60)
    generate.add_argument(
        "--entity-selection-strategy",
        choices=sorted(ENTITY_SELECTION_STRATEGIES),
        default="llm-order",
        help="How to choose entities when extraction returns more than max_entities.",
    )
    generate.add_argument("--max-combos-per-doc", type=int, default=None)
    generate.add_argument("--sample-combos", action="store_true")
    generate.add_argument("--cross-doc-min-shared-entities", type=int, default=1)
    generate.add_argument("--cross-doc-max-shared-entities", type=int, default=12)
    generate.add_argument("--cross-doc-max-pairs", type=int, default=None)
    generate.add_argument("--cross-doc-sample-pairs", action="store_true")
    generate.add_argument("--sog-path-length", type=int, default=3)
    generate.add_argument("--sog-max-paths", type=int, default=1000)
    generate.add_argument("--sog-max-section-chars", type=int, default=1800)
    generate.add_argument("--random-seed", type=int, default=13)
    generate.add_argument("--max-workers", type=int, default=8)
    generate.add_argument("--max-in-flight", type=int, default=None, help="Bound queued LLM requests; defaults to 4x workers.")
    generate.add_argument("--entity-temperature", type=float, default=0.0)
    generate.add_argument("--relation-temperature", type=float, default=1.0)
    generate.add_argument("--entity-max-tokens", type=int, default=2048)
    generate.add_argument("--relation-max-tokens", type=int, default=2048)
    generate.add_argument("--no-resume", action="store_true")
    generate.add_argument("--include-source-text", action="store_true")
    generate.add_argument("--no-progress", action="store_true", help="Disable interactive progress bars.")
    evaluate = subparsers.add_parser("evaluate", help="Evaluate generated EntiGraph JSONL before training.")
    evaluate.add_argument("--input", required=True, type=Path, help="Original source JSONL.")
    evaluate.add_argument("--generated", required=True, type=Path, help="Generated synthetic JSONL.")
    evaluate.add_argument("--output-dir", type=Path, default=Path("evaluate"), help="Directory for scores and reports.")
    evaluate.add_argument("--entity-cache", type=Path, default=None, help="Optional entity cache JSONL.")
    evaluate.add_argument("--text-key", default="text")
    evaluate.add_argument("--title-key", default="title")
    evaluate.add_argument("--id-key", default=None)
    evaluate.add_argument("--min-overall-score", type=float, default=0.75)
    evaluate.add_argument("--max-unsupported-proper-nouns", type=int, default=2)
    evaluate.add_argument("--max-redundancy-overlap", type=float, default=0.65)
    return parser


def run_generate(args: argparse.Namespace) -> int:
    llm_config = LLMConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        max_retries=args.max_retries,
        json_mode=args.json_mode,
    )
    pipeline_config = EntiGraphConfig(
        input_path=args.input,
        output_path=args.output,
        entity_cache_path=args.entity_cache,
        text_key=args.text_key,
        title_key=args.title_key or None,
        id_key=args.id_key,
        combo_sizes=parse_combo_sizes(args.combo_sizes),
        mode=args.mode,
        max_docs=args.max_docs,
        max_entities=args.max_entities,
        entity_selection_strategy=args.entity_selection_strategy,
        max_combos_per_doc=args.max_combos_per_doc,
        sample_combos=args.sample_combos,
        cross_doc_min_shared_entities=args.cross_doc_min_shared_entities,
        cross_doc_max_shared_entities=args.cross_doc_max_shared_entities,
        cross_doc_max_pairs=args.cross_doc_max_pairs,
        cross_doc_sample_pairs=args.cross_doc_sample_pairs,
        sog_path_length=args.sog_path_length,
        sog_max_paths=args.sog_max_paths,
        sog_max_section_chars=args.sog_max_section_chars,
        random_seed=args.random_seed,
        max_workers=args.max_workers,
        max_in_flight=args.max_in_flight,
        entity_temperature=args.entity_temperature,
        relation_temperature=args.relation_temperature,
        entity_max_tokens=args.entity_max_tokens,
        relation_max_tokens=args.relation_max_tokens,
        json_mode=args.json_mode,
        resume=not args.no_resume,
        include_source_text=args.include_source_text,
        show_progress=not args.no_progress,
        metadata={
            "provider": args.provider,
            "model": args.model,
            "generator": "entigraph-pipeline",
        },
    )
    client = OpenAICompatibleClient(llm_config)
    stats = EntiGraphPipeline(pipeline_config, client).run()
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0 if stats["relations_failed"] == 0 and stats["cross_doc_failed"] == 0 and stats["sog_lite_failed"] == 0 else 1


def run_evaluate(args: argparse.Namespace) -> int:
    config = EvaluationConfig(
        input_path=args.input,
        generated_path=args.generated,
        output_dir=args.output_dir,
        entity_cache_path=args.entity_cache,
        text_key=args.text_key,
        title_key=args.title_key or None,
        id_key=args.id_key,
        min_overall_score=args.min_overall_score,
        max_unsupported_proper_nouns=args.max_unsupported_proper_nouns,
        max_redundancy_overlap=args.max_redundancy_overlap,
    )
    summary = EntiGraphEvaluator(config).run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed"] == 0 else 1


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


if __name__ == "__main__":
    raise SystemExit(main())
