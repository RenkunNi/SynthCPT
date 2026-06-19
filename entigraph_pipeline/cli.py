"""Command-line interface for EntiGraph generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .context_graph import ContextGraphBuilder, ContextGraphConfig
from .entity_selection import ENTITY_SELECTION_STRATEGIES
from .evaluator import EntiGraphEvaluator, EvaluationConfig
from .llm import LLMConfig, OpenAICompatibleClient
from .pipeline import EntiGraphConfig, EntiGraphPipeline
from .prompts import GENERATION_STYLES


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return run_generate(args)
    if args.command == "evaluate":
        return run_evaluate(args)
    if args.command == "build-context-graph":
        return run_build_context_graph(args)
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
        choices=["single-doc", "cross-doc", "sog-lite", "long-context", "longfaith-qa", "both", "all"],
        default="single-doc",
        help="Generate single-doc, cross-doc, SoG-lite graph paths, long-context graph synthesis, LongFaith-style QA, both single/cross, or all modes.",
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
    generate.add_argument("--sog-path-strategy", choices=["dfs", "bridge", "coverage"], default="dfs")
    generate.add_argument("--sog-candidate-multiplier", type=int, default=8)
    generate.add_argument("--sog-min-shared-entities", type=int, default=0)
    generate.add_argument(
        "--generation-styles",
        type=parse_generation_styles,
        default=("default",),
        help=f"Comma-separated style variants for graph modes. Choices: {', '.join(GENERATION_STYLES)}",
    )
    generate.add_argument("--long-context-paths-per-example", type=int, default=3)
    generate.add_argument("--long-context-max-examples", type=int, default=100)
    generate.add_argument("--long-context-max-chars", type=int, default=6000)
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
    graph = subparsers.add_parser(
        "build-context-graph",
        help="Build an entity-level context graph from source JSONL and an entity cache.",
    )
    graph.add_argument("--input", required=True, type=Path, help="Input JSONL; each row must contain text_key.")
    graph.add_argument("--entity-cache", required=True, type=Path, help="Entity extraction cache JSONL path.")
    graph.add_argument("--output", required=True, type=Path, help="Output context graph JSON path.")
    graph.add_argument("--text-key", default="text")
    graph.add_argument("--title-key", default="title")
    graph.add_argument("--id-key", default=None)
    graph.add_argument("--max-section-chars", type=int, default=1800)
    graph.add_argument("--min-shared-contexts", type=int, default=1)
    graph.add_argument("--min-edge-weight", type=float, default=1.0)
    graph.add_argument("--max-evidence-per-fact", type=int, default=4)
    graph.add_argument("--max-contexts-per-entity", type=int, default=20)
    graph.add_argument("--no-context-text", action="store_true", help="Omit raw section text from context nodes.")
    graph.add_argument("--graph-name", default="entity_context_graph")
    graph.add_argument(
        "--typed-facts",
        choices=["off", "heuristic", "llm"],
        default="heuristic",
        help="Add typed semantic facts in addition to co-occurrence facts.",
    )
    graph.add_argument("--min-typed-confidence", type=float, default=0.4)
    graph.add_argument("--max-typed-facts-per-context", type=int, default=12)
    graph.add_argument("--typed-fact-provider", choices=["local", "openai"], default="local")
    graph.add_argument("--typed-fact-model", default=None)
    graph.add_argument("--typed-fact-base-url", default=None)
    graph.add_argument("--typed-fact-api-key", default=None)
    graph.add_argument("--typed-fact-api-key-env", default=None)
    graph.add_argument("--typed-fact-temperature", type=float, default=0.0)
    graph.add_argument("--typed-fact-max-tokens", type=int, default=2048)
    graph.add_argument("--typed-fact-json-mode", action="store_true")
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
        sog_path_strategy=args.sog_path_strategy,
        sog_candidate_multiplier=args.sog_candidate_multiplier,
        sog_min_shared_entities=args.sog_min_shared_entities,
        generation_styles=args.generation_styles,
        long_context_paths_per_example=args.long_context_paths_per_example,
        long_context_max_examples=args.long_context_max_examples,
        long_context_max_chars=args.long_context_max_chars,
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
    return (
        0
        if stats["relations_failed"] == 0
        and stats["cross_doc_failed"] == 0
        and stats["sog_lite_failed"] == 0
        and stats["long_context_failed"] == 0
        and stats["longfaith_qa_failed"] == 0
        else 1
    )


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


def run_build_context_graph(args: argparse.Namespace) -> int:
    client = None
    if args.typed_facts == "llm":
        if not args.typed_fact_model:
            raise SystemExit("--typed-fact-model is required when --typed-facts llm")
        client = OpenAICompatibleClient(
            LLMConfig(
                provider=args.typed_fact_provider,
                model=args.typed_fact_model,
                base_url=args.typed_fact_base_url,
                api_key=args.typed_fact_api_key,
                api_key_env=args.typed_fact_api_key_env,
                json_mode=args.typed_fact_json_mode,
            )
        )
    config = ContextGraphConfig(
        input_path=args.input,
        entity_cache_path=args.entity_cache,
        output_path=args.output,
        text_key=args.text_key,
        title_key=args.title_key or None,
        id_key=args.id_key,
        max_section_chars=args.max_section_chars,
        min_shared_contexts=args.min_shared_contexts,
        min_edge_weight=args.min_edge_weight,
        max_evidence_per_fact=args.max_evidence_per_fact,
        max_contexts_per_entity=args.max_contexts_per_entity,
        include_context_text=not args.no_context_text,
        graph_name=args.graph_name,
        typed_fact_mode=args.typed_facts,
        min_typed_confidence=args.min_typed_confidence,
        max_typed_facts_per_context=args.max_typed_facts_per_context,
        typed_fact_temperature=args.typed_fact_temperature,
        typed_fact_max_tokens=args.typed_fact_max_tokens,
        json_mode=args.typed_fact_json_mode,
    )
    summary = ContextGraphBuilder(config, client=client).run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


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


def parse_generation_styles(value: str) -> tuple[str, ...]:
    styles = tuple(part.strip() for part in value.split(",") if part.strip())
    if not styles:
        raise argparse.ArgumentTypeError("at least one generation style is required")
    invalid = sorted(set(styles) - set(GENERATION_STYLES))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"invalid generation style(s): {', '.join(invalid)}; choices are: {', '.join(GENERATION_STYLES)}"
        )
    return styles


if __name__ == "__main__":
    raise SystemExit(main())
