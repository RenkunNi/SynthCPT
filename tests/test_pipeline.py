import json
import tempfile
import unittest
from pathlib import Path

from entigraph_pipeline.context_graph import ContextGraphBuilder, ContextGraphConfig
from entigraph_pipeline.entity_selection import select_entities
from entigraph_pipeline.evaluator import EntiGraphEvaluator, EvaluationConfig
from entigraph_pipeline.graph_paths import split_markdown_sections
from entigraph_pipeline.pipeline import (
    EntiGraphConfig,
    EntiGraphPipeline,
    normalize_entities,
    parse_entity_response,
)
from entigraph_pipeline.titles import infer_title_from_text


class FakeClient:
    def chat(self, messages, *, temperature, max_tokens=None, response_format=None):
        if "Extract Entities" in messages[0]["content"]:
            user = messages[1]["content"]
            if "Ada Lovelace" in user:
                entities = ["Ada Lovelace", "Analytical Engine", "Charles Babbage"]
            elif "Difference Engine" in user:
                entities = ["Charles Babbage", "Difference Engine", "Analytical Engine"]
            else:
                entities = ["Alpha", "Beta", "Alpha", "Gamma"]
            return json.dumps(
                {
                    "summary": "A short document.",
                    "entities": entities,
                }
            )
        user = messages[1]["content"]
        if "Shared entities:" in user:
            return "Cross-document synthetic text\n" + user.split("Shared entities:", 1)[1].strip()
        if "Global entities:" in user:
            return "Long-context synthetic text\n" + user.split("Global entities:", 1)[1].strip()
        if "Path entities:" in user:
            if "faithful long-context qa" in messages[0]["content"].lower():
                return json.dumps(
                    {
                        "question": "How are Ada Lovelace and the Analytical Engine connected?",
                        "answer": "Ada Lovelace is connected to the Analytical Engine through the cited source chunks.",
                        "reasoning": "According to [1], Ada Lovelace appears in the path. According to [2], the Analytical Engine appears in a connected chunk.",
                        "support_ids": ["1", "2"],
                    }
                )
            return "SoG-lite synthetic text\n" + user.split("Path entities:", 1)[1].strip()
        return "Synthetic relation text\n" + user.split("Entities:", 1)[1].strip()


class PipelineTest(unittest.TestCase):
    def test_parse_entity_response_from_fence(self):
        parsed = parse_entity_response('```json\n{"summary": "s", "entities": ["A"]}\n```')
        self.assertEqual(parsed["entities"], ["A"])

    def test_normalize_entities_dedupes(self):
        values = normalize_entities([" Alpha ", "alpha", "B", "Gamma"], min_chars=2)
        self.assertEqual(values, ["Alpha", "Gamma"])

    def test_importance_entity_selection_prefers_mentions(self):
        selected = select_entities(
            ("Rare Name", "Alpha", "Beta"),
            "Alpha appears several times. Alpha connects to Beta. Alpha matters here.",
            doc_frequency={},
            total_docs=1,
            strategy="importance",
            max_entities=1,
        )
        self.assertEqual(selected, ("Alpha",))

    def test_rarity_entity_selection_prefers_corpus_rare_terms(self):
        selected = select_entities(
            ("Common", "Rare"),
            "Common and Rare are both mentioned.",
            doc_frequency={"common": 10, "rare": 1},
            total_docs=10,
            strategy="rarity",
            max_entities=1,
        )
        self.assertEqual(selected, ("Rare",))

    def test_title_fallback_uses_source_text(self):
        title = infer_title_from_text("# Ada Lovelace and the Analytical Engine\nMore text.", 0)
        self.assertEqual(title, "Ada Lovelace and the Analytical Engine")

    def test_pipeline_without_title_infers_readable_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "out.jsonl"
            entity_path = root / "entities.jsonl"
            input_path.write_text(
                json.dumps({"id": "doc-1", "text": "Ada Lovelace studied the Analytical Engine. More text."}) + "\n"
            )
            config = EntiGraphConfig(
                input_path=input_path,
                output_path=output_path,
                entity_cache_path=entity_path,
                combo_sizes=(2,),
                max_combos_per_doc=1,
                max_workers=1,
                show_progress=False,
            )
            stats = EntiGraphPipeline(config, FakeClient()).run()
            self.assertEqual(stats["relations_generated"], 1)
            row = json.loads(output_path.read_text().splitlines()[0])
            self.assertEqual(row["title"], "Ada Lovelace studied the Analytical Engine")

    def test_pipeline_generates_pairs_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "out.jsonl"
            entity_path = root / "entities.jsonl"
            input_path.write_text(json.dumps({"id": "doc-1", "title": "T", "text": "Alpha meets Beta and Gamma."}) + "\n")
            config = EntiGraphConfig(
                input_path=input_path,
                output_path=output_path,
                entity_cache_path=entity_path,
                combo_sizes=(2,),
                max_workers=2,
            )
            stats = EntiGraphPipeline(config, FakeClient()).run()
            self.assertEqual(stats["relations_generated"], 3)
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["combo_size"], 2)

            stats = EntiGraphPipeline(config, FakeClient()).run()
            self.assertEqual(stats["relations_skipped"], 3)
            self.assertEqual(stats["relations_generated"], 0)

    def test_cross_document_mode_generates_shared_entity_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "out.jsonl"
            entity_path = root / "entities.jsonl"
            rows = [
                {
                    "id": "ada",
                    "title": "Ada Lovelace",
                    "text": "Ada Lovelace wrote notes about the Analytical Engine with Charles Babbage.",
                },
                {
                    "id": "babbage",
                    "title": "Charles Babbage",
                    "text": "Charles Babbage designed the Difference Engine and the Analytical Engine.",
                },
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = EntiGraphConfig(
                input_path=input_path,
                output_path=output_path,
                entity_cache_path=entity_path,
                mode="cross-doc",
                cross_doc_min_shared_entities=1,
                max_workers=2,
            )
            stats = EntiGraphPipeline(config, FakeClient()).run()
            self.assertEqual(stats["relations_generated"], 0)
            self.assertEqual(stats["cross_doc_generated"], 1)
            output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(output_rows[0]["generation_mode"], "cross_doc")
            self.assertEqual(set(output_rows[0]["doc_ids"]), {"ada", "babbage"})
            self.assertIn("Analytical Engine", output_rows[0]["shared_entities"])

    def test_markdown_sections_assign_entities(self):
        nodes = split_markdown_sections(
            doc_id="doc",
            source_index=0,
            source_sha256="abc",
            doc_title="Doc",
            text="# Title\n\n## Engine\nAda Lovelace studied the Analytical Engine.\n\n## Loom\nThe Jacquard loom used cards.",
            entities=("Ada Lovelace", "Analytical Engine", "Jacquard loom"),
            max_section_chars=500,
        )
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].section_title, "Engine")
        self.assertEqual(nodes[0].entities, ("Ada Lovelace", "Analytical Engine"))

    def test_sog_lite_mode_generates_graph_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "out.jsonl"
            entity_path = root / "entities.jsonl"
            rows = [
                {
                    "id": "ada",
                    "title": "Ada Lovelace",
                    "text": "Ada Lovelace wrote notes about the Analytical Engine with Charles Babbage.",
                },
                {
                    "id": "babbage",
                    "title": "Charles Babbage",
                    "text": "Charles Babbage designed the Difference Engine and the Analytical Engine.",
                },
                {
                    "id": "engine",
                    "title": "Analytical Engine",
                    "text": "Ada Lovelace and Charles Babbage discussed the Analytical Engine.",
                },
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = EntiGraphConfig(
                input_path=input_path,
                output_path=output_path,
                entity_cache_path=entity_path,
                mode="sog-lite",
                sog_path_length=2,
                sog_max_paths=2,
                max_workers=2,
                show_progress=False,
            )
            stats = EntiGraphPipeline(config, FakeClient()).run()
            self.assertEqual(stats["sog_lite_generated"], 2)
            output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(output_rows[0]["generation_mode"], "sog_lite")
            self.assertEqual(output_rows[0]["task_type"], "graph_path_synthesis")
            self.assertIn("path_id", output_rows[0])
            self.assertIn("source_sha256", output_rows[0])
            self.assertIn("path_metrics", output_rows[0])

    def test_sog_lite_styles_generate_distinct_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "out.jsonl"
            entity_path = root / "entities.jsonl"
            rows = [
                {
                    "id": "ada",
                    "title": "Ada Lovelace",
                    "text": "Ada Lovelace wrote notes about the Analytical Engine with Charles Babbage.",
                },
                {
                    "id": "babbage",
                    "title": "Charles Babbage",
                    "text": "Charles Babbage designed the Difference Engine and the Analytical Engine.",
                },
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = EntiGraphConfig(
                input_path=input_path,
                output_path=output_path,
                entity_cache_path=entity_path,
                mode="sog-lite",
                sog_path_length=2,
                sog_max_paths=1,
                generation_styles=("default", "contrastive"),
                max_workers=2,
                show_progress=False,
            )
            stats = EntiGraphPipeline(config, FakeClient()).run()
            self.assertEqual(stats["sog_lite_generated"], 2)
            output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual({row["style"] for row in output_rows}, {"default", "contrastive"})
            self.assertEqual(len({row["path_id"] for row in output_rows}), 2)

    def test_long_context_mode_groups_graph_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "out.jsonl"
            entity_path = root / "entities.jsonl"
            rows = [
                {
                    "id": "ada",
                    "title": "Ada Lovelace",
                    "text": "Ada Lovelace wrote notes about the Analytical Engine with Charles Babbage.",
                },
                {
                    "id": "babbage",
                    "title": "Charles Babbage",
                    "text": "Charles Babbage designed the Difference Engine and the Analytical Engine.",
                },
                {
                    "id": "engine",
                    "title": "Analytical Engine",
                    "text": "Ada Lovelace and Charles Babbage discussed the Analytical Engine.",
                },
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = EntiGraphConfig(
                input_path=input_path,
                output_path=output_path,
                entity_cache_path=entity_path,
                mode="long-context",
                sog_path_length=2,
                sog_max_paths=2,
                long_context_paths_per_example=2,
                long_context_max_examples=1,
                max_workers=2,
                show_progress=False,
            )
            stats = EntiGraphPipeline(config, FakeClient()).run()
            self.assertEqual(stats["long_context_generated"], 1)
            output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(output_rows[0]["generation_mode"], "long_context")
            self.assertEqual(output_rows[0]["task_type"], "multi_path_long_context_synthesis")
            self.assertEqual(output_rows[0]["path_count"], 2)

    def test_longfaith_qa_mode_generates_cited_qa(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "out.jsonl"
            entity_path = root / "entities.jsonl"
            rows = [
                {
                    "id": "ada",
                    "title": "Ada Lovelace",
                    "text": "Ada Lovelace wrote notes about the Analytical Engine with Charles Babbage.",
                },
                {
                    "id": "babbage",
                    "title": "Charles Babbage",
                    "text": "Charles Babbage designed the Difference Engine and the Analytical Engine.",
                },
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = EntiGraphConfig(
                input_path=input_path,
                output_path=output_path,
                entity_cache_path=entity_path,
                mode="longfaith-qa",
                sog_path_length=2,
                sog_max_paths=1,
                max_workers=2,
                show_progress=False,
            )
            stats = EntiGraphPipeline(config, FakeClient()).run()
            self.assertEqual(stats["longfaith_qa_generated"], 1)
            output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(output_rows[0]["generation_mode"], "longfaith_qa")
            self.assertEqual(output_rows[0]["task_type"], "cited_long_context_qa")
            self.assertIn("question", output_rows[0])
            self.assertIn("[1]", output_rows[0]["reasoning"])
            self.assertIn("source_sha256", output_rows[0])

    def test_failed_output_rows_are_retried_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "out.jsonl"
            entity_path = root / "entities.jsonl"
            input_path.write_text(json.dumps({"id": "doc-1", "title": "T", "text": "Alpha meets Beta and Gamma."}) + "\n")
            config = EntiGraphConfig(
                input_path=input_path,
                output_path=output_path,
                entity_cache_path=entity_path,
                combo_sizes=(2,),
                max_combos_per_doc=1,
                max_workers=1,
            )
            first_task = next(EntiGraphPipeline(config, FakeClient()).iter_relation_tasks(
                list(EntiGraphPipeline(config, FakeClient()).iter_documents()),
                {
                    "doc-1": type(
                        "Record",
                        (),
                        {"doc_id": "doc-1", "entities": ("Alpha", "Beta"), "error": ""},
                    )()
                },
            ))
            output_path.write_text(json.dumps({"relation_id": first_task.relation_id, "error": "temporary"}) + "\n")
            stats = EntiGraphPipeline(config, FakeClient()).run()
            self.assertEqual(stats["relations_skipped"], 0)
            self.assertEqual(stats["relations_generated"], 1)

    def test_evaluator_writes_scores_and_passes_grounded_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            generated_path = root / "generated.jsonl"
            output_dir = root / "evaluate"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "doc-1",
                        "title": "Analytical Engine",
                        "entities": ["Analytical Engine", "Ada Lovelace"],
                        "text": "Ada Lovelace studied the Analytical Engine.",
                    }
                )
                + "\n"
            )
            generated_path.write_text(
                json.dumps(
                    {
                        "relation_id": "r1",
                        "doc_id": "doc-1",
                        "generation_mode": "single_doc",
                        "title": "Analytical Engine",
                        "entities": ["Analytical Engine", "Ada Lovelace"],
                        "text": "Analytical Engine and Ada Lovelace are connected in the article.",
                    }
                )
                + "\n"
            )
            summary = EntiGraphEvaluator(
                EvaluationConfig(input_path=input_path, generated_path=generated_path, output_dir=output_dir)
            ).run()
            self.assertEqual(summary["passed"], 1)
            self.assertTrue((output_dir / "rows.jsonl").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "probes.jsonl").exists())

    def test_evaluator_scores_longfaith_without_copied_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            generated_path = root / "generated.jsonl"
            output_dir = root / "evaluate"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "doc-1",
                        "title": "Analytical Engine",
                        "entities": ["Analytical Engine", "Ada Lovelace"],
                        "text": "Ada Lovelace studied the Analytical Engine.",
                    }
                )
                + "\n"
            )
            generated_path.write_text(
                json.dumps(
                    {
                        "qa_id": "qa-1",
                        "path_id": "path-1",
                        "doc_ids": ["doc-1"],
                        "generation_mode": "longfaith_qa",
                        "shared_entities": ["Analytical Engine", "Ada Lovelace"],
                        "context": "Ada Lovelace studied the Analytical Engine.",
                        "question": "How are Ada Lovelace and the Analytical Engine connected?",
                        "answer": "Ada Lovelace studied the Analytical Engine.",
                        "reasoning": "According to [1], Ada Lovelace studied the Analytical Engine. According to [2], the cited path supports the connection.",
                        "support_ids": ["1", "2"],
                        "text": "### Context\nAda Lovelace studied the Analytical Engine.\n\n### Question\n...",
                    }
                )
                + "\n"
            )
            summary = EntiGraphEvaluator(
                EvaluationConfig(input_path=input_path, generated_path=generated_path, output_dir=output_dir)
            ).run()
            self.assertEqual(summary["passed"], 1)
            row = json.loads((output_dir / "rows.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["generated_id"], "qa-1")
            self.assertEqual(row["scores"]["citation_support"], 1.0)

    def test_context_graph_builder_creates_entity_facts_with_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            entity_path = root / "entities.jsonl"
            graph_path = root / "context_graph.json"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "ada",
                        "title": "Ada Lovelace",
                        "text": "# Ada Lovelace\n\n## Notes\nAda Lovelace wrote Notes about the Analytical Engine with Charles Babbage.",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "id": "engine",
                        "title": "Analytical Engine",
                        "text": "# Analytical Engine\n\n## Design\nCharles Babbage designed the Analytical Engine.",
                    }
                )
                + "\n"
            )
            entity_path.write_text(
                json.dumps(
                    {
                        "doc_id": "ada",
                        "source_index": 0,
                        "source_sha256": "h1",
                        "title": "Ada Lovelace",
                        "summary": "",
                        "entities": ["Ada Lovelace", "Analytical Engine", "Charles Babbage", "Notes"],
                        "raw_response": "{}",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "doc_id": "engine",
                        "source_index": 1,
                        "source_sha256": "h2",
                        "title": "Analytical Engine",
                        "summary": "",
                        "entities": ["Analytical Engine", "Charles Babbage"],
                        "raw_response": "{}",
                    }
                )
                + "\n"
            )
            summary = ContextGraphBuilder(
                ContextGraphConfig(
                    input_path=input_path,
                    entity_cache_path=entity_path,
                    output_path=graph_path,
                    max_section_chars=500,
                )
            ).run()
            graph = json.loads(graph_path.read_text())
            self.assertEqual(summary["documents"], 2)
            self.assertGreater(summary["typed_facts"], 0)
            self.assertGreater(summary["cooccurrence_facts"], 0)
            self.assertEqual(graph["schema_version"], "entity-context-graph-v1")
            self.assertGreaterEqual(len(graph["entities"]), 4)
            self.assertEqual(len(graph["contexts"]), 2)
            fact = next(
                row
                for row in graph["facts"]
                if {row["head"], row["tail"]} == {"Analytical Engine", "Charles Babbage"}
            )
            self.assertEqual(fact["relation"], "co_occurs_with")
            self.assertGreaterEqual(fact["context_count"], 2)
            self.assertIn("evidence", fact)
            self.assertIn("source_sha256", fact["evidence"][0])
            typed_fact = next(row for row in graph["facts"] if row["edge_type"] == "typed_contextual_fact")
            self.assertIn("relation_category", typed_fact)
            self.assertIn(typed_fact["relation_category"], graph["build_config"]["relation_categories"])
            self.assertIn("confidence", typed_fact)
            context = graph["contexts"][0]
            self.assertIn("source_sha256", context)
            self.assertIn("text", context)


if __name__ == "__main__":
    unittest.main()
