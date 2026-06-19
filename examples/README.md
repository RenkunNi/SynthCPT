# Examples

## Synthetic wiki fixture

`wiki_fixture.jsonl` is a tiny offline corpus with short wiki-like pages about early computing. Each JSONL row includes the pipeline-required `text` field plus `id`, `title`, and an `entities` list for deterministic local inspection.

The overlapping entities make it useful for comparing ordinary single-document EntiGraph relation generation with cross-document ideas without calling a network service first.

Inspect the fixture:

```bash
python examples/compare_wiki_fixture.py
```

Use it with the pipeline when an OpenAI-compatible local or hosted model is available:

```bash
entigraph generate \
  --input examples/wiki_fixture.jsonl \
  --output /tmp/wiki_synth.jsonl \
  --entity-cache /tmp/wiki_entities.jsonl \
  --provider local \
  --base-url http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --combo-sizes 2,3 \
  --max-entities 12
```

Run paper-style single-document, cross-document, SoG-lite, long-context, and LongFaith-style QA modes offline:

```bash
python examples/run_offline_wiki_pipeline.py \
  --mode all \
  --sog-path-strategy bridge \
  --generation-styles default,contrastive
python examples/compare_wiki_fixture.py --generated /tmp/wiki_synth_offline.jsonl
entigraph evaluate \
  --input examples/wiki_fixture.jsonl \
  --generated /tmp/wiki_synth_offline.jsonl \
  --entity-cache /tmp/wiki_entities_offline.jsonl \
  --output-dir evaluate
```

Build the reusable entity-level context graph from the same fixture and entity cache:

```bash
entigraph build-context-graph \
  --input examples/wiki_fixture.jsonl \
  --entity-cache /tmp/wiki_entities_offline.jsonl \
  --output /tmp/wiki_context_graph.json \
  --max-section-chars 1800 \
  --min-edge-weight 1.0 \
  --typed-facts heuristic
```

Cross-document mode is available in the real CLI with:

```bash
entigraph generate \
  --input examples/wiki_fixture.jsonl \
  --output /tmp/wiki_cross_doc.jsonl \
  --entity-cache /tmp/wiki_entities.jsonl \
  --provider local \
  --base-url http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --mode cross-doc \
  --cross-doc-max-pairs 20
```

After a run, summarize generated output:

```bash
python examples/compare_wiki_fixture.py --generated /tmp/wiki_synth.jsonl
```

Compare quality signals across multiple generation methods:

```bash
python examples/compare_generation_quality.py \
  baseline=/tmp/wiki_synth_offline.jsonl \
  sog-lite=/tmp/wiki_sog_lite.jsonl \
  long-context=/tmp/wiki_long_context.jsonl \
  longfaith-qa=/tmp/wiki_longfaith_qa.jsonl
```

The comparison helper is dependency-free and reports local grounding signals:
entity support, entity mentions, unsupported proper nouns, source 5-gram
overlap, relation wording, length, duplicate texts, and warning counts. It can
be run for single-document/cross-document, SoG-lite graph-path,
long-context, and LongFaith-style QA outputs.

If you want the helper to use extracted entities instead of the fixture metadata:

```bash
python examples/compare_wiki_fixture.py --entity-cache /tmp/wiki_entities.jsonl
```
