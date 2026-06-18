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

Run both the paper-style single-document path and the experimental cross-document path offline:

```bash
python examples/run_offline_wiki_pipeline.py --mode both
python examples/compare_wiki_fixture.py --generated /tmp/wiki_synth_offline.jsonl
entigraph evaluate \
  --input examples/wiki_fixture.jsonl \
  --generated /tmp/wiki_synth_offline.jsonl \
  --entity-cache /tmp/wiki_entities_offline.jsonl \
  --output-dir evaluate
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
  longfaith-qa=/tmp/wiki_longfaith_qa.jsonl
```

The comparison helper is dependency-free and reports local grounding signals:
entity support, entity mentions, unsupported proper nouns, source 5-gram
overlap, relation wording, length, duplicate texts, and warning counts. It can
be run now for the single-document/cross-document baseline and reused for
SoG-lite graph-path or LongFaith-style QA outputs once those JSONL files exist.

If you want the helper to use extracted entities instead of the fixture metadata:

```bash
python examples/compare_wiki_fixture.py --entity-cache /tmp/wiki_entities.jsonl
```
