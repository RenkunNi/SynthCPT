# Generated Samples

These files are tiny offline samples produced from `examples/wiki_fixture.jsonl` with:

```bash
python examples/run_offline_wiki_pipeline.py \
  --mode all \
  --combo-sizes 2 \
  --max-combos-per-doc 2 \
  --cross-doc-max-pairs 4 \
  --sog-max-paths 4 \
  --sog-path-strategy bridge \
  --generation-styles default,contrastive
```

- `wiki_synth_offline_sample.jsonl`: 36 generated rows across single-document, cross-document, SoG-lite, long-context, and LongFaith-style QA modes.
- `wiki_entities_offline_sample.jsonl`: entity extraction cache used by the sample run.
- `context_graph_sample.json`: entity-level context graph built from the same fixture and entity cache, including co-occurrence facts plus heuristic typed facts.

They are intentionally small and deterministic, so they are useful for inspecting schemas and for running the evaluator without external API calls.
