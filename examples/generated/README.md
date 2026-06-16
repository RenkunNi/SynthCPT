# Generated Samples

These files are tiny offline samples produced from `examples/wiki_fixture.jsonl` with:

```bash
python examples/run_offline_wiki_pipeline.py \
  --mode both \
  --combo-sizes 2 \
  --max-combos-per-doc 2 \
  --cross-doc-max-pairs 4
```

- `wiki_synth_offline_sample.jsonl`: 16 generated rows, including 12 single-document relation rows and 4 cross-document graph rows.
- `wiki_entities_offline_sample.jsonl`: entity extraction cache used by the sample run.

They are intentionally small and deterministic, so they are useful for inspecting the JSONL schema and for running the evaluator without external API calls.
