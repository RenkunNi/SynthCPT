# SynthCPT

SynthCPT is a dependency-light Python pipeline for generating synthetic continued-pretraining data from JSONL corpora. It reimplements the EntiGraph-style augmentation from Yang et al., **Synthetic Continued Pretraining** ([arXiv:2409.07431](https://arxiv.org/pdf/2409.07431)) and adds an optional cross-document graph mode plus a pre-training quality gate.

Input rows are JSON objects with a source document in the `text` key:

```json
{"id": "doc-1", "title": "Example", "text": "Source document text..."}
```

## Install

```bash
git clone https://github.com/RenkunNi/SynthCPT.git
cd SynthCPT
python -m pip install -e .
```

After install, use the `entigraph` command. Without installing, use:

```bash
python -m entigraph_pipeline.cli --help
```

## How The Pipeline Works

### 1. Load Source JSONL

The pipeline reads one document per JSONL row from `--input`. By default it uses:

- `text`: document body
- `title`: optional document title
- `id`, `doc_id`, or `document_id`: optional stable id

Missing/empty `text` rows are skipped. If no title field is present, SynthCPT infers a short title from the first heading, first non-empty line, or first sentence of the source text. Only fully empty title candidates fall back to `Untitled source N`.

### 2. Extract Entities

For each document, the augmenter LLM receives the full source document and returns:

```json
{
  "summary": "short document summary",
  "entities": ["entity1", "entity2"]
}
```

The entity extraction result is written to `--entity-cache`. Cache rows include `source_sha256`, so if a source document changes, the pipeline re-extracts entities instead of silently reusing stale data.

### 3. Build Single-Document EntiGraph Tasks

In `single-doc` mode, the pipeline enumerates entity combinations inside each document:

- pairs: `(entity1, entity2)`
- triples: `(entity1, entity2, entity3)`

For every pair/triple, the LLM is prompted to rephrase the document around each entity and discuss their interaction. These rows are the paper-faithful EntiGraph-style augmentation.

Useful controls:

- `--combo-sizes 2,3`
- `--max-entities 60`
- `--max-combos-per-doc 5000`
- `--sample-combos`
- `--no-progress`

### 4. Build Cross-Document Graph Tasks

In `cross-doc` mode, the pipeline links documents that share extracted entities. For each document pair with enough shared entities, it prompts the LLM to synthesize a grounded cross-document relation.

This mode is an extension, not part of the paper baseline.

Useful controls:

- `--mode cross-doc`
- `--cross-doc-min-shared-entities 1`
- `--cross-doc-max-shared-entities 12`
- `--cross-doc-max-pairs 10000`
- `--cross-doc-sample-pairs`

Use `--mode both` to generate both single-document and cross-document rows into the same output JSONL.

### 5. Write Synthetic JSONL

Generated rows are appended to `--output`. Resume is automatic:

- single-document rows resume by `relation_id`
- cross-document rows resume by `graph_id`
- rows containing `error` are retried

### 6. Evaluate Before Training

The `evaluate` command scores generated rows before any model training. It writes all scores under `evaluate/` by default:

- `evaluate/rows.jsonl`: row-level scores and pass/fail decisions
- `evaluate/summary.json`: aggregate score summary
- `evaluate/probes.jsonl`: simple QA-style probes
- `evaluate/report.md`: human-readable report

The default quality gate checks selected-entity support, selected-entity mentions, unsupported proper nouns, copying from source text, relation signal, specificity, structure, and length.

## Run With Local vLLM

Start a vLLM OpenAI-compatible server separately, then run:

```bash
entigraph generate \
  --input data/source.jsonl \
  --output data/entigraph_synth.jsonl \
  --entity-cache data/entities.jsonl \
  --provider local \
  --base-url http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --mode both \
  --max-workers 8 \
  --max-in-flight 32 \
  --max-entities 60 \
  --combo-sizes 2,3 \
  --max-combos-per-doc 5000 \
  --sample-combos \
  --cross-doc-max-pairs 10000
```

The default local API key is `EMPTY`, matching common vLLM setups. Set `VLLM_API_KEY` or pass `--api-key` if your server requires one.

Interactive runs show progress bars for entity extraction, single-document relation generation, and cross-document generation. Add `--no-progress` for quiet batch jobs or logs.

## Run With Hosted OpenAI API

```bash
export OPENAI_API_KEY=...

entigraph generate \
  --input data/source.jsonl \
  --output data/entigraph_synth.jsonl \
  --entity-cache data/entities.jsonl \
  --provider openai \
  --model gpt-4.1 \
  --json-mode \
  --mode both \
  --max-workers 4 \
  --max-entities 60 \
  --combo-sizes 2,3 \
  --max-combos-per-doc 5000 \
  --sample-combos
```

## Evaluate Generated Data

```bash
entigraph evaluate \
  --input data/source.jsonl \
  --generated data/entigraph_synth.jsonl \
  --entity-cache data/entities.jsonl \
  --output-dir evaluate
```

The evaluator exits nonzero if any generated row fails the gate, which makes it usable in scripts or CI.

## Offline Example

The `examples/` directory has a tiny wiki-like corpus about early computing. It can run without any external API:

```bash
python examples/run_offline_wiki_pipeline.py \
  --mode both \
  --combo-sizes 2 \
  --max-combos-per-doc 2 \
  --cross-doc-max-pairs 4
```

Inspect single-document and cross-document opportunities:

```bash
python examples/compare_wiki_fixture.py \
  --generated /tmp/wiki_synth_offline.jsonl
```

Evaluate the generated sample:

```bash
entigraph evaluate \
  --input examples/wiki_fixture.jsonl \
  --generated /tmp/wiki_synth_offline.jsonl \
  --entity-cache /tmp/wiki_entities_offline.jsonl \
  --output-dir evaluate
```

Small generated samples are included in [examples/generated](examples/generated):

- `wiki_synth_offline_sample.jsonl`
- `wiki_entities_offline_sample.jsonl`

The checked-in evaluation report in [evaluate](evaluate) was produced from the offline sample.

## Output Schemas

Single-document generated row:

```json
{
  "relation_id": "stable resume id",
  "doc_id": "source document id",
  "source_index": 0,
  "source_sha256": "...",
  "title": "Document title",
  "combo_size": 2,
  "entities": ["entity A", "entity B"],
  "prompt_name": "entigraph_relation_2",
  "generation_mode": "single_doc",
  "task_type": "entity_relation",
  "provider": "local",
  "model": "...",
  "generator": "entigraph-pipeline",
  "text": "synthetic CPT text"
}
```

Cross-document generated row:

```json
{
  "graph_id": "stable resume id",
  "doc_ids": ["source A", "source B"],
  "source_indices": [0, 1],
  "source_sha256": ["...", "..."],
  "titles": ["Title A", "Title B"],
  "shared_entities": ["entity A"],
  "shared_entity_count": 1,
  "prompt_name": "cross_document_graph_pair",
  "generation_mode": "cross_doc",
  "task_type": "cross_document_graph",
  "text": "cross-document synthetic CPT text"
}
```

Evaluation row:

```json
{
  "generated_id": "relation_id or graph_id",
  "generation_mode": "single_doc",
  "scores": {
    "overall": 0.93,
    "selected_entity_support": 1.0,
    "selected_entity_mention": 1.0,
    "source_5gram_overlap": 0.0
  },
  "pass_gate": true,
  "warnings": []
}
```

## Practical Notes

- Use `single-doc` for the paper-faithful EntiGraph baseline.
- Use `cross-doc` separately when you want broader document-level graph synthesis.
- The pair/triple space grows quickly: all pairs are `O(n^2)`, triples are `O(n^3)`.
- To mirror the paper's practical setup, run all pairs plus sampled triplets.
- Keep source text in prompts to reduce hallucination.
- Before CPT, run `entigraph evaluate` and inspect failed rows.
- For actual CPT, shuffle generated JSONL, tokenize the `text` field, and optionally mix replay data to reduce distribution drift.

## Development Checks

```bash
python -m unittest discover -s tests
python -m compileall entigraph_pipeline tests examples
```
