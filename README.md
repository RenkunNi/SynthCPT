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
- `--entity-selection-strategy hybrid`
- `--max-combos-per-doc 5000`
- `--sample-combos`
- `--no-progress`

When extraction returns many entities, `--max-entities` limits the selected entities used for graph generation. The cache still stores the full extracted list, so you can rerun with a different selection strategy without extracting entities again:

- `llm-order`: keep the LLM's extracted order. This is the default and closest to the paper prompt.
- `importance`: prefer entities mentioned frequently and early in the source document.
- `rarity`: prefer entities that appear in fewer documents across the corpus.
- `hybrid`: combine document importance with corpus rarity.

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

### 5. Build SoG-lite Graph Paths

In `sog-lite` mode, the pipeline splits documents into markdown sections or paragraph chunks, assigns selected entities to each chunk, builds a section-level entity graph, samples graph paths, and asks the LLM to synthesize a grounded long-context document from the path.

This is a practical, dependency-free version of Synthesize-on-Graph style path sampling:

- nodes: document sections / paragraphs
- edges: shared selected entities
- paths: multi-section, preferably cross-document context chains

Useful controls:

- `--mode sog-lite`
- `--sog-path-length 3`
- `--sog-max-paths 1000`
- `--sog-max-section-chars 1800`
- `--sog-path-strategy dfs|bridge|coverage`
- `--sog-min-shared-entities 1`

Path strategy choices:

- `dfs`: compatibility mode; returns the first deterministic graph paths.
- `bridge`: ranks paths with more shared entities, cross-document edges, and bridge evidence.
- `coverage`: ranks paths that cover more documents and entities.

Each SoG-lite row includes `path_metrics`, `path_strategy`, `source_sha256`, and `style`, so you can inspect why a path was selected and safely resume after source text changes.

### 6. Generate Long-context Graph Rows

In `long-context` mode, the pipeline groups multiple SoG-lite graph paths into one larger grounded context and asks the LLM to synthesize across paths. This is useful when you want long CPT examples that cover several connected sections instead of one short path.

Useful controls:

- `--mode long-context`
- `--long-context-paths-per-example 3`
- `--long-context-max-examples 100`
- `--long-context-max-chars 6000`
- `--sog-path-strategy bridge`

Each row includes `long_context_id`, `path_ids`, `path_count`, `chunk_count`, `path_metrics`, and the same source hashes used by SoG-lite rows.

### 7. Generate LongFaith-style QA

In `longfaith-qa` mode, the pipeline reuses SoG-lite graph paths as cited support contexts. It asks the LLM to generate one question, answer, and cited reasoning chain grounded in the supplied chunks.

This produces instruction-style long-context data rather than plain CPT prose. Each row includes:

- `context`: cited source chunks
- `question`
- `answer`
- `reasoning`
- `support_ids`
- `target_entities`: the entities the QA row is expected to mention and reason about
- `text`: a combined context/question/answer/reasoning field for evaluation or training

Useful controls are shared with SoG-lite:

- `--mode longfaith-qa`
- `--sog-path-length 3`
- `--sog-max-paths 1000`
- `--sog-max-section-chars 1800`

Use `--mode all` to generate single-document, cross-document, SoG-lite, long-context, and LongFaith-style QA rows.

### 8. Multi-style Outputs

Graph modes support multiple prompt styles with `--generation-styles`. Styles are generated as separate rows with style-specific IDs:

- `default`
- `synthesis`
- `contrastive`
- `timeline`
- `causal`

Example:

```bash
entigraph generate \
  --input data/source.jsonl \
  --output data/graph_styles.jsonl \
  --entity-cache data/entities.jsonl \
  --provider local \
  --base-url http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --mode all \
  --sog-path-strategy bridge \
  --generation-styles default,contrastive,causal
```

### 9. Write Synthetic JSONL

Generated rows are appended to `--output`. Resume is automatic:

- single-document rows resume by `relation_id`
- cross-document rows resume by `graph_id`
- SoG-lite rows resume by source-hash-sensitive `path_id`
- long-context rows resume by `long_context_id`
- LongFaith-style QA rows resume by `qa_id`
- rows containing `error` are retried

### 10. Evaluate Before Training

The `evaluate` command scores generated rows before any model training. It writes all scores under `evaluate/` by default:

- `evaluate/rows.jsonl`: row-level scores and pass/fail decisions
- `evaluate/summary.json`: aggregate score summary
- `evaluate/probes.jsonl`: simple QA-style probes
- `evaluate/report.md`: human-readable report

The default quality gate checks selected-entity support, selected-entity mentions, unsupported proper nouns, copying from source text, relation signal, specificity, structure, and length.

### 11. Build an Entity-level Context Graph

The `build-context-graph` command creates a reusable graph artifact inspired by entity-level Context Graphs. It keeps both:

- entity nodes: canonical entity names, aliases, document frequency, context IDs
- context nodes: source sections/chunks with provenance and selected entities
- fact edges: entity-to-entity contextual facts with evidence snippets, source hashes, context IDs, document IDs, and weights

This graph is different from the temporary section graph used by SoG-lite. The section graph is rebuilt during generation; the context graph is saved and can be reused for agentic data generation, graph retrieval, hard negatives, and coverage analysis.

```bash
entigraph build-context-graph \
  --input data/source.jsonl \
  --entity-cache data/entities.jsonl \
  --output data/context_graph.json \
  --max-section-chars 1800 \
  --min-shared-contexts 1 \
  --min-edge-weight 1.0 \
  --typed-facts heuristic
```

Useful controls:

- `--no-context-text`: omit raw chunk text if the graph artifact should only carry metadata/evidence snippets.
- `--min-shared-contexts 2`: keep only stronger entity pairs that appear together in multiple contexts.
- `--min-edge-weight 2.0`: prune weak edges.
- `--max-evidence-per-fact 4`: cap evidence snippets per edge.
- `--typed-facts off|heuristic|llm`: choose whether to add typed facts in addition to `co_occurs_with`.
- `--min-typed-confidence 0.4`: filter low-confidence typed facts.
- `--max-typed-facts-per-context 12`: cap typed facts extracted from each context.

Typed fact definition is hybrid:

- `relation_category` is selected from a small controlled set: `created_or_designed_by`, `authored_or_translated_by`, `located_in`, `part_of`, `used_for`, `inspired_by`, `causes_or_enables`, `compares_or_contrasts`, `temporal`, `associated_with`, `other`.
- `relation` is an open short phrase, such as `designed`, `translated`, `published_account_of`, or `associated_with`.

This gives you stable buckets for analysis while preserving corpus-specific relation wording. Heuristic mode is deterministic and high precision but conservative. LLM mode is better for open-domain typed relations:

```bash
entigraph build-context-graph \
  --input data/source.jsonl \
  --entity-cache data/entities.jsonl \
  --output data/context_graph_typed.json \
  --typed-facts llm \
  --typed-fact-provider local \
  --typed-fact-base-url http://localhost:8000/v1 \
  --typed-fact-model meta-llama/Llama-3.1-70B-Instruct \
  --typed-fact-json-mode
```

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
  --mode all \
  --max-workers 8 \
  --max-in-flight 32 \
  --max-entities 60 \
  --entity-selection-strategy hybrid \
  --combo-sizes 2,3 \
  --max-combos-per-doc 5000 \
  --sample-combos \
  --cross-doc-max-pairs 10000 \
  --sog-path-length 3 \
  --sog-max-paths 10000 \
  --sog-path-strategy bridge \
  --generation-styles default,synthesis
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
  --mode all \
  --max-workers 4 \
  --max-entities 60 \
  --entity-selection-strategy hybrid \
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

## Build a Context Graph

```bash
entigraph build-context-graph \
  --input data/source.jsonl \
  --entity-cache data/entities.jsonl \
  --output data/context_graph.json \
  --max-section-chars 1800 \
  --min-edge-weight 1.0
```

## Offline Example

The `examples/` directory has a tiny wiki-like corpus about early computing. It can run without any external API:

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
- `context_graph_sample.json`

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
- Use `sog-lite` when you want long-context graph-path synthesis from section-level chunks.
- Use `long-context` when you want multi-path, longer CPT examples from the same graph.
- Use `longfaith-qa` when you want cited long-context instruction data instead of CPT prose.
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
