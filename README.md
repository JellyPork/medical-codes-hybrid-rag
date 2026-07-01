# Medical Codes Hybrid RAG

This project is a simple starting point for searching medical codes with a hybrid retrieval pipeline. It combines plain keyword matching with embedding-based similarity so the system can handle both exact code lookups and more natural language questions.

The current version is intentionally small. It uses a JSONL dataset, builds an in-memory index, returns the best matches, and can optionally pass the retrieved context to an Ollama chat model for answer generation.

## What is in the project

```text
medical_codes_hybrid_rag/
  README.md
  requirements.txt
  hybrid_rag.py
  data/
    medical_codes_sample.jsonl
```

- `hybrid_rag.py` runs indexing, retrieval, and optional answer generation.
- `data/medical_codes_sample.jsonl` is a small sample dataset with ICD-10-CM, CPT, and HCPCS-style records.
- `requirements.txt` lists the Python dependency used by the project.

## How retrieval works

The pipeline uses two retrieval signals:

1. Keyword scoring for exact matches, code lookups, and term overlap.
2. Semantic scoring with embeddings for related phrases and looser wording.

Those two scores are normalized and blended into one hybrid score. If Ollama is not available, the script falls back to keyword-only retrieval so the search still works.

## Getting started

From the project folder:

```bash
cd /Users/alqmmba04/code/misc/machinelearning/medical_codes_hybrid_rag
../venv/bin/python hybrid_rag.py --self-check
../venv/bin/python hybrid_rag.py --query "type 2 diabetes"
../venv/bin/python hybrid_rag.py
```

Useful options:

- `--keyword-only` runs retrieval without embeddings.
- `--no-chat` returns matches without sending them to the chat model.
- `--data-path path/to/file.jsonl` points to a different dataset.
- `--top-n 10` returns more retrieval results.
- `--embed-model` and `--chat-model` let you swap models without editing the script.

## Data format

Each line in the dataset is one JSON object:

```json
{"code":"E11.9","title":"Type 2 diabetes mellitus without complications","description":"Use for type 2 diabetes when no complication is documented in the record.","synonyms":["type 2 diabetes","adult-onset diabetes"],"source":"ICD-10-CM sample"}
```

Required fields:

- `code`
- `title`
- `description`

Optional fields:

- `synonyms`
- `source`

## Expected use

This repo is best treated as a foundation, not a finished coding tool. It is a good place to start if you want to:

- test different retrieval strategies
- swap in a larger medical-code dataset
- add chunking or reranking later
- separate ICD, CPT, and HCPCS sources more cleanly
- build a small API or UI on top of the retrieval layer

## Notes

The sample data is only there to make the pipeline easy to run. For anything real, replace it with a proper source dataset and validate the output against your coding rules and documentation process.
