# Dataset factory

Phase 2 uses `hyperjev.dataset_factory.build_dataset` and the CLI wrapper:

```bash
uv run hyperjev dataset build \
  --seed runs/phase2/seed.jsonl \
  --output runs/phase2/dataset.jsonl
uv run hyperjev dataset validate --samples runs/phase2/dataset.jsonl
```

The factory calls Qwen and Gemma independently, validates typed outputs,
redacts common email/phone/token patterns, gates silver labels on agreement,
keeps disagreement in a review JSONL, deduplicates normalized state/question
pairs, and assigns source/entity groups to deterministic train/validation/test
splits. User or secret sources are excluded by default; enabling user-data
training is an explicit configuration change.

Teacher reasoning is never copied into the dataset. Only normalized typed
labels, minimal teacher metadata, hashes, and provenance are emitted.
