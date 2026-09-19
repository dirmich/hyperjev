# Phase 2 dataset factory

`hyperjev.dataset_factory.build_dataset` implements the first reproducible
dataset path over seed JSONL records:

```text
seed records
  -> source/privacy gate
  -> deterministic redaction
  -> Qwen label + Gemma independent label
  -> typed schema validation
  -> agreement/confidence filter
  -> deduplication
  -> source/entity grouped split
  -> normalized dataset + review queue
```

Silver records contain only normalized typed results and minimal provenance;
teacher reasoning and raw teacher text are never copied. Human corrections in
`labels.human` override teacher disagreement after the same typed validation.
Disagreements, malformed teacher responses, and low-confidence pairs remain in
the review JSONL with redacted state and response hashes/metadata.

The current implementation is a builder and validator, not a claim that the
20k–50k POC dataset or the human-reviewed golden set already exists. Production
generation should run with an explicit seed snapshot and preserve the emitted
dataset hash.
