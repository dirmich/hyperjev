# Phase 1 router and ingestion boundary

Phase 1 keeps the product decision boundary typed while the encoder Student is
still under development:

```text
request
  -> high-precision deterministic rule
  -> Qwen primary teacher
  -> Gemma independent judge
  -> abstain + append-only human review
```

Every attempt records provider, model, schema validity, response hash, latency,
acceptance status, and a SHA-256 state identifier. Raw state and raw teacher
responses are not written to the review or feedback JSONL stores.

The HTTP server exposes the existing `POST /v1/decide` contract in
`hyperjev serve --mode router`. Evidence traces are returned only when the
request sets `options.return_evidence=true`.

The Hyper Memory boundary is implemented by `IngestionGate`:

- empty or oversized input is handled by a deterministic filter;
- an accepted false `remember_worthy` result returns `retain_raw` without an
  external write;
- an abstention or failed chain returns `review`;
- only an accepted true result may call `POST /v1/documents` through
  `HyperMemoryClient`.

This keeps the Hyper Memory database separate from the review store and makes
teacher outages fail closed for memory extraction.
