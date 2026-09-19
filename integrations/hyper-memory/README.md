# Hyper Memory integration

The Phase 1 integration is intentionally source-bound and write-gated:

```text
raw input -> deterministic filter -> HyperJev remember_worthy
          -> retain_raw | review | extract -> Hyper Memory /v1/documents
```

Use `hyperjev.integration.IngestionGate` from the Hyper Memory ingestion
worker. The gate never writes raw input to the HyperJev review or feedback
stores. It only sends the original content to Hyper Memory when the typed
decision is accepted as `extract`; `retain_raw` and `review` are returned to
the caller without an external write.

`HyperMemoryClient` targets the documented REST contract:

```text
POST http://127.0.0.1:6767/v1/documents
{
  "container": "project_alpha",
  "content": "...",
  "title": "optional",
  "source_type": "conversation"
}
```

The client is dependency-free and can be replaced by a native Hyper Memory
adapter later. The default HyperJev config points at
`http://127.0.0.1:6767`; deployment-specific values belong in TOML or the
service environment, not in source code.
