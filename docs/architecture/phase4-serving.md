# Phase 4 serving and shadow boundary

The local API now supports both the single request and batch contracts:

```text
POST /v1/decide
POST /v1/batch/decide
        -> typed request validation
        -> shared-state router execution
        -> typed responses
```

Batch requests are a transport optimization only; each item retains its own
request identity, route, latency, and evidence boundary. They do not bypass
the rule/Qwen/Gemma/human fallback policy.

`ShadowLogger` is the promotion boundary for future Student-vs-router
comparison. It persists request/state hashes, model identifiers, and result
signatures without raw state or raw result payloads. A model can therefore be
evaluated in shadow traffic before it is allowed to change Hyper Memory writes.
