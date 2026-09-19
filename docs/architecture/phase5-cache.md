# Phase 5 cache boundary

Cache keys are composed from model/calibration versions, task references,
normalized question text, candidates, and SHA-256 digests. Raw state is never
used as a persisted key or emitted in cache metadata.

`BoundedCache` is an in-process LRU primitive for the first serving path. The
default config enables it with 1,024 entries, but it can be disabled or resized
under `[cache]`. The cache is intentionally bounded and version-scoped so model
promotion and calibration changes cannot reuse stale decisions. Sensitive
deployments can disable the cache or replace it with an encrypted
implementation without changing the key contract.
