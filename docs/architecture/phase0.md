# Phase 0 architecture and environment baseline

## Fixed decisions

- Product path: encoder + typed decision heads.
- Research path: `HyperJev-D` remains a separate diffusion experiment.
- Qwen is the primary data generator, labeler, and fallback teacher.
- Gemma is an independent cross-validator and judge; it is not silently
  treated as the primary generator.
- Teacher access is OpenAI-compatible HTTP and model aliases are configuration
  values, not code constants.
- `/models` health probes use a short timeout, while generation requests use a
  separate per-teacher read timeout (300 seconds for Qwen and 900 seconds for
  Gemma) because llama.cpp generation can
  legitimately take much longer than endpoint discovery.
- Raw state and teacher output are not written to baseline run files by
  default. A SHA-256 response hash plus latency/token metadata is retained.

## Observed DGX development environment

The initial setup was checked on an ARM64 Linux host with an NVIDIA GB10,
CUDA driver 580.142, Docker 29.2.1, Python 3.10 available on the host, and
the project-managed Python environment supplied by `uv`.

The configured endpoints responded to `GET /v1/models` during setup:

| Teacher | Endpoint | Configured alias | Roles |
| --- | --- | --- | --- |
| Qwen | `http://127.0.0.1:8081/v1` | `qwen38fn` | data generation, labeler, fallback |
| Gemma | `http://macmini:11434/v1` | `google/gemma-4-12b` | cross-validator, judge |

The Gemma server advertises more than one model. The alias above is a
configuration default because it was present in the observed model list; it
must be confirmed before a production baseline is announced. The Qwen model
metadata also does not independently prove the PRD's “8B” description, so the
endpoint alias is recorded as evidence rather than a model-size claim.

## Phase 0 gate

The repository now has the runner needed to reproduce the measurement, but the
Phase 0 gate is still open until a human-reviewed 1,000-sample set is supplied.
The six-record fixture is only a contract smoke set. A complete gate run must
record schema-valid rate, Qwen/Gemma agreement, task quality against human
targets, p50/p95/p99 latency, token counts, timeout rate, and the environment
manifest for the same dataset hash.
