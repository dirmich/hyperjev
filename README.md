# HyperJev

HyperJev is the local-first typed decision layer for Hyper Memory. The product
model is an encoder with typed decision heads; `HyperJev-D` is kept as a
separate research track and is not part of the product baseline.

## Phase 1 status

The repository now contains the Phase 0 foundation, Phase 1 local router, and
the Phase 2 dataset builder:

- reproducible Python package configuration for ARM64/Linux;
- environment-specific Qwen/Gemma endpoint configuration without credentials;
- versioned task registry for the six initial decision tasks;
- typed request/response validation and a privacy-preserving baseline evaluator;
- Rust workspace placeholders for the future API, core, client, and registry;
- a deterministic `rule → Qwen → Gemma → human` decision router;
- append-only provenance/review/feedback stores without raw state persistence;
- a Hyper Memory ingestion gate and `/v1/documents` REST adapter.
- a Qwen/Gemma agreement-gated dataset factory with redaction, deduplication,
  privacy filtering, and grouped deterministic splits.

The human-reviewed 1,000-sample golden set and the measured teacher baseline
are intentionally not claimed complete yet. They are the next Phase 0 work
items after the contracts and connectivity checks are in place.

## Local setup

```bash
uv sync --extra dev
uv run pytest
uv run hyperjev registry validate
uv run hyperjev teacher check
uv run hyperjev benchmark --dry-run --limit 1
uv run hyperjev serve --host 127.0.0.1 --port 6777
uv run hyperjev serve --mode router --host 127.0.0.1 --port 6777
```

The default configuration is `configs/phase0.toml`. Override teacher endpoint
or model aliases with the `HYPERJEV_*` environment variables shown in
`.env.example`. The defaults reflect the endpoints observed during Phase 0
setup: Qwen on `127.0.0.1:8081/v1` and Gemma on `macmini:11434/v1`.

## Repository layout

```text
src/hyperjev/       Python Phase 0 contracts, registry, and CLI
registry/tasks/     Versioned task definitions
configs/            Environment and model configuration
crates/             Rust workspace skeleton for the production path
python/             Dataset/training/calibration/export work areas
tests/              Unit, contract, golden, and performance checks
docs/               Architecture, ADRs, and runbooks
```

Teacher requests are OpenAI-compatible and use Qwen for generation/fallback
and Gemma for independent cross-checking/judging, as specified by the PRD.

`hyperjev golden validate` intentionally fails until the required 1,000
human-reviewed samples are supplied. The repository contains only a six-record
non-sensitive smoke fixture, so it does not pretend that the Phase 0 quality
gate has passed.

The default `serve` mode is the Phase 0 contract-only mock server. The
`--mode router` server executes the Phase 1 rule/teacher chain; it is still not
the encoder Student inference runtime.

Create the deterministic synthetic review queue with:

```bash
uv run hyperjev golden generate --count 1000 --seed 7
```

The generated queue remains pending until reviewers populate `labels.human`;
synthetic targets never count as human review.
