# DGX Spark Phase 0 runbook

## Check the environment

```bash
uv sync --frozen --extra dev
uv run hyperjev doctor
uv run hyperjev teacher check
```

`doctor` is read-only. It checks ARM64, Docker, `nvidia-smi`, disk space,
registry loading, and both configured teacher aliases. It does not start or
stop llama.cpp, modify model files, or emit raw prompts.

## Validate the six task registry

```bash
uv run hyperjev registry validate
```

The initial tasks are `memory.remember_worthy`, `memory.type`,
`memory.importance`, `query.route`, `memory.relation`, and
`wiki.semantic_change`.

Run the contract-only mock API for client integration checks:

```bash
uv run hyperjev serve
curl http://127.0.0.1:6777/health/ready
```

## Exercise the runner without teacher calls

```bash
uv run hyperjev benchmark --dry-run --limit 1
```

## Run a live smoke request

Use a non-sensitive fixture first. The run file contains hashes and metrics,
not raw state or raw teacher output.

```bash
uv run hyperjev benchmark --limit 1
```

Teacher generation uses the configured 300-second read timeout by default. Use
`--timeout` only when deliberately overriding it for a controlled experiment;
the 3-second timeout used by `doctor` is only for the lightweight `/models`
probe.

For a real Phase 0 baseline, replace the smoke fixture with the reviewed
dataset through a config override and preserve the printed dataset hash.

Evaluate a completed run without exposing raw state in the report:

```bash
uv run hyperjev evaluate --run runs/phase0/teacher-baseline-<run-id>.jsonl
uv run hyperjev golden validate
```

The golden command must report `ready: true` before the Phase 0 gate can be
called complete. The checked-in smoke fixture is expected to report `ready:
false`.

Generate the reproducible synthetic queue that reviewers can work through:

```bash
uv run hyperjev golden generate --count 1000 --seed 7
```

This creates a pending queue under `runs/phase0/`. Its synthetic targets are
useful for runner smoke tests but do not satisfy the human-label requirement.

## Container check

The Compose service is an explicit one-shot check and uses host networking so
it can reach the existing localhost Qwen service. It does not run a teacher
container.

```bash
docker compose --profile phase0 run --rm phase0-check
```
