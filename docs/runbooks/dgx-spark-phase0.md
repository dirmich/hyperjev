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

## Exercise the runner without teacher calls

```bash
uv run hyperjev benchmark --dry-run --limit 1
```

## Run a live smoke request

Use a non-sensitive fixture first. The run file contains hashes and metrics,
not raw state or raw teacher output.

```bash
uv run hyperjev benchmark --limit 1 --timeout 60
```

For a real Phase 0 baseline, replace the smoke fixture with the reviewed
dataset through a config override and preserve the printed dataset hash.

## Container check

The Compose service is an explicit one-shot check and uses host networking so
it can reach the existing localhost Qwen service. It does not run a teacher
container.

```bash
docker compose --profile phase0 run --rm phase0-check
```
