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

Teacher generation uses the per-teacher read timeout by default: 300 seconds for
Qwen and 900 seconds for Gemma. Use `--timeout` only when deliberately overriding it for a controlled experiment;
the 3-second timeout used by `doctor` is only for the lightweight `/models`
probe.

Some OpenAI-compatible Gemma servers spend the whole completion budget in
reasoning unless the server supports an explicit reasoning control. For a
controlled probe, pass the option through the environment and record the model
alias separately; this does not change the repository default:

```bash
HYPERJEV_GEMMA_MODEL=gemma4-26b-a4b-uncensored-hauhaucs-balanced \
HYPERJEV_GEMMA_REASONING_EFFORT=none \
uv run hyperjev benchmark --provider gemma --limit 6 --timeout 30
```

The resulting manifest records `model`, `reasoning_effort`, and completion
metrics. A working response is still only a teacher draft, not a human golden
label or a production promotion decision.

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

Append each human correction to a separate feedback log. The correction is
validated against the task registry before it is accepted:

```bash
uv run hyperjev golden review \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --feedback-output runs/phase0/golden-feedback.jsonl \
  --sample-id phase0-synthetic-0001 \
  --reviewer reviewer@example.test \
  --correction-file /path/to/typed-correction.json
```

Build a reviewed copy only after collecting feedback. The source queue remains
unchanged and feedback is rejected if its queue hash is stale:

```bash
uv run hyperjev golden apply-feedback \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --feedback runs/phase0/golden-feedback.jsonl \
  --output runs/phase0/phase0-review-queue-reviewed.jsonl
uv run hyperjev golden validate \
  --samples runs/phase0/phase0-review-queue-reviewed.jsonl
```

## Container check

The Compose service is an explicit one-shot check and uses host networking so
it can reach the existing localhost Qwen service. It does not run a teacher
container.

```bash
docker compose --profile phase0 run --rm phase0-check
```
