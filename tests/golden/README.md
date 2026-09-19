# Phase 0 fixtures

`phase0_smoke.jsonl` contains one non-sensitive contract fixture per initial
task. It is for schema and runner tests only; it is not the immutable,
human-reviewed 1,000-sample golden set required by the PRD. The real golden
set belongs in the DGX data volume with a manifest and hash once its review
process is defined.

Use `hyperjev golden generate --count 1000 --seed 7` to create a deterministic
pending synthetic review queue under `runs/phase0/`. Synthetic targets are not
human labels and therefore do not make the golden gate pass.
