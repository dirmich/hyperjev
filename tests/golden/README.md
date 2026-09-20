# Phase 0 fixtures

`phase0_smoke.jsonl` contains one non-sensitive contract fixture per initial
task. It is for schema and runner tests only; it is not the immutable,
human-reviewed 1,000-sample golden set required by the PRD. The real golden
set belongs in the DGX data volume with a manifest and hash once its review
process is defined.

Use `hyperjev golden generate --count 1000 --seed 7` to create a deterministic
pending synthetic review queue under `runs/phase0/`. Synthetic targets are not
human labels and therefore do not make the golden gate pass.

Gemma를 human 검수용 초안으로 사용할 수 있다. 초안은 human label이 아니며,
사용자가 원문 `state`와 `question`을 확인한 뒤에만 `golden review`로 승인한다.

```bash
uv run hyperjev golden draft \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --output runs/phase1/gemma-golden-draft.jsonl \
  --limit 20 --timeout 900
```

draft record는 `sample_id`, `task`, `normalized_result`, `schema_valid`와
response hash를 포함하지만 raw 입력과 raw teacher 응답은 저장하지 않는다.
Gemma가 timeout/error를 낸 sample은 자동으로 human label이 되지 않는다.

Reviewers append typed corrections without modifying the source queue:

```bash
uv run hyperjev golden review \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --feedback-output runs/phase0/golden-feedback.jsonl \
  --sample-id phase0-synthetic-0001 \
  --reviewer reviewer@example.test \
  --reason "checked against source" \
  --correction '{"type":"boolean","value":true,"probability":1.0,"abstained":false}'
```

After reviews are collected, create a separate reviewed queue copy and validate
it. Feedback is bound to the source queue hash, so a regenerated or edited
queue cannot silently receive labels from another queue:

```bash
uv run hyperjev golden apply-feedback \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --feedback runs/phase0/golden-feedback.jsonl \
  --output runs/phase0/phase0-review-queue-reviewed.jsonl
uv run hyperjev golden validate \
  --samples runs/phase0/phase0-review-queue-reviewed.jsonl
```
