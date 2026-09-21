# Phase 0 fixtures

`phase0_smoke.jsonl` contains one non-sensitive contract fixture per initial
task. It is for schema and runner tests only; it is not the immutable,
human-reviewed 1,000-sample golden set required by the PRD. The real golden
set belongs in the DGX data volume with a manifest and hash once its review
process is defined.

Use `hyperjev golden generate --count 1000 --seed 7` to create a deterministic
pending synthetic review queue under `runs/phase0/`. Synthetic targets are not
human labels and therefore do not make the golden gate pass.

Qwen `qwen38fn` 또는 Gemma4를 human 검수용 초안으로 사용할 수 있다. Qwen은
빠른 초안 생성에 적합하지만, 초안은 human label이 아니며,
사용자가 원문 `state`와 `question`을 확인한 뒤에만 `golden review`로 승인한다.
구체적인 `memory.remember_worthy` 판정 기준은
[`docs/human_labeling.md`](../../docs/human_labeling.md)를 따른다.

```bash
uv run hyperjev golden draft \
  --provider qwen \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --output runs/phase1/qwen-golden-draft.jsonl \
  --limit 20 --timeout 300
```

draft record는 `sample_id`, `task`, `normalized_result`, `schema_valid`와
response hash를 포함하지만 raw 입력과 raw teacher 응답은 저장하지 않는다.
Gemma 초안은 `--provider gemma`로 실행한다. 두 provider 모두 draft를
`labels.human`으로 자동 승격하지 않는다. teacher가 timeout/error를 낸
sample은 자동으로 human label이 되지 않는다.

사람이 원문과 Qwen 초안을 한 화면/파일에서 검수하려면 raw 입력 저장을
명시적으로 허용해 review pack을 만든다. pack에는 synthetic `target`과 queue
`labels`를 넣지 않는다.

```bash
uv run hyperjev golden review-pack \\
  --queue runs/phase0/phase0-review-queue.jsonl \\
  --draft runs/phase1/qwen-golden-draft-1000-prompt-v2.jsonl \\
  --output runs/phase1/qwen-golden-review-pack.jsonl \\
  --allow-raw
```

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

1,000건을 한 번의 세션에서 검수하려면 다음 명령을 사용한다. 화면에는
`state`, `question`, Qwen draft와 현재 human label이 표시된다.

```bash
uv run hyperjev golden review-session \
  --review-pack runs/phase1/qwen-golden-review-pack.jsonl \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --feedback-output runs/phase0/golden-feedback.jsonl \
  --reviewer dirmich \
  --deduplicate-exact
```

키 입력은 `a`(draft 승인), `e`(값 수정), `n`(다음), `p`(이전),
`s`(보류), `q`(저장 후 종료)다. 승인·수정은 즉시 feedback에 append되며,
이전에 저장된 item도 `p`로 돌아가 다시 수정할 수 있다.
`e`를 누르면 boolean은 `true/false`, choice는 후보 문자열, score는 `0~1`
숫자만 입력한다. JSON 전체를 작성할 필요가 없다.

`--deduplicate-exact`를 사용하면 `task`, `language`, `domain`, `state`,
`question`이 모두 같은 item을 한 그룹으로 표시한다. 대표 item을 승인하거나
수정하면 같은 그룹의 sample IDs에도 동일한 typed feedback이 기록되므로,
반복되는 synthetic sample을 다시 판단하지 않아도 된다. 이 전파 label은
동일 원문에 대한 파생 label이지 독립 human sample이 아니므로, production
golden 정확도 보고서에서는 unique group 수를 함께 표시해야 한다.

control task만 검수할 때는 nested teacher 결과를 숨기는 flat-value 쉘을 사용할 수
있다. exact duplicate는 기본으로 하나의 그룹으로 접힌다.

```bash
uv run hyperjev control review-shell \
  --review-pack /tmp/control-review-v3-test-pack-v137-student.jsonl \
  --queue /tmp/control-review-v3-test-384.jsonl \
  --feedback-output runs/control/control-human-feedback.jsonl \
  --reviewer dirmich --offset 0 --limit 50
```

각 화면에서 `state`, `question`, `allowed values`만 보고 `value>`에
`STOP`/`MOVE` 등의 값을 직접 입력한다. `n`/`p`/`s`/`q`는 다음/이전/보류/종료다.
쉘은 teacher draft를 승인하지 않으며, 입력값을 registry 기준으로 검증한 뒤
sample별 append-only feedback으로 저장한다.
