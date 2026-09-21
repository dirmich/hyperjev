# 4장. Qwen과 Gemma를 안전하게 쓰는 법

## 4.1 역할 분리

Qwen은 generation/label/fallback의 빠른 초안 역할이다. Gemma는 독립적인
cross-validation과 judge 역할이므로 Qwen과 같은 오류를 반복하지 않도록
prompt와 endpoint metadata를 분리한다. Router는 둘의 typed output을
`parse_decision_result`와 task registry로 검증한 뒤 agreement와 confidence를
판정한다.

Teacher response에 대해 보존하는 것은 모델명, latency, token count,
response hash, schema validity다. raw response와 raw state는 baseline run과
review log에 넣지 않는다.

## 4.2 실행 예

```bash
uv run hyperjev teacher check
uv run hyperjev benchmark --dry-run --limit 1
uv run hyperjev benchmark --provider qwen --limit 1 \
  --output runs/phase0/qwen-smoke.jsonl
uv run hyperjev evaluate \
  --run runs/phase0/qwen-smoke.jsonl \
  --samples tests/golden/phase0_smoke.jsonl
```

실제 Qwen one-sample benchmark는 DGX Spark의 ARM64/Linux 환경에서 실행됐고
typed boolean 결과, schema validity, latency, hash-only metadata를 반환했다.
Gemma 전체 benchmark와 production quality 수치는 아직 이 저장소에서
완료했다고 주장하지 않는다.

## 4.3 Router 흐름

```text
request
  → registry/task validation
  → high-precision rule
  → Qwen
  → Gemma cross-check/judge
  → human review queue
  → typed response + trace
```

rule은 정밀도를 우선한다. Qwen이 오류를 내거나 deadline을 넘기면 Gemma로
넘기고, 두 teacher의 label 또는 confidence가 정책상 불일치하면 human route로
보낸다. `allow_fallback=false`와 deadline은 request contract에 포함되어
있으므로 “항상 모델을 호출한다”는 암묵적 동작을 허용하지 않는다.

## 4.4 Golden review workflow

queue를 생성한다.

```bash
uv run hyperjev golden generate --count 1000 --seed 7
```

reviewer는 typed correction 하나를 append한다.

```bash
uv run hyperjev golden review \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --feedback-output runs/phase0/golden-feedback.jsonl \
  --sample-id phase0-synthetic-0001 \
  --reviewer reviewer@example.test \
  --reason "checked against source" \
  --correction '{"type":"boolean","value":true,"probability":1.0,"abstained":false}'
```

그 다음 원본을 건드리지 않는 reviewed copy를 만든다.

```bash
uv run hyperjev golden apply-feedback \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --feedback runs/phase0/golden-feedback.jsonl \
  --output runs/phase0/phase0-review-queue-reviewed.jsonl
uv run hyperjev golden validate \
  --samples runs/phase0/phase0-review-queue-reviewed.jsonl
```

이 명령은 correction의 타입과 candidate를 registry 기준으로 확인한다.
feedback은 queue SHA-256과 함께 저장되고, queue가 바뀌면 apply가 실패한다.
따라서 mutable JSONL을 사람 검수의 유일한 audit trail로 사용하지 않는다.

v1.138.0에서는 DGX Spark의 Qwen endpoint를 300초 timeout으로 probe하고,
Qwen+Gemma adjudication과 Student checkpoint provenance를 target-excluded
review pack에 묶었다. teacher disagreement `1/384`와 Student uncertainty
`0/384`는 검수 순서를 정하는 신호다. teacher가 synthetic target과 일치해도
`labels.human`을 자동으로 만들지 않으며, 실제 사람이 state/question을 보고
typed value를 입력해야만 golden gate가 열린다.

v1.140.0에서는 control review 입력을 한두 글자 alias로 단축했다. `a/h/i/m/s`와
`ro/rt/rc`는 각각 registry action으로 변환되고, 입력 비교는 대소문자를
무시한다. `s`가 `STOP`이므로 보류 명령은 `sk`/`skip`이다. alias도 full action과
마찬가지로 registry validation을 거치며, reviewer가 실제 입력한 값만 golden
feedback으로 남는다.

v1.141.0에서는 review pack의 teacher normalized result에서 scalar만 추려
`configured value`로 표시한다. reviewer는 설정된 `APPROACH`나 `STOP`을 확인할
수 있지만 nested 확률, provider metadata, synthetic target은 보지 않는다.
설정값이 없는 item은 `not provided`로 표시되며, 사람의 correction만 feedback에
기록된다.

v1.142.0에서는 첫 control human review 2건이 feedback에 기록됐다. 두 건 모두
Student prediction과 일치했지만 test queue는 `2/384`만 검수되어 아직 human
99% gate를 계산할 수 없다. synthetic target과 human target을 분리해 보고하는
원칙을 유지한다.

v1.143.0에서는 NVIDIA GB10에서 CUDA synchronize를 포함한 partial replay를
실행했다. p95 `162.480µs`, STOP recall `1.0`, human `2/2`는 각각 runtime,
safety, human-label 신호이며 서로 합쳐서 99% accuracy로 계산하지 않는다.

v1.144.0에서는 BOW weight2, BOW weight4, hybrid weight2 후보를 같은 입력으로
비교했다. hybrid는 human 2건은 맞췄지만 mixed test `343/384`로 탈락했고,
최종 선택은 v1.137 sparse BOW weight2로 고정했다.

v1.145.0에서는 선택된 checkpoint를 `5192`-row augmented targeted queue 전체에
재생했다. synthetic target은 `5192/5192`, STOP recall은 `1.0`이었고 CUDA
synchronized p95는 `163.376µs`였다. 그러나 이 queue의 human label은 `0/5192`이며
별도 test review도 `2/384`뿐이다. teacher가 보여준 configured value나 synthetic
target과 사람이 입력한 correction은 서로 다른 provenance로 끝까지 분리한다.

control용 `review-shell`은 teacher JSON을 숨기고 `state`, `question`, 허용 action만
보여준다. reviewer가 `STOP` 또는 `MOVE`처럼 flat value를 입력하면 registry가
검증한 typed feedback을 append한다. exact duplicate는 한 번만 묻지만 audit trail은
각 sample ID에 남기므로 review group 수와 label 수를 분리해 보고한다.
