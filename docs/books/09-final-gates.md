# 9장. 최종 gate 체크리스트

이 장은 책의 마지막에 붙일 실행 가능한 release checklist다. 각 명령은
성공/실패를 JSON 또는 테스트 출력으로 남기고, “실행하지 않음”을 “통과”로
바꾸지 않는다.

## 9.1 코드와 환경

```bash
uv sync --frozen --extra dev
uv run hyperjev --version
uv run hyperjev doctor --timeout 3
uv run hyperjev registry validate
uv run ruff check src tests
uv run pytest -q
```

성공 조건은 version, aarch64/Linux, Docker/NVIDIA, teacher `/models` probe,
registry 6개, Ruff, 전체 테스트의 증거가 모두 존재하는 것이다. 이 단계는
현재 DGX Spark에서 통과했다.

## 9.2 Teacher baseline

Qwen은 다음처럼 실행한다.

```bash
uv run hyperjev benchmark --provider qwen --limit 1 \
  --output runs/phase0/qwen-smoke.jsonl
uv run hyperjev evaluate \
  --run runs/phase0/qwen-smoke.jsonl \
  --samples tests/golden/phase0_smoke.jsonl
```

현재 확인된 Qwen one-sample은 약 0.6초 latency, typed boolean,
`schema_valid=true`였다. 이는 full quality baseline이 아니라 connectivity와
contract smoke다.

Gemma는 probe가 정상이어도 generation timeout을 별도로 판정한다.

```bash
uv run hyperjev benchmark --provider gemma --limit 1 \
  --output runs/phase0/gemma-smoke.jsonl
```

현재 config는 Gemma 900초 timeout이다. 2026-09-20 실행에서도 한 sample이
900초 안에 끝나지 않아 `TimeoutError`가 기록됐다. 따라서 Gemma full baseline,
teacher agreement, 품질 수치는 아직 gate 미통과다. 다음 시도 전에 thinking
설정, model alias, remote server queue, GPU utilization을 측정한다.

## 9.3 Human golden

```bash
uv run hyperjev golden generate --count 1000 --seed 7 \
  --output runs/phase0/phase0-review-queue.jsonl
```

사람 reviewer가 각 샘플에 대해 `golden review`를 실행하고, feedback을 모두
모은 뒤 reviewed copy를 만든다.

```bash
uv run hyperjev golden apply-feedback \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --feedback runs/phase0/golden-feedback.jsonl \
  --output runs/phase0/phase0-review-queue-reviewed.jsonl
uv run hyperjev golden validate \
  --samples runs/phase0/phase0-review-queue-reviewed.jsonl
```

`sample_count >= 1000`이고 `human_labeled_count == sample_count`일 때만
`ready: true`다. synthetic `target`만 존재하는 queue는 이 gate를 통과하지
못한다.

## 9.4 Dataset과 Student

```bash
uv run hyperjev dataset build \
  --seed runs/phase2/seed.jsonl \
  --output runs/phase2/dataset.jsonl
uv run hyperjev dataset validate --samples runs/phase2/dataset.jsonl
uv run hyperjev train plan \
  --dataset runs/phase2/dataset.jsonl \
  --output runs/phase3/training-plan.json
uv run hyperjev train run \
  --dataset runs/phase2/dataset.jsonl \
  --output runs/phase3/reference-student.pt \
  --device auto
```

`train run`은 PyTorch가 있는 DGX에서 실행되며 reference byte encoder를
사용한다. 현재 synthetic reference checkpoint와 manifest는 `trained`로
등록됐지만 test 정확도가 33.33%라 production 승격하지 않았다. production
multilingual tokenizer/backbone을 선택하고 사람 golden으로 checkpoint를
검증한 뒤에만 `evaluated` 이후 상태로 이동한다.

## 9.5 Serving과 promotion

```bash
uv run hyperjev serve --host 127.0.0.1 --port 6777
curl http://127.0.0.1:6777/health/ready
curl http://127.0.0.1:6777/v1/tasks
curl http://127.0.0.1:6777/metrics
```

검증된 Student checkpoint를 붙이는 smoke는 다음과 같다.

```bash
HYPERJEV_STUDENT_CHECKPOINT=runs/phase3/reference-ngram-student.pt \
HYPERJEV_STUDENT_MIN_CONFIDENCE=0.95 \
uv run hyperjev decide --request runs/phase3/student-smoke-request.json
```

응답 trace의 `route=student`와 `attempts[0].provider=student`를 확인한다.
confidence 미달은 `accepted=false`이고 Qwen fallback으로 이어져야 하며,
teacher까지 실패하면 `route=human`이어야 한다.

모델은 `registered → evaluated → calibrated → candidate → canary → active`를
순서대로 이동한다. quality/calibration, latency, privacy, rollback evidence가
없는 manifest를 active로 만들지 않는다. drift report는 promotion의 입력이지
자동 승격 명령이 아니다.

## 9.6 출시 판정

아래 중 하나라도 없으면 “HyperJev v1 완료” 대신 해당 gate의 상태를 보고한다.

- human-reviewed 1,000 golden과 immutable hash
- Qwen/Gemma/Rule/Student 동일 schema 비교 결과
- Student checkpoint와 calibration regression
- DGX 동시성별 p95/throughput/unified-memory 측정
- 큰 LLM 호출 감소율과 fallback/OOD 비율
- model registry canary/rollback/drift 증거
- Hyper Memory ingestion corruption 및 failure-injection 결과

현재 실행된 parent/teacher 및 rule/mock 경계 시험의 원본 기록은
[`docs/test_result.md`](../test_result.md)다. 이 기록은 production Student의
성능 통과를 의미하지 않으며, checkpoint·동일 golden·동시성 matrix가 추가로
필요하다.
