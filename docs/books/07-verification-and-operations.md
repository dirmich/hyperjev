# 7장. 검증, DGX 운영, 그리고 남은 gate

## 7.1 반복 검증 명령

코드 변경 후 최소 검증은 아래 세 가지다.

```bash
uv run ruff check src tests
uv run pytest -q
git diff --check
```

CLI와 mock server를 바꾼 단계에서는 다음 smoke도 추가한다.

```bash
uv run hyperjev --version
uv run hyperjev registry validate
uv run hyperjev benchmark --dry-run --limit 1
uv run hyperjev golden --help
```

현재 마지막 확정 증거는 1.21.0 기준 Ruff 통과와 Python 테스트 70개 통과,
1개 skip이다. Golden review, metric, training validator, reference torch
checkpoint, calibration manifest gate가 포함되어 있다.

1.18.0부터 `hyperjev student evaluate`가 checkpoint 품질을 전체 정확도와
자동 수락 정확도로 분리한다. 자동 수락 정확도만 99%를 넘긴 경우에도
coverage와 fallback count를 함께 보지 않으면 제품 품질을 과대평가하게 된다.
1.20.1의 synthetic guarded path는 100%였지만 coverage 78.70%이며, human
golden 검증 전에는 production quality 통과로 기록하지 않는다.

1.21.0부터 실제 router에 optional Student checkpoint를 연결했다. checkpoint가
설정되면 route trace에서 `rule → student → qwen → gemma → human` 순서를 확인할
수 있다. reference n-gram checkpoint smoke는 약 5ms에 Student accepted를
반환했으며, confidence를 0.9999로 올린 smoke는 Student uncertain 뒤 teacher
connection error를 거쳐 human으로 fail-closed 되었다. 이 결과는 fallback
계약의 smoke이지 human golden 정확도 결과가 아니다.

1.22.0에서는 production accuracy의 정답 출처도 고정했다. human typed label이
있으면 그 label로만 `correct`를 계산하고, synthetic target은 human label이
없는 exploratory dataset에서만 사용한다. report의 `target_source`와
`golden.human_labeled`를 함께 확인해야 한다.

1.24.0의 `golden draft`는 Qwen 또는 Gemma를 reviewer assistant로만 사용한다.
Qwen `qwen38fn`은 빠른 초안 생성에 적합하고 Gemma는 독립 교차검증/judge
역할을 유지한다. 두 모델의 초안은 schema-valid 여부와 hash를 보존하지만
`labels.human`으로 복사되지 않는다. 사용자가 원문을 확인하고 correction을
별도로 제출해야 한다. 실제 Gemma4 1건은 60초 timeout으로 끝나 generation
병목이 재현되었으므로 Qwen retry를 우선 실행한다.

Parent LLM과 HyperJev 경계의 실제 성능 측정은 [`docs/test_result.md`](../test_result.md)에
기록했다. Qwen endpoint는 6개 smoke에서 HTTP 6/6, typed schema 5/6,
p50 1,127.542ms였고, 30개 synthetic 확장에서는 schema 27/30, p50
1,121.314ms였다. Gemma는 model discovery 후 generation timeout 상태였다.
확장 결과에서 score auto-accept가 부정확한 interval을 통과시킨 문제도 확인되어,
상용 정책은 confidence를 높이는 대신 calibration과 abstain/risk-coverage를
우선하도록 기록했다.
Rule/mock 수치는 production Student 성능이 아니므로 Student checkpoint가 준비될
때까지 별도 gate로 남긴다.

v1.132.0의 control replay는 두 층으로 보고한다. integrated 경로의 `40/40`과
약 `34~38µs`는 deterministic safety/phrase rule이 해결한 fast-path 결과이고,
Student-only 경로의 English `35/40`, Korean `34/40`이 encoder의 현재 OOD
기준선이다. integrated 숫자만 보고 Student가 모든 새로운 state를 100% 판단한다고
해석하지 않도록 source count와 fallback 비율을 함께 기록한다.

v1.135.0부터 CUDA latency는 warmup을 분모에서 제외하고 측정 전후
`torch.cuda.synchronize()`를 호출한다. 이 기준에서 OOD p95 약 10.4ms가 남았으므로
Student-only control tick은 아직 5ms actuator gate를 통과하지 못했다. fast-path
rule latency와 encoder latency를 같은 숫자로 합치지 않고, 각각의 source와
warmup/sync 조건을 report에 남긴다.

v1.137.0에서는 동일한 32k/256 checkpoint를 유지하면서 dense BOW count tensor를
만들지 않는 sparse projection을 검증했다. dense reference와의 수치 동치 test,
hard/English/Korean `100%` OOD, safety `1000/1000`과 STOP recall `500/500`을
확인했고 synchronized Student-only p95는 `0.159/0.182/0.183ms`였다. 이 결과는
5ms model latency gate는 통과하지만 human label `0/5192` 상태이므로 실기기
제어 또는 production promotion을 허용하지 않는다.

## 7.2 DGX Spark runbook

```bash
uv sync --frozen --extra dev
uv run hyperjev doctor
uv run hyperjev teacher check
uv run hyperjev benchmark --limit 1
```

`doctor`는 ARM64, Docker, `nvidia-smi`, disk, registry, teacher probe를 보고한다.
실제 generation은 300초 read timeout을 사용하고, connectivity probe는 3초로
분리한다. `HYPERJEV_QWEN_BASE_URL`, `HYPERJEV_GEMMA_BASE_URL`, model alias는
환경 변수로 덮어쓸 수 있다.

컨테이너 검사는 기존 localhost Qwen에 접근할 수 있도록 host networking을
사용한다.

```bash
docker compose --profile phase0 run --rm phase0-check
```

## 7.3 완료와 미완료를 구분하는 표

| 항목 | 현재 증거 | 판정 |
| --- | --- | --- |
| Python package/CLI | uv build, version command | 완료 |
| registry/contracts | unit/contract tests | 완료 |
| Qwen smoke | DGX one-sample live run, schema-valid | smoke 완료, 품질 gate 미완료 |
| Gemma baseline | `/models` probe 정상, 300초/900초 generation timeout | endpoint 정상, generation 병목 해결 필요 |
| synthetic queue | deterministic 1,000 queue | human golden 아님 |
| human golden | review/apply workflow | 1,000개 검수 필요 |
| dataset factory | unit tests와 privacy/agreement code | 실데이터 run 필요 |
| Student model | reference checkpoint/manifest | synthetic quality gate 미통과, production checkpoint 필요 |
| calibration/metrics | deterministic code/tests | held-out model data 필요 |
| Rust crates | workspace skeleton | host cargo 부재, DGX/toolchain compile 필요 |
| cache/registry/drift | local tests/metrics | production load gate 필요 |

이 표의 `필요`를 코드가 있다고 바꾸어 쓰지 않는 것이 책의 중요한 원칙이다.

2026-09-20의 live check에서 Gemma `google/gemma-4-12b`는 model discovery에는
응답했지만 한 개의 boolean sample도 900초 안에 completion하지 못했다. 이는
연결 실패가 아니라 현재 model/server 설정의 generation latency 문제다. 다음
운영 작업은 thinking 설정, model alias, server queue/GPU 사용량을 별도
benchmark하고, 실제 응답이 생기기 전에는 Gemma 품질 수치를 기록하지 않는
것이다.

## 7.4 실패를 기록하는 방법

실패는 숨기지 않고 원인 층을 분리한다.

- schema failure: typed parser/registry 계약 문제
- timeout: endpoint/GPU queue 또는 timeout 설정 문제
- disagreement: teacher label/confidence 불일치, review로 보냄
- privacy exclusion: training에 포함시키지 않는 정상 결과
- dependency unavailable: torch/cargo 같은 실행환경 준비 문제
- quality gate failure: code bug가 아니라 데이터/모델 품질 미달일 수 있음

## 7.5 실시간 control 적용의 검증 순서

게임과 로봇에서는 HyperJev를 저수준 actuator 대신 고수준 skill supervisor로
배치한다. `ControlObservation`은 한 시점의 bounded state를 받고,
`ControlStudentClient`는 `control.skill@1` typed head에서 `MOVE`, `ROTATE`,
`APPROACH`, `RETREAT`, `HOLD`, `INTERACT`, `RECOVER`, `STOP` 중 하나를
선택한다. 실제 PWM/토크/키 입력은 별도의 rate-limited controller가 담당한다.

검증은 다음 순서로 쪼갠다.

1. contract: 잘못된 parameter, 잘못된 clock, stale frame, emergency stop을
   모두 `STOP`으로 바꾸는지 unit test한다.
2. model: train이 아닌 validation/test의 unique state에서 typed accuracy와
   calibration을 측정한다. confidence가 높은 오답은 별도 실패로 기록한다.
3. simulator: action latency, collision/unsafe-action rate, stop recall,
   recovery latency를 episode 단위로 측정한다.
4. shadow: 실제 게임/로봇 제어기에는 연결하지 않고 parent policy와 결과를
   비교하며 HyperMemory read/write 및 fallback을 포함한 end-to-end 비용을
   측정한다.
5. canary: 승인된 skill subset과 속도/TTL 상한으로 제한된 환경에서만 실제
   실행한다. 모든 예외와 stale frame은 STOP 또는 controller hold로 보낸다.

v1.40.0의 48-sample synthetic smoke는 train `100%`였지만 validation
`37.5%`, test `12.5%`에 불과했다. 따라서 latency `p50 0.128ms`만으로 제품
적용을 선언하지 않는다. 현재 checkpoint는 reference encoder 실험 artifact이고,
사람 라벨과 simulator gate 전에는 actuator 연결 금지 상태다.

v1.41.0의 HyperMemory adapter는 `POST /v1/context`에서 compact context를
prefetch하는 경계를 제공한다. context는 최대 8개 summary/2,048자이며, 이를
실제 loop에 연결할 때는 memory timeout, cache hit, stale context age, context
사용/미사용별 action accuracy를 분리해서 기록해야 한다. HyperMemory 장애가
제어기를 멈추게 하지 않도록 bounded timeout 뒤에는 이전 context의 유효기간을
검사하고, 유효하지 않으면 Student confidence gate와 safety shield를 거쳐
STOP 또는 deterministic HOLD로 내려보낸다.

v1.139.0의 `control review-shell`은 human review를 batch 단위로 나누고 exact
duplicate를 접어 입력량을 줄인다. `state/question`과 flat candidate만 보여주며
teacher JSON은 숨긴다. 입력된 값은 registry validation을 통과한 뒤 append-only
feedback으로 저장되므로, 사람이 실제로 입력한 label만 다음 materialize gate에
사용된다.

v1.140.0의 review shell은 action을 `a`, `h`, `i`, `m`, `s`, `ro`, `rt`, `rc`로
입력할 수 있고 대소문자를 무시한다. `s`는 안전 동작인 `STOP`이므로 보류는
`sk`/`skip`으로 분리했다. alias는 표시 편의와 입력 속도만 바꾸며, typed registry
검증과 human-label gate는 그대로 유지한다.

v1.141.0의 review shell은 teacher draft에서 configured scalar도 표시한다.
`configured value`는 reviewer에게 현재 설정된 선택을 알려주지만 nested
probability와 synthetic target은 노출하지 않는다. 이 값은 참고용이며 human
correction을 대신하지 않는다.

v1.142.0의 partial human replay는 2개 human label 중 `2/2` correct였고, 전체
test coverage는 `2/384`였다. CPU model-only-with-safety-policy replay latency는
p95 `2.339ms`, p99 `2.839ms`, STOP recall `1.0`이었다. 이 latency와 safety
결과는 bounded-runtime 회귀 방지용이며 human test completeness와 별도 gate다.

v1.143.0의 NVIDIA GB10 CUDA synchronized replay는 p95 `162.480µs`, p99
`182.256µs`, max `221.263µs`였다. 같은 실행에서 STOP recall `1.0`과 human
label `2/2`를 확인했지만, test review status는 `2/384`이므로 promotion gate는
계속 닫혀 있다.

v1.144.0 candidate comparison에서 선택된 v1.137 weight2는 p95 `162.480µs`와
mixed test `384/384`를 유지했다. hybrid 후보의 `343/384` 회귀를 감지해 폐기한
것은 accuracy와 bounded latency를 함께 검증하는 모델 선택 gate다.

v1.145.0에서는 partial test와 구분해 `5192`-row augmented targeted queue 전체를
같은 checkpoint로 재생했다. model-only-with-safety-policy runtime source는
`hyperjev-control 4810`과 `safety-rule 382`였고, synthetic `5192/5192`, STOP
recall `1.0`, CUDA p95 `163.376µs`였다. 다만 human label `0/5192`와 test
coverage `2/384`는 그대로이므로 latency와 synthetic gate만 통과한 상태다.

v1.146.0에서 `runs/phase3/human-reviewed-1000-dataset.jsonl`을 대조해 보니,
사용자의 1,000개 human label은 존재하며 Phase 3 rule+Student는 `1000/1000`,
accepted `822/822`, coverage `82.20%`였다. 그러나 이는 control action과 다른
task registry이므로 control review status에는 포함하지 않는다. 평가 보고서는
row 수뿐 아니라 task registry, unique group, split까지 확인해야 한다.

v1.147.0의 독립 control replay는 v3 seed `4000/4000`, hard-combined `1800/1800`
및 STOP recall `1.0`을 확인했다. CUDA p95는 각각 `158.369µs`와 `162.017µs`다.
이 결과는 bounded latency와 synthetic 회귀를 보강하지만, human gate가 닫힌
상태에서 actuator production을 허용하는 근거는 아니다.

v1.148.0 공식 quality gate는 validation/test Wilson lower `0.992126`, safety
500개 STOP recall Wilson lower `0.992376`, false STOP `0`을 기록했다.
`passed=true`와 `production_ready=false`를 동시에 반환하는 이유는 synthetic
quality와 human provenance가 서로 다른 gate이기 때문이다.

v1.149.0에서 held-out human test 384개 전체의 acceptance boundary를 고정했다.
`384/384`이면 Wilson lower `0.990095`, `383/384`이면 `0.985398`이므로
point accuracy만 보고 승격하지 않는다. 현재 status `2/384`는 아직 이 gate를
계산할 충분한 human sample이 아니다.

v1.150.0 fresh adversarial replay는 새 hard-negative `1000/1000`과 세 가지
64-row queue `64/64`를 확인했다. 각 replay의 STOP recall은 `1.0`이고 CUDA
p95는 모두 `176.112µs` 이하였지만, human gate가 닫힌 상태는 변하지 않는다.
