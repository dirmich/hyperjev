# HyperJev를 처음부터 끝까지 만드는 책

이 디렉터리는 HyperJev를 빈 저장소에서 시작해 DGX Spark의 실제 서비스로
가져가기까지의 과정을 책으로 엮기 위한 원고 저장소다. 구현이 진행될 때마다
설계 이유, 실행 명령, 테스트 증거, 커밋을 이 원고에 누적한다. 최종 출판본은
이 파일을 목차로 사용하고 각 장의 `상태`와 `남은 gate`를 최신 값으로 갱신해
만든다.

## 독자가 얻게 될 것

- `docs/prd.md`를 구현 가능한 계약과 단계별 gate로 분해하는 방법
- Qwen3 8B와 Gemma 4를 teacher로 배치하는 local-first 개발 환경
- rule → optional Student → Qwen → Gemma → human fallback의 typed decision router
- encoder + typed decision heads Student와 별도 연구 트랙 HyperJev-D의 경계
- privacy-safe dataset factory, calibration, golden review, shadow serving,
  registry, drift monitoring을 한 흐름으로 연결하는 방법
- DGX Spark에서 실제로 통과시켜야 하는 검증과 아직 외부 자원이 필요한 검증

## 원문과 구현의 우선순위

1. 제품 계약은 [`docs/prd.md`](../prd.md)가 기준이다.
2. 실제 실행 계약은 `registry/tasks/`, `configs/phase0.toml`,
   `src/hyperjev/contracts.py`가 기준이다.
3. 이 책은 왜 그렇게 구현했는지와 어떤 증거를 얻었는지를 설명하는
   학습 자료다. 책과 코드가 어긋나면 코드를 고친 뒤 이 원고를 갱신한다.

## 장 목록

| 장 | 파일 | 내용 | 현재 상태 |
| --- | --- | --- | --- |
| 1 | [범위와 문제](01-scope-and-problem.md) | HyperJev의 역할과 경계 | 초안 완료 |
| 2 | [PRD 해부](02-prd-analysis.md) | 요구사항, 출력 계약, gate | 초안 완료 |
| 3 | [Phase 0 기반](03-phase0-foundation.md) | 환경, 구조, 첫 테스트 | 초안 완료 |
| 4 | [Teacher와 Golden](04-teachers-and-golden.md) | Qwen/Gemma baseline과 human review | 초안 완료 |
| 5 | [Router에서 Serving까지](05-router-to-serving.md) | Phase 1~7의 누적 설계 | 초안 완료 |
| 6 | [Dataset, Student, Training](06-dataset-student-training.md) | 학습 데이터와 typed-head 준비 | 진행 중 |
| 7 | [검증과 운영](07-verification-and-operations.md) | 테스트, DGX, 관측성, 남은 gate | 진행 중 |
| 8 | [구현 일지](08-implementation-ledger.md) | 단계별 version/commit/push 증거 | 계속 갱신 |
| 9 | [최종 gate 체크리스트](09-final-gates.md) | 재현 명령과 출시 전 판정 | 초안 완료 |

## 현재 스냅샷

현재 `main`은 origin에 push될 1.77.0까지 진행되어 있다. Python 테스트와 skip
수는 최신 전체 suite 실행 결과를 기준으로 기록하며, Ruff 검사가 통과한
상태이다. dataset validator와 training plan,
reference train CLI, reference Student checkpoint 생성, n-gram Student 비교와
guarded 99% 평가, optional Student router 연결, parent LLM 대 HyperJev
경계의 성능 시험 기록까지 완료됐다. training plan·calibration·checkpoint를
registry manifest로 묶는 contract와 30-sample confidence/coverage 분석도
추가됐다. memory/query checkpoint는 synthetic 36개 group의
reference-ngram-encoder이고 control checkpoint는 별도 48개 smoke fixture다.
둘 다 human golden이 없거나 일반화 gate를 통과하지 않았으므로 production
모델로 승격하지 않았다. hard-negative synthetic checkpoint는 validation/test
100%였지만 human label 0건이라 production-ready가 아니다. 실제
사람 검수 golden set, Gemma live baseline 전수 실행, production multilingual
PyTorch checkpoint 학습, Student GPU inference benchmark, Rust toolchain compile은
이 책에서 성공했다고 가장하지 않고 외부 의존성 gate로 표시한다. 상세 결과는
[`docs/test_result.md`](../test_result.md)에 있다.

v1.58.0에서는 control evaluator에 threshold별 risk-coverage를 추가했다.
synthetic combined checkpoint가 confidence `0.50/0.90/0.95/0.99`에서 모두
100% accepted accuracy를 보였어도 class-balanced 후보의 safety action은
75%였으므로, confidence와 정확도·안전성을 같은 숫자로 취급하지 않는 원칙을
책의 검증 장에 반영했다.

v1.59.0에서는 held-out logits에서 task별 temperature를 계산하는 control
calibration manifest를 추가했다. 실제 synthetic validation 180건에서 NLL은
줄었지만 temperature가 검색 하한에 붙고 human label이 0건이어서 calibration
artifact를 production 품질 증거로 승격하지 않았다.

v1.60.0에서는 calibration manifest를 Student runtime에 연결했다. task별
temperature를 확률/abstain에 적용하되 checkpoint SHA가 맞지 않으면 거부해,
calibration 파일만 바꾸어 품질을 위장하거나 다른 모델에 잘못 적용하는 경로를
막았다.

v1.61.0에서는 Qwen 800건 control draft를 완료했다. synthetic target은
800/800이었지만 2건은 확률합 repair가 필요했고, p95 2.36초라 실시간 motor
loop가 아니라 비동기 teacher 경로로 분리했다. teacher의 100% 결과는 human
golden 정확도 증거가 아니다.

v1.62.0에서는 teacher draft를 반복 측정하는 evaluator를 추가했다. queue와
draft hash를 고정하고 synthetic match, schema repair, latency, human gate를
분리하므로 Qwen 초안의 100%를 production 정확도로 오해하지 않는다.

v1.63.0 hard-negative 실험에서 Qwen은 STOP에는 100%였지만 HOLD 0%,
APPROACH 49.70%였다. 이 결과를 통해 teacher가 만든 counterfactual label도
사람 검수 전에는 학습 target이 될 수 없고, 오류 pair를 우선 검수해야 한다는
운영 규칙을 책에 추가했다.

v1.64.0에서는 `control review-pack --prioritize`를 추가했다. review pack은
invalid schema, repair가 필요한 teacher 결과, 낮은 typed confidence 순으로
정렬하고 manifest에 `uncertain_first`를 기록한다. 이 정렬은 synthetic target을
보지 않으므로 target leakage가 없으며, 사람이 먼저 확인할 위험 row를 줄여준다.
다만 아직 human label은 0건이어서 이 기능은 검수 throughput 개선이지 정확도
증거가 아니다.

v1.65.0에서는 hard-negative pair의 provenance를 review item에 안전하게 남기고,
priority 정렬 시 counterfactual group 전체를 연속 배치했다. 따라서 한쪽만 보고
다음 500개 뒤에 반대쪽을 보는 검수 순서를 피할 수 있다. target과 labels는 여전히
pack에서 제외되므로, 이 기능은 human 검수 품질과 일관성을 높이는 장치이지
pseudo-label 생성기가 아니다.

v1.66.0에서는 실제 `control review-pack` CLI handler가 `--prioritize`를
`export_review_pack`까지 전달하는지 고쳤다. Qwen hard-negative 1,000건을 실제로
재생성해 `uncertain_first`, 500 pair, adjacency violation 0, target/labels
누출 false를 확인했다. 이 smoke가 helper만 통과하고 CLI가 queue order를 내는
회귀를 잡았으므로 이후 단계의 운영 검증 기준에 포함한다.

v1.67.0에서는 generic `golden review-pack` parser에도 `--prioritize`를 추가해
control과 memory review 도구의 CLI 계약을 일치시켰다. handler만 수정하고
parser를 놓치는 회귀를 막기 위해 두 parser flag를 각각 테스트한다.

v1.68.0에서는 confidence만 믿을 수 없는 teacher 오류를 잡기 위해
counterfactual pair collision을 최우선 검수로 보냈다. 실제 Qwen hard-negative에서
HOLD 오답 confidence가 1.0이었지만 pair collision 167개가 검출되어 앞쪽에
배치됐다. 이는 target을 노출하지 않고도 semantic contradiction을 이용하는
active-review 개선이다.

v1.69.0에서는 review manifest에 active-review 범위를 구조화했다. 실제 Qwen
hard-negative는 500 counterfactual groups, 167 collision groups, 334 collision
items로 기록되므로, 운영자가 JSONL을 재분석하지 않고도 우선 검수량을 알 수 있다.

v1.70.0에서는 hard-negative provenance를 loss 가중치로 반영하는 연구 옵션을
추가했다. weight 2.0/4.0은 synthetic split accuracy 100%를 유지했지만 safety
action accuracy를 75%로 떨어뜨려 탈락했다. 이 실패를 기록한 이유는 human
hard-negative가 들어온 뒤 같은 실험을 재현하되, STOP 안전 gate를 절대 양보하지
않기 위해서다.

v1.71.0에서는 Gemma4 hard-negative 1건 probe가 30초 timeout으로 실패한 결과를
기록했다. Gemma는 control tick의 parent가 아니라 async judge로 격리하고, 실제
golden target은 human reviewer가 확정해야 한다는 운영 경계를 재검증했다.

v1.72.0에서는 hard-negative의 `APPROACH↔MOVE`, `HOLD↔STOP` 혼동을 보완하는
compound-phrase control fast path를 추가했다. hard-negative 1,000행은 synthetic
target 기준 100%를 약 10.11µs/행 평균으로 처리했고, seed 800행은 750행을
규칙으로 확정했다. 이는 human label이 0건인 synthetic regression 결과이므로
production accuracy로 해석하지 않는다. 모호한 입력은 model/fallback으로
남기고 explicit STOP은 safety rule로 먼저 처리한다.

v1.73.0에서는 부정문·위험문이 compound phrase를 포함해도 fast path가 오작동하지
않도록 skill별 contradiction blocker를 추가했다. 정상 HOLD의 `no hazard`가
차단되지 않는 회귀와, 접근 불가/비상 위험 반례의 fast path 우회 테스트를 모두
통과했다. 이는 coverage 확장이 아니라 안전 false-positive를 줄이는 단계다.

v1.74.0에서는 fast path 전용 evaluator를 저장소에 추가했다. queue SHA와
source별 coverage, p50/p95/p99 latency, synthetic/human target을 분리해 같은
명령으로 재현할 수 있다. hard-negative는 100% coverage와 100% synthetic match를
보였지만 human `0/1000`, production `false`를 유지한다.

v1.75.0에서는 evaluator에 실제 checkpoint runtime replay를 추가했다. 같은
hard-negative를 `ControlStudentClient`로 재생해 rule/safety source, synthetic
accuracy, STOP recall, runtime latency를 별도로 확인한다. 이번 queue는
100%였지만 모든 row가 known fast path였고 human `0/1000`이므로, 실제 일반화
정확도와 model fallback latency gate는 여전히 남아 있다.

v1.76.0에서는 runtime latency를 source별로 나눴다. seed queue의 local
reference Student fallback은 p50/p95/p99 `293.486/421.518/2972.318µs`였고,
control-rule p95는 `14.112µs`였다. 이 차이는 teacher가 아니라 local Student
fallback 비용이며, human golden과 GPU/e2e gate가 없으므로 production 수치가
아니다.

v1.77.0에서는 40개 compositional OOD fixture에서 model-only 정확도 `10%`,
STOP recall `0%`와 integrated runtime 정확도 `100%`, STOP recall `100%`를
나란히 측정했다. 이 차이는 fast path가 모델 오류를 가리는 것이 아니라 안전
경계의 일부라는 사실을 보여준다. fixture는 synthetic/human `0/40`이므로
production 정확도 증거가 아니다.

v1.78.0에서는 OOD 자체를 train에 복사하지 않은 64개 compositional train 문장을
추가해 model-only 정확도를 `95%`까지 올렸고, integrated runtime은 `100%`를
유지했다. model-only STOP recall은 명시적 collision safety interlock을 항상 켜도록
고정해 `100%`가 됐다. 다만 human label `0/40`이고 model-only HOLD 오분류가 남아
있으므로 상용 99% gate나 모델 정확도 100%로 승격하지 않는다.

v1.79.0에서는 모든 skill의 semantic boundary 문장을 균형 있게 보강해 model-only
OOD 정확도를 `97.5%`까지 높였다. integrated runtime은 `100%`를 유지했지만 HOLD
한 건이 STOP으로 보수 처리됐고, 독립 safety scenario의 정상 action accuracy는
`75%`였다. STOP recall `100%`와 정상 action precision을 따로 최적화해야 하며,
human label `0` 때문에 상용 gate는 계속 닫혀 있다.

v1.80.0에서는 명확한 정상 APPROACH compound signal을 fast path에 추가해 safety
scenario action accuracy를 `100%`로 복구하고 STOP recall `100%`를 유지했다. 추가
boundary/seed/class-balanced 학습은 v1.79 best `97.5%`를 넘지 못해 승격하지
않았다. 이 단계에서도 human label이 없으므로 합성 OOD와 통합 rule 경로의 수치를
상용 모델 정확도로 부르지 않는다.

v1.40.0에서는 실시간 적용을 위한 `control.skill@1` typed head와 safety-bounded
`control decide` 경로를 추가했다. 48개 synthetic smoke checkpoint의 train은
100%였지만 validation 37.5%, test 12.5%였으므로 production actuator에 연결하지
않는다. control latency p50 0.128ms는 in-process CPU model-only 측정이며,
simulator와 저수준 controller를 포함한 end-to-end 결과가 아니다.

v1.41.0에서는 HyperMemory `POST /v1/context`를 control skill tick에 연결할 수
있는 adapter를 추가했다. context는 최대 8개 summary와 2,048자로 잘라 모델에
전달하며, 매 frame의 raw transcript를 motor loop에 넣지 않는다.

v1.42.0에서는 `control simulate`로 정상 action과 safety STOP 시나리오를 같은
JSONL에서 재생하고 latency/recall을 기록한다. 이 harness의 100% 결과는
contract regression 증거이지, 새로운 상태에 대한 model 일반화 정확도는 아니다.

v1.43.0에서는 validation/test 정확도와 safety STOP recall을 함께 판정하는
독립 evaluator를 추가하고, dataset/checkpoint SHA와 split/group/human-label
메타데이터를 report에 고정했다. `reference-token-encoder` 실험은 validation
50.00%, test 62.50%로 baseline보다 좋아졌지만 99% gate에는 실패했으며,
human golden 데이터가 없는 synthetic 결과로 production 승격하지 않는다.

v1.44.0에서는 control dataset에 scenario/episode/semantic-group provenance를
요구하고, normalized exact input과 episode/group이 split 사이에 섞이면
실패시키는 validator를 추가했다. 기존 smoke fixture는 이 계약을 만족하지
않으므로 human golden seed를 만들기 전까지 품질 dataset으로 취급하지 않는다.

v1.45.0에서는 `hyperjev control seed`로 8개 skill을 균형 있게 만들고, 기존
human review/apply 흐름에 넣을 수 있는 pending queue를 추가했다. seed target은
synthetic draft이고 human label은 비어 있으므로, validator의 human-required
gate를 통과하기 전에는 정확도 학습/출시 데이터로 사용하지 않는다.

v1.46.0에서는 evaluator가 safety 전체 action accuracy 100%와 human label gate를
함께 확인하도록 강화했다. 따라서 synthetic seed의 split 100% 결과가 있어도
safety action 50% 또는 human label 0개이면 production-ready가 아니다.

v1.47.0에서는 control queue를 Qwen/Gemma draft 경로에 직접 연결했다. Qwen
16개는 schema-valid 16/16이었지만 Gemma는 think 활성화 generation이 매우
느려 비동기 judge queue가 필요하다는 운영 결과를 남겼다. 두 teacher 모두
human label을 대체하지 않는다.

v1.48.0에서는 BOW count encoder를 ablation했다. validation 50%로 token/ngram과
같아 승격하지 않았으며, control accuracy 상승의 우선순위를 human semantic
group과 hard-negative 데이터로 확정했다.

v1.49.0에서는 혼동쌍 counterfactual hard-negative queue를 추가했다. pair는
같은 episode/group과 split에 고정해 leakage를 막고, human label 전에는
synthetic seed로만 취급한다.

각 구현 단계는 다음 순서를 따른다.

```text
요구사항 확인 → 작은 변경 → targeted test → 전체 test/lint
→ version bump → commit → push → 원고 갱신
```
