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

현재 `main`은 origin에 push될 1.53.0까지 진행되어 있다. Python 테스트 111개와
1개 skip, Ruff 검사가 통과한 상태이며, dataset validator와 training plan,
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
