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

현재 `main`은 origin에 push될 1.25.0까지 진행되어 있다. Python 테스트 73개와
1개 skip, Ruff 검사가 통과한 상태이며, dataset validator와 training plan,
reference train CLI, reference Student checkpoint 생성, n-gram Student 비교와
guarded 99% 평가, optional Student router 연결, parent LLM 대 HyperJev
경계의 성능 시험 기록까지 완료됐다. training plan·calibration·checkpoint를
registry manifest로 묶는 contract와 30-sample confidence/coverage 분석도
추가됐다. 다만 checkpoint는 synthetic 36개 샘플의 reference-ngram-encoder이고
human golden이 없으므로 production 모델로 승격하지 않았다. 실제
사람 검수 golden set, Gemma live baseline 전수 실행, production multilingual
PyTorch checkpoint 학습, Student GPU inference benchmark, Rust toolchain compile은
이 책에서 성공했다고 가장하지 않고 외부 의존성 gate로 표시한다. 상세 결과는
[`docs/test_result.md`](../test_result.md)에 있다.

각 구현 단계는 다음 순서를 따른다.

```text
요구사항 확인 → 작은 변경 → targeted test → 전체 test/lint
→ version bump → commit → push → 원고 갱신
```
