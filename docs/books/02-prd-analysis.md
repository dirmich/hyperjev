# 2장. PRD를 구현 계약으로 바꾸기

## 2.1 PRD를 읽는 순서

`docs/prd.md`는 제품 정의에서 시작해 출력 타입, Hyper Memory 통합, 모델
아키텍처, registry, 데이터, 학습, API, 평가, human review, 운영, 단계별
개발 계획, 최종 승인 기준까지 이어진다. 처음부터 코드를 쓰기보다 다음 네
층으로 다시 읽었다.

1. **제품 계약**: 무엇을 판단하고 누가 결과를 소비하는가.
2. **모델 계약**: teacher, Student, HyperJev-D의 책임이 어디까지인가.
3. **운영 계약**: timeout, fallback, privacy, cache, registry, drift를 어떻게
   통제하는가.
4. **증거 계약**: 어떤 테스트와 golden/calibration gate가 완료를 증명하는가.

이 재분류가 필요한 이유는 PRD의 기능 목록을 그대로 파일 목록으로 옮기면
계약과 구현이 섞이기 때문이다. 예를 들어 “calibration”은 모델의 내부
구현이면서 동시에 API의 probability 해석과 promotion gate의 운영 계약이다.

## 2.2 출력 타입

초기 6개 task는 다음 typed heads로 매핑된다.

| task | 타입 | 중요한 검증 |
| --- | --- | --- |
| `memory.remember_worthy` | boolean | value는 bool, probability는 0~1 |
| `memory.type` | choice | selected와 모든 registered candidate probability |
| `memory.importance` | score | value와 interval_90의 순서/포함 관계 |
| `query.route` | choice | registry candidate 외 선택 금지 |
| `memory.relation` | choice | relation enum 외 선택 금지 |
| `wiki.semantic_change` | boolean | abstain과 confidence를 함께 기록 |

`src/hyperjev/contracts.py`는 이 표를 실행 가능한 경계로 만들었다. legacy
`noul` boolean alias를 읽을 수 있지만 외부 출력은 typed `boolean`으로
정규화한다. 잘못된 task reference, 중복 question id, probability 합계,
interval 범위는 teacher가 생성한 값이어도 통과시키지 않는다.

## 2.3 모델 결정

PRD의 모델 장은 diffusion을 논의하지만 제품 MVP에는 cross-encoder와
shared-state multi-question, 이후 bi-encoder cascade의 경로를 둔다. 현재
repository에서 실제로 고정한 것은 다음이다.

```text
제품 baseline: encoder + typed decision heads
teacher: Qwen3 8B 생성/label/fallback, Gemma 4 검증/judge
비교 연구: HyperJev-D diffusion track
```

이 구분은 이름만의 문제가 아니다. Student manifest에는 `track`과
`diffusion_track`을 둘 다 기록하지만 serving route는 encoder typed-heads를
제품 모델로 간주한다.

## 2.4 데이터와 human gate

PRD는 한국어 중심 human-reviewed golden set 1,000개를 요구한다. 따라서
synthetic queue generator는 개발 편의를 위한 pending queue만 만든다. queue의
`target`은 기대 동작을 테스트하는 참고값이지 `labels.human`이 아니다.

실제 검수 흐름은 다음이다.

```text
synthetic/source queue
  → typed correction 검증
  → append-only golden feedback
  → source queue hash 확인
  → reviewed queue copy 생성
  → golden validate
```

feedback은 원본 queue의 SHA-256에 묶인다. queue가 재생성되거나 수정되면
오래된 label을 조용히 재사용하지 못한다. 전체 1,000개가 검수되기 전에는
`ready: true`를 보고하지 않는다.

## 2.5 단계별 gate로 번역

| PRD 영역 | 구현 gate |
| --- | --- |
| 환경/기본 구조 | `doctor`, registry validation, unit/contract tests |
| baseline | Qwen/Gemma probe와 run JSONL, latency/schema report |
| router | rule precision, provider fallback, human review trace |
| dataset | redaction, privacy exclusion, agreement, deterministic split |
| Student | registry-derived head manifest, optional torch dependency boundary |
| calibration/evaluation | temperature fitting, typed quality/calibration metrics |
| serving | single/batch API, health, metrics, shadow logs |
| promotion | model registry state transition과 rollback |
| operations | cache privacy, drift snapshots, reproducible runbook |
| final approval | human golden, live baselines, checkpoint, latency/quality targets |

