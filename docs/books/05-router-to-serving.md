# 5장. Router에서 Serving과 운영으로 확장하기

## 5.1 Phase 1: rule + Student + teacher router

초기 router는 모델을 바로 제품으로 승격하지 않고, rule과 teacher의 결과를
같은 typed response로 통합한다. request ID, state hash, route, deadline,
fallback reason, review reason을 trace로 보존한다. raw state는 trace에 넣지
않는다. checkpoint가 설정된 경우 고정밀 rule 다음에 typed-head Student를
시도하고, confidence 미달/abstain은 자동 수락하지 않고 Qwen으로 넘긴다.

```text
rule → Student → Qwen → Gemma → human
```

Student는 기본 비활성이다. `HYPERJEV_STUDENT_CHECKPOINT`를 설정해야만
checkpoint manifest와 registry head를 로드하며, 이 설정이 없으면 기존
`Qwen → Gemma → human` 경로가 그대로 유지된다. score head는 calibration과
별도 acceptance 정책이 준비되기 전까지 자동 수락하지 않는다.

Hyper Memory ingestion gate는 decision 결과가 `extract` 정책을 통과할 때만
`/v1/documents`로 전달한다. `ignored`, `retain_raw`, `review`, `extract`를
분리해 “판단”과 “기억 저장”을 같은 부작용으로 취급하지 않는다.

## 5.2 Dataset factory

dataset factory는 seed JSONL을 읽고 다음 순서를 지킨다.

```text
privacy exclusion → deterministic redaction → dedup
→ Qwen/Gemma independent typed labels → agreement gate
→ human/review routing → source/entity grouped split
→ normalized JSONL + dataset hash
```

user/secret source는 기본 제외하고, 이메일·전화번호·token 패턴을 보수적으로
redact한다. teacher disagreement는 억지로 silver label로 만들지 않고 review
JSONL에 남긴다. split은 row random split이 아니라 source/entity hash group으로
정한다.

## 5.3 Student과 serving 경계

`student_manifest`는 registry에서 boolean/choice/score head를 만든다. optional
`build_torch_model`은 shared embedding/encoder와 typed `ModuleDict` head를
만들 수 있지만, torch가 없는 호스트에서는 명시적으로 dependency error를
낸다. 이것은 실패를 숨기지 않는 설계다.

Serving 계층은 다음 endpoint를 제공한다.

- `/v1/decide`
- `/v1/batch/decide`
- `/v1/tasks`, `/v1/tasks/validate`
- `/v1/models`, `/v1/models/{id}/activate`
- `/health/live`, `/health/ready`, `/metrics`

현재 mock mode는 contract integration용이고 router mode는 rule/Student/teacher
chain용이다. Student는 checkpoint 환경 변수가 있을 때만 활성화되며, checkpoint
artifact가 없는 기본 실행은 teacher-only fallback을 유지한다.

## 5.4 Cache, shadow, registry, drift

cache key는 raw state가 아니라 privacy-safe hash 조합이다. bounded LRU로 제한하고
cache hit도 route와 metrics에 반영한다. shadow log는 Student와 baseline의 결과를
hash-only로 비교해 raw decision payload가 운영 log로 새지 않게 한다.

model registry는 `registered → evaluated → calibrated → candidate → canary →
active → retired` 상태를 명시적으로 관리하고 promote/rollback 이유를 남긴다.
traffic monitor는 response-only counters와 latency/route/error를 모아 snapshot
사이의 drift를 계산한다. drift가 발견되면 자동으로 model을 바꾸지 않고 review와
promotion gate로 넘긴다.
