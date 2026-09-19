# 5장. Router에서 Serving과 운영으로 확장하기

## 5.1 Phase 1: rule + teacher router

초기 router는 모델을 바로 제품으로 승격하지 않고, rule과 teacher의 결과를
같은 typed response로 통합한다. request ID, state hash, route, deadline,
fallback reason, review reason을 trace로 보존한다. raw state는 trace에 넣지
않는다.

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

현재 mock mode는 contract integration용이고 router mode는 rule/teacher chain용이다.
둘 다 encoder Student runtime의 존재를 의미하지 않는다.

## 5.4 Cache, shadow, registry, drift

cache key는 raw state가 아니라 privacy-safe hash 조합이다. bounded LRU로 제한하고
cache hit도 route와 metrics에 반영한다. shadow log는 Student와 baseline의 결과를
hash-only로 비교해 raw decision payload가 운영 log로 새지 않게 한다.

model registry는 `registered → evaluated → calibrated → candidate → canary →
active → retired` 상태를 명시적으로 관리하고 promote/rollback 이유를 남긴다.
traffic monitor는 response-only counters와 latency/route/error를 모아 snapshot
사이의 drift를 계산한다. drift가 발견되면 자동으로 model을 바꾸지 않고 review와
promotion gate로 넘긴다.

