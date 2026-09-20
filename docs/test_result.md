# HyperJev 성능 테스트 결과

## 1. 시험 목적과 판정 범위

이 문서는 2026-09-20 DGX Spark 개발 환경에서 실행한 성능 시험의 방법과
결과를 기록한다. 목적은 다음 세 경로를 분리해서 비교하는 것이다.

1. 현재 parent/teacher LLM인 Qwen endpoint의 실제 end-to-end 지연시간과 typed
   schema 품질
2. 독립 judge/fallback인 Gemma endpoint의 연결성과 generation 상태
3. HyperJev의 deterministic rule fast-path 및 typed API contract 비용

중요한 제한이 있다. 현재 저장소에는 production encoder + typed decision heads
checkpoint와 Student inference runtime이 없다. 따라서 이 문서의 rule/mock 수치는
Student 모델의 성능이 아니며, Student가 parent LLM보다 빠르다고 결론내리지
않는다. Student 대 parent의 공정한 비교는 checkpoint와 동일 입력·동일 품질
기준을 갖춘 뒤 별도 gate로 실행해야 한다.

## 2. 실행 환경

| 항목 | 관측값 |
| --- | --- |
| 실행 시각 | 2026-09-20 (Asia/Seoul) |
| host | Linux aarch64, Python 3.11.14 |
| GPU | NVIDIA GB10, driver 580.142 |
| Docker | 29.2.1 |
| HyperJev | 1.14.0, commit `87a77d5` |
| task registry | 6 tasks |
| smoke dataset | `tests/golden/phase0_smoke.jsonl`, 6 samples |
| dataset SHA-256 | `cc467ab358d41dff53ccff5ee551366bf59b851d4176091f028a7f9c23025187` |
| model track | `encoder-typed-heads` |
| research track | `HyperJev-D` |

GPU 관측 명령은 다음과 같다.

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,pstate --format=csv,noheader
nvidia-smi -L
docker version --format '{{.Server.Version}}'
```

시험 시점의 결과는 `NVIDIA GB10, 580.142`, `GPU 0: NVIDIA GB10`, Docker
`29.2.1`이었다. 단, 이 짧은 시험에서 GPU memory/utilization을 시계열로
수집하지 않았으므로 peak memory나 saturation을 주장할 수 없다.

## 3. llama.cpp를 사용해야 하는가?

### 결론

- parent Qwen의 실제 서비스 성능을 비교할 때는 현재처럼 OpenAI-compatible
  HTTP endpoint를 호출하는 것이 맞다. 이것이 HyperJev가 운영 중 실제로 보는
  end-to-end 비용이다.
- llama.cpp 자체의 prompt processing/decode kernel, batch, context, speculative
  decoding을 분리 측정할 때는 `llama-bench`와 `llama-server --metrics`가
  필요하다.
- 현재 Qwen은 이미 llama.cpp `llama-server`로 실행 중이다. 프로세스 관측값은
  다음과 같다.

```text
/home/dirmich/work/llama.cpp/build-gb10-mtp/bin/llama-server
  ... --alias qwen38fn ... --port 8081 ... --draft-mtp ...
```

따라서 Qwen parent 비교에 llama.cpp를 별도로 다시 띄울 필요는 없었다. 다만
현재 PATH에는 `llama-bench`가 없고 해당 build의 `bin/`에도 `llama-server`만
있었다. `/metrics`도 `--metrics` 없이 시작되어 501을 반환했다. 다음 low-level
시험에서는 서버를 `--metrics`로 재시작하고, 동일 GGUF/동일 context/동일
parallel slot 조건에서 benchmark binary를 별도로 빌드해야 한다.

### 모델 식별 주의

Qwen `/v1/models` metadata는 다음을 반환했다.

```json
{
  "id": "qwen38fn",
  "owned_by": "llamacpp",
  "meta": {
    "n_params": 176943899520,
    "size": 89975329280,
    "n_ctx": 262144,
    "ftype": "Q3_K - Medium"
  }
}
```

이는 PRD에 기록된 “Qwen3 8B”와 일치하지 않는다. 현재 성능 결과의 모델명은
PRD 추정치가 아니라 endpoint가 보고한 alias와 metadata인 `qwen38fn`, 약
176.9B parameter metadata, 약 89.98GB GGUF로 기록한다. 이 discrepancy를
해결하기 전에는 “Qwen3 8B 기준” 성능표로 발표하지 않는다.

Gemma endpoint는 `http://macmini:11434/v1`의 remote service다. `/v1/models`
에는 `google/gemma-4-12b`가 존재했지만, 서버 backend가 llama.cpp인지 이
호스트에서 확인하지 않았다.

## 4. 시험 방법

### 4.1 Parent Qwen full smoke

동일한 6개 canonical sample에 대해 prompt version 1, temperature 0, top-p 1,
seed 0, max tokens 256으로 Qwen을 순차 호출했다.

```bash
uv run hyperjev benchmark --provider qwen --limit 6 \
  --output /tmp/hyperjev-qwen-perf-20260920.jsonl
uv run hyperjev evaluate \
  --run /tmp/hyperjev-qwen-perf-20260920.jsonl \
  --samples tests/golden/phase0_smoke.jsonl \
  --output /tmp/hyperjev-qwen-perf-20260920-report.json
```

측정값은 HTTP 호출부터 completion 수신, JSON 추출, typed contract validation까지
포함한다. raw state와 raw completion은 repository run file에 기록하지 않고,
response hash와 latency/token metadata만 보존한다.

### 4.2 Parent Qwen repeated sample

동일한 boolean sample을 10회 warm endpoint에 호출했다. 반복성 확인용이며,
6개 서로 다른 task의 대표 quality benchmark를 대체하지 않는다.

### 4.3 Gemma generation probe

먼저 `/v1/models` connectivity/model discovery를 3초 probe로 확인하고, 실제
generation은 같은 smoke sample 하나에 `--timeout 15`를 적용했다. 기존 운영
설정 900초에서도 동일한 한 sample이 완료되지 않았던 증거도 함께 보존한다.

```bash
uv run hyperjev teacher check --timeout 3
uv run hyperjev benchmark --provider gemma --limit 1 --timeout 15 \
  --output /tmp/hyperjev-gemma-perf-20260920.jsonl
```

### 4.4 HyperJev deterministic rule fast-path

registry task와 smoke sample을 그대로 사용해 `match_rule()`만 1,000회씩
반복했다. network, tokenizer, GPU, LLM generation은 포함하지 않는다. 6개
중 현재 rule이 match하는 sample 비율과 primitive latency를 측정했다.

### 4.5 HyperJev typed mock API

`create_server()`의 mock mode를 loopback에서 실행하고 `/v1/decide` 15회 및
6개 요청을 묶은 `/v1/batch/decide` 100회를 측정했다. 이는 typed request
validation·JSON serialization·HTTP transport의 contract 비용만 보여주며,
Student encoder 계산량은 포함하지 않는다.

## 5. 결과

### 5.1 Qwen parent: 6-sample smoke

| 지표 | 결과 |
| --- | ---: |
| 요청 수 | 6 |
| HTTP/completion 완료 | 6/6 (100%) |
| typed schema valid | 5/6 (83.33%) |
| transport/generation error | 0 |
| schema failure | 1/6 (`memory.type`, probabilities sum validation) |
| total tokens | 1,332 |
| 평균 total tokens/request | 222 |
| p50 latency | 1,127.542 ms |
| p95 latency | 2,448.624 ms |
| p99 latency | 2,448.624 ms |

sample별 결과는 다음과 같다.

| sample | task | latency | schema | 결과 |
| --- | --- | ---: | --- | --- |
| `smoke-remember-001` | `memory.remember_worthy` | 1,127.542 ms | valid | true, p=.95 |
| `smoke-type-001` | `memory.type` | 2,448.624 ms | invalid | probability 합이 1에서 벗어남 |
| `smoke-importance-001` | `memory.importance` | 1,034.547 ms | valid | value=.90 |
| `smoke-route-001` | `query.route` | 2,130.025 ms | valid | TEMPORAL, p=.60 |
| `smoke-relation-001` | `memory.relation` | 1,710.467 ms | valid | CONTRADICT, p=.90 |
| `smoke-wiki-change-001` | `wiki.semantic_change` | 849.634 ms | valid | false, p=.95 |

이 결과는 “Qwen이 6개를 모두 성공적으로 판단했다”가 아니다. HTTP level은
완료했지만 typed output gate는 1개 실패했다. 또한 각 task가 1개뿐이어서
accuracy, calibration, p95 품질 결론을 내릴 수 없다.

### 5.2 Qwen parent: 동일 sample 10회

| 지표 | 결과 |
| --- | ---: |
| sample | `smoke-remember-001` |
| 반복 수 | 10 |
| schema valid | 10/10 |
| error | 0 |
| p50 | 583.500 ms |
| p95 | 869.537 ms |
| p99 | 869.537 ms |
| 평균 | 611.423 ms |
| total tokens/request | 171 |

서버가 warm 상태이고 prompt가 동일해도 반복 latency가 약 0.58~0.87초였다.
이는 Qwen endpoint의 현재 관측치이지, 모든 context 길이와 동시성의 대표값은
아니다.

### 5.3 Gemma parent/judge

| 시험 | 결과 |
| --- | --- |
| `/v1/models` probe | 성공, configured alias `google/gemma-4-12b` 발견 |
| generation timeout 15초 | 1/1 timeout, completion 없음 |
| 기존 generation timeout 900초 | 1 sample도 completion하지 못함 |
| schema valid | 측정 불가 |
| quality/agreement | 측정 불가 |

즉 Gemma는 “연결 불가”가 아니라 model discovery는 통과하고 generation이
완료되지 않는 상태다. 따라서 Gemma를 Qwen의 parent quality 비교군으로
계산하지 않았으며, 현재는 production fallback/judge latency를 산출할 수 없다.

### 5.4 HyperJev rule fast-path

6개 중 2개가 rule match되어 smoke 기준 match rate는 33.33%였다.

| 구분 | p50 | p95 | p99 | 평균 |
| --- | ---: | ---: | ---: | ---: |
| `memory.remember_worthy` rule | 0.001472 ms | 0.001536 ms | 0.001712 ms | 0.001483 ms |
| `wiki.semantic_change` rule | 0.001472 ms | 0.001520 ms | 0.001568 ms | 0.001489 ms |
| 전체 sample 중 no-match task primitive | 약 0.0002 ms | 약 0.0002 ms | 약 0.0003 ms | 약 0.0002 ms |

이는 Qwen generation보다 압도적으로 작지만, deterministic rule이 적용 가능한
입력에 한정된 수치다. rule이 못 푸는 task를 이 숫자로 대체하면 안 된다.

### 5.5 HyperJev typed mock API

| 경로 | 측정량 | p50 | p95 | p99 | 평균 처리량 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `/v1/decide` | 15 requests, 1 decision/request | 0.329519 ms | 0.593950 ms | 0.593950 ms | 2,531.57 req/s |
| `/v1/batch/decide` | 100 requests, 6 decisions/request | 0.370879 ms/request | 0.863821 ms/request | 1.100428 ms/request | 12,324.42 decisions/s |

이 수치는 model inference가 빠르다는 결과가 아니라 mock handler의 contract
overhead다. 실제 Student inference가 연결되면 encoder, head, calibration,
batch scheduling, memory transfer 비용을 다시 포함해야 한다.

## 6. Parent와 HyperJev 비교 해석

현재 관측 가능한 비교는 다음과 같다.

| 경로 | 품질 측정 | latency 측정 | 해석 |
| --- | --- | --- | --- |
| Qwen parent | 5/6 schema valid, 각 task 1 sample | 6-sample p50 1.128 s, p95 2.449 s | 실제 generation baseline |
| Qwen repeated | 10/10 schema valid | p50 583.5 ms, p95 869.5 ms | warm 동일 입력 반복성 |
| Gemma parent | generation 없음 | 15초 timeout, 과거 900초 timeout | judge baseline gate blocked |
| HyperJev rule | smoke target과 같은 입력이지만 2개만 rule match | 약 0.0015 ms primitive | high-precision fast-path만 해당 |
| HyperJev mock API | mock typed result | p50 0.33 ms single | Student가 아닌 contract overhead |

Qwen 반복 p50 583.5ms와 rule primitive p50 0.001472ms를 직접 나누면 큰
비율이 나오지만, 이는 GPU LLM generation과 Python 문자열 rule 함수를 비교한
것이다. Student encoder와의 공정한 speedup이 아니므로 제품 성능 주장으로
사용하지 않는다.

참고로 smoke 6개에서 rule이 match한 두 sample을 Qwen 호출 없이 처리한다고
가정하면, 나머지 4개 Qwen latency 합으로 계산한 이상적 평균은 약 1,220.61ms로
Qwen 전체 평균 약 1,550.14ms보다 21.3% 낮다. 이 값도 실제 router 측정값이
아니며 fallback, API server, cache, deadline, batch 효과를 제외한 산술적인
fast-path 시나리오다.

## 7. 현재 판정

### 통과한 것

- DGX Spark host, NVIDIA, Docker, registry 환경 확인
- Qwen llama.cpp endpoint discovery 및 실제 6-sample completion
- Qwen repeated sample latency 측정
- typed schema validation이 malformed probability를 차단하는 것 확인
- deterministic rule primitive와 mock typed API의 lower-bound 측정
- Gemma endpoint discovery 확인

### 아직 통과하지 못한 것

- production Student checkpoint/inference benchmark
- Qwen/Gemma/Student 동일 golden set 기반 accuracy/calibration 비교
- Gemma generation latency와 teacher agreement
- 동시성별 p95, throughput, GPU memory peak, unified-memory pressure
- batch size 1/8/32/64/128 비교
- fallback 비율, abstain 비율, Hyper Memory write reduction
- human-reviewed 1,000-sample quality gate

현재 결론은 다음과 같다.

> HyperJev의 rule/typed boundary는 parent LLM 호출보다 훨씬 작은 계산 경로를
> 제공할 가능성이 확인됐지만, production Student의 성능·품질 우위는 아직
> 측정되지 않았다. Qwen은 실제 6개 smoke에서 schema 83.33%였고, Gemma는
> discovery 이후 generation timeout 상태이므로, 지금은 “완료”가 아니라
> Student/Gemma gate가 남아 있는 baseline 결과다.

## 8. 다음 재현 시험 계획

Student checkpoint가 준비되면 같은 dataset hash와 같은 request envelope로
아래 matrix를 실행한다.

| 축 | 값 |
| --- | --- |
| warm-up | 100 requests 제외 |
| 반복 | 각 cell 1,000 requests 이상 |
| concurrency | 1, 2, 4, 8, 16 |
| batch size | 1, 8, 32, 64, 128 |
| state length | 128, 512, 2,048, 8,192 chars |
| outputs | p50/p95/p99, req/s, decision/s, error, schema, abstain |
| resource | GPU util, GPU/UMA memory peak, power, host RSS |
| quality | accuracy, F1, Brier, ECE, NLL, AUROC/AUPRC, risk-coverage |
| routing | rule, Student, Qwen, Gemma, human route 비율 |

parent 비교는 Qwen/Gemma와 Student가 같은 canonical input, 같은 task version,
같은 timeout policy, 같은 warm-up/반복 수를 사용해야 한다. 결과는 평균 latency
하나가 아니라 tail latency와 fallback까지 포함해야 한다. llama.cpp low-level
결과는 별도 표로 prompt eval tok/s, generation tok/s, context, `-np`, batch,
GPU offload 조건을 함께 기록한다.
