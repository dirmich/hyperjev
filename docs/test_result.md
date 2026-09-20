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

## 9. 추가 확장 시험: 성능 차이와 confidence/coverage

### 9.1 Qwen synthetic queue 30개

6개 smoke만으로 확률 품질을 판단하지 않기 위해, 사람이 검수하지 않은
synthetic review queue의 앞 30개를 같은 Qwen 설정으로 실행했다. 이 데이터는
반복 패턴을 포함하므로 production quality evidence가 아니라 threshold와
failure-mode를 확인하는 확장 시험이다.

```text
run_id: 20260920T012034Z
source: runs/phase0/phase0-review-queue.jsonl (first 30 records)
dataset_sha256: 463b00f91257b34a837f8eaa184d5fb4efec49f2028f086dd78aa0e3bdbd873f
provider/model: qwen/qwen38fn
```

| 지표 | 결과 |
| --- | ---: |
| completion | 30/30 (100%) |
| typed schema valid | 27/30 (90.00%) |
| schema failure | 3/30 (10.00%), 모두 `memory.type` |
| 평균 latency | 1,469.105 ms |
| p50 / p95 / p99 | 1,121.314 / 2,368.689 / 2,380.245 ms |
| sequential throughput | 0.681 requests/s |
| total tokens | 6,580 |
| 평균 tokens/request | 219.33 |

task별 확장 결과는 다음과 같다. Boolean과 choice의 `correct`는 typed
prediction이 target과 일치하는지, score의 `correct`는 target과 ±0.10 이내인지
계산했다.

| task | valid/전체 | task quality | 현재 정책 accepted | accepted 결과 |
| --- | ---: | --- | ---: | --- |
| `memory.remember_worthy` | 5/5 | 5/5 correct | 5/5 | 5/5 correct |
| `memory.type` | 2/5 | 2/2 valid correct | 2/5 | 2/2 correct, 3 schema failure |
| `memory.importance` | 5/5 | MAE 0.210, RMSE 0.222486 | 5/5 | 0/5 within ±0.10 |
| `query.route` | 5/5 | 5/5 correct | 3/5 | 3/3 accepted correct, 2 low confidence |
| `memory.relation` | 5/5 | 5/5 correct | 4/5 | 4/4 accepted correct, 1 low confidence |
| `wiki.semantic_change` | 5/5 | 5/5 correct | 5/5 | 5/5 correct |

`memory.importance`의 90% interval coverage는 0/5였다. 즉 score 값이 typed
schema를 통과하고 interval도 형식상 유효해도, held-out target을 포함하지
않았다. 이 결과 때문에 score head를 “확률이 높다”는 이유만으로 자동 저장하는
것은 상용 정책으로 부적합하다.

### 9.2 답변 확률과 자동 수락률

여기서 boolean의 `probability`와 choice의 선택 확률은 “반환한 답이 맞을
자신감”으로 취급했다. Boolean metric 계산 때만 false prediction을 positive
class probability로 변환한다. Score는 단일 answer probability가 없으므로
90% interval width를 불확실성 proxy로 사용하며, 이것은 calibration을 대체하지
않는다.

현재 task-specific acceptance policy를 그대로 적용하면:

| 정책 | 자동 수락 | 전체 coverage | accepted quality |
| --- | ---: | ---: | ---: |
| 현재 정책, score 포함 | 24/30 | 80.00% | 19/24 correct = 79.17% |
| valid output 중 confidence >= 0.95 | 17/27 | 63.00% (전체 기준 56.67%) | 17/17 correct = 100% |
| score 자동 수락 금지 + 기존 boolean/choice threshold | 19/30 | 63.33% | 19/19 correct = 100% |

이 표의 100%는 반복 synthetic 30개에 대한 결과일 뿐이며, 사람이 검수한
독립 test set의 100%를 의미하지 않는다. 그러나 운영 의사결정에는 중요한
방향을 보여준다.

- 낮은 confidence와 schema failure를 fallback/review로 보내면 coverage는
  감소하지만 관측된 accepted risk가 줄어든다.
- 현재 구현의 score `_accepted`가 모든 유효 score를 통과시키는 것은 위험하다.
  score는 held-out calibration set에서 interval coverage가 확인될 때까지
  자동 수락하지 않는 것이 안전하다.
- confidence를 0.99로 강제로 올리는 방식은 해결책이 아니다. confidence는
  held-out correctness와 calibration error에 맞춰야 한다.

### 9.3 성능 차이의 수치 해석

동일한 30개 시험의 Qwen parent와 앞서 측정한 deterministic/mock 경로를
비교하면 다음과 같다.

| 경로 | 대표 p50 | 대표 p95 | 의미 |
| --- | ---: | ---: | --- |
| Qwen parent, 30 sequential generations | 1,121.314 ms | 2,368.689 ms | 실제 parent generation |
| rule primitive, matched samples | 약 0.001472 ms | 약 0.001536 ms | network/LLM 없는 rule 계산 |
| mock `/v1/decide` | 0.329519 ms | 0.593950 ms | typed HTTP contract만 |

Qwen p50과 rule primitive의 단순 비율은 약 76만 배, Qwen p50과 mock API의
단순 비율은 약 3,400배다. 하지만 이 숫자는 Student encoder가 아니라 각각
Python rule과 mock handler를 비교한 것이므로 product speedup으로 발표하지
않는다. 실제 Student 비교를 위해서는 동일 입력을 encoder + typed heads로
실행하고 calibration/fallback 비용까지 포함해야 한다.

현재 측정으로 확정할 수 있는 것은 다음뿐이다.

1. parent generation은 대략 1~2.4초 tail을 가진다.
2. deterministic fast-path가 적용되는 입력은 parent 호출을 피할 수 있다.
3. typed contract overhead는 parent generation보다 작지만 Student inference
   비용을 아직 모른다.
4. 품질을 100%로 보이게 만드는 가장 안전한 방법은 confidence를 조작하는
   것이 아니라 낮은 confidence를 abstain/fallback으로 보내 coverage를
   관리하는 것이다.

### 9.4 상용 release를 위한 probability gate

다음 gate를 production Student에 적용해야 한다. 수치는 실제 human golden과
validation set에서 확정해야 하며, 현재 30개 결과가 통과했다는 뜻이 아니다.

1. **Schema gate:** typed schema valid rate와 choice probability sum failure를
   task별로 측정한다. schema failure는 confidence가 높아도 자동 수락하지
   않는다.
2. **Calibration gate:** task별 held-out set에서 Brier, NLL, ECE/adaptive ECE,
   interval-90 coverage를 계산하고 calibration manifest를 고정한다.
3. **Risk-coverage gate:** threshold별 coverage와 accepted error를 함께 보고,
   고위험 task는 낮은 coverage를 감수하고 낮은 risk threshold를 사용한다.
4. **Score gate:** interval coverage가 검증되기 전에는 `memory.importance`를
   자동 저장하지 않고 Qwen/Gemma 또는 human review로 보낸다.
5. **Fallback gate:** schema invalid, threshold 미달, OOD, teacher disagreement는
   모두 fallback/review로 기록하며 “답을 만들기” 위해 confidence를 높이지
   않는다.
6. **Release gate:** human-reviewed 1,000개 이상에서 Student, Qwen, Gemma,
   rule을 같은 task/version으로 비교하고, 동시에 p50/p95/p99, throughput,
   abstain, fallback, Hyper Memory write reduction을 기록한다.

현재 결과를 기준으로 한 보수적 운영 선택은 `score auto-accept off`,
boolean/choice는 task별 threshold 적용, schema-invalid는 즉시 fallback이다.
이 선택은 답변 coverage를 낮추지만, 잘못된 기억을 자동 저장하는 위험을
줄이는 방향이다.

## 10. Reference Student checkpoint 생성 및 정확도

사용자가 요구한 실제 checkpoint를 빠르게 생성하기 위해 synthetic queue의
중복을 제거한 36개 typed dataset으로 reference Student를 학습했다. 이 경로는
PRD의 production multilingual encoder가 아니라 `reference-byte-encoder`와
registry-derived typed heads를 검증하는 최소 실행 경로다.

### 10.1 생성 명령과 artifact

```bash
uv run hyperjev train plan \
  --dataset runs/phase2/reference-dataset.jsonl \
  --output runs/phase3/reference-training-plan.json \
  --model-id hyperjev-reference-synthetic \
  --backbone reference-byte-encoder \
  --epochs 3 --batch-size 1 --precision fp32

uv run hyperjev train run \
  --dataset runs/phase2/reference-dataset.jsonl \
  --output runs/phase3/reference-student.pt \
  --model-id hyperjev-reference-synthetic \
  --backbone reference-byte-encoder \
  --epochs 3 --batch-size 1 --precision fp32 --device cpu
```

| artifact | 값 |
| --- | --- |
| model id | `hyperjev-reference-synthetic` |
| backbone | `reference-byte-encoder` |
| dataset samples | 36 (train 31, validation 2, test 3) |
| dataset SHA-256 | `c207d558e8381c29bb9a881d0d8c2bc42ba2a97c36adce59d671ed0b26ed3c9f` |
| checkpoint | `runs/phase3/reference-student.pt` |
| checkpoint SHA-256 | `7a4bb4d47a58295db479cb56ee675335e45706d80d54a87359b87a30f678e19a` |
| PyTorch | `2.14.0+cu130` |
| training device | CPU |
| train loss | 1.431961 → 1.023529 → 0.821674 |

GB10 CUDA는 정상 인식됐지만 이미 실행 중인 Qwen llama-server가 GPU/통합
메모리를 점유해 Student를 CUDA로 올릴 때 `CUDA error: out of memory`가
발생했다. Qwen을 중단하지 않고 CPU에서 checkpoint를 완성했다. 이 때문에
이 수치는 GPU inference latency 결과가 아니다.

### 10.2 Reference Student 정확도

Boolean/choice는 exact match, score는 target과 ±0.10 이내를 correct로
계산했다.

| split | correct | total | accuracy |
| --- | ---: | ---: | ---: |
| train | 26 | 31 | 83.87% |
| validation | 1 | 2 | 50.00% |
| test | 1 | 3 | 33.33% |
| 전체 | 28 | 36 | 77.78% |

task별 전체 결과는 다음과 같다.

| task | correct/total | accuracy 또는 score 지표 |
| --- | ---: | --- |
| `memory.remember_worthy` | 5/6 | 83.33% |
| `memory.type` | 5/6 | 83.33% |
| `memory.importance` | 4/6 | ±0.10 기준 66.67% |
| `query.route` | 4/6 | 66.67% |
| `memory.relation` | 5/6 | 83.33% |
| `wiki.semantic_change` | 5/6 | 83.33% |

이 결과는 checkpoint가 실제로 load되고 모든 dotted registry task head가
forward되는 것을 증명하지만, test 3개 중 1개만 맞았으므로 상용 정확도로
사용할 수 없다. 특히 validation/test가 너무 작고 synthetic pattern을
기반으로 하므로 generalization을 판단하기 어렵다.

### 10.3 Calibration과 registry 상태

checkpoint에서 분리한 3개 held-out boolean logits로 calibration manifest를
생성했다.

| 항목 | 값 |
| --- | --- |
| calibration version | `cal-824aeb36ba59` |
| temperature | 4.0 (검색 상한) |
| NLL before → after | 0.880682 → 0.701236 |
| held-out count | 3 |
| registry status | `trained` |

NLL은 개선됐지만 held-out가 3개뿐이고 temperature가 상한에 도달했으므로
production calibration으로 승인하지 않는다. model manifest는 생성·등록만
했고 `evaluated → calibrated → candidate → canary → active`로 승격하지
않았다.

### 10.4 현재 HyperJev rule 경로 정확도

Student와 별도로 실제 제품 router의 deterministic rule fast-path를 전체
synthetic queue 1,000개에서 재측정했다. style-only 우선순위 bug를 수정한 뒤:

| 지표 | 결과 |
| --- | ---: |
| rule coverage | 197/1,000 = 19.70% |
| covered rule accuracy | 197/197 = 100% |
| abstain/fallback 대상 | 803/1,000 = 80.30% |
| 전체를 abstain failure로 계산한 accuracy | 19.70% |

이 100%는 rule이 판단한 입력에만 해당한다. 나머지 80.3%를 Student나
teacher가 처리하지 않으면 제품 전체 정확도가 아니다. 현재 가장 정확한
상용 형태는 rule의 100% covered path를 유지하고, 나머지는 낮은 confidence로
자동 수락하지 않는 것이다.

### 10.5 판단

현재 즉시 사용할 수 있는 숫자는 다음과 같다.

- **Reference Student:** 전체 77.78%, test 33.33% — production 사용 불가
- **Rule fast-path:** covered 100%, 전체 coverage 19.70% — 제한된 입력에만 사용
- **Qwen parent:** 30-sample valid classification 22/22, 전체 schema/score 포함
  보수적 22/30 = 73.33% — teacher baseline이지 Student가 아님

따라서 checkpoint는 생성됐지만 아직 상용 모델로 승격하지 않았다. 다음 개선은
synthetic 중복 데이터가 아니라 사람이 검수한 task별 train/validation/test를
확대하고, production multilingual backbone·GPU inference·calibration·risk
coverage를 다시 측정하는 것이다.

## 11. n-gram Student와 99% guarded path 결과

기존 평균 byte embedding의 underfit을 확인한 뒤, 기존 checkpoint 경로를
깨지 않는 `reference-ngram-encoder`를 추가했다. byte embedding 뒤에
depthwise 3-gram convolution, mean/max pooling, typed heads를 연결했다.

### 11.1 학습 조건

```bash
uv run hyperjev train run \
  --dataset runs/phase2/reference-dataset.jsonl \
  --output runs/phase3/reference-ngram-student.pt \
  --model-id hyperjev-reference-ngram \
  --backbone reference-ngram-encoder \
  --epochs 100 --batch-size 1 \
  --learning-rate 0.01 --weight-decay 0 \
  --precision fp32 --device cpu
```

| 항목 | 결과 |
| --- | --- |
| checkpoint | `runs/phase3/reference-ngram-student.pt` |
| backbone | `reference-ngram-encoder` |
| dataset | 36개 (train 31, validation 2, test 3) |
| device | CPU (Qwen llama-server가 GB10 memory 점유) |
| epochs | 100 |
| checkpoint SHA-256 | `030cdc2e270a3cfb4a28756dd36f5a3991dfbce28ae178243f88a089cdf5c0ba` |

### 11.2 정확도와 guarded acceptance

`hyperjev student evaluate`로 동일 checkpoint를 측정했다. boolean/choice는
exact match, score는 ±0.10 이내를 correct로 계산했고 score는 calibration
검증 전 자동 수락하지 않았다.

| 경로 | correct | accuracy | accepted | coverage | fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| Student 전체 | 35/36 | 97.22% | - | - | - |
| Student test split | 3/3 | 100% | 2/3 | 66.67% | 1 |
| Student confidence ≥ 0.95 | 27/27 | 100% | 27/36 | 75.00% | 9 |
| rule + Student guarded | 36/36 | 100% | 29/29 | 80.56% | 7 |

`--with-rules` 경로에서 rule은 8개를 처리했고 8/8 정확했다. 나머지는
Student가 처리하며, score task는 자동 수락하지 않고 fallback으로 남겼다.
따라서 현재는 “99% 이상”을 **synthetic accepted path에서 달성**했지만,
사람 golden 전체 정확도 99%라는 주장은 아직 할 수 없다.

### 11.3 다음 품질 gate

다음 실행은 synthetic 데이터를 더 반복해 숫자를 키우는 것이 아니라,
사람이 검수한 최소 1,000개 golden으로 같은 명령을 반복하는 것이다.

```bash
uv run hyperjev student evaluate \
  --checkpoint runs/phase3/reference-ngram-student.pt \
  --dataset runs/phase2/human-golden-dataset.jsonl \
  --minimum-confidence 0.95 --with-rules \
  --output runs/phase3/human-golden-student-evaluation.json
```

overall accuracy >= 99%, task별 >= 98%, accepted accuracy >= 99.5%를 모두
충족하기 전에는 registry를 `candidate` 이상으로 승격하지 않는다.

`--production-gate`를 synthetic dataset에 실행한 결과는 overall 100%,
accepted 100%였지만 human label 0/36이라 exit code 1과
`human_labels_required`를 반환했다. 이는 99% 숫자와 production 승인을
분리하는 의도된 실패다.

### 11.4 1,000개 synthetic stress

동일한 n-gram checkpoint를 원래의 1,000개 synthetic review queue 전체에
적용했다. 중복 제거 dataset과 달리 sample ID는 유지하고 split만 deterministic
하게 부여했다.

| 지표 | 결과 |
| --- | ---: |
| 전체 | 1000/1000 = 100% |
| accepted | 787/787 = 100% |
| coverage | 78.70% |
| fallback | 213/1000 |
| rule covered | 197/197 = 100% |
| human labels | 0/1000 |

따라서 synthetic stress에서는 99% 목표를 넘겼지만, `--production-gate`는
`human_labels_required`로 실패한다. 다음 정확도 작업은 이 수치를 더 높이는
것이 아니라 이 queue를 실제 사람이 검수한 golden dataset으로 교체하는 것이다.

## 12. Student router 실제 연결 및 confidence fallback

### 12.1 구현 범위

1.21.0에서 checkpoint를 단순히 evaluator로 읽는 데서 끝내지 않고 실제
`DecisionRouter`에 optional provider로 연결했다. 환경 변수가 없으면 기존
teacher-only 경로를 유지하고, 설정되면 다음 순서로 시도한다.

```text
high-precision rule → Student → Qwen → Gemma → human
```

Student는 `student_manifest`와 registry task definition을 이용해 typed head를
복원한다. boolean/choice는 `minimum_confidence` 미만이면
`abstained=true`를 반환하고 score는 `allow_score=false` 정책으로 자동
수락하지 않는다. router의 `_accepted`는 모든 abstention을 먼저 거부하므로
확률 필드가 높아도 fallback guard를 우회할 수 없다.

### 12.2 실제 checkpoint smoke

실행 대상은 local ignored artifact인
`runs/phase3/reference-ngram-student.pt`이며, 공개 저장소에는 checkpoint를
push하지 않고 코드·명령·hash만 기록한다. request는
`memory.type@1` 한 건이고 state는 `The service uses PostgreSQL for durable
storage.`였다.

```bash
HYPERJEV_STUDENT_CHECKPOINT=runs/phase3/reference-ngram-student.pt \
HYPERJEV_STUDENT_MIN_CONFIDENCE=0.95 \
uv run hyperjev decide --request runs/phase3/student-smoke-request.json
```

관측 결과:

| 항목 | 결과 |
| --- | --- |
| route | `student` |
| model | `hyperjev-reference-ngram` |
| selected | `fact` |
| top probability | `0.9993294477` |
| accepted | `true` |
| latency | 약 `5.04 ms` |
| parent calls | 0 |

confidence gate를 의도적으로 `0.9999`로 올리고 Qwen/Gemma 주소를 연결 거부
endpoint로 격리한 두 번째 smoke도 실행했다.

```bash
HYPERJEV_STUDENT_CHECKPOINT=runs/phase3/reference-ngram-student.pt \
HYPERJEV_STUDENT_MIN_CONFIDENCE=0.9999 \
HYPERJEV_QWEN_BASE_URL=http://127.0.0.1:9/v1 \
HYPERJEV_GEMMA_BASE_URL=http://127.0.0.1:9/v1 \
uv run hyperjev decide --request runs/phase3/student-smoke-request.json
```

결과는 Student가 동일한 `0.9993294477` 확률을 내고 `accepted=false`가 된 뒤,
Qwen/Gemma connection error를 기록하고 `route=human`,
`review_required=true`로 종료되었다. 즉 confidence 미달 결과를 억지로
자동 수락하지 않고 fail-closed 한다.

### 12.3 회귀 및 정합성 검증

```bash
uv run ruff check src tests
git diff --check
uv run pytest -q
```

결과: **70 passed, 1 skipped**. 새 테스트는 optional config 환경 변수,
checkpoint load/typed result, Student 우선 수락, Student abstention의 Qwen
fallback을 고정했다. Torch는 현재 NumPy 미설치 warning만 출력했으며 테스트
실패는 없었다.

이 단계의 정확도 해석은 명확하다. 실제 router 연결과 fallback 계약은 검증했지만,
smoke 한 건은 모델 정확도 benchmark가 아니다. synthetic 1,000개 stress의
100% accepted accuracy와 human golden 0/1,000이라는 기존 제한은 그대로이며,
`student evaluate --production-gate`는 human label이 채워질 때까지 실패해야
한다.

## 13. production 정확도 기준을 human label로 고정

### 13.1 발견된 문제

기존 evaluator는 `human_labels_required`를 gate 조건으로 확인하면서도,
정작 `correct` 계산에는 dataset의 synthetic `target`을 사용했다. reviewer가
synthetic target을 수정한 경우에도 synthetic 정답으로 99%를 계산할 수 있는
구조였으므로 production quality evidence로 사용할 수 없었다.

### 13.2 수정된 기준

각 sample에 대해 다음 기준을 적용한다.

| human label 상태 | 정확도 기준 | 의미 |
| --- | --- | --- |
| typed `labels.human` 존재 | human label의 value/selected/score | production 평가 기준 |
| human label 없음 | `target` | synthetic/exploratory 결과, gate 미충족 |
| human label이 schema 오류 | 평가 중단 | 잘못된 golden을 숨기지 않음 |
| human label이 `abstained=true` | 평가 중단 | 정답으로 쓸 수 없는 review 상태 |

report의 각 prediction에는 `target_source`를 기록한다. 따라서 전체 accuracy가
높아도 `golden.human_labeled=false`이면 production gate가 통과하지 않는다.

### 13.3 검증

synthetic `target=true`와 human label `false`가 충돌하는 fixture를 추가했고,
report가 `target=false`, `target_source=human`을 사용하는지 검증했다.

```bash
uv run ruff check src tests
git diff --check
uv run pytest -q
```

결과: **71 passed, 1 skipped**. 이 단계는 accuracy 숫자를 99%로 만드는
작업이 아니라 99% 주장의 분모와 정답 출처를 올바르게 만드는 품질 gate
수정이다. 실제 1,000개 human golden 라벨이 채워지기 전에는 production
승격을 주장하지 않는다.

### 13.4 1,000개 재평가 결과

human-source 수정 이후 동일한 1,000개 queue를 재실행했다.

| 항목 | 결과 |
| --- | --- |
| checkpoint SHA-256 | `030cdc2e270a3cfb4a28756dd36f5a3991dfbce28ae178243f88a089cdf5c0ba` |
| dataset SHA-256 | `139e5c2f63de52811eab3845f19a4b42e4c7c0d6d7b61f1d979af292e08438ff` |
| target source | `sample.target` only |
| overall | 1000/1000 = 100% |
| accepted | 787/787 = 100% |
| coverage | 78.70% |
| fallback | 213 |
| human labels | 0/1000 |
| production gate | exit 1: `human_labels_required` |

이 결과는 evaluator가 synthetic source를 정확히 표시한다는 회귀 증거이며,
human golden 정확도 99%의 증거가 아니다.

## 14. Gemma human-review draft probe

### 14.1 목적과 경계

Gemma4는 Qwen과 독립적인 cross-validator/judge이므로 human reviewer가 볼
초안 라벨을 만드는 데 사용할 수 있다. 그러나 Gemma output을 그대로
`labels.human`으로 복사하면 teacher agreement일 뿐 human golden이 아니다.
`hyperjev golden draft`는 이 경계를 코드로 분리한다.

```bash
uv run hyperjev golden draft \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --output runs/phase1/gemma-golden-draft.jsonl \
  --limit 20 --timeout 900
```

draft 파일은 queue SHA-256, sample ID, Gemma model, prompt version, latency,
response hash, normalized typed result, schema validity와 timeout/error 사유만
기록한다. raw state/question과 raw Gemma 응답은 저장하지 않는다. reviewer는
원본 queue의 `state`/`question`을 보고 초안과 비교한 뒤 `golden review`로
correction을 append해야 한다.

### 14.2 실제 Gemma4 probe

첫 sample에 대해 60초 timeout으로 실행했다.

| 항목 | 결과 |
| --- | --- |
| model | `google/gemma-4-12b` |
| sample count | 1 |
| timeout | 60초 |
| completed | 0 |
| schema-valid | 0 |
| error | `TimeoutError: timed out` |
| queue SHA-256 | `463b00f91257b34a837f8eaa184d5fb4efec49f2028f086dd78aa0e3bdbd873f` |

따라서 현재는 pipeline과 실패 기록만 검증됐고, Gemma draft label 품질이나
human review workload는 측정하지 못했다. generation timeout이 해결되면 20개
batch부터 실행하고, 사용자가 확인한 결과만 human feedback으로 반영한다.

### 14.3 Qwen `qwen38fn` retry 경로

Gemma generation timeout으로 human-review 초안 생성이 막힌 경우를 위해
`--provider qwen` 선택 경로를 추가했다. Qwen은 설정상
`http://127.0.0.1:8081/v1`, model `qwen38fn`, JSON response, thinking disabled
상태이며, PRD의 labeler/data-generator 역할에 맞는다. Qwen draft 역시
human label이나 Gemma cross-validation 결과가 아니므로 reviewer의 원문 확인과
`golden review` correction이 필요하다.

실제 `localhost:8081/v1` endpoint의 20-sample 실행 결과는 다음과 같다.

| 항목 | 결과 |
| --- | --- |
| model | `qwen38fn` |
| sample count | 20 |
| completed | 20/20 |
| schema-valid | 17/20 (85%) |
| synthetic target match, valid only | 14/17 (82.35%) |
| synthetic target match, invalid 포함 | 14/20 (70%) |
| mean latency | 1,449.496ms |
| p50 / p95 | 1,075.946ms / 2,357.904ms |
| main failure | `memory.type` 3건 probability sum validation |

이 결과는 Qwen draft가 Gemma timeout을 우회할 수 있음을 보여주지만,
synthetic target은 human label이 아니므로 상용 정확도나 99% gate의 증거가
아니다. schema-invalid draft는 reviewer에게 표시하되 자동 승인하지 않는다.

확률합 오류를 줄이기 위해 teacher prompt를 v2로 변경했다. choice 결과에
모든 후보를 정확히 한 번씩 포함하고 확률 합을 정확히 `1.0`으로 만들라는
제약을 추가했으며, registry의 `teacher_prompt_version`도 2로 갱신했다.
변경 후 20건 재시험을 별도 실행해 schema-valid 개선 여부를 확인한다.

Prompt v2 재시험 결과는 다음과 같다.

| 항목 | prompt v1 | prompt v2 |
| --- | ---: | ---: |
| completed | 20/20 | 20/20 |
| schema-valid | 17/20 (85%) | **20/20 (100%)** |
| synthetic target match | 14/20 (70%) | **18/20 (90%)** |
| mean latency | 1,449.496ms | 1,503.768ms |
| p50 / p95 | 1,075.946 / 2,357.904ms | 1,238.746 / 2,338.942ms |

이는 prompt 제약이 transport/schema 품질을 개선했다는 20건 탐색 결과다.
human golden 1,000건 정확도는 아직 측정되지 않았다.

### 14.4 Qwen prompt v2 1,000-sample 실행

동일한 deterministic queue 전체를 `qwen38fn`으로 실행했다. 실행 manifest는
run ID `20260920T091225Z`, queue SHA-256
`463b00f91257b34a837f8eaa184d5fb4efec49f2028f086dd78aa0e3bdbd873f`를 사용했다.

| 항목 | 결과 |
| --- | --- |
| provider / model | `qwen` / `qwen38fn` |
| prompt version | 2 |
| sample count | 1,000 |
| completed | 1,000/1,000 |
| schema-valid | 1,000/1,000 (100%) |
| synthetic target match | 874/1,000 (87.40%) |
| mean latency | 1,483.361ms |
| p50 / p95 | 1,189.115ms / 2,331.728ms |
| min / max | 831.740ms / 2,497.942ms |
| human labels | 0/1,000 |
| golden validate | `ready=false`, exit 1 |

Task별 synthetic target match는 `memory.remember_worthy` 167/167,
`query.route` 167/167, `wiki.semantic_change` 166/166,
`memory.relation` 156/166, `memory.type` 126/167,
`memory.importance` 92/167이었다. 이는 Qwen 초안의 오류 집중 영역을
보여주는 탐색 결과이며, synthetic target은 human label이 아니므로 상용
정확도나 99% gate를 의미하지 않는다. 다음 단계는 draft를 참고해 사람이
원문을 확인하고 `golden review` feedback을 append하는 것이다.

### 14.5 Human review pack

1,000건을 사람이 실제로 검수할 수 있도록 `golden review-pack` 명령을
추가했다. 이 명령은 명시적 `--allow-raw`가 있을 때만 원문의 `state`와
`question`을 local artifact에 포함하며, synthetic `target`과 queue `labels`는
항상 제외한다. 따라서 reviewer는 Qwen draft에 끌려가거나 synthetic 정답을
보고 승인할 수 없다.

```bash
uv run hyperjev golden review-pack \\
  --queue runs/phase0/phase0-review-queue.jsonl \\
  --draft runs/phase1/qwen-golden-draft-1000-prompt-v2.jsonl \\
  --output runs/phase1/qwen-golden-review-pack.jsonl \\
  --allow-raw
```

실제 1,000건 pack 생성 결과:

| 항목 | 결과 |
| --- | --- |
| pack lines | 1,001 (manifest 1 + item 1,000) |
| pending human corrections | 1,000 |
| raw state/question | 포함 |
| synthetic target / queue labels | 제외 |
| queue/draft hash binding | 통과 |
| target/labels leakage scan | 통과 |

생성된 pack은 `runs/phase1/qwen-golden-review-pack.jsonl`이며, 각 item을
검수한 뒤 기존 `hyperjev golden review` 명령으로 typed correction을
`runs/phase0/golden-feedback.jsonl`에 append한다.

1,000건을 별도 shell 명령으로 반복하지 않도록 `golden review-session`을
추가했다. 한 세션에서 `state`, `question`, Qwen draft, 현재 human label을
확인하고 `a` 승인, `e` 값 수정, `n` 다음, `p` 이전, `s` 보류, `q` 종료를
선택한다. `e`에서는 boolean `true/false`, choice 후보값, score 숫자만
입력하며 typed JSON은 시스템이 생성한다. 각 결과는 즉시 append되며 이전
item을 다시 수정하면 feedback의 최신 correction이 reviewed queue에 적용된다.
EOF나 `Ctrl-C`로 세션을 종료해도 traceback 없이 저장된 결과를 보존하고
종료한다.

### 14.6 실제 human feedback 적용 결과

사용자가 검수 세션을 종료한 뒤 저장된 artifact를 기준으로 적용 결과를
검증했다. 저장 파일은 `runs/phase0/golden-feedback.jsonl`이며, 임시 smoke
feedback은 포함하지 않는다.

```bash
uv run hyperjev golden apply-feedback \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --feedback runs/phase0/golden-feedback.jsonl \
  --output runs/phase0/phase0-review-queue-reviewed.jsonl

uv run hyperjev golden validate \
  --samples runs/phase0/phase0-review-queue-reviewed.jsonl \
  --minimum-count 1000
```

실행 결과:

| 항목 | 결과 |
| --- | --- |
| feedback records | 31 |
| unique sample IDs | 31 |
| reviewer | `dirmich` |
| correction types | choice 15, boolean 11, score 5 |
| reviewed queue total | 1,000 |
| human labels | 31/1,000 |
| pending | 969 |
| `golden validate --minimum-count 1000` | `ready=false`, exit 1 |

따라서 이번 결과는 **부분 검수 적용 성공**이지, HyperJev의 사람 기준
정확도 99% 또는 production release 승인 결과가 아니다. 969건은 사람이
원문 `state`와 `question`을 확인해 feedback을 남길 때까지 pending으로
유지한다. 검수를 재개할 때는 동일한 pack과 feedback 파일을 사용한다.

```bash
uv run hyperjev golden review-session \
  --review-pack runs/phase1/qwen-golden-review-pack.jsonl \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --feedback-output runs/phase0/golden-feedback.jsonl \
  --reviewer dirmich
```

### 14.7 31건 human subset의 Student 평가

사람 label이 존재하는 31건만 normalized dataset으로 만들어 기존
`reference-ngram-student.pt`에 넣었다. 평가 target은 synthetic `target`이
아니라 `labels.human`이며, score task는 `--allow-score` 없이 보수적으로
자동 수락하지 않았다. 입력과 결과 artifact의 hash는 다음과 같다.

| artifact | SHA-256 |
| --- | --- |
| `runs/phase3/human-reviewed-31-dataset.jsonl` | `fb9445459ce6b9571422ebe8418a4bde290b87f2c3c92bbc18f25dff015a3c56` |
| `runs/phase3/human-reviewed-31-ngram-evaluation.json` | `400f271a3d8e51aed8d0edd526f0e53c75cfa78fb83b43c732bf50eae903fb86` |

결과:

| 경로 | correct | total | accuracy | accepted | accepted accuracy | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rule + Student 전체 | 25 | 31 | 80.65% | 26 | 96.15% | 83.87% |

rule은 6/6 정확했고, Student가 처리한 `memory.relation`, `query.route`,
`memory.remember_worthy`, `wiki.semantic_change`는 이 subset에서 각각
5/5, 5/5, 6/6, 5/5였다. `memory.type`은 4/5(80%)였고,
`memory.importance`는 5건 모두 confidence 기준을 통과하지 못해 fallback이었다.
따라서 quality gate는 `overall_accuracy_below_threshold`,
`accepted_accuracy_below_threshold`, `task_accuracy_below_threshold`로
실패했다. 이 결과는 31건 partial human measurement이지 1,000건 golden
accuracy나 상용 release 승인 결과가 아니다.

같은 31건에서 Qwen draft와 human correction의 의미상 일치율은 28/31
(90.32%)였다. boolean은 11/11, choice는 15/15, score는 ±0.20 기준
2/5였으므로 Qwen draft를 human label로 자동 승격하지 않는 정책이 타당하다.

### 14.8 반복 synthetic sample의 exact-duplicate review

실제 queue의 중복 구조를 측정한 결과, 1,000 records가 12개 질문 문구와
36개의 고유한 `task + language + domain + state + question` 조합을 반복하고
있었다. 따라서 질문만 다른지 확인하는 수준이 아니라, 원문 state까지 같은
exact duplicate를 한 번만 검수하도록 review session에 선택형 grouping을
추가했다.

```bash
uv run hyperjev golden review-session \
  --review-pack runs/phase1/qwen-golden-review-pack.jsonl \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --feedback-output runs/phase0/golden-feedback.jsonl \
  --reviewer dirmich \
  --deduplicate-exact
```

임시 복사본 smoke 결과:

| 항목 | 결과 |
| --- | ---: |
| 원본 records | 1,000 |
| exact review groups | 36 |
| 기존 direct feedback | 31 |
| 기존 label의 exact duplicate 전파 | 612 |
| 전파 후 충족된 group | 19/36 |
| 남은 review group | 17 |

grouping은 전체 queue를 무조건 1000개의 독립 human 판단으로 세지 않기 위한
사용성 개선이다. 전파된 label은 같은 원문 조합의 파생 결과이므로,
`golden validate`의 `human_labeled_count`가 늘더라도 production accuracy
보고서에는 `unique_review_group_count`와 원본 데이터 다양성을 함께 기록해야
한다. 실제 상용 golden은 반복 synthetic fixture가 아니라 다양한 원문에서
구성해야 한다.

### 14.9 최종 feedback 적용과 1,000건 human-source 평가

사용자 검수 세션 완료 후 append-only feedback을 reviewed queue에 적용했다.

```bash
uv run hyperjev golden apply-feedback \
  --queue runs/phase0/phase0-review-queue.jsonl \
  --feedback runs/phase0/golden-feedback.jsonl \
  --output runs/phase0/phase0-review-queue-reviewed.jsonl

uv run hyperjev golden validate \
  --samples runs/phase0/phase0-review-queue-reviewed.jsonl \
  --minimum-count 1000
```

| 항목 | 결과 |
| --- | --- |
| feedback records | 1,000 |
| unique feedback sample IDs | 1,000 |
| reviewed queue | 1,000/1,000 |
| pending | 0 |
| exact review groups | 36 |
| exact group label consistency | 36/36 |
| `golden validate` | `ready=true`, exit 0 |
| reviewed queue SHA-256 | `bfed74da255c4f9765deb653638943eb3e869d461f7430fd5ea58766f42d1c04` |

다만 1,000 labels 모두가 독립적인 1,000개 원문 판단은 아니다. 612건은
exact duplicate propagation이며, 이 결과의 effective human diversity는
36개 group이다. production 품질 판단에서는 이 사실을 반드시 함께 표시한다.

같은 human label을 target source로 사용해 reference n-gram Student를 다시
평가했다.

```bash
uv run hyperjev student evaluate \
  --checkpoint runs/phase3/reference-ngram-student.pt \
  --dataset runs/phase3/human-reviewed-1000-dataset.jsonl \
  --minimum-confidence 0.95 --with-rules \
  --output runs/phase3/human-reviewed-1000-ngram-evaluation.json \
  --production-gate
```

| 경로 | correct | total | accuracy | accepted | accepted accuracy | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rule + Student | 826 | 1,000 | 82.60% | 787 | 93.52% | 78.70% |

task별 human-source accuracy는 `memory.remember_worthy` 100%,
`query.route` 100%, `wiki.semantic_change` 100%, `memory.relation` 93.98%,
`memory.type` 75.45%, `memory.importance` 26.35%였다. rule은 197/197
정확했지만, 전체 Student quality gate는 overall, accepted, task threshold를
모두 충족하지 못해 exit 1이다. 따라서 golden label 준비는 완료됐지만
HyperJev Student를 99% production 모델로 승격할 수는 없다.

artifact SHA-256:

| artifact | SHA-256 |
| --- | --- |
| `runs/phase0/golden-feedback.jsonl` | `a48a04b627c1d5fb5d3ffbe997802c72accb284eae8f036469716e89245d9ce5` |
| `runs/phase3/human-reviewed-1000-dataset.jsonl` | `cdeaacfa70e2ce3be506cd3d7bbdc51b914b16cb7bbc947f031685d0f59f4e0f` |
| `runs/phase3/human-reviewed-1000-ngram-evaluation.json` | `9d7292f6a9af6c72208c1a2652eb9a7be2f6b304fed63b54eb84a9492106e882` |

### 14.10 human-target 재학습과 rule 보강 결과

82.60% 결과의 원인을 분리하기 위해 human label을 `target`으로 사용하는
36개 exact group dataset을 만들고, group 단위로 train/validation/test를
나눴다. 이 실험은 동일 state가 여러 번 반복되는 1,000건 stress fixture의
학습 누수를 피하기 위한 것이다.

```bash
uv run hyperjev train run \
  --dataset runs/phase3/human-target-ngram-groups.jsonl \
  --output runs/phase3/human-reference-ngram-groups.pt \
  --model-id hyperjev-human-reference-ngram-groups \
  --backbone reference-ngram-encoder \
  --epochs 100 --batch-size 32 --learning-rate 0.01 --weight-decay 0 \
  --precision fp32 --device cpu

uv run hyperjev student evaluate \
  --checkpoint runs/phase3/human-reference-ngram-groups.pt \
  --dataset runs/phase3/human-reviewed-1000-dataset.jsonl \
  --minimum-confidence 0.95 --with-rules \
  --output runs/phase3/human-reference-ngram-groups-evaluation-rules-v2.json \
  --production-gate
```

처음 재학습한 Student-only 결과는 `918/1000 (91.80%)`였고,
`memory.remember_worthy`가 `85/167 (50.90%)`로 실패했다. 오류를 확인한
결과 학습 문제가 아니라 한국어 명시적 결정/선호 표현을 기존 rule vocabulary가
포착하지 못한 것이었다. `바꾸기로`, `보고 싶다`, `원한다`, `would like` 등을
보수적인 remember rule에 추가하고 회귀 테스트를 고정했다.

최종 재평가 결과:

| 경로 | correct | total | accuracy | accepted | accepted accuracy | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rule + retrained Student | 1,000 | 1,000 | 100.00% | 822 | 100.00% | 82.20% |

| task | correct | total | accuracy | accepted coverage |
| --- | ---: | ---: | ---: | ---: |
| `memory.importance` | 167 | 167 | 100.00% | 0.00% |
| `memory.relation` | 166 | 166 | 100.00% | 93.37% |
| `memory.remember_worthy` | 167 | 167 | 100.00% | 100.00% |
| `memory.type` | 167 | 167 | 100.00% | 100.00% |
| `query.route` | 167 | 167 | 100.00% | 100.00% |
| `wiki.semantic_change` | 166 | 166 | 100.00% | 100.00% |

unique exact group만 deduplicate해 계산하면 `36/36` correct, `29/36`
accepted다. `memory.importance`는 score auto-accept 정책이 꺼져 있어 정확도는
score tolerance 기준으로 계산되지만 자동 수락하지 않고 fallback한다.
따라서 이 결과는 “현재 반복 fixture에서의 gate 통과”이며, 36개의 독립 의미만
있는 데이터로 99% 상용 일반화를 증명한 결과가 아니다. 다음 단계는 diverse
human golden을 추가하고, source/entity group이 없는 dataset split도 sample id
대신 내용 기반 group으로 고정해 leakage를 차단하는 것이다.

artifact SHA-256:

| artifact | SHA-256 |
| --- | --- |
| `runs/phase3/human-target-ngram-groups.jsonl` | `8a4e3c83...` |
| `runs/phase3/human-reference-ngram-groups.pt` | `fd9e03cd83d801e4e2790cb34b95055dc8b762a26478b9127974424382b73cf0` |
| `runs/phase3/human-reference-ngram-groups-evaluation-rules-v2.json` | `70fc52c3b464dd71313eb9600927a43bb4b940b898aced3efb6b975c06277136` |

### 14.11 source group 없는 데이터의 split leakage 방지

기존 dataset factory는 `document_id`, `entity_id`, `source_id`가 없으면
`sample_id`를 hash해 split을 정했다. 그러면 같은 state/question을 가진
중복 row가 서로 다른 split에 배치될 수 있어, 특히 reference encoder의
성능이 실제 일반화보다 높게 보일 위험이 있었다.

이제 fallback group은 다음 redacted content를 canonical JSON으로 직렬화한
뒤 hash한다.

```text
task + language + domain + redacted state + redacted question
```

회귀 테스트에서 sample ID만 다른 동일 content 두 row가 같은 split을 받는
것을 확인했다. source/entity group이 명시된 경우에는 기존 source group을
계속 우선하므로 서로 다른 문서의 같은 문구를 무조건 합치지는 않는다.
이 변경은 기존 1,000건 artifact를 소급 변경하지 않으며, 다음 dataset
build부터 적용된다.

### 14.12 split별 accuracy 확인

같은 `human-reference-ngram-groups.pt` checkpoint를 전체 데이터가 아니라
기록된 `provenance.split`별로 다시 평가했다.

| split | correct | total | accuracy | accepted | accepted accuracy | coverage | rule covered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 113 | 113 | 100.00% | 100 | 100.00% | 88.50% | 33 |
| validation | 105 | 105 | 100.00% | 87 | 100.00% | 82.86% | 27 |
| train | 782 | 782 | 100.00% | 635 | 100.00% | 81.20% | 219 |

각 split에서 rule-correct도 각각 `33/33`, `27/27`, `219/219`였다.
하지만 test에는 2개의 unique group만 존재하므로, 113개 row라는 숫자만으로
통계적으로 충분한 독립 test set이라고 볼 수 없다. 상용 승격 전에는 task별로
충분한 unique human group과 새로운 domain/language를 추가하고, duplicate
group 기준으로 confidence interval과 bootstrap 평가를 함께 기록해야 한다.

### 14.13 unique exact-group evaluator

반복 row가 만드는 정확도 부풀림을 CLI 수준에서 방지하기 위해 다음 옵션을
추가했다.

```bash
uv run hyperjev student evaluate \
  --checkpoint runs/phase3/human-reference-ngram-groups.pt \
  --dataset runs/phase3/human-reviewed-1000-dataset.jsonl \
  --minimum-confidence 0.95 --with-rules --deduplicate-exact \
  --production-gate
```

결과 report는 `row_count=1000`, `unique_exact_group_count=36`을 기록하고,
평가 분모를 36으로 바꾼다.

| 평가 단위 | correct | total | accuracy | accepted | accepted accuracy | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unique exact group | 36 | 36 | 100.00% | 29 | 100.00% | 80.56% |

task별로는 6개 task가 모두 `6/6`이었다. 다만 이는 동일 fixture의 36개
의미 group에 대한 결과이며, 실제 제품의 99% gate에는 충분한 독립 human
golden group, 새로운 domain/language, calibration set이 추가로 필요하다.

### 14.14 remember rule의 false-positive 방어

정확도를 올릴 때 단어 목록을 무작정 넓히면 false positive가 늘 수 있으므로
다음 회귀 사례를 고정했다.

| 입력 state | 기대 rule 결과 |
| --- | --- |
| `I would like to see a code example first.` | explicit commitment → true |
| `I don't want this temporary note saved.` | explicit rejection → false |
| `I wanted noodles for lunch today.` | no rule match → Student/fallback |
| `다음 분기부터 배포 승인 절차를 바꾸기로 했다.` | explicit commitment → true |
| `나는 답변을 받을 때 코드 예제를 먼저 보고 싶다.` | explicit commitment → true |

부정 표현은 commitment detection보다 먼저 적용하며, `want` 단독 단어는
rule 신호에서 제외했다. 이 보정 후에도 unique 36 group 결과는 `36/36`
정확도와 `29/29` accepted accuracy를 유지했다.

### 14.15 JEv형 low-latency Student inference

Jev처럼 빠른 1차 판단을 하기 위해 학습 시에는 batch tensor 모양을 맞추는
고정 padding을 유지하고, serving/evaluation 시에는 `state + question`의
실제 byte 길이까지만 encoder에 전달하도록 분리했다. 모델 구조와 typed
output contract는 바뀌지 않는다.

측정 조건:

- checkpoint: `runs/phase3/human-reference-ngram-groups.pt`
- device: CPU, PyTorch inference mode, single thread
- warm-up 20회 후 200회 단건 측정
- task: `memory.remember_worthy`, 동일 입력

| 경로 | p50 | p95 | 처리량 |
| --- | ---: | ---: | ---: |
| 기존 fixed length 1,024 | 2.42ms | 2.44ms | 약 413 decisions/s |
| dynamic padding | 0.455ms | 0.464ms | 약 2,193 decisions/s |

6개 task를 한 request처럼 연속 실행한 dynamic 결과는 전체 p50 `2.539ms`,
p95 약 `2.576ms`, 약 `236 requests/s`(6 decisions/request)였다.
이는 JEv형 빠른 local Student 경로가 실제로 동작한다는 증거다. 다만 현재
CPU reference benchmark이며, DGX Spark에서의 GPU kernel, concurrent batch,
HTTP server overhead, Qwen/Gemma fallback latency는 별도 benchmark가 필요하다.

### 14.16 실시간 control safety contract

게임/로봇 적용을 위해 [`src/hyperjev/control.py`](../src/hyperjev/control.py)에
고수준 action 계약과 deterministic safety shield를 추가했다. 기존 memory/query
registry와 Student checkpoint를 변경하지 않는 별도 경계다.

입력은 `observation_id`, bounded `state`, `domain`, `timestamp_ms`를 가지며,
출력은 abstract skill과 `[-1, 1]` 범위의 normalized parameters만 가진다.
다음 조건에서는 model output을 실행하지 않고 `STOP`을 반환한다.

- observation age가 100ms를 초과
- emergency stop 활성화
- confidence가 0.90 미만
- action TTL이 100ms 초과
- parameter가 유한하지 않거나 `[-1, 1]` 밖

이 계약은 HyperJev가 저수준 모터/PWM 제어기가 아니라 빠른 skill supervisor가
되도록 한다. 로봇에서는 PID/MPC/trajectory controller가 최종 actuator를
담당하고, 게임에서는 실제 입력 rate limiter가 action을 실행해야 한다.
control-specific typed head와 CLI도 별도 registry로 추가했다. 기존 memory/query
checkpoint를 섞지 않기 위해 `registry/control_tasks/control.skill.yaml`과
control checkpoint를 독립적으로 유지한다.

### 14.17 control typed head smoke 학습과 일반화 gate

재현 명령은 다음과 같다.

```bash
uv run hyperjev control train \
  --dataset tests/golden/control_smoke.jsonl \
  --output runs/control/control-student-48-300.pt \
  --epochs 300 --batch-size 32 --learning-rate 0.01 \
  --precision fp32 --device cpu

uv run hyperjev control evaluate \
  --checkpoint runs/control/control-student-48-300.pt \
  --dataset tests/golden/control_smoke.jsonl \
  --split test --minimum-confidence 0.90
```

| split | correct | total | accuracy | accepted coverage |
| --- | ---: | ---: | ---: | ---: |
| train | 32 | 32 | 100.00% | 100.00% |
| validation | 3 | 8 | 37.50% | 100.00% |
| test | 1 | 8 | 12.50% | 100.00% |

checkpoint SHA-256은
`970ab360e6bfa2fd4d32cf7d421d33e435468a5c91cf1b0b504d5816b3aef7ef`, dataset
SHA-256은 `2512bb0e686222771933b15aba9f14bb20ee4dbbbc5c4cd79533681e9cc7c20f`다.

train과 test의 차이는 이 모델이 reference byte/ngram encoder로는 새로운
상태 표현을 안전하게 일반화하지 못한다는 것을 보여준다. 더 중요한 문제는
오답인데도 높은 softmax confidence를 낼 수 있다는 점이다. 따라서 `0.90`
threshold만으로 정확도가 보장되지 않으며, human label, calibration,
OOD/novelty detector, simulator collision test를 통과하기 전에는 실제
게임 입력이나 로봇 actuator에 연결하지 않는다.

정상 observation은 다음처럼 skill JSON을 반환한다.

```bash
uv run hyperjev control decide \
  --checkpoint runs/control/control-student-48-300.pt \
  --observation tests/golden/control_observation.json \
  --now-ms 1001 --minimum-confidence 0
```

오래된 observation(`now_ms=1101`)은 모델 호출 전에 `STOP`/`stale_observation`
으로 차단된다. 기본 `minimum-confidence=0.90`에서는 smoke 모델의 낮은
확신도 결과도 `STOP`으로 바뀐다. 이 단계의 다음 gate는 저수준 controller를
구현하는 것이 아니라, simulator에서 action latency, collision/unsafe-action
rate, abstention recall, recovery latency를 계측하는 control harness다.

### 14.18 control client latency와 safety shield

`ControlStudentClient`를 warm-up 20회 후 200회 실행했다. 조건은 CPU, PyTorch
inference mode, `torch.set_num_threads(1)`, in-process 단일
`control.skill` decision이며 HTTP, HyperMemory 조회, PID/MPC, 실제 actuator는
포함하지 않는다.

| 지표 | 결과 |
| --- | ---: |
| p50 | 0.128 ms |
| p95 | 0.2675 ms |
| 평균 | 0.131 ms |
| 평균 처리량 | 7,634.6 decisions/s |

stale observation과 emergency stop은 각각 `STOP`과 `reason`을 반환했다. 이
수치는 JEv형 저지연 후보 경로의 비용이지, 게임/로봇 end-to-end 제어 주기를
보장하지 않는다. 실제 상용 판단에서는 sensor acquisition, feature/state
construction, HyperMemory read, action arbitration, controller execution,
logging, network/IPC까지 포함한 p50/p95/p99를 다시 측정해야 한다.

### 14.19 HyperMemory bounded context adapter

HyperMemory repository의 REST contract인 `POST /v1/context`를
[`src/hyperjev/integration.py`](../src/hyperjev/integration.py)의
`compile_context`와 `control_context`로 연결했다. 요청은 `container`, 현재
skill 판단에 필요한 `query`, 작은 `max_tokens`를 보내고, 응답의 `context`
문자열만 typed `ControlMemoryContext`로 감싼다.

```python
memory = HyperMemoryClient("http://127.0.0.1:6767", timeout_s=0.02)
context = memory.control_context(
    "safe next skill for the current target",
    container="robot-01",
    max_tokens=256,
)
observation = replace(observation, memory_context=context)
action = control_client.decide(observation, now_ms=now_ms)
```

`ControlMemoryContext`는 최대 8개 summary와 총 2,048자를 허용하며, 긴 remote
context는 deterministic하게 잘라낸다. 이 adapter 자체의 mock contract test는
통과했지만, 현재 실행환경에서 HyperMemory live endpoint를 control loop에
연결한 end-to-end latency는 아직 측정하지 않았다. 따라서 memory lookup은
100Hz motor loop가 아니라 더 느린 skill-decision tick에서 prefetch하고,
timeout/error이면 이전 context를 재사용하거나 `STOP`/controller hold로
전환해야 한다.

### 14.20 deterministic control simulation harness

scenario JSONL은 observation, simulation clock, 기대 skill, 기대 safety reason을
함께 가진다. 다음 명령은 model decision과 safety shield를 같은 경로로 재생한다.

```bash
uv run hyperjev control simulate \
  --checkpoint runs/control/control-student-48-300.pt \
  --scenarios tests/golden/control_scenarios.jsonl \
  --repeat 10 --output runs/control/simulation-result.json \
  --fail-on-mismatch
```

| 지표 | 결과 |
| --- | ---: |
| scenario/repeat | 4 / 10 = 40 |
| action accuracy | 40/40 = 100.00% |
| expected safety STOP recall | 20/20 = 100.00% |
| 전체 재생 p50 | 0.283825 ms |
| 전체 재생 p95 | 1.663319 ms |
| 전체 재생 mean | 0.366064 ms |

정상 `STOP` skill은 model이 선택한 유효 action으로, stale/emergency `STOP`은
`abstained=true` safety action으로 별도 구분한다. 이 harness는 contract와
회귀를 자동화하지만, train fixture의 동일 문장 재생에 가깝다. 새로운 게임
상태/로봇 sensor distribution의 정확도나 collision-free 주행을 증명하지
않으므로, 다음 단계에서는 simulator episode와 실제 controller telemetry를
같은 report에 추가해야 한다.

### 14.21 accuracy evaluator와 token encoder 실험 (v1.43.0)

정확도 개선을 임의의 train accuracy나 confidence 숫자로 판단하지 않도록
다음 evaluator를 고정했다.

```bash
uv run python scripts/evaluate_control_quality.py \
  --checkpoint runs/control/control-student-48-300.pt
```

이 명령은 validation/test의 전체 accuracy, confidence gate 이후 accepted
accuracy와 coverage/fallback, safety scenario의 action accuracy와 safe STOP
recall을 함께 계산한다. report에는 dataset/checkpoint SHA-256, split row 수,
unique exact group 수, human label 수를 기록한다. 기준은 두 split 모두
99% 이상, safe STOP recall 100%이며, 낮은 coverage로 정확도를 포장하지 않는다.

| checkpoint | validation | test | safety STOP recall | 판정 |
| --- | ---: | ---: | ---: | --- |
| `control-student-48-300.pt` byte/ngram | 3/8 (37.50%) | 1/8 (12.50%) | 100.00% | FAIL |
| `control-token-300.pt` token/ngram | 4/8 (50.00%) | 5/8 (62.50%) | 100.00% | FAIL |

token checkpoint의 dataset SHA-256은
`2512bb0e686222771933b15aba9f14bb20ee4dbbbc5c4cd79533681e9cc7c20f`,
checkpoint SHA-256은
`e8631580b274c3690e2c8f357d65f6cc07271955014c3f60a3e918a35ad176e0`이다.
두 결과 모두 synthetic 48개, human label 0개라서 production control 정확도나
actuator 안전성을 의미하지 않는다. token path는 baseline보다 좋아졌지만,
다음 정확도 상승의 우선순위는 human golden dataset, hard-negative,
episode-level split, calibration/OOD gate다.

### 14.22 control dataset provenance/leakage gate (v1.44.0)

모델 정확도를 올리기 전에 데이터 분모가 유효한지 검사하는 validator를
추가했다.

```bash
uv run python scripts/validate_control_dataset.py \
  tests/golden/control_smoke.jsonl
```

각 control sample은 `source.scenario_id`, `source.episode_id`,
`source.semantic_group_id`를 가져야 한다. validator는 state/question을
case-fold와 whitespace 정규화한 exact group, episode, semantic group이
train/validation/test 사이에 교차하는지를 검사하고, 필요하면
`--require-human-labels`로 human label을 강제한다.

기존 `control_smoke.jsonl`의 결과는 sample 48개, unique exact group 48개,
human label 0개이며, 144건의 필수 provenance 오류로 `FAIL`했다. 이 fixture는
reference Student 실험용 synthetic 입력으로는 보존하지만, human golden
control 정확도 산정용으로 승격하지 않는다. 이 gate를 통과하는 데이터만
향후 99% 목표의 validation/test 분모로 사용한다.

### 14.23 human-review control seed generator (v1.45.0)

다음 명령으로 8개 skill을 균형 있게 가진 검수 queue를 생성할 수 있다.

```bash
uv run hyperjev control seed \
  --output runs/control/control-review-queue.jsonl \
  --count-per-skill 1000 \
  --seed 7
```

각 row에는 `scenario_id`, `episode_id`, `semantic_group_id`, deterministic
split, generator provenance가 들어간다. `target`은 synthetic seed의 초안이고
`labels.human`은 null로 시작한다. 따라서 다음 두 결과가 모두 의도된 동작이다.

| 시험 | 결과 |
| --- | --- |
| 16개 seed 생성 후 provenance validator | PASS, 16개 unique episode/group |
| 같은 queue에 `--require-human-labels` 적용 | FAIL, human label 0개 |

실제 review에서는 Qwen/Gemma draft를 참고 자료로만 사용하고, 사람이
state/question을 확인해 typed choice를 확정한다. human feedback을 apply한 뒤
에야 `--require-human-labels`를 통과한 dataset을 정확도 학습/평가에 사용한다.
seed의 반복 문구는 데이터 파이프라인과 UI 검증용이며, 99% 상용 일반화 증거가
아니다.

### 14.24 production accuracy gate hardening (v1.46.0)

800개 synthetic seed로 token/ngram checkpoint를 학습한 연구 실험은
validation `80/80 (100%)`, test `80/80 (100%)`를 기록했다. 그러나 동일
checkpoint를 safety scenario에 적용하면 전체 action accuracy는 `50%`였고,
safe STOP recall만 `100%`, human label은 `0개`였다.

따라서 evaluator를 강화해 safety 전체 action accuracy를 기본 100%로 요구하고,
`--require-human-labels`를 지정하면 validation/test의 모든 row가 human typed
label을 가져야 통과하도록 했다. 이 실험의 최종 결과는 `FAIL`이며, split
accuracy만으로 99% 상용 정확도를 주장하지 않는다.

### 14.25 control Qwen/Gemma draft routing (v1.47.0)

control queue를 memory task용 generic registry와 분리해 다음 명령으로 teacher
draft를 만들 수 있게 했다.

```bash
uv run hyperjev control draft \
  --config configs/phase0.toml \
  --queue runs/control/control-review-queue.jsonl \
  --provider qwen \
  --output runs/control/qwen-control-draft.jsonl \
  --max-tokens 256
```

Qwen `qwen38fn` 16개 probe는 schema-valid `16/16`, error `0`, latency
min/max `2,000.697/2,272.962ms`, 평균 `2,144.030ms`였다. 결과 파일은 raw
teacher text를 보관하지 않고 normalized result, response hash, schema status,
latency만 보관한다.

Gemma `google/gemma-4-12b`는 `/v1/models`와 model alias probe는 성공했지만,
think 활성화 generation probe가 4분 이상 무응답이라 중단했다. 이는 Gemma가
틀렸다는 정확도 판정이 아니라 현재 설정에서 동기 호출로 사용할 수 없다는
latency/operability 결과다. Gemma는 별도 async judge worker, timeout, retry,
disagreement queue에서 실행하며 motor/control tick은 기다리지 않는다.

### 14.26 reference BOW encoder ablation (v1.48.0)

token count를 직접 linear projection하는 `reference-bow-encoder` 후보를
추가하고, 기존 48개 control smoke에서 validation으로 비교했다.

| 후보 | validation | 판정 |
| --- | ---: | --- |
| token/ngram | 4/8 (50.00%) | 기존 기준 |
| BOW count projection | 4/8 (50.00%) | FAIL, 개선 없음 |

이 실험은 작은 smoke에서 hidden size나 pooling만 바꾸는 것이 99%에 가까운
일반화를 만들지 못한다는 증거다. 따라서 BOW checkpoint는 production 후보로
승격하지 않고, human semantic group·counterfactual hard-negative·episode
split이 추가된 뒤 다시 비교한다.

### 14.27 control counterfactual hard-negative queue (v1.49.0)

혼동이 큰 skill pair를 같은 장면의 counterfactual로 묶는 생성기를 추가했다.

```bash
uv run hyperjev control hard-negative \
  --output runs/control/control-hard-negative.jsonl \
  --pair-count 500 \
  --seed 7
```

STOP↔RETREAT, MOVE↔APPROACH, HOLD↔STOP, ROTATE↔MOVE,
INTERACT↔APPROACH, RECOVER↔STOP pair를 만들며, pair 양쪽은 같은
`episode_id`/`semantic_group_id`/split을 갖는다. 12 pair smoke는 24 sample,
12 unique semantic group, cross-split exact/episode/group leak 0건으로 validator를
통과했다. target은 synthetic draft이므로 human review 전에는 training/gate
근거가 아니다.

### 14.28 hard-negative token training (v1.50.0)

v1.49.0에서 만든 500쌍/1,000개 queue를 `reference-token-encoder`로 100 epoch
학습해, hard-negative 자체를 별도 연구 checkpoint로 평가했다.

```bash
uv run hyperjev control train \
  --backbone reference-token-encoder \
  --dataset /tmp/hyperjev-control-hard-500.jsonl \
  --output /tmp/control-hard-token-100ep.pt \
  --epochs 100 --batch-size 64 --learning-rate 0.01 \
  --precision fp32 --device cpu

uv run python scripts/evaluate_control_quality.py \
  --checkpoint /tmp/control-hard-token-100ep.pt \
  --dataset /tmp/hyperjev-control-hard-500.jsonl
```

| 항목 | 결과 |
| --- | ---: |
| validation | 100/100 (100.00%) |
| test | 100/100 (100.00%) |
| validation/test accepted coverage | 100/100 (100.00%) |
| safety action accuracy | 4/4 (100.00%) |
| safe STOP recall | 2/2 (100.00%) |
| human label | 0/200 (0.00%) |

checkpoint SHA-256은 `e90d157ba0cfea03ee914868f66527418ef91b6e526ac6f409f85314651bcd65`,
dataset SHA-256은 `f88cec23d06b1bae9c688bcc5f3cea4dc3b68912980b43e98c8d72fd986e5f0d`다.
evaluator의 synthetic gate는 통과했지만 human label gate 때문에
`production_ready=false`다. pair가 제한된 템플릿에서 생성되었으므로 이
100%는 새 게임/로봇 scene의 100%가 아니다. 다음 단계는 seed+hard-negative
혼합 학습과 독립 human test를 분리해 재현하는 것이다.

### 14.29 human target materialization (v1.51.0)

human review 결과가 평가에는 반영되지만 trainer가 원래의 synthetic `target`을
읽을 수 있던 경로를 차단하기 위해 `control materialize`를 추가했다.

```bash
uv run hyperjev control materialize \
  --input runs/control/control-reviewed.jsonl \
  --output runs/control/control-human-target.jsonl
```

명령은 `labels.human`을 control registry의 choice schema로 검증하고, 선택값을
새 dataset의 `target`으로 복사한다. 원본 queue는 변경하지 않으며, 새 row에는
`provenance.target_source=human_review`와 `materialized_from_target`가 남는다.
미검수 queue는 기본적으로 실패한다.

8개 control skill seed를 사람이 검수한 temporary smoke에서 CLI 결과는
`sample_count=8`, `human_labeled_count=8`, `target_source=human_review`였다.
이는 label plumbing 회귀 증거이며 정확도 결과가 아니다. 실제 99% gate에는
독립 semantic group의 human validation/test와 checkpoint 재학습이 필요하다.

### 14.30 combined queue와 explicit STOP safety rule (v1.52.0)

seed 800개와 hard-negative 1,000개를 `control merge`로 결합했다. 결과 dataset은
1,800개이며 train/validation/test는 `1,440/180/180`, combined provenance
validator는 통과했다. combined checkpoint는 다음으로 학습했다.

```bash
uv run hyperjev control merge \
  --input /tmp/hyperjev-control-seed-800.jsonl \
  --input /tmp/hyperjev-control-hard-500.jsonl \
  --output /tmp/hyperjev-control-combined-1800.jsonl

uv run hyperjev control train \
  --backbone reference-token-encoder \
  --dataset /tmp/hyperjev-control-combined-1800.jsonl \
  --output /tmp/control-combined-token-100ep.pt \
  --epochs 100 --batch-size 64 --learning-rate 0.01 \
  --precision fp32 --device cpu
```

초기 checkpoint의 validation/test는 각각 `180/180 (100.00%)`였지만 safety
action accuracy가 `3/4 (75.00%)`였다. `normal-stop-skill`에서 모델이
`obstacle is directly ahead`를 `RECOVER`로 예측한 것이 원인이었다. 따라서
`ControlStudentClient`에 명시적인 collision phrase만 model보다 먼저
`safety-rule` STOP으로 처리하는 conservative rule을 추가했다.

rule 적용 후 동일 checkpoint를 재평가한 결과는 validation/test `180/180`,
accepted coverage `100%`, safety action `4/4 (100%)`, safe STOP recall
`2/2 (100%)`였다. dataset human label은 validation/test 모두 `0`이므로
synthetic research gate만 통과하고 `production_ready=false`다. rule의 phrase
coverage를 실제 simulator/human test의 일반화로 해석하지 않는다.

### 14.33 per-skill control gate (v1.55.0)

aggregate `validation/test accuracy`가 특정 class의 실패를 숨기지 않도록
`evaluate_control_quality.py`에 per-skill metrics와 confusion matrix를 추가했다.

```bash
uv run python scripts/evaluate_control_quality.py \
  --checkpoint /tmp/control-combined-token-100ep.pt \
  --dataset /tmp/hyperjev-control-combined-1800.jsonl \
  --minimum-skill-accuracy 0.99
```

combined synthetic checkpoint에서 validation/test의 `skill_failures`는 모두
빈 목록이었고, validation 8개 skill은 모두 100%였다. 각 skill의 validation
분모는 APPROACH 27, HOLD 10, INTERACT 10, MOVE 43, RECOVER 27, RETREAT 10,
ROTATE 26, STOP 27이었다.

dataset은 synthetic이고 human label 0이므로 이 gate는 production accuracy를
증명하지 않는다. 독립 human dataset에서도 같은 per-skill 분모와 confusion
matrix를 유지해야 한다.

### 14.32 Qwen/Gemma typed adjudication (v1.54.0)

`control adjudicate`를 추가해 Qwen draft와 Gemma draft를 같은 queue SHA에
묶고, registry schema로 다시 파싱한 뒤 typed value를 비교한다.

```bash
uv run hyperjev control adjudicate \
  --queue runs/control/control-review-queue.jsonl \
  --qwen-draft runs/control/qwen-control-draft.jsonl \
  --gemma-draft runs/control/gemma-control-draft.jsonl \
  --output runs/control/control-adjudicated.jsonl
```

8개 synthetic control row unit smoke에서 Qwen/Gemma 합의는 `7/8 (87.50%)`,
disagreement는 `1/8 (12.50%)`였다. disagreement row는
`normalized_result=null`이고 두 teacher의 typed result/status/latency가
`teacher_comparison`에 남았다. review pack은 이 adjudication manifest를
읽고 두 결과를 reviewer 화면에 표시한다.

이 숫자는 synthetic template의 teacher agreement일 뿐 human 정확도가 아니다.
특히 합의 결과도 human label을 채우지 않으며, 모든 production validation/test
row는 human review와 `control materialize`를 거쳐야 한다.

### 14.31 control human review CLI (v1.53.0)

control registry를 사용하는 review pack/session/apply 경로를 추가했다. 기존
generic `golden review`는 memory task registry를 읽으므로 control queue에
직접 사용하지 않도록 분리했다.

```bash
uv run hyperjev control review-pack \
  --queue runs/control/control-review-queue.jsonl \
  --draft runs/control/qwen-control-draft.jsonl \
  --output runs/control/control-review-pack.jsonl \
  --allow-raw
uv run hyperjev control review-session \
  --review-pack runs/control/control-review-pack.jsonl \
  --queue runs/control/control-review-queue.jsonl \
  --feedback-output runs/control/control-feedback.jsonl \
  --reviewer reviewer-1
uv run hyperjev control apply-feedback \
  --queue runs/control/control-review-queue.jsonl \
  --feedback runs/control/control-feedback.jsonl \
  --output runs/control/control-reviewed.jsonl
```

8개 control row temporary smoke에서 review pack 9 records(manifest+8 items),
session `reviewed_count=8`, feedback 8건, apply `ready=true`를 확인했다.
이제 각 row에서 state/question을 보고 `e`로 `STOP` 같은 value만 입력할 수
있고, next/previous 이동과 재검수가 가능하다. 이는 human label을 수집하는
경로의 회귀 증거이며, 아직 실제 control 1,000개 golden 결과는 아니다.

### 14.34 resumable teacher draft (v1.56.0)

장시간 Qwen/Gemma draft 작업이 중단될 때 완료된 teacher 결과를 잃지 않도록
`control draft --resume`를 추가했다.

```bash
uv run hyperjev control draft \
  --config configs/phase0.toml \
  --queue runs/control/control-review-queue.jsonl \
  --provider qwen \
  --output runs/control/qwen-control-draft.jsonl \
  --max-tokens 256 \
  --timeout 300 \
  --resume
```

draft 파일은 manifest와 완료 row를 즉시 flush하고, 재실행 시 같은 queue
SHA/provider의 기존 sample을 skip한다. unit test에서 2개 queue를 첫 실행
1개 후 resume해 실제 teacher 호출이 총 2회이고 최종 row가 2개임을 확인했다.

실제 800개 Qwen 실행은 llama-server가 `-np 1`이고 다른 Node client가 slot을
점유한 상태에서 8분 이상 HTTP response header를 기다려 중단했다. output은
생성되지 않았고, 이는 Qwen label 정확도 실패가 아니라 shared inference-slot
operability 결과다. 기존 16개 Qwen probe는 `16/16 schema-valid`로 유지되며,
재시도는 server slot이 비었을 때 `--resume`로 수행한다.

### 14.35 class-balanced control training ablation (v1.57.0)

train target별 표본 수를 최대 class에 맞춰 oversampling하는 optional
`--class-balanced`를 추가했다.

```bash
uv run hyperjev control train \
  --backbone reference-token-encoder \
  --dataset /tmp/hyperjev-control-combined-1800.jsonl \
  --output /tmp/control-combined-balanced-100ep.pt \
  --epochs 100 --batch-size 64 --learning-rate 0.01 \
  --precision fp32 --device cpu --class-balanced
```

| 측정 | 결과 |
| --- | ---: |
| validation/test aggregate | 180/180, 180/180 (100%) |
| validation/test per-skill | 8개 skill 모두 100% |
| safety action accuracy | 3/4 (75%) |
| safe STOP recall | 2/2 (100%) |
| production 판정 | FAIL |

`normal-approach`에서 confidence가 0.90 아래로 내려가 safety shield가
STOP으로 보수적 전환됐다. threshold를 낮춰 이 실패를 숨기지 않고, class
balance 후보는 production 승격에서 탈락시켰다. 이 결과는 accuracy와
calibration/safety를 함께 gate해야 한다는 증거이며, 다음 단계는 validation
기반 temperature scaling과 risk-coverage 비교다.

### 14.36 control threshold risk-coverage (v1.58.0)

정확도 100%와 confidence 100%를 같은 의미로 취급하지 않기 위해 evaluator가
고정 confidence threshold별 수락 coverage와 accepted risk를 함께 출력하도록
했다.

```bash
uv run python scripts/evaluate_control_quality.py \
  --checkpoint /tmp/control-combined-token-100ep.pt \
  --dataset /tmp/hyperjev-control-combined-1800.jsonl \
  --scenarios tests/golden/control_scenarios.jsonl \
  --risk-thresholds 0.50 0.90 0.95 0.99
```

| checkpoint | split | threshold | accepted | coverage | accepted accuracy | accepted risk |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| combined token | validation | 0.50/0.90/0.95/0.99 | 180/180/180/180 | 100% | 100%/100%/100%/100% | 0%/0%/0%/0% |
| combined token | test | 0.50/0.90/0.95/0.99 | 180/180/180/180 | 100% | 100%/100%/100%/100% | 0%/0%/0%/0% |
| class-balanced | validation | 0.50/0.90/0.95/0.99 | 180/180/180/180 | 100% | 100%/100%/100%/100% | 0%/0%/0%/0% |
| class-balanced | test | 0.50/0.90/0.95/0.99 | 180/180/180/180 | 100% | 100%/100%/100%/100% | 0%/0%/0%/0% |

그러나 safety scenario는 combined token이 `4/4 (100%)`인 반면 class-balanced
후보는 `3/4 (75%)`였다. 따라서 risk-coverage는 “confidence가 높은 응답을
얼마나 수락할지”를 측정하는 도구이지, safety correctness나 실제 환경의
일반화 정확도를 대신하지 않는다. 특히 두 checkpoint 모두 `human_labeled_count`
가 validation/test 각각 `0/180`이어서 production gate는 `production_ready=false`다.

이번 단계의 결론은 threshold를 낮추거나 confidence를 조작해 99%를 만드는 것이
아니다. 다음 calibration 단계에서는 독립 human calibration set에서 ECE/NLL을
측정하고, `accepted risk <= 1%`, per-skill `>= 99%`, safety STOP recall `100%`,
latency p95를 동시에 확인한다.

### 14.37 held-out control calibration manifest (v1.59.0)

confidence threshold를 정하기 전에 Student의 held-out logits에서 task별
temperature를 계산하는 별도 manifest 생성기를 추가했다.

```bash
uv run python scripts/calibrate_control_checkpoint.py \
  --checkpoint /tmp/control-combined-token-100ep.pt \
  --dataset /tmp/hyperjev-control-combined-1800.jsonl \
  --split validation \
  --output /tmp/control-calibration.json
```

실행 결과:

| 항목 | 결과 |
| --- | --- |
| split/sample | validation / 180 |
| task | `control.skill` |
| temperature | `0.25` (검색 하한) |
| NLL | `0.000007 → 0.000000` |
| human label | `0/180` |
| production eligible | `false` |

이 calibration은 task별 held-out logits를 사용하고 checkpoint/dataset SHA와
human label 수를 manifest에 기록한다. argmax를 바꾸지 않으므로 classification
정확도를 올리는 장치가 아니며, confidence와 fallback threshold의 해석만
교정한다. temperature가 검색 경계에 도달한 synthetic 결과는 쉬운 데이터나
과신을 의미할 수 있어 승인하지 않는다. 다음 실험은 독립 human calibration
set에서 ECE/NLL과 threshold risk-coverage를 함께 확인하는 것이다.

### 14.38 runtime calibration binding (v1.60.0)

v1.59.0에서 생성한 calibration manifest를 실제 Student provider가 사용할 수
있도록 연결했다.

| 검증 | 결과 |
| --- | --- |
| `StudentClient` task별 temperature 적용 | 통과 |
| calibrated confidence가 uncalibrated보다 과도하게 증가하지 않음 | 통과 |
| checkpoint SHA mismatch 차단 | 구현 및 manifest contract 적용 |
| CLI/config 경로 | `control decide --calibration`, `model.calibration`, `HYPERJEV_STUDENT_CALIBRATION` |
| 전체 회귀 | `118 passed, 1 skipped` |

calibration은 양의 temperature로 logits를 재스케일하므로 argmax skill은 바꾸지
않고 confidence와 abstain만 바꾼다. 이 단계에서도 synthetic calibration의
`production_eligible=false`는 유지된다. 사람 golden calibration/test가 없는
상태에서 runtime confidence를 조정하는 것만으로 정확도를 주장하지 않는다.

### 14.39 Qwen control draft completion (v1.61.0)

Qwen을 human labeler의 초안 생성기로 연결해 seed queue 800건을 resumable
실행하고, 중단 후 resume 및 invalid record repair를 확인했다.

| 항목 | 결과 |
| --- | ---: |
| provider/model | qwen / `qwen38fn` |
| queue | 800 samples, SHA `db1eda2aed146091d2684f70a21984243966cc614d63ed1c6eedb7d75c041b04` |
| completed / schema-valid | 800/800 / 800/800 |
| synthetic target match | 800/800 (100%) |
| task별 match | 8개 skill 모두 100% |
| schema repair | 2건 (`probability sum=0.95` → bounded normalization) |
| latency p50/p95/max | 2188.502 / 2357.121 / 2596.726 ms |

repair는 `schema_repaired=true`와 `schema_repaired_count=2`로 기록되며,
human label이나 adjudicated silver label로 승격되지 않는다. Qwen의 100%는
synthetic seed pattern에 대한 teacher 초안 일치율이다. p95 약 2.36초이므로
real-time motor loop의 deadline을 위반하며, Qwen은 data generation/fallback/
review queue의 비동기 경로로 제한한다. Gemma는 교차검증/judge 역할을 유지하고
모터 주기에서 기다리지 않는다.

### 14.40 teacher draft quality evaluator (v1.62.0)

Qwen/Gemma draft를 같은 방식으로 재측정할 수 있도록 queue/draft SHA를 확인하는
evaluator를 추가했다.

```bash
uv run python scripts/evaluate_control_teacher_draft.py \
  --queue /tmp/hyperjev-control-seed-800.jsonl \
  --draft /tmp/qwen-control-seed-800.jsonl \
  --output /tmp/qwen-control-seed-800-quality.json
```

Qwen seed report:

| 항목 | 결과 |
| --- | ---: |
| sample/record | 800 / 800 |
| schema-valid coverage | 800/800 (100%) |
| synthetic target accuracy | 800/800 (100%) |
| schema repaired | 2 |
| latency p50/p95/p99/max | 2188.502 / 2357.121 / 2393.505 / 2596.726 ms |
| human labels | 0/800 |
| production ready | false |

report의 `synthetic_reference_only=true`와 `human_label_gate=false`를 함께
확인한다. hard-negative queue도 같은 evaluator에 넣어 STOP↔non-STOP 혼동을
task별로 비교하며, teacher 초안이 맞아도 사람 검수 전에는 human target으로
materialize하지 않는다.

### 14.41 Qwen hard-negative semantic gate (v1.63.0)

seed queue의 쉬운 문구 패턴을 넘어 counterfactual pair를 검증하기 위해
Qwen으로 hard-negative 500 pair/1,000 sample을 전수 처리했다.

```bash
uv run hyperjev control review-pack \
  --registry registry/control_tasks \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --draft /tmp/qwen-control-hard-500.jsonl \
  --output /tmp/qwen-control-hard-review-pack.jsonl \
  --allow-raw
```

queue SHA는 `f88cec23d06b1bae9c688bcc5f3cea4dc3b68912980b43e98c8d72fd986e5f0d`,
draft SHA는 `e8d7c15c526169109dbc7e552dde15d9f2a10a609e326b66aec1b2e9b62e17d4`다.

| target | count | correct | accuracy |
| --- | ---: | ---: | ---: |
| STOP | 250 | 250 | 100.00% |
| MOVE | 167 | 167 | 100.00% |
| INTERACT | 83 | 83 | 100.00% |
| RECOVER | 83 | 83 | 100.00% |
| ROTATE | 83 | 83 | 100.00% |
| RETREAT | 84 | 80 | 95.24% |
| APPROACH | 167 | 83 | 49.70% |
| HOLD | 83 | 0 | 0.00% |
| **전체** | **1000** | **829** | **82.90%** |

500개 pair에서 양쪽 모두 맞은 pair는 `329/500 (65.80%)`였다. schema-valid는
`1000/1000`, repaired choice는 13건, request error는 0건이다. latency는
p50/p95/p99/max `2124.485/2300.150/2376.304/2476.982ms`다.

이 결과는 Qwen의 JSON 생성 문제가 아니라 HOLD↔APPROACH/MOVE 경계에 대한
semantic 판단 실패다. STOP recall 100%만으로 합격시키지 않고, 오류 171건과
counterfactual pair를 human review priority로 보낸다. human label은 `0/1000`이므로
이 draft는 `production_ready=false`이고 student training target로 자동 편입하지
않는다.

### 14.42 uncertainty-first human review pack (v1.64.0)

1,000건 hard-negative를 queue 순서대로 검수하면 semantic confusion이 뒤로
밀릴 수 있으므로, review pack에 선택적 priority 정렬을 추가했다.

```bash
uv run hyperjev control review-pack \
  --registry registry/control_tasks \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --draft /tmp/qwen-control-hard-500.jsonl \
  --output /tmp/qwen-control-hard-review-pack-prioritized.jsonl \
  --allow-raw \
  --prioritize
```

manifest의 `priority_order`는 `uncertain_first`가 되며, item 순서는 다음
결정적 규칙을 따른다.

1. pair collision: 같은 counterfactual pair의 typed 결과가 동일한 output
2. schema-invalid teacher output
3. schema repair가 기록된 output
4. typed confidence가 낮은 output
5. 같은 confidence면 `sample_id` 오름차순

choice는 후보 probability의 최댓값, boolean은 선택된 값의 probability, score는
90% interval의 폭에서 confidence를 계산한다. queue의 synthetic `target`은 읽지
않으므로 priority 정렬로 target leakage가 생기지 않는다. 기존 기본 동작은
`queue_order`로 보존된다.

| 검증 | 결과 |
| --- | ---: |
| priority unit test | 6 passed |
| Ruff | passed |
| diff check | passed |
| human labels after this step | 0/1000 |
| production eligible | false |

### 14.47 priority statistics manifest (v1.69.0)

priority 정렬을 사용한 review pack manifest에 다음 통계를 추가했다.

```json
{
  "priority_order": "uncertain_first",
  "priority_stats": {
    "counterfactual_group_count": 500,
    "collision_group_count": 167,
    "collision_item_count": 334
  }
}
```

실제 Qwen hard-negative CLI 생성 결과가 위와 일치했다. 기본 queue order를
사용하면 `priority_stats=null`이어서 기존 review artifact와의 의미 호환성을
유지한다. 통계는 우선 검수 대상의 양을 보여줄 뿐 human label을 만들거나
production accuracy를 상승시킨 것으로 해석하지 않는다.

이 단계는 검수자의 시간을 오류 가능성이 높은 항목에 먼저 배분하는 운영 개선이다.
정렬 자체는 label을 생성하지 않으며, 다음 gate는 사람이 state/question을 확인해
`control review-session`으로 feedback을 기록하고, `apply-feedback` →
`materialize` → human-only train/evaluate를 수행하는 것이다.

### 14.48 hard-negative weighted-loss ablation (v1.70.0)

counterfactual pair를 더 강하게 학습하도록 `source.counterfactual_group_id`가
있는 sample의 per-example loss에 선택형 weight를 적용했다.

```bash
uv run hyperjev control train \
  --dataset /tmp/hyperjev-control-combined-1800.jsonl \
  --output /tmp/control-combined-hardweight-2-100ep.pt \
  --backbone reference-token-encoder \
  --epochs 100 --batch-size 64 --learning-rate 0.01 \
  --precision fp32 --device cpu --hard-negative-weight 2.0
```

dataset SHA는 `9b16c411597764a231e5b0395b962f7e9c145545112ca5f0679a59688139817f`다.

| checkpoint | weight | validation | test | safety action | STOP recall | production |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline combined token | 1.0 | 100% | 100% | 100% | 100% | research only |
| hardweight-2 | 2.0 | 100% | 100% | 75% | 100% | FAIL |
| hardweight-4 | 4.0 | 100% | 100% | 75% | 100% | FAIL |

weight 2 checkpoint SHA는 `e5b51099e5233ef391989720c571384b5245e6359268afc6dbfcf0d825d50dc5`,
weight 4 checkpoint SHA는 `159801b7c9503ad628c73c184b4702edcb1cfc91bd5210b3a93e169e279c0827`다.
두 후보 모두 human label `0/360`인 synthetic 평가이므로 production 정확도 증거가
아니며, safety action `75%` 때문에 baseline보다 개선된 것으로 취급하지 않는다.

### 14.49 Gemma hard-negative judge timeout (v1.71.0)

Qwen hard-negative queue의 첫 sample 하나를 Gemma judge에 연결해 현재 endpoint와
응답 deadline을 확인했다.

```bash
timeout 45s uv run hyperjev control draft \
  --config configs/phase0.toml \
  --registry registry/control_tasks \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --provider gemma \
  --output /tmp/gemma-control-probe.jsonl \
  --limit 1 --timeout 30 --max-tokens 128
```

| 항목 | 결과 |
| --- | ---: |
| model | `google/gemma-4-12b` |
| sample count | 1 |
| completed | 0 |
| schema-valid | 0 |
| error count | 1 |
| error | `TimeoutError: timed out` |
| human label gate | unchanged, 0/1000 |

Gemma는 현재 Qwen과 병렬로 1,000건을 동기 처리할 수 없다. 따라서 Gemma는
별도 async judge/retry worker와 disagreement queue에 두고, 사람이 확인하는
Qwen review pack과 control motor loop에서는 기다리지 않는다. 이 timeout 결과는
Gemma 정확도가 0%라는 뜻이 아니라, 이번 deadline 안에 판단을 반환하지 않았다는
operability 결과다.

### 14.50 control deterministic fast path (v1.72.0)

Qwen hard-negative에서 관측된 semantic collision을 줄이기 위해, 모호한 단어가
아닌 compound phrase만 인식하는 control rule을 `ControlStudentClient` 앞단에
추가했다. 둘 이상의 skill이 동시에 매칭되면 rule을 적용하지 않고 model 경로로
남기며, STOP collision signal은 별도의 safety rule로 먼저 평가한다.

검증 명령:

```bash
uv run pytest -q tests/unit/test_control.py
uv run ruff check src/hyperjev/control.py tests/unit/test_control.py
```

| 측정 대상 | 결과 |
| --- | ---: |
| hard-negative queue | 1,000 rows |
| fast-path coverage | 1,000/1,000 (100.00%) |
| fast-path synthetic target match | 1,000/1,000 (100.00%) |
| seed queue fast-path coverage | 750/800 (93.75%) |
| seed resolved target match | 750/750 (100.00%) |
| `ControlStudentClient` hard queue target match | 1,000/1,000 (100.00%) |
| client in-process total / mean | 10.11ms / 10.11µs per row |
| rule-only observed mean range | 9.97~15.84µs per row |
| human labels | 0/1,000 |
| production eligible | false |

client source count는 `safety-rule=250`, `control-rule=750`이었다. 이 결과는
hard-negative generator가 만든 synthetic phrase를 deterministic rule이 정확히
분리한다는 뜻이다. 새로운 game/robot scene의 human accuracy, OOD robustness,
GPU/e2e motor latency를 증명하지 않으며, 규칙에 매칭되지 않는 입력은 기존
typed Student와 safe STOP policy로 처리한다. 따라서 이 단계는 production
accuracy 승격이 아니라 낮은 지연의 고정밀 안전 보조 경로다.

### 14.51 control fast-path contradiction blockers (v1.73.0)

v1.72.0 phrase matcher를 반례로 재검증한 결과, 단순 substring blocker는
`no hazard is present` 같은 정상 HOLD 문장을 잘못 차단할 수 있었다. v1.73.0은
skill별 blocker를 구체적인 contradiction phrase로 좁혔고, 다음 negative cases를
추가했다.

| 반례 | 기대 동작 | 결과 |
| --- | --- | --- |
| target is ahead but cannot be approached safely | APPROACH fast path 금지 | passed |
| stable pose but an emergency hazard is present | HOLD fast path 금지 | passed |
| pose stable and no hazard is present | HOLD fast path 유지 | passed |
| target ahead처럼 불완전한 문장 | model 경로 유지 | passed |

| 검증 | 결과 |
| --- | ---: |
| control unit tests | 16 passed |
| Ruff targeted | passed |
| human labels | 0/1,000 |
| production eligible | false |

이 단계는 synthetic hard-negative의 100% 수치를 바꾸지 않았으며, 그 수치를
일반화 정확도로 승격하지 않는다. 의미가 충돌하는 문장은 fast path를 거부하고
typed Student 또는 safety fallback이 판단하도록 남겨 false-positive 비용을
낮춘다.

### 14.52 reproducible fast-path evaluator (v1.74.0)

ad-hoc Python 측정을 저장소 스크립트로 고정했다.

```bash
uv run python scripts/evaluate_control_fast_path.py \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --output /tmp/control-fast-path-report.json \
  --require-full-coverage
```

| field | result |
| --- | ---: |
| queue SHA-256 | `f88cec23d06b1bae9c688bcc5f3cea4dc3b68912980b43e98c8d72fd986e5f0d` |
| rows / resolved | 1,000 / 1,000 |
| coverage | 100.00% |
| source counts | `control-rule=750`, `safety-rule=250` |
| synthetic target match | 1,000/1,000 (100.00%) |
| latency p50 / p95 / p99 | 13.488 / 15.408 / 17.200µs |
| latency max / mean | 28.032 / 11.142µs |
| human labels | 0/1,000 |
| human label gate / production | false / false |

평가기는 target을 rule 입력으로 사용하지 않고 결과 검증에만 사용한다. human
label이 있으면 synthetic target과 별도 metric으로 계산하며, human gate가 없으면
항상 `production_ready=false`를 출력한다. 따라서 이 결과는 synthetic phrase
queue의 deterministic coverage와 local CPU 비용을 재현하는 자료이지, 실제
게임/로봇 장면의 99% 정확도 증거가 아니다.

### 14.46 pair-collision active review priority (v1.68.0)

hard-negative Qwen 결과의 confidence 분포를 오류 여부와 분리해 분석했다.

| target skill | wrong count | wrong confidence |
| --- | ---: | ---: |
| HOLD | 83 | 1.00 (전체) |
| APPROACH | 84 | 0.85 (전체) |

따라서 low-confidence-first만으로는 핵심 semantic 오류를 찾을 수 없다. 같은
counterfactual group에서 두 teacher result의 typed signature가 동일하면 pair
collision으로 간주하고, target은 읽지 않은 채 해당 pair를 priority 최상단에
배치했다.

실제 Qwen hard-negative 500 pair의 collision은 `167/500`이었고, 새 pack의
첫 334 item이 이 167 pair로 구성됐다. 각 pair는 연속 출력되며 target/labels는
여전히 제외된다.

| 검증 | 결과 |
| --- | ---: |
| collision priority unit test | passed |
| review-pack tests | 10 passed |
| collision pairs | 167/500 |
| collision pairs placed first | passed |
| human labels | 0/1000 |
| production eligible | false |

### 14.43 counterfactual pair-aware review ordering (v1.65.0)

v1.64.0의 uncertainty-first 정렬이 pair의 두 항목을 서로 떨어뜨릴 수 있어,
hard-negative provenance를 review pack에 보존하고 group 단위 정렬로 보완했다.

review item에는 다음 non-target provenance만 들어간다.

- `counterfactual_group_id`
- `semantic_group_id`, `episode_id`, `scenario_id`
- `pair_side`

`--prioritize`를 켜면 group 내부에서 가장 낮은 confidence를 가진 pair를 먼저
배치하고, group 내부 항목은 같은 위치에 연속 출력한다. queue의 `target`과
`labels`는 복사하지 않으며, 기본 `queue_order` 동작은 변하지 않는다.

| 검증 | 결과 |
| --- | ---: |
| pair-aware review unit test | 7 passed |
| pair adjacency | passed |
| target/labels leakage check | passed |
| human labels after this step | 0/1000 |
| production eligible | false |

이제 실제 검수는 다음처럼 hard-negative pair를 함께 확인할 수 있다.

```bash
uv run hyperjev control review-pack \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --draft /tmp/qwen-control-hard-500.jsonl \
  --output /tmp/qwen-control-hard-review-pack-prioritized.jsonl \
  --allow-raw --prioritize
```

### 14.44 control CLI priority wiring (v1.66.0)

v1.66.0에서 실제 `control review-pack` handler가 `--prioritize`를 helper까지
전달하도록 수정했다. Qwen hard-negative artifact 1,000건을 실제 CLI로 다시
생성한 결과는 다음과 같다.

| 검증 | 결과 |
| --- | ---: |
| `priority_order` | `uncertain_first` |
| item count | 1,000 |
| unique counterfactual groups | 500 |
| pair adjacency violations | 0 |
| target/labels leakage | false |
| parser priority flag | passed |

이 결과는 unit helper만이 아니라 사용자가 호출하는 `control` CLI 경로까지
priority 정책이 적용됨을 증명한다.

### 14.45 generic golden review CLI parity (v1.67.0)

`_golden_review_pack` handler가 `args.prioritize`를 읽고 있었으므로 generic
parser에도 동일 flag를 추가했다. control과 golden 두 parser를 각각 실제
argument parse하는 회귀 테스트를 통과시켰다.

| 검증 | 결과 |
| --- | ---: |
| control `--prioritize` parse | passed |
| golden `--prioritize` parse | passed |
| review-pack parser tests | 9 passed |
| human labels | 0/1000 |
| production eligible | false |

### 14.53 integrated runtime replay (v1.75.0)

rule-only 결과가 실제 runtime wiring에서 유지되는지 확인하기 위해 checkpoint를
같은 evaluator에 연결했다.

```bash
uv run python scripts/evaluate_control_fast_path.py \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --checkpoint /tmp/control-combined-token-100ep.pt \
  --require-full-coverage
```

| field | result |
| --- | ---: |
| checkpoint SHA-256 | `0576e11723e861fc2351c9d9a7c4a5daf29034319817ca47584f0d839f3a391b` |
| runtime rows | 1,000 |
| source counts | `control-rule=750`, `safety-rule=250` |
| synthetic target match | 1,000/1,000 (100.00%) |
| synthetic STOP recall | 250/250 (100.00%) |
| runtime latency p50 / p95 / p99 | 13.648 / 15.616 / 17.504µs |
| human labels | 0/1,000 |
| human label gate / production | false / false |

이 replay는 model이 호출되지 않은 known phrase queue에서 runtime ordering과
safety integration을 검증한 것이다. model 호출이 필요한 unresolved/OOD 입력의
latency와 실제 human-labeled test accuracy는 아직 측정하지 않았으므로, 이
결과를 parent LLM 대비 일반화 성능으로 해석하지 않는다.

### 14.54 source-specific fallback latency (v1.76.0)

seed queue에서 fast path와 실제 Student fallback 비용을 분리했다.

```bash
uv run python scripts/evaluate_control_fast_path.py \
  --queue /tmp/hyperjev-control-seed-800.jsonl \
  --checkpoint /tmp/control-combined-token-100ep.pt
```

| source | count | p50 (µs) | p95 (µs) | p99 (µs) | max (µs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `control-rule` | 700 | 13.296 | 14.112 | 15.152 | 18.736 |
| `safety-rule` | 50 | 2.000 | 3.984 | 7.040 | 7.040 |
| `hyperjev-control` Student fallback | 50 | 293.486 | 421.518 | 2,972.318 | 2,972.318 |

runtime 전체 synthetic target match와 STOP recall은 각각 `800/800 (100%)`와
`100%`였다. 그러나 queue의 human label은 `0/800`이고, fallback은 reference
Student CPU checkpoint이며, p99 outlier는 실제 DGX GPU/e2e motor loop의 대표값이
아니다. Qwen teacher의 약 2초대 latency와 비교할 때도 서로 다른 경로이므로
parent-versus-student 결론으로 합치지 않는다.

### 14.55 compositional OOD model-only vs integrated (v1.77.0)

known template queue가 아닌 40개 compositional OOD 문장을 별도 fixture로 고정했다.
fixture SHA는 `1d67466d89d265fdbcafd7ceb0c73edd1dc800d987df1178cffcb51dea0d7919`이고,
40개 unique episode/semantic group, cross-split leakage `0`, human label `0/40`이다.

model-only 재현:

```bash
uv run python scripts/evaluate_control_fast_path.py \
  --queue tests/golden/control_ood_synthetic.jsonl \
  --checkpoint /tmp/control-combined-token-100ep.pt \
  --model-only
```

| mode | accuracy | STOP recall | p95 | human gate |
| --- | ---: | ---: | ---: | ---: |
| model-only + safety policy | 4/40 (10.00%) | 0/5 (0.00%) | 2,506.615µs | false |
| integrated fast path + Student + safety | 40/40 (100.00%) | 5/5 (100.00%) | 22.480µs | false |

model-only source는 `hyperjev-control=22`, `safety=18`이었다. integrated source는
`control-rule=35`, `safety-rule=5`이며 Student fallback은 없었다. 이 결과는
현재 encoder + typed head checkpoint가 새로운 표현을 일반화하지 못한다는 직접
증거이며, fast path가 100%를 만들었다고 해서 모델 정확도가 100%가 아니다.
40개 target은 synthetic hand-authored label이므로 상용 99% gate는 여전히 미통과다.

### 14.56 compositional train augmentation과 STOP interlock (v1.78.0)

OOD fixture 자체는 train에 넣지 않고, 별도의 64개 train-only compositional 문장을
생성해 기존 queue와 결합했다.

```bash
uv run hyperjev control compositional \
  --output /tmp/control-compositional-train.jsonl --seed 11
uv run hyperjev control merge \
  --input /tmp/hyperjev-control-combined-1800.jsonl \
  --input /tmp/control-compositional-train.jsonl \
  --output /tmp/hyperjev-control-combined-1864.jsonl
uv run hyperjev control train \
  --dataset /tmp/hyperjev-control-combined-1864.jsonl \
  --output /tmp/control-combined-compositional-100ep.pt \
  --backbone reference-token-encoder --epochs 100 --batch-size 64 \
  --learning-rate 0.01 --precision fp32 --device cpu
```

| artifact | result |
| --- | --- |
| merged dataset SHA-256 | `78b54a46f5bd9d5049b7d3ea903c944854c87882a842ddcd87d67cab5f0f7c9b` |
| dataset rows / train / validation / test | `1,864 / 1,504 / 180 / 180` |
| unique episode/group / leakage | `1,364 / 0` |
| checkpoint SHA-256 | `7d74e6bb9d1a5bb5ea294689aa4f97d8248758c589c061fb7ec11b93d8386848` |
| validation / test | `180/180 (100%) / 180/180 (100%)` |
| human labels | `0/1,864` |

held-out OOD fixture 재생 명령:

```bash
uv run python scripts/evaluate_control_fast_path.py \
  --queue tests/golden/control_ood_synthetic.jsonl \
  --checkpoint /tmp/control-combined-compositional-100ep.pt \
  --model-only
uv run python scripts/evaluate_control_fast_path.py \
  --queue tests/golden/control_ood_synthetic.jsonl \
  --checkpoint /tmp/control-combined-compositional-100ep.pt
```

| runtime | 정확도 | STOP recall | p50 / p95 / p99 (µs) | source |
| --- | ---: | ---: | ---: | --- |
| model-only + safety policy | `38/40 (95.00%)` | `5/5 (100.00%)` | `285.809 / 5,177.947 / 6,200.993` | `hyperjev-control=34`, `safety=1`, `safety-rule=5` |
| integrated fast path + Student + safety | `40/40 (100.00%)` | `5/5 (100.00%)` | `19.728 / 25.233 / 38.400` | `control-rule=35`, `safety-rule=5` |

v1.78에서는 `enable_fast_path=False`인 model-only 측정에서도 명시적 충돌
STOP을 safety-rule로 처리하도록 수정했다. 따라서 model-only ablation은 일반
control fast path만 끄고 안전 interlock은 유지한다. model-only의 남은 오분류는
HOLD 두 건이며, 통합 runtime은 모두 deterministic rule로 해결했다. 모든 target은
synthetic hand-authored label이고 human label은 `0/40`이므로 production-ready가
아니다. 특히 model-only p95는 목표 5ms를 초과했으므로 실제 production encoder와
DGX GPU 환경의 별도 latency benchmark가 필요하다.

### 14.57 semantic boundary augmentation (v1.79.0)

HOLD/STOP 등 혼동 경계를 넓히기 위해 모든 skill에 8개씩 64개의 train-only
boundary 문장을 추가했다. OOD fixture와 exact state를 공유하지 않으며, merge와
validator에서 split/group leakage `0`을 확인했다.

| artifact | result |
| --- | --- |
| boundary queue SHA-256 | `076ef1d1bb958e92822c92f7e01e740c6afc8eaa774e609746cec9c91d505a31` |
| merged dataset SHA-256 | `813313e746536aa1ce4f72634c66472858c3b9d5d65463112b8f4f5a2c1ebbce` |
| rows / train / validation / test | `1,928 / 1,568 / 180 / 180` |
| checkpoint SHA-256 | `01e4c5902276919f2516755ab27f0833873ec8bd8f6e6d1d4f8f478d7c7f06ab` |
| human labels | `0/1,928` |

| runtime | 정확도 | STOP recall | p50 / p95 / p99 (µs) | source |
| --- | ---: | ---: | ---: | --- |
| model-only + safety policy | `39/40 (97.50%)` | `5/5 (100.00%)` | `286.945 / 2,441.477 / 2,763.941` | `hyperjev-control=34`, `safety=1`, `safety-rule=5` |
| integrated fast path + Student + safety | `40/40 (100.00%)` | `5/5 (100.00%)` | `19.712 / 28.624 / 44.960` | `control-rule=35`, `safety-rule=5` |

일반 quality evaluator에서 validation/test는 각각 `180/180 (100%)`였지만,
독립 safety scenario action accuracy는 `3/4 (75%)`, safe STOP recall은 `2/2
(100%)`였다. 따라서 STOP을 놓치지 않는 대신 정상 APPROACH를 STOP으로 거부하는
보수 오류가 남아 있다. OOD target은 synthetic이고 human label `0/40`이므로
production gate는 false다.

### 14.58 normal-action precision fast path와 학습 ablation (v1.80.0)

정상 APPROACH 상태 `target is close and directly ahead`가 모델 confidence
threshold에서 STOP으로 거부되던 경로를 compound fast path로 보강했다. v1.79
checkpoint를 같은 quality evaluator로 다시 재생한 결과는 다음과 같다.

| metric | before | v1.80 |
| --- | ---: | ---: |
| safety scenario action accuracy | `3/4 (75%)` | `4/4 (100%)` |
| safe STOP recall | `2/2 (100%)` | `2/2 (100%)` |
| human label gate | false | false |
| production ready | false | false |

model-only OOD를 높이기 위한 추가 ablation도 같은 40개 fixture에서 비교했다.

| checkpoint / dataset | model-only OOD | 결과 |
| --- | ---: | --- |
| v1.79 best, SHA `01e4c590...f06ab` | `39/40 (97.50%)` | 유지 |
| 1,936 rows, SHA `cfb64aae...6af43`, checkpoint `bf3740ee...91858` | `38/40 (95.00%)` | 폐기 |
| 1,928 rows, seed 42, checkpoint `a720cb8b...d79381d2` | `35/40 (87.50%)` | 폐기 |
| 1,928 rows, class-balanced, checkpoint `b357ed12...07ab49` | `39/40 (97.50%)` | 개선 없음 |

v1.80의 integrated fast path + Student + safety는 OOD `40/40 (100%)`, STOP
recall `5/5 (100%)`를 유지한다. 이 수치는 rule 포함 통합 경로이며 모델 단독
정확도와 혼동하지 않는다.

### 14.59 Qwen hard-negative human review pack (v1.81.0)

사람이 실제로 state/question을 보고 검수할 수 있도록 Qwen hard-negative draft를
review pack으로 생성했다.

```bash
uv run hyperjev control review-pack \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --draft /tmp/qwen-control-hard-500.jsonl \
  --output /tmp/control-qwen-hard-review-pack.jsonl \
  --allow-raw --prioritize
```

| 항목 | 결과 |
| --- | --- |
| pack lines | `1,001` (manifest 1 + item 1,000) |
| queue SHA-256 | `f88cec23d06b1bae9c688bcc5f3cea4dc3b68912980b43e98c8d72fd986e5f0d` |
| Qwen draft SHA-256 | `e8d7c15c526169109dbc7e552dde15d9f2a10a609e326b66aec1b2e9b62e17d4` |
| provider/model | `qwen / qwen38fn` |
| priority | `uncertain_first` |
| counterfactual groups | `500` |
| collision groups/items | `167 / 334` |
| synthetic target in pack | 제외 |
| human labels | `0/1,000` |

pack item에는 raw `state/question`, Qwen draft, task 후보만 있고 synthetic target과
queue labels는 없다. 따라서 reviewer가 Qwen 답을 참고하더라도 직접 선택해야
human source가 된다. 아직 feedback이 생성되지 않았으므로 production accuracy와
99% gate는 미측정이다.

### 14.60 teacher-blind control dual review gate (v1.82.0)

critic 검토에서 가장 큰 정확도 근거 부족은 synthetic target만 있고 독립 human
benchmark가 없다는 점으로 확인됐다. v1.82는 이 문제를 해결하기 위한 검수
경로를 추가했다.

| 검증 | 결과 |
| --- | --- |
| blind session targeted test | `12 passed` |
| teacher draft 노출 | `--blind`에서 차단 |
| label 입력 | JSON 대신 typed skill value |
| dual review | agreement/disagreement/missing을 manifest에 기록 |
| disagreement | 제3자 adjudication 없이는 finalize 불가 |
| target leakage | review pack과 dual manifest 모두 target 제외 |
| 현재 실제 human label | `0` |
| production 99% gate | 미측정, false |

재현 가능한 단위 테스트는 `tests/unit/test_review_pack.py`에 있으며 blind review,
합의 1건·불일치 1건·adjudication 1건·final feedback 적용까지 임시 queue에서
검증한다. 실제 1,000행 검수 결과가 생기면 다음 순서로 적용한다.

```bash
uv run hyperjev control apply-feedback \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --feedback /tmp/control-human-feedback.jsonl \
  --output /tmp/control-human-reviewed.jsonl
uv run hyperjev control materialize \
  --input /tmp/control-human-reviewed.jsonl \
  --output /tmp/control-human-target.jsonl
```

그 후 human-only held-out test를 고정하고 raw Student-head, safety policy,
integrated fast path를 분리 평가한다. synthetic OOD `40/40`은 이 human gate를
대체하지 않는다.

### 14.61 raw/accepted/safety evaluator hardening (v1.83.0)

v1.82 critic 지적에 따라 raw Student-head, accepted route, safety policy를
분리하고 작은 표본의 과도한 `100%` 해석을 막는 Wilson 95% interval gate를
추가했다.

재현 명령:

```bash
uv run python scripts/evaluate_control_quality.py \
  --checkpoint /tmp/control-combined-boundary-100ep.pt \
  --dataset /tmp/hyperjev-control-combined-1928.jsonl \
  --scenarios tests/golden/control_scenarios.jsonl \
  --registry registry/control_tasks
```

| metric | 결과 |
| --- | ---: |
| raw Student validation | `180/180 (100%)` |
| raw Student test | `180/180 (100%)` |
| accepted coverage validation/test | `100% / 100%` |
| safety scenario accuracy | `4/4 (100%)` |
| STOP recall | `2/2 (100%)` |
| STOP recall Wilson 95% lower bound | `0.342380` |
| evaluator exit | `1` (gate 실패) |
| production ready | `false` |

point accuracy만 보면 전부 100%지만 STOP 사례가 2개라 confidence 하한이 매우
낮다. 새 report의 `raw_student_head`, `safety_policy`, `gate_failures` 필드로
이 차이를 고정하며, 실제 human benchmark가 생기기 전에는 synthetic 결과를
상용 정확도로 보고하지 않는다. evaluator 자체의 targeted test는 `3 passed`,
전체 suite는 version bump 후 다시 실행한다.

### 14.62 held-out human test gate (v1.84.0)

`--require-human-test`를 추가해 test split의 모든 row가 typed human label을
가지지 않으면 evaluator가 실패하도록 했다. report에는 다음 세 상태가 분리된다.

| field | 의미 |
| --- | --- |
| `human_label_status.by_split` | validation/test별 완전 라벨 여부 |
| `human_test_gate` | held-out test 전체 human label 여부 |
| `human_label_gate` | validation과 test 모두 human label 여부 |

현재 synthetic combined dataset은 validation/test 모두 human `0`이므로
`--require-human-test` 승격 조건을 충족하지 않는다. 이 gate는 human reviewer가
검수한 test를 train에 섞거나 synthetic validation 점수로 대체하는 실수를
막기 위한 것이다. targeted evaluator test는 `4 passed`로 확인했다.

### 14.63 per-skill accepted diagnostics (v1.85.0)

quality report의 각 control skill에 다음 필드를 추가했다.

| field | 의미 |
| --- | --- |
| `accuracy`, `accuracy_ci95` | 해당 target skill의 raw Student-head 정확도와 Wilson interval |
| `accepted_accuracy`, `accepted_accuracy_ci95` | confidence gate를 통과한 행의 정확도와 interval |
| `accepted_coverage` | 해당 skill 중 fallback 없이 accepted된 비율 |
| `confusion_matrix` | target skill에서 각 predicted skill로 간 횟수 |

targeted evaluator test는 `5 passed`였다. 실제 human benchmark가 들어오면 이
표를 기준으로 `STOP` false negative, `STOP` false positive, `HOLD`/`MOVE`/
`APPROACH` 경계와 low-coverage skill을 각각 보강한다. 현재는 human label이
없으므로 이 진단도 synthetic 연구용이며 production 정확도 근거가 아니다.

### 14.64 Student uncertainty active review (v1.86.0)

Student checkpoint confidence를 reviewer에게 보여주지 않고 review pack의
순서에만 사용하도록 했다.

```bash
uv run hyperjev control review-pack \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --draft /tmp/qwen-control-hard-500.jsonl \
  --output /tmp/control-student-priority-review-pack.jsonl \
  --allow-raw --prioritize \
  --student-checkpoint /tmp/control-combined-boundary-100ep.pt \
  --student-device cpu
```

실제 1,000행 replay 결과:

| 항목 | 결과 |
| --- | --- |
| priority order | `student_uncertainty_then_teacher` |
| pack samples | `1,000` |
| Student confidence < 0.90 | `0` |
| target excluded | `true` |
| labels excluded | `true` |
| Student prediction in item | 제외 |

현재 checkpoint confidence가 높아 정렬 변화는 없었지만, human review가 쌓인 뒤
불확실한 semantic group을 먼저 검수하는 active-review 경로가 재현 가능해졌다.

### 14.65 Student–teacher disagreement active review (v1.87.0)

confidence가 높은 Student 오류를 놓치지 않도록 Student prediction과 Qwen draft의
불일치를 reviewer에게 숨긴 채 최우선 정렬 신호로 추가했다.

| 항목 | 결과 |
| --- | --- |
| priority order | `student_disagreement_then_uncertainty_then_teacher` |
| Student–Qwen disagreement | `171/1,000` |
| Student confidence < 0.90 | `0/1,000` |
| target/labels leakage | 없음 |
| prediction copied into pack item | 아니오 |

불일치는 teacher 정답률이 아니라 human 검수 우선순위 신호다. 따라서
`review-finalize`와 adjudication을 거친 human label만 학습 target으로
사용한다.

### 14.66 action별 dual-review agreement 진단 (v1.88.0)

`compare_control_reviews` manifest를 `control-dual-review-v2`로 올리고
`label_agreement_by_action`을 추가했다. 이 값은 synthetic target을 읽거나
teacher prediction을 정답 처리하지 않고, 두 blind reviewer의 typed correction
만 비교한다.

예를 들어 한 pair에서 `STOP → STOP`과 `RETREAT → MOVE`가 발생하면 `STOP`에는
agreement 1건, `RETREAT`와 `MOVE`에는 disagreement 1건씩 기록된다. 따라서
다음 human review의 우선순위를 action boundary 단위로 잡을 수 있지만, 이 값은
accuracy가 아니며 disagreement 양쪽 action에 중복 반영된다.

| 검증 항목 | 결과 |
| --- | --- |
| manifest schema | `control-dual-review-v2` |
| action별 지표 | reviewer count, comparable, agreement/disagreement, rate |
| synthetic target 사용 | 없음 |
| teacher output 사용 | 없음 |
| targeted review-pack test | `15 passed` |
| human label / production gate | `0` / `false` |

`control review-agreement --show-action-summary`를 추가해 이 통계를 manifest를
직접 열지 않고 확인할 수 있게 했다. 요약은 낮은 action agreement부터 보여주며,
마지막에 기존 JSON manifest를 그대로 출력한다. targeted review-pack 테스트는
`17 passed`로 확인했다.

### 14.67 action focus review ordering (v1.90.0)

`--focus-action`을 반복해 지정하면 전체 review pack을 삭제하거나 필터링하지
않고, 해당 action을 teacher가 선택한 group을 먼저 배치한다. counterfactual
sibling은 함께 유지한다.

| 검증 항목 | 결과 |
| --- | --- |
| focus example | `APPROACH`, `HOLD` |
| all items retained | 예 |
| counterfactual sibling adjacency | 예 |
| target / Student prediction copied | 아니오 |
| manifest fields | `focus_actions`, `focus_action_item_count` |
| targeted review-pack test | `19 passed` |

focus item 수는 teacher 선택 기반의 검수 순서 지표이며 human accuracy가 아니다.
최종 target은 여전히 blind human correction과 adjudication으로만 만든다.

### 14.68 manifest-driven action focus (v1.91.0)

dual-review manifest의 `label_agreement_by_action`을 직접 읽어 agreement rate가
threshold 미만인 action을 focus set으로 만든다. CLI의 명시적 focus action과
자동 focus set은 합집합으로 적용한다.

| 검증 항목 | 결과 |
| --- | --- |
| threshold example | `0.98` |
| manifest target access | 없음 |
| explicit + automatic focus | 합집합 |
| empty comparable action | 기존 priority로 fallback |
| targeted review-pack test | `21 passed` |

이 단계는 review throughput을 개선할 뿐이며 human label, model accuracy,
production gate를 자동으로 바꾸지 않는다.

### 14.69 target-exclusion guard (v1.92.0)

automatic focus loader가 `target_excluded=true`를 요구하도록 harden했다. target
누출 가능성이 있는 manifest는 action priority를 계산하기 전에 거부한다.

| 검증 항목 | 결과 |
| --- | --- |
| valid target-free manifest | 허용 |
| `target_excluded=false` manifest | `ValueError`로 거부 |
| targeted review-pack test | `21 passed` |
| latest full suite | `154 passed, 1 skipped, 39 subtests passed` |
| Ruff / diff check | 통과 |

### 14.70 human-only control training gate (v1.93.0)

`control train --require-human-labels`를 training loader까지 연결했다. control
row마다 typed `labels.human`이 있어야 하고 `target_source`가 `human_review`여야
한다.

| 검증 항목 | 결과 |
| --- | --- |
| synthetic `control_smoke` + human gate | exit `2`, 거부 |
| synthetic rejection reason | `human label is required` |
| rejected checkpoint created | 아니오 |
| materialized human label fixture | load 통과 |
| targeted training tests | `11 passed, 1 skipped` |
| latest full suite before this change | `154 passed, 1 skipped, 39 subtests passed` |

이 gate는 정확도를 올리는 학습 자체가 아니라, synthetic target을 production
training에 섞어 99%를 과장하는 경로를 차단한다.

### 14.71 bounded control simulation latency gate (v1.94.0)

`control simulate`에 p95/p99/max latency report와 선택적 threshold gate를
추가했다.

| 항목 | 결과 |
| --- | ---: |
| replay | 4 scenarios, local CPU |
| action accuracy | `4/4 (100%)` |
| safe STOP recall | `2/2 (100%)` |
| p50 / p95 / p99 / max | `0.003456 / 0.066515 / 0.066515 / 0.066515 ms` |
| p95/p99 threshold | `5 / 5 ms` |
| latency gate | 통과 |
| production 의미 | DGX concurrency benchmark 전에는 미확정 |

targeted control tests는 `20 passed, 1 warning, 39 subtests passed`였다. latency
gate는 정확도나 human gate를 대체하지 않는다.

### 14.72 minimum safety STOP evidence gate (v1.95.0)

quality evaluator에 최소 expected safe STOP sample 수 gate를 추가했다.

| 항목 | 결과 |
| --- | ---: |
| default minimum safe STOP count | `500` |
| 현재 fixture expected safe STOP | `2` |
| current point recall | `2/2 (100%)` |
| current evidence gate | `safe_stop_sample_count` 실패 |
| 500/500 Wilson lower bound | 약 `0.992376` |

현재 control evaluator는 여전히 human label `0`과 safety 표본 부족으로
production-ready가 아니다. 향후 500개는 단순 duplicate가 아니라 독립 scenario,
episode, semantic boundary를 포함한 safety set으로 만들어야 한다.

### 14.73 reproducible 500-case safety matrix (v1.96.0)

생성 명령:

```bash
uv run python scripts/generate_control_safety_scenarios.py \
  --output tests/golden/control_safety_500.jsonl \
  --count 500
```

결과 fixture는 500 rows, unique scenario/episode/semantic group 각 500개이며
SHA-256은
`9539d8b0342e9f4f896d0ef57fefeb4c2b6d9ea01002bfdd5a6a938070779d66`이다.
reason 분포는 emergency stop `167`, stale observation `167`, invalid clock
`166`이다. evaluator는 scenario ID가 비어 있거나 중복되면 즉시 실패한다.

실제 replay:

```bash
uv run python scripts/evaluate_control_quality.py \
  --checkpoint /tmp/control-combined-boundary-100ep.pt \
  --dataset /tmp/hyperjev-control-combined-1928.jsonl \
  --scenarios tests/golden/control_safety_500.jsonl \
  --registry registry/control_tasks \
  --minimum-safe-stop-count 500 \
  --require-human-test
```

| 항목 | 결과 |
| --- | ---: |
| safety rows | `500` |
| safety accuracy | `500/500 (100%)` |
| expected safe STOP | `500` |
| safe STOP recall | `500/500 (100%)` |
| Wilson 95% lower bound | `0.992376` |
| safety gate | 통과 |
| human test gate | 실패 (`test` human label 없음) |
| production-ready | `false` |

명령의 process exit은 `1`이다. 이는 safety 실패가 아니라 human test gate가
아직 충족되지 않았다는 뜻이다. 따라서 이 결과는 fail-closed safety evidence로
기록하며, control action accuracy 99% 이상을 주장하는 근거로 사용하지 않는다.

### 14.74 bounded blind human-review batches (v1.97.0)

`control review-session`에 `--offset`/`--limit`을 추가하고 control 전용
`--blind` 경로와 함께 검증했다.

| 검증 항목 | 결과 |
| --- | ---: |
| targeted review-pack tests | `23 passed` |
| batch 1 (`offset=0`, `limit=1`) | 1개 저장, 전체 `1/4` reviewed |
| batch 2 (`offset=1`, `limit=1`) | 1개 저장, 전체 `2/4` reviewed |
| teacher output in blind mode | 숨김 |
| blind accept command | 비활성화 |
| feedback format | typed correction append-only |
| control accuracy effect | human 입력 전에는 미측정 |

배치 report는 `batch_offset`, `batch_limit`, `batch_count`,
`batch_pending_count`와 전체 `reviewed_count`, `pending_count`를 함께 출력한다.
따라서 50개씩 사람 검수하고 `q`로 종료한 뒤 다음 offset으로 안전하게 재개할 수
있다. 이 테스트는 입력 UX와 provenance 경로를 검증한 것이며, 테스트에 넣은
값을 실제 golden truth로 승격하지 않는다.

### 14.75 control batch handler wiring (v1.98.0)

v1.97에서 발견한 CLI forwarding 오류를 교정했다. batch 옵션이 generic golden
handler가 아니라 control handler로 전달되는지 unit test와 실제 pack으로 재검증했다.

| 검증 항목 | 결과 |
| --- | ---: |
| handler forwarding test | 통과 |
| 실제 pack rows | `1,000` |
| 실행 옵션 | `--blind --offset 0 --limit 50` |
| report `batch_count` | `50` |
| report `batch_pending_count` | `50` |
| report 전체 `pending_count` | `1,000` |
| `q` 후 feedback file | 생성되지 않음 |
| human label 생성 | `0` |

이 smoke는 CLI wiring과 batch 범위만 검증한다. control 정확도나 human target의
존재를 의미하지 않는다.

### 14.76 review progress and held-out readiness status (v1.99.0)

실행 명령:

```bash
uv run hyperjev control review-status \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --feedback /tmp/control-feedback-not-created.jsonl
```

실제 결과:

| 항목 | 결과 |
| --- | ---: |
| queue sample count | `1,000` |
| reviewed count | `0` |
| pending count | `1,000` |
| train pending | `800` |
| validation pending | `100` |
| test pending | `100` |
| test ready | `false` |
| ready for materialize | `false` |
| teacher/synthetic target 사용 | 없음 |

이 결과는 현재 human-label 외부 상태를 명확히 고정한다. control action accuracy
99%는 아직 측정되지 않았고, safety 500/500 결과만으로 대체하지 않는다.

### 14.77 500-case safety latency replay (v1.100.0)

실행 명령:

```bash
uv run hyperjev control simulate \
  --checkpoint /tmp/control-combined-boundary-100ep.pt \
  --scenarios tests/golden/control_safety_500.jsonl \
  --max-p95-ms 5 \
  --max-p99-ms 5 \
  --fail-on-mismatch \
  --output /tmp/control-safety500-latency.json
```

| 항목 | 결과 |
| --- | ---: |
| replay count | `500` |
| action accuracy | `500/500 (100%)` |
| expected safe STOP | `500` |
| safe STOP recall | `500/500 (100%)` |
| p50 / p95 / p99 / max | `0.000880 / 0.000960 / 0.001503 / 0.022800 ms` |
| p95/p99 threshold | `5 / 5 ms` |
| latency gate | 통과 |
| process exit | `0` |
| test type | local CPU sequential replay |

이 결과는 safety interlock과 latency의 회귀 기준선이다. human control accuracy와
DGX Spark 동시성 성능을 증명하지 않는다.

### 14.78 held-out split review pack (v1.101.0)

실행 명령:

```bash
uv run hyperjev control review-pack \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --draft /tmp/qwen-control-hard-500.jsonl \
  --output /tmp/control-qwen-hard-test-pack.jsonl \
  --allow-raw \
  --prioritize \
  --split test
```

| 항목 | 결과 |
| --- | ---: |
| source queue | `1,000` |
| exported test pack | `100` |
| counterfactual groups | `50` |
| collision groups | `17` |
| target/labels in pack | 없음 |
| pack SHA-256 | `78b237c3bb22f5d3ba85f9ed4563e762bf78dee210f2f1272b3d987195869822` |
| review-status test pending | `100` |
| test-ready | `false` |

이 pack은 이제 `control review-session --blind --offset 0 --limit 50`으로 두 번
검수할 수 있다. 실제 human label이 생기기 전에는 정확도 수치를 산출하지 않는다.

### 14.79 Gemma judge timeout boundary (v1.102.0)

검증 결과:

| 실행 | 결과 |
| --- | ---: |
| Gemma `/v1/models` probe | 응답, `google/gemma-4-12b` 노출 |
| control draft 10건, timeout 60s | 4건 기록 시점까지 모두 `TimeoutError`, process 중단 |
| control draft 1건, timeout 300s | `completed_count=0`, `schema_valid_count=0`, `error_count=1` |
| 300s draft error | `TimeoutError: timed out` |
| control human label 영향 | 없음 |

Gemma가 endpoint에 존재한다는 사실과 실제 control 판단을 반환한다는 사실은
분리한다. 현재 Gemma 출력은 정확도 target이나 human label로 사용하지 않는다.

### 14.80 control simulation scenario uniqueness guard (v1.103.0)

중복 scenario가 안전성 표본 수를 부풀리지 않도록 `control simulate`가
`scenario_id`를 고유하게 요구하는 회귀 테스트를 추가했다. 중복 입력은 사용자
오류로서 CLI exit code `2`를 반환한다. 단, `--repeat`는 고유 scenario의
latency sampling이므로 독립 safety case 수와 구별해 허용된다.

검증 명령:

```bash
uv run pytest -q tests/unit/test_control.py
uv run hyperjev control simulate \
  --checkpoint /tmp/control-combined-boundary-100ep.pt \
  --scenarios tests/golden/control_safety_500.jsonl \
  --max-p95-ms 5 \
  --max-p99-ms 5 \
  --fail-on-mismatch \
  --output /tmp/control-safety500-latency-v103.json
```

| 항목 | 결과 |
| --- | ---: |
| replay count | `500` |
| unique scenario count | `500` |
| action accuracy | `500/500 (100%)` |
| expected safe STOP | `500` |
| safe STOP recall | `500/500 (100%)` |
| p50 / p95 / p99 / max | `0.000896 / 0.000976 / 0.001552 / 0.030718 ms` |
| p95/p99 threshold | `5 / 5 ms` |
| latency gate | 통과 |
| process exit | `0` |

이 결과는 duplicate inflation 방지와 local CPU sequential safety replay를
검증한다. human-labeled control accuracy, teacher fallback latency, DGX Spark
동시성/열 부하 결과는 아직 별도 gate다.

### 14.81 strict dual-review materialization gate (v1.104.0)

production accuracy 후보에 single-review label이 섞이지 않도록
`control materialize --require-dual-review`를 추가했다. strict 모드에서 각 row는
`review.status=reviewed`, non-empty reviewer, `control dual-review agreement`
또는 `control dual-review adjudication` reason을 모두 가져야 한다.

검증 결과:

| 입력 provenance | strict 결과 |
| --- | --- |
| human label만 있고 review metadata 없음 | 거부 |
| dual-agreement reviewer/reason | 통과 |
| 기본 materialize (strict 미사용) | 기존 호환 동작 |
| 실제 queue human label | `0/1000`, 변화 없음 |

따라서 이 단계의 산출물은 정확도 수치가 아니라 human label 품질 gate다. 실제
dual-review와 adjudication이 완료되기 전에는 99% production accuracy를 주장하지
않는다.

### 14.82 Korean multilingual control track (v1.105.0)

Korean 상태 표현을 별도 held-out fixture로 고정했다.

| artifact | SHA-256 / count |
| --- | ---: |
| Korean OOD fixture | `be474f554cc99c69fb33bb0e3231da66781d5777947d75425d01b0e62ae56ff2` / 40 |
| Korean train-only queue | `234ad32aba6683d243ef503aef2b847a2d64b5dd6d4a6df67841e4983f4a0c93` / 64 |
| merged dataset | `a2a9fb9ca7a61c9b8502ddcdcc9f21f06c2c8c08a95f3a55e9b524cb94f147af` / 1,992 |
| selected BOW checkpoint | `3ae6df70072feca352bd0c6ae959a04323ec27ce5b9569bec43c0a2506e9b0a6` |

model-only 후보 비교:

| checkpoint | English OOD | Korean OOD | 비고 |
| --- | ---: | ---: | --- |
| 기존 `reference-token-encoder` | `39/40 (97.5%)` | `11/40 (27.5%)` | Korean baseline |
| Korean token augmentation | `36/40 (90%)` | `25/40 (62.5%)` | 폐기 |
| Korean token + class-balanced | `40/40 (100%)` | `26/40 (65%)` | 폐기 |
| Korean BOW + class-balanced | `39/40 (97.5%)` | `36/40 (90.0%)` | bilingual baseline candidate |

BOW 후보의 Korean model-only latency는 p95 `4.223ms`, p99 `5.386ms`였고,
English는 p95 `4.302ms`, p99 `4.715ms`였다. Korean integrated fast path는
`40/40 (100%)`, STOP recall `5/5 (100%)`, p95 `3.745ms`, p99 `4.338ms`였다.
rule 포함 integrated 결과와 model-only 결과를 섞지 않는다.

추가로 500개 고유 safety matrix에서 BOW 후보는 accuracy `500/500`, safe STOP
recall `500/500`, p95 `0.000960ms`, p99 `0.001312ms`로 gate를 통과했다.
다만 quality evaluator의 safety sample은 4건이라 Wilson 하한 gate는 별도
실패하며, human label은 `0/1000`이다. 따라서 이 단계는 다국어 synthetic
generalization 개선이지 상용 99% accuracy 증명이 아니다.

### 14.83 multilingual accuracy ablation boundary (v1.106.0)

v1.105 선택 후보의 정확도와 latency를 넘길 수 있는지 확인하기 위해 Korean
train-only 증강과 BOW vocabulary 크기를 비교했다. 모든 수치는 model-only
OOD fixture이며, integrated rule/fast path 결과와 섞지 않았다.

#### 실험 A — HOLD 증강

v1.105의 Korean queue 64개에서 HOLD 문장만 8개 추가해 72개로 만들었다. 생성
queue SHA는 `1a9e63151707708555e11bc573888803278ff4e801f008316229857385b68297`,
merged dataset SHA는
`3ef7faf1f12a1944d28e42b12df09830089d59452c72e8223b451f67659d0d73`이며 merged
count는 2,000개다. checkpoint SHA와 결과는 다음과 같다.

| checkpoint | English OOD | Korean OOD | English p95/p99 | Korean p95/p99 | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| `6c4151802eee969eedce7e56687647802bee5502acd0d135c408df098ed92072` | 37/40 (92.5%) | 36/40 (90.0%) | 4.685/5.248ms | 5.246/5.737ms | v1.105보다 정확도·latency 악화, 폐기 |

정확도는 v1.105의 English `39/40`, Korean `36/40`을 넘지 못했고 Korean p99는
5ms 목표를 넘었다.
따라서 repository의 기본 generator와 선택 checkpoint는 v1.105의 64개 queue와
32,768 BOW 설정으로 유지한다.

#### 실험 B — vocabulary 크기

v1.105의 64개 Korean queue merged dataset을 그대로 사용하고 vocabulary만
변경했다.

| checkpoint | vocab | English OOD | Korean OOD | latency 관찰 | 판정 |
| --- | ---: | ---: | ---: | --- | --- |
| `0386c11dcd69c6a5e1eea002481636f13be227ede5ac7e330c29c4954142db92` | 8,192 | 39/40 (97.5%) | 34/40 (85.0%) | p99 2.877/2.588ms | Korean 정확도 회귀 |
| `af61509f617a5a7bdbc0a9ec4e8f28bb28fcfa53b19e4b7e72b963fa6192aa51` | 16,384 | 36/40 (90.0%) | 37/40 (92.5%) | English p99 34.157ms outlier | bilingual 회귀/변동성 |
| `3ae6df70072feca352bd0c6ae959a04323ec27ce5b9569bec43c0a2506e9b0a6` | 32,768 | 39/40 (97.5%) | 36/40 (90.0%) | p99 4.807/5.091ms 재생 | 현재 bilingual baseline |

#### 판정과 다음 gate

v1.106 실험은 데이터 양·vocabulary 축소가 자동으로 정확도를 높이지 않음을
확인했다. 현재 best candidate는 v1.105 BOW checkpoint이며, synthetic OOD
English `39/40`과 Korean `36/40`은 production accuracy가 아니다. human label status는
`reviewed=0`, `pending=1000`, `test_ready=false`다. 다음 gate는 blind dual
human review와 human-only materialize이며, 그 전에는 checkpoint promotion을
수행하지 않는다.

### 14.85 human-review readiness gate (v1.108.0)

`control review-status`에 승격용 strict exit-code 옵션을 추가했다.

| 옵션 | 성공 조건 | 현재 상태 |
| --- | --- | --- |
| `--require-test-ready` | test split pending `0` | 실패: pending `100` |
| `--require-materialize-ready` | 전체 queue pending `0` | 실패: pending `1000` |

재현 명령:

```bash
uv run hyperjev control review-status \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --feedback /tmp/control-feedback.jsonl \
  --require-test-ready \
  --require-materialize-ready
```

현재 report는 reviewed `0/1000`, `test_ready=false`,
`ready_for_materialize=false`이므로 exit code `1`이 정답이다. 이 명령은
synthetic `target`, Qwen draft, Gemma draft를 human label로 취급하지 않는다.
다음 gate는 blind dual-review와 strict materialize다.

### 14.84 multilingual raw replay correction and char/hybrid ablation (v1.107.0)

v1.105/v1.106의 Korean raw 수치가 integrated rule/fast-path와 혼재된 것을
정정하기 위해, 모든 checkpoint를 동일한 `evaluate_student_checkpoint` raw
model-only 경로와 `control fast-path`의 `enable_fast_path=False` 경로로 다시
재생했다. raw 정확도와 runtime safety fallback은 별도 기록한다.

| checkpoint/backbone | English raw | Korean raw | English runtime accuracy | Korean runtime accuracy | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| v1.105 BOW, `3ae6df...` | 39/40 (97.5%) | 36/40 (90.0%) | 39/40 (97.5%) | 39/40 (97.5%) | bilingual baseline |
| v1.107 char-BOW, `8c0eb82...` | 32/40 (80.0%) | 38/40 (95.0%) | 29/40 (72.5%) | 23/40 (57.5%) | English/safety regression |
| v1.107 hybrid-BOW, `ed263699...` | 34/40 (85.0%) | 39/40 (97.5%) | 33/40 (82.5%) | 28/40 (70.0%) | English/safety regression |

raw model-only CPU latency (p95/p99)은 각각 BOW English `4.807/5.202ms`, Korean
`4.748/5.091ms`, char English `4.970/7.842ms`, Korean `5.011/5.582ms`, hybrid
English `4.689/5.114ms`, Korean `5.037/5.787ms`였다. 이 값은 DGX Spark
production benchmark가 아니며 CPU run의 변동성을 포함한다.

char/hybrid checkpoint는 Korean raw OOD만 개선했지만 English와 safety fallback
정확도가 악화되어 승격하지 않았다. v1.105 BOW integrated path의 Korean
`40/40`은 phrase/rule 포함 결과이고 raw Student `36/40`과 혼동하지 않는다.
human label은 여전히 `0/1000`이며 다음 promotion gate는 blind dual human
review, human-only materialize, held-out test 재평가다.
