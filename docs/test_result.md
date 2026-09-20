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
