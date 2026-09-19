# PRD — HyperJev

**문서 상태:** Draft v1.0  
**제품명:** HyperJev  
**부제:** Hyper Memory용 로컬 우선 초고속 확률 판단 모델  
**목표 환경:** NVIDIA DGX Spark 1대, Linux/ARM64, CUDA, Docker  
**기존 모델:** llama.cpp로 실행 중인 Qwen3 8B 계열과 Gemma 4 계열  
**연동 대상:** Hyper Memory, Obsidian, MCP Agent, REST/gRPC Client  
**라이선스 목표:** 코드 Apache-2.0, 모델·데이터는 원본 라이선스에 따라 별도 표기  

hypermemory는 이미 만들어뒀다. https://github.com/dirmich/hypermemory
qwen3.8은 http://localhost:8081/v1에 실행중이고 gemma4는 http://macmini:11434/v1에 실행중이다
---

## 1. 제품 정의

HyperJev는 긴 문장을 생성하는 일반 LLM 대신, 입력 상태와 질문을 받아 **타입이 정해진 판단값과 보정된 확률**을 한 번에 반환하는 로컬 AI 판단 엔진이다.

```text
State + Questions + Output Schema
                ↓
            HyperJev
                ↓
Boolean/NOUL + Choice + Score + Abstain + Confidence
```

HyperJev의 첫 사용처는 Hyper Memory다. 새 대화·문서·코드가 들어올 때 다음 결정을 빠르게 처리한다.

- 기억할 가치가 있는가?
- 사실, 선호, 결정, 목표, 작업 중 어느 유형인가?
- 기존 기억과 중복·갱신·모순 관계인가?
- 중요도, 민감도, 만료 가능성은 어느 정도인가?
- 어떤 프로젝트·사람·주제에 속하는가?
- 작은 모델이 바로 처리해도 되는가, 큰 LLM의 검토가 필요한가?

HyperJev는 챗봇을 대체하지 않는다. Qwen/Gemma 같은 큰 로컬 LLM 앞에 놓이는 **smart decision layer**다.

---

## 2. 문제 정의

Hyper Memory가 모든 입력을 Qwen/Gemma에 보내 JSON을 생성하게 하면 다음 문제가 발생한다.

1. 토큰을 순차 생성하므로 지연시간이 길다.
2. JSON 문법 오류와 필드 누락이 생긴다.
3. 단순한 yes/no 판단에도 큰 모델을 계속 점유한다.
4. 여러 질문이 있어도 같은 원문을 반복 처리한다.
5. 모델이 말하는 0.9 확률이 실제 90% 정확도를 뜻하지 않는다.
6. 대규모 수집 중에는 검색용 LLM과 ingestion용 LLM이 GPU를 서로 방해한다.
7. 분류 프롬프트가 바뀔 때 결과의 일관성과 회귀 테스트가 어렵다.
8. 판단 근거와 모델 버전이 기록되지 않으면 기억의 변경을 감사하기 어렵다.

HyperJev는 문자열 생성 문제를 typed decision 문제로 바꾼다.

```text
기존 방식: 입력 → LLM → JSON 문자열 30~300 tokens
HyperJev:   입력 1회 인코딩 → 여러 decision head 병렬 계산
```

---

## 3. 목표와 비목표

### 3.1 제품 목표

- 완전 로컬 실행 및 학습
- DGX Spark 1대로 데이터 생성, 미세조정, 평가, 추론 가능
- Hyper Memory ingestion의 큰 LLM 호출량을 MVP 기준 70% 이상 감소
- 출력 문법 오류가 구조적으로 발생하지 않는 typed output
- 하나의 state에 최대 128개 질문을 batch로 평가
- Boolean/Choice/Score 확률의 calibration 제공
- confidence가 낮으면 자동으로 Qwen/Gemma에 위임
- 새로운 task를 코드 재배포 없이 registry로 추가
- 모든 판단에 모델·스키마·출처·시간·확률 기록
- 한국어와 영어를 우선 지원

### 3.2 비목표

- Qwen/Gemma를 대체하는 범용 대화 모델
- 초기 버전에서 Jev의 비공개 구조를 그대로 복제
- 자유 형식 장문 생성
- 확률만으로 안전이 중요한 결정을 무조건 자동 실행
- 첫 버전에서 1억 후보를 cross-encoder로 전수 평가
- GGUF 파일 자체를 직접 full fine-tuning

---

## 4. 핵심 원칙

1. **Generate less, decide more.** 생성이 필요한 작업만 큰 LLM에 맡긴다.
2. **State는 한 번, 질문은 병렬로.** 같은 문서를 질문마다 다시 인코딩하지 않는다.
3. **확률은 검증 가능한 값이어야 한다.** accuracy뿐 아니라 ECE/Brier/NLL을 관리한다.
4. **모르면 위임한다.** abstain은 실패가 아니라 핵심 기능이다.
5. **Teacher와 Student를 분리한다.** llama.cpp의 Qwen/Gemma는 데이터 생성·검수·fallback에 사용하고, HyperJev Student는 별도로 학습한다.
6. **모든 결과는 재현 가능해야 한다.** prompt, seed, teacher model, schema version, dataset hash를 기록한다.
7. **Hyper Memory의 원본 출처를 보존한다.** 판단은 source span과 연결한다.
8. **작게 시작해 측정 후 확장한다.** 100M 샘플을 먼저 만들지 않는다.

---

## 5. 사용자와 사용 사례

### 5.1 Hyper Memory 엔진

입력 메시지에서 기억 후보를 거르고, 유형·중요도·민감도·수명을 판단한다.

### 5.2 Agent Router

GitHub issue, 이메일, 문서 요청을 severity, category, worker, approval 필요 여부로 분류한다.

### 5.3 개발자

새로운 decision schema를 등록하고 REST/SDK로 호출한다.

### 5.4 모델 운영자

Teacher 데이터 생성, human review, student 학습, calibration, canary 배포를 수행한다.

---

## 6. 출력 타입

### 6.1 NOUL / Boolean

문장이 참일 확률을 반환한다.

```json
{
  "type": "noul",
  "value": true,
  "probability": 0.973,
  "abstained": false
}
```

내부 명칭은 호환성을 위해 `noul`을 허용하되 공개 API에서는 `boolean`도 alias로 지원한다.

### 6.2 Choice

동적 후보 중 하나 또는 복수 선택을 반환한다.

```json
{
  "type": "choice",
  "selected": "decision",
  "probabilities": {
    "fact": 0.08,
    "preference": 0.03,
    "decision": 0.84,
    "task": 0.05
  }
}
```

### 6.3 Score

0~1의 연속값과 불확실성 구간을 반환한다.

```json
{
  "type": "score",
  "value": 0.81,
  "interval_90": [0.72, 0.88]
}
```

### 6.4 Relation

두 memory 사이의 관계를 선택한다.

```text
DUPLICATE | UPDATE | EXTEND | CONTRADICT | RELATED | NONE
```

### 6.5 Abstain

다음 중 하나이면 자동화하지 않는다.

- 최대 확률이 task threshold 미만
- 상위 두 choice의 margin이 너무 작음
- 입력이 학습 분포 밖으로 감지됨
- schema나 language가 지원 범위를 벗어남
- ensemble disagreement가 큼

---

## 7. Hyper Memory 통합 범위

### 7.1 Ingestion Gate

```text
Raw input
  ↓ deterministic filter
  ↓ HyperJev remember_worthy
  ├─ 낮음 → 원문만 보관/종료
  ├─ 중간 → teacher 검토
  └─ 높음 → memory extraction 단계
```

### 7.2 Memory Metadata

한 번의 요청에서 병렬로 판단한다.

- `remember_worthy`: boolean
- `memory_type`: choice
- `importance`: score
- `sensitivity`: choice
- `temporary`: boolean
- `expiration_days`: score를 정책으로 변환
- `requires_source_quote`: boolean
- `project_scope`: choice

### 7.3 Temporal Relation

Hyper Memory가 ANN/FTS로 top-K 기존 기억을 찾은 후 HyperJev가 pair를 평가한다.

```text
새 memory → ANN/FTS top 20 → HyperJev relation scorer → top 3
                                               ↓
                                    확신 낮으면 teacher
```

### 7.4 Query Router

질의를 `PROFILE`, `EXACT`, `LEXICAL`, `SEMANTIC`, `TEMPORAL`, `GRAPH`, `HYBRID`로 분류하고 검색 경로와 token budget을 결정한다.

### 7.5 Retrieval Reranker

검색 후보 top 50만 HyperJev로 재정렬한다. 전체 DB를 직접 점수화하지 않는다.

### 7.6 Wiki Change Detector

Obsidian 수정이 스타일 변경인지 실제 사실 변경인지 판단하고, 사실 변경이면 자동 반영·검토·거절로 routing한다.

---

## 8. 전체 시스템 아키텍처

```text
┌──────────────────────────────────────────────────────────┐
│ DGX Spark                                                │
│                                                          │
│  llama.cpp Server A — Qwen3 8B                           │
│  llama.cpp Server B — Gemma 4                            │
│        │ teacher / judge / fallback                      │
│        ▼                                                 │
│  Dataset Factory → Review Store → Training Pipeline      │
│                                      │                   │
│                                      ▼                   │
│                              HyperJev Student            │
│                         encoder + decision heads          │
│                                      │                   │
│                         Inference / Calibration API       │
└──────────────────────────────────────┼───────────────────┘
                                       │
                               Hyper Memory Core
                   FTS + ANN + Temporal Graph + Obsidian
```

서비스는 다음 컨테이너로 분리한다.

| 서비스 | 책임 | 기본 자원 정책 |
|---|---|---|
| `hyperjev-api` | typed decision API, batching | 항상 실행 |
| `hyperjev-trainer` | student 학습 | 예약 실행 |
| `teacher-qwen` | 라벨 생성·fallback | 필요 시 실행 |
| `teacher-gemma` | 교차 검수 | 필요 시 실행 |
| `dataset-worker` | 샘플 생성·검증 | background |
| `calibration-worker` | threshold/temperature 산출 | 배포 전 |
| `registry` | task/model/schema version | 항상 실행 |
| `hypermemory-core` | 저장·검색·graph/wiki | 항상 실행 |

GPU 메모리 경합을 막기 위해 기본 운영 모드는 다음과 같다.

- Online: HyperJev + Hyper Memory 우선, teacher는 1개만 상주
- Training: ingestion 속도를 제한하고 teacher를 내린 뒤 trainer에 자원 배정
- Dataset: teacher 1개씩 순차 실행, 결과 캐시
- Benchmark: 다른 GPU 작업 중지 후 고정 조건에서 측정

---

## 9. 기존 llama.cpp 모델 활용

사용자가 가진 Qwen3 8B와 Gemma 4의 실제 model alias와 GGUF 경로는 환경설정으로 주입한다. 제품 코드는 특정 파일명에 의존하지 않는다.

```toml
[teachers.qwen]
base_url = "http://127.0.0.1:8081/v1"
model = "qwen3-8b-local"
role = ["labeler", "fallback"]

[teachers.gemma]
base_url = "http://127.0.0.1:8082/v1"
model = "gemma4-local"
role = ["judge", "fallback"]
```

### 9.1 역할 분담

- Qwen: 한국어 중심 1차 라벨, 질문 변형, 어려운 샘플 추론
- Gemma: 독립 2차 판정, disagreement 탐지, 영어 샘플 보강
- 두 모델 합의: 높은 신뢰의 silver label
- 두 모델 불일치: human review 또는 보류

### 9.2 주의사항

GGUF는 빠른 추론용이다. HyperJev Student의 본격 학습은 대응되는 원본 학습 체크포인트 또는 별도 encoder 체크포인트를 사용한다. llama.cpp teacher는 OpenAI-compatible endpoint로만 접근하여 교체 가능하게 한다.

---

## 10. 모델 아키텍처

### 10.0 Diffusion 채택 결정

HyperJev v1의 주력 판단 모델에는 diffusion을 사용하지 않는다. Boolean, Choice, Score처럼 출력 크기가 작은 판단은 encoder가 한 번의 forward pass로 모두 계산할 수 있는 반면, diffusion language model은 출력 token을 병렬로 복원하더라도 일반적으로 여러 denoising step이 필요하기 때문이다. 따라서 짧은 typed output에서는 diffusion의 병렬 생성 이점보다 반복 계산 비용이 더 클 가능성이 높다.

```text
Encoder HyperJev
state encode 1회 → heads 병렬 계산 → 완료

Diffusion HyperJev 후보
masked/noisy output → denoise 1 → denoise 2 → ... → 완료
```

Diffusion은 다음 조건에서만 별도 `HyperJev-D` 연구 트랙으로 검토한다.

- 수십~수백 개의 상호 의존적인 label을 동시에 생성해야 함
- 고정 head가 아니라 가변 길이의 구조화된 결과가 필요함
- 첫 판단을 반복적으로 수정하는 self-correction이 accuracy를 유의미하게 높임
- 여러 단계 denoising을 포함해도 encoder baseline보다 end-to-end latency 또는 품질이 우수함

검토하지 않을 용도:

- 단일 yes/no
- 10개 이내의 독립 score
- query routing
- 단순 memory type 분류
- top-K pair relation scoring

`HyperJev-D` 실험은 동일 파라미터 수, 동일 입력 길이, 동일 하드웨어에서 다음과 비교한다.

| 후보 | 계산 방식 | 예상 강점 | 예상 약점 |
|---|---|---|---|
| Encoder + heads | 1 forward | 최소 latency, 간단한 calibration | 가변 출력에 제한 |
| Masked diffusion | N회 병렬 denoise | 구조 전체의 동시 수정 | 반복 forward 비용 |
| Autoregressive student | token 순차 생성 | 구현·생태계 용이 | 짧은 JSON도 decoding 필요 |
| Hybrid | encoder 판단 + diffusion 구조 보완 | 쉬운 판단과 복잡한 생성을 분리 | 운영 복잡도 증가 |

채택 기준은 "autoregressive보다 빠른가"가 아니라 **encoder + typed heads보다 정확도-지연시간 곡선이 우수한가**로 한다. MVP 이후 4-step, 8-step, 16-step masked diffusion prototype을 만들 수 있으나, encoder baseline을 이기지 못하면 제품 경로에서 제외한다.

### 10.1 MVP: Cross-Encoder Decision Model

가장 먼저 정확도를 확보하는 구조다.

```text
[STATE] input text
[QUESTION] question
[CANDIDATES] optional candidates
             ↓
      Small Transformer
             ↓ pooled representation
   ┌─────────┼──────────┬──────────┐
 boolean   choice      score     abstain
 sigmoid   softmax     sigmoid    energy
```

권장 시작 크기는 300M~600M다. 한국어 성능을 반드시 사전 평가한 encoder 또는 작은 multilingual backbone을 고른다. 특정 backbone은 벤치마크 후 ADR로 확정한다.

### 10.2 v1: Shared-State Multi-Question Model

state를 한 번 인코딩하고 질문 1~128개를 batch 처리한다.

```text
State Encoder → state tokens/cache
                    │
Question Encoder → Q1..Qn
                    │
           lightweight cross-attention
                    │
          typed task-specific adapters
```

### 10.3 v2: Bi-Encoder + Cross-Encoder Cascade

후보가 많은 entity/duplicate 검색에 사용한다.

```text
bi-encoder ANN top 100
        ↓
cross-encoder top 20
        ↓
teacher only if uncertain
```

### 10.4 Head 설계

- Binary head: 2-logit softmax 또는 single sigmoid
- Choice head: candidate representation과 state-question vector의 dot product
- Score head: Beta distribution의 alpha/beta 예측 또는 scalar regression
- Relation head: 6-way classifier
- OOD head: energy score + embedding distance
- Evidence head: 입력 token별 relevance score; MVP 후 적용

### 10.5 Adapter 구조

backbone은 공유하고 task family별 작은 adapter/head만 둔다.

```text
memory_gate
memory_metadata
temporal_relation
query_router
retrieval_rerank
wiki_change
```

새 task는 가능한 한 backbone 재학습 없이 adapter/head 추가로 지원한다.

---

## 11. Task Registry

모든 판단은 코드에 하드코딩하지 않고 versioned YAML로 등록한다.

```yaml
id: memory.remember_worthy
version: 1
input:
  max_chars: 12000
output:
  type: boolean
thresholds:
  auto_true: 0.92
  auto_false: 0.08
  otherwise: fallback
cost:
  false_positive: 2
  false_negative: 5
languages: [ko, en]
teacher_prompt_version: 3
```

Choice 예:

```yaml
id: memory.type
version: 1
output:
  type: choice
  candidates:
    - fact
    - preference
    - episode
    - decision
    - goal
    - task
    - relationship
    - temporary
    - none
```

Registry 변경은 dataset과 model compatibility 검사를 통과해야 한다.

---

## 12. 데이터 모델

학습 샘플의 canonical 형식:

```json
{
  "sample_id": "uuid",
  "task_id": "memory.remember_worthy",
  "task_version": 1,
  "state": "프로젝트 DB를 PostgreSQL로 바꾸기로 했다.",
  "question": "장기 기억으로 저장할 가치가 있는가?",
  "candidates": null,
  "target": true,
  "soft_target": 0.97,
  "language": "ko",
  "domain": "conversation",
  "source": {
    "kind": "synthetic",
    "document_id": null,
    "span": null
  },
  "labels": {
    "qwen": {"value": true, "confidence": 0.96},
    "gemma": {"value": true, "confidence": 0.93},
    "human": null
  },
  "provenance": {
    "prompt_version": 3,
    "teacher_model": "...",
    "created_at": "..."
  }
}
```

원문에 개인정보가 있으면 dataset export 전에 redaction policy를 적용한다. secret 등급 데이터는 합성 데이터 생성과 학습에서 기본 제외한다.

---

## 13. Dataset Factory

### 13.1 데이터 원천

1. Hyper Memory 실제 입력 중 사용자가 학습 허용한 데이터
2. 기존 memory와 관계 데이터
3. 공개 라이선스 문서·대화·코드
4. Qwen/Gemma가 생성한 합성 샘플
5. 규칙 기반 hard negative
6. 운영 중 fallback/human correction 로그

### 13.2 초기 데이터 규모

| 단계 | 샘플 수 | 목적 |
|---|---:|---|
| POC | 20k~50k | 구조 검증 |
| Alpha | 200k~500k | 주요 6 task 학습 |
| Beta | 1M~5M | domain/language 확장 |
| Scale | 측정 후 결정 | 실제 병목에만 추가 |

처음부터 100M 샘플 생성은 금지한다. 라벨 품질, 중복률, coverage를 먼저 측정한다.

### 13.3 생성 절차

```text
Seed documents
  ↓ scenario generator
  ↓ Qwen labels + rationales
  ↓ Gemma independent labels
  ↓ schema validator
  ↓ agreement filter
  ↓ dedup / contamination check
  ↓ human review sample
  ↓ train/dev/test split
```

### 13.4 라벨 합의 정책

- 정답과 확률 방향이 일치: silver 데이터 채택
- choice 상위 1위가 불일치: review queue
- confidence 차이가 0.35 이상: review queue
- 두 teacher 모두 낮은 confidence: 학습 제외 또는 abstain positive로 사용
- human correction이 있으면 teacher보다 우선

Teacher가 생성한 장문 reasoning은 Student 입력에 넣지 않는다. 필요하면 감사용으로 별도 보관한다.

### 13.5 Split 정책

random row split을 금지하고 source/time/entity 단위로 분리한다.

- train 80%
- validation 10%
- test 10%
- 별도 temporal holdout
- 별도 Korean colloquial set
- 별도 adversarial/OOD set

---

## 14. 학습 방법

### 14.1 Stage A — Supervised Multi-task Distillation

Loss:

```text
L = w_bin * BCE
  + w_choice * CE/KL
  + w_score * BetaNLL or MSE
  + w_relation * CE
  + w_ood * EnergyMargin
  + w_cal * Brier
```

hard label과 teacher soft probability를 함께 사용한다. teacher의 언어적 자신감 표현을 확률 정답으로 그대로 믿지 않고, held-out 정답에 기반해 teacher 자체도 보정한다.

### 14.2 Stage B — Hard-Negative Mining

Student가 자신 있게 틀린 샘플, temporal relation 혼동, entity alias 충돌, 짧은 한국어 구어체를 수집해 재학습한다.

### 14.3 Stage C — Calibration

학습 완료 checkpoint를 고정한 뒤 별도 calibration set으로 수행한다.

- temperature scaling
- vector scaling 또는 isotonic regression 검토
- task별 threshold
- class별 threshold
- conformal prediction을 이용한 prediction set 검토

### 14.4 Stage D — 선택적 Preference/RL

MVP에는 RL을 넣지 않는다. supervised + calibration으로 기준을 만든 후 다음 조건에서만 도입한다.

- 자동 위임 비용을 직접 최적화할 필요가 있음
- human preference 데이터가 충분함
- reward hacking을 감시할 평가셋이 있음

가능한 reward:

```text
correct decision reward
- false automation cost
- unnecessary fallback cost
- calibration error penalty
```

---

## 15. DGX Spark 학습 운영

### 15.1 저장 구조

```text
/data/hyperjev/
├── datasets/raw/
├── datasets/normalized/
├── datasets/splits/
├── models/base/
├── models/checkpoints/
├── models/calibrated/
├── exports/onnx/
├── exports/tensorrt/
├── registry/
├── runs/
└── cache/
```

### 15.2 학습 기본값

초기값이며 OOM/throughput 측정 후 자동 조정한다.

```yaml
precision: bf16
gradient_checkpointing: true
optimizer: adamw_8bit
sequence_length: 1024
micro_batch_size: 8
gradient_accumulation: 8
eval_every_steps: 500
save_every_steps: 1000
early_stopping_metric: validation_composite
```

300M~600M 모델은 full fine-tuning을 우선 시험하고, 더 큰 backbone은 LoRA/QLoRA를 먼저 사용한다. unified memory가 넉넉해도 CPU-GPU page migration이 잦으면 속도가 크게 떨어질 수 있으므로 메모리 점유율만이 아니라 throughput을 기준으로 설정한다.

### 15.3 Job Scheduler

- 검색 요청 P0
- HyperJev inference P0
- memory write P1
- teacher fallback P1
- dataset generation P3
- training P4
- nightly benchmark P4

training은 예약 시간 또는 수동 명령으로 시작한다. online p95가 한도를 넘으면 training을 pause한다.

---

## 16. 추론 최적화

### 16.1 단계

1. PyTorch BF16 baseline
2. `torch.compile` 가능성 검증
3. ONNX Runtime CUDA export
4. TensorRT engine 생성
5. FP16/INT8 비교
6. 정확도 손실이 허용될 때만 FP8/더 낮은 정밀도 검토

### 16.2 Dynamic Batching

- 대기창 2~8ms
- 동일 task/비슷한 길이끼리 bucket
- interactive request는 batch 상한 전에 즉시 처리
- ingestion은 큰 batch 허용

### 16.3 Cache

```text
state_cache_key = model_version + normalized_state_hash
question_cache_key = task_version + question_hash + candidate_hash
result_cache_key = all of above + calibration_version
```

민감 데이터는 cache encryption 또는 cache disable 정책을 따른다.

### 16.4 Early Exit

얕은 layer에서 확신이 매우 높은 쉬운 task를 종료하는 구조는 v2 실험으로 제한한다. calibration이 악화되면 사용하지 않는다.

---

## 17. API

### 17.1 단일/다중 판단

```http
POST /v1/decide
```

```json
{
  "state": "PostgreSQL 대신 ClickHouse로 변경하기로 했다.",
  "context": {"workspace_id": "hyper-memory"},
  "questions": [
    {"id": "remember", "task": "memory.remember_worthy@1"},
    {"id": "type", "task": "memory.type@1"},
    {"id": "importance", "task": "memory.importance@1"}
  ],
  "options": {
    "allow_fallback": true,
    "return_evidence": true,
    "deadline_ms": 500
  }
}
```

응답:

```json
{
  "request_id": "...",
  "model": "hyperjev-0.5b-v1",
  "calibration": "cal-2026-09-19",
  "results": {
    "remember": {"value": true, "probability": 0.991},
    "type": {"value": "decision", "probability": 0.934},
    "importance": {"value": 0.86, "interval_90": [0.77, 0.91]}
  },
  "route": "hyperjev",
  "latency_ms": 18
}
```

### 17.2 Pair Relation

```http
POST /v1/relations/score
```

### 17.3 Batch

```http
POST /v1/batch/decide
```

### 17.4 Registry

```http
GET  /v1/tasks
POST /v1/tasks/validate
GET  /v1/models
POST /v1/models/{version}/activate
```

### 17.5 Health/Metrics

```http
GET /health/live
GET /health/ready
GET /metrics
```

Python, TypeScript, Rust SDK를 순서대로 제공한다.

---

## 18. Fallback 정책

```text
HyperJev result
  ├─ probability/coverage 충족 → 바로 사용
  ├─ 애매함 → Qwen teacher
  ├─ Qwen도 애매함 → Gemma judge
  └─ 고위험 또는 불일치 → human review
```

fallback 결과는 온라인 응답에 사용하면서 동시에 future training candidate로 저장한다. 단, 사용자 데이터의 학습 허용 정책을 반드시 확인한다.

Task별 비용 행렬을 둔다. 예를 들어 기억 누락 비용이 과잉 저장보다 크면 `remember_worthy`의 false-negative threshold를 더 보수적으로 설정한다.

---

## 19. 평가

### 19.1 품질 지표

- Boolean: precision, recall, F1, AUROC, AUPRC
- Choice: accuracy, macro-F1, top-2 accuracy
- Score: MAE, RMSE, Spearman correlation
- Relation: macro-F1, UPDATE/CONTRADICT recall
- Rerank: MRR, NDCG@10, Recall@10
- Calibration: Brier, NLL, ECE, adaptive ECE
- Selective prediction: risk-coverage curve

### 19.2 시스템 지표

- p50/p95/p99 latency
- states/sec, questions/sec
- peak unified memory
- CPU/GPU utilization
- cache hit rate
- fallback rate
- energy per 1k decisions 가능 시 측정

### 19.3 비교 기준

1. Qwen이 JSON 생성
2. Gemma가 JSON 생성
3. 규칙 기반 classifier
4. HyperJev PyTorch
5. HyperJev optimized runtime

같은 데이터와 같은 output schema로 비교한다. 빠르다는 주장보다 **동일 품질에서의 latency/cost**, 또는 동일 latency에서의 품질을 보고한다.

### 19.4 MVP 목표

| 항목 | 목표 |
|---|---:|
| 핵심 task macro-F1 | ≥ 0.90 |
| remember_worthy recall | ≥ 0.95 |
| relation macro-F1 | ≥ 0.85 |
| ECE | ≤ 0.05 |
| 쉬운 task fallback 비율 | ≤ 20% |
| typed output schema error | 0% |
| 1 state + 8 questions p95 | ≤ 100ms |
| 큰 LLM 호출 감소 | ≥ 70% |

Latency는 실제 DGX Spark에서 입력 512/1024 tokens, 동시성 1/8/32 조건을 각각 공개한다.

---

## 20. Human Review

Review UI는 다음을 한 화면에 보여준다.

- 원문과 source
- 질문과 후보
- HyperJev 확률
- Qwen/Gemma 결과
- 기존 관련 memory
- accept/correct/abstain/ignore

review sampling:

- 모든 disagreement
- 자동 처리 결과의 무작위 1~5%
- confidence가 threshold 근처인 샘플
- 새 domain/language
- drift가 감지된 task

수정된 라벨은 append-only로 저장하며 누가 언제 왜 수정했는지 기록한다.

---

## 21. 모델 Registry와 배포

각 모델 manifest:

```yaml
model_id: hyperjev-0.5b-v1
base_model: ...
dataset_hash: ...
git_commit: ...
tasks:
  memory.remember_worthy: [1]
  memory.type: [1]
calibration_version: cal-2026-09-19
runtime: tensorrt
precision: fp16
status: candidate
```

배포 상태:

```text
trained → evaluated → calibrated → candidate → canary → active → retired
```

Canary는 5% shadow traffic으로 시작한다. 기존 active 모델 결과와 비교하되 실제 동작에는 반영하지 않는다. 기준을 통과하면 10% → 50% → 100%로 승격한다. 즉시 rollback 가능해야 한다.

---

## 22. Drift와 지속 학습

다음을 감지한다.

- 입력 embedding 분포 변화
- task별 fallback 증가
- calibration 악화
- human correction 증가
- 언어/도메인 비율 변화
- 특정 choice 비율 급변

자동 재학습은 하지 않는다. dataset snapshot을 만들고 평가 보고서를 생성한 뒤 사용자 승인으로 새 candidate를 학습한다.

---

## 23. 보안과 개인정보

- 기본 bind address는 localhost
- 외부 접속은 API key + TLS
- workspace별 scoped token
- prompt/data/model artifact의 접근권한 분리
- secret 데이터는 teacher prompt와 training dataset에서 기본 제외
- 원문 로그는 opt-in
- inference log는 text 대신 hash와 최소 metadata 저장 가능
- model inversion 및 memorization 검사용 canary string test
- 데이터 삭제 요청 시 raw, normalized, cache, review, dataset lineage를 추적
- 외부 telemetry 기본 off

HyperJev가 반환한 확률을 의료·법률·금융 등 고위험 자동결정의 근거로 단독 사용하지 않는다.

---

## 24. Observability

필수 metrics:

```text
hyperjev_request_total
hyperjev_latency_ms
hyperjev_batch_size
hyperjev_questions_per_state
hyperjev_fallback_total
hyperjev_abstain_total
hyperjev_task_probability_histogram
hyperjev_cache_hit_total
hyperjev_model_memory_bytes
hyperjev_teacher_queue_depth
hyperjev_review_queue_depth
hyperjev_drift_score
```

Trace에는 request_id, task_version, model_version, calibration_version, fallback chain을 남긴다. 원문은 기본 trace에 남기지 않는다.

---

## 25. Configuration 예시

```toml
[server]
host = "127.0.0.1"
port = 6777
max_batch_size = 64
batch_wait_ms = 4

[model]
active = "hyperjev-0.5b-v1"
runtime = "tensorrt"
precision = "fp16"
max_sequence_length = 1024

[fallback]
enabled = true
primary = "qwen"
judge = "gemma"
deadline_ms = 5000

[teachers.qwen]
base_url = "http://127.0.0.1:8081/v1"
model = "qwen3-8b-local"

[teachers.gemma]
base_url = "http://127.0.0.1:8082/v1"
model = "gemma4-local"

[privacy]
store_raw_inputs = false
allow_training_from_user_data = false

[hypermemory]
base_url = "http://127.0.0.1:6767"
feedback_enabled = true
```

---

## 26. 저장소 구조

```text
hyperjev/
├── README.md
├── Cargo.toml
├── pyproject.toml
├── docker-compose.yml
├── configs/
├── registry/tasks/
├── crates/
│   ├── hyperjev-api/
│   ├── hyperjev-core/
│   ├── hyperjev-client/
│   └── hyperjev-registry/
├── python/
│   ├── dataset_factory/
│   ├── training/
│   ├── calibration/
│   ├── evaluation/
│   └── export/
├── web/review-ui/
├── integrations/hyper-memory/
├── tests/
│   ├── contract/
│   ├── golden/
│   ├── calibration/
│   └── performance/
└── docs/
    ├── architecture/
    ├── adr/
    └── runbooks/
```

Rust는 API, registry, batching, Hyper Memory integration을 담당한다. Python은 학습·평가·export를 담당한다. 초기 POC는 Python inference로 시작할 수 있지만 API contract는 처음부터 고정한다.

---

## 27. CLI

```bash
hyperjev doctor
hyperjev teacher check
hyperjev dataset build --task memory.remember_worthy
hyperjev dataset validate
hyperjev train --config configs/train-0.5b.yaml
hyperjev calibrate --checkpoint ...
hyperjev evaluate --suite all
hyperjev export --runtime onnx
hyperjev serve
hyperjev benchmark --profile dgx-spark
hyperjev model promote hyperjev-0.5b-v1 --to canary
hyperjev model rollback
```

`doctor`는 CUDA, ARM64 wheel, llama.cpp endpoint, disk, model files, ports를 점검한다.

---

## 28. 테스트 전략

### 28.1 Unit

- schema validation
- task registry compatibility
- cache key correctness
- threshold routing
- redaction

### 28.2 Contract

- llama.cpp OpenAI-compatible response adapter
- Hyper Memory API
- SDK 간 동일 결과

### 28.3 Golden Set

최소 1,000개의 사람이 검수한 한국어 중심 샘플을 변경 금지 golden set으로 유지한다.

### 28.4 Calibration Regression

accuracy가 올라도 ECE가 나빠지면 자동 승격하지 않는다.

### 28.5 Failure Injection

- teacher timeout
- CUDA OOM
- model load failure
- registry mismatch
- corrupted checkpoint
- Hyper Memory unavailable
- request cancellation

---

## 29. 단계별 개발 계획

### Phase 0 — 환경 확인 및 Baseline (1주)

- Qwen/Gemma llama.cpp endpoints 확인
- 6개 핵심 task schema 정의
- 1,000개 golden set 작성
- teacher JSON baseline latency/품질 측정
- DGX Spark benchmark script 작성

**완료 기준:** 동일 입력으로 Qwen/Gemma 결과와 비용·속도를 재현 가능.

### Phase 1 — Rule + Teacher Router (1~2주)

- HyperJev API contract 구현
- 아직 Student 없이 rule → Qwen → Gemma chain 구현
- provenance, review queue, feedback 저장
- Hyper Memory ingestion gate 연결

**완료 기준:** Hyper Memory를 깨지 않고 전체 데이터 흐름이 작동.

### Phase 2 — Dataset Factory (2주)

- 20k~50k POC dataset
- teacher agreement, dedup, split, redaction
- review UI 최소 버전
- 데이터 품질 보고서

**완료 기준:** label distribution과 오류 유형을 설명할 수 있음.

### Phase 3 — HyperJev 0.3B~0.6B POC (2~3주)

- cross-encoder multi-task 학습
- boolean/choice/score/relation heads
- calibration
- PyTorch API 연결

**Gate:** macro-F1 0.85 미만이면 최적화 전에 데이터/태스크를 수정.

### Phase 4 — Hyper Memory Alpha (2주)

- remember/type/importance/query-router 실사용
- confidence 기반 fallback
- shadow logging
- error analysis dashboard

**Gate:** 큰 LLM 호출 50% 감소, 치명적 memory update 오류 없음.

### Phase 5 — Shared-State와 최적화 (2~4주)

- multi-question shared state
- ONNX/TensorRT
- dynamic batching/cache
- p95와 throughput 최적화

**Gate:** 8-question p95 100ms 이하 목표 및 baseline 대비 명확한 이점.

### Phase 6 — Beta (4주)

- 200k~500k dataset
- relation/rerank/wiki-change 강화
- canary/rollback/drift
- MCP/SDK 문서

### Phase 7 — v1

- 1M+ 고품질 샘플 여부를 성능 근거로 결정
- adapter task extension
- 선택적 evidence span
- 외부 프로젝트가 사용할 수 있는 안정 API

### Research Track — HyperJev-D

- 작은 masked diffusion backbone 또는 기존 공개 diffusion LM adapter 조사
- 4/8/16 denoising step 비교
- 상호 의존적 multi-label 및 가변 schema에 한정해 평가
- encoder + heads와 quality/latency/calibration/메모리 비교
- encoder baseline을 이긴 task에만 선택적 적용

---

## 30. 우선 구현할 6개 Task

| 우선순위 | Task | 출력 | 자동화 위험 |
|---:|---|---|---|
| 1 | `memory.remember_worthy` | boolean | 낮음 |
| 2 | `memory.type` | choice | 낮음 |
| 3 | `memory.importance` | score | 낮음 |
| 4 | `query.route` | choice/multi-choice | 낮음 |
| 5 | `memory.relation` | choice | 높음 |
| 6 | `wiki.semantic_change` | boolean/choice | 중간 |

relation은 높은 정확도가 확보되기 전까지 자동 UPDATE/CONTRADICT를 실행하지 않고 candidate만 만든다.

---

## 31. 주요 리스크와 대응

### Teacher 오류 증류

대응: 두 teacher 독립 판정, human golden set, disagreement mining, 출처 기반 검증.

### 한국어 성능 부족

대응: backbone 사전 벤치마크, 한국어 구어체/오타/존댓말 별도 set, data balancing.

### 확률 과신

대응: post-hoc calibration, risk-coverage, threshold별 자동화, drift 재보정.

### DGX Spark 소프트웨어 호환성

대응: ARM64/CUDA용 lockfile과 container image, `doctor`, PyTorch baseline을 항상 유지.

### teacher와 training 자원 경합

대응: 운영 모드 분리, 예약 학습, latency-triggered pause, model unload 정책.

### Dynamic candidate 확장성

대응: ANN retrieval 후 top-K만 cross-encode; 후보 전수 평가는 금지.

### Hyper Memory 오염

대응: source-bound decision, high-risk task review, append-only revision, rollback.

### 모델 라이선스

대응: base/teacher/dataset별 manifest에 라이선스 기록, 재배포 가능 범위 분리.

---

## 32. 의사결정 기록(ADR) 필요 항목

1. Student backbone: encoder-only vs 작은 decoder backbone
2. 300M/600M/1B 중 첫 제품 크기
3. PyTorch/ONNX/TensorRT 최종 serving runtime
4. Score head: scalar vs Beta distribution
5. Rust-Python 경계
6. dataset 저장 포맷: Parquet + manifest 권장
7. review DB와 Hyper Memory DB 분리 여부
8. task adapter 전략
9. HyperJev-D의 채택 또는 연구 종료 결정

각 ADR은 benchmark, 선택지, 결정, 되돌림 조건을 포함한다.

---

## 33. 최종 승인 기준

HyperJev v1은 다음을 모두 만족해야 한다.

1. DGX Spark 한 대에서 offline end-to-end 구축 가능.
2. 기존 llama.cpp Qwen/Gemma endpoint를 변경 없이 teacher/fallback으로 사용.
3. Hyper Memory의 6개 핵심 task를 typed API로 제공.
4. golden test에서 사전 정의 품질과 calibration 기준 통과.
5. Qwen/Gemma 단독 JSON 방식보다 동일 품질 구간에서 명확히 빠름.
6. 운영 중 큰 LLM 호출을 70% 이상 줄임.
7. confidence가 낮거나 OOD인 입력을 자동 위임.
8. source, model, schema, probability, fallback 이력을 추적 가능.
9. 사용자 원문을 학습에 쓰지 않는 기본 설정 제공.
10. canary, rollback, drift monitoring과 재현 가능한 benchmark 제공.

---

## 34. 최종 제품 흐름

```text
대화·문서·코드
      ↓
Hyper Memory normalize/dedup
      ↓
HyperJev — 빠른 병렬 판단
      ├─ 기억 여부
      ├─ 유형·중요도·민감도
      ├─ 검색 경로
      ├─ 기존 기억과 관계
      └─ confidence/abstain
              ↓
       확신이 충분함?
       ├─ Yes → Memory Graph/Index/Wiki
       └─ No  → Qwen → Gemma → Human
                         ↓
                 correction/feedback
                         ↓
                  다음 dataset 버전
```

HyperJev의 핵심 가치는 단순히 작은 모델을 만드는 데 있지 않다. **Hyper Memory가 반복 수행하는 수많은 판단을 빠르고, 타입 안전하고, 확률적으로 검증 가능하게 만들며, 어려운 문제에만 기존 Qwen/Gemma를 집중시키는 것**이 제품의 본질이다.

---

## 35. 구현 시작 시 첫 작업 목록

- [ ] 실제 Qwen/Gemma llama.cpp alias, port, context 설정 기록
- [ ] `memory.remember_worthy@1` schema 확정
- [ ] 한국어 500개 + 영어 200개 수작업 golden set 작성
- [ ] Qwen/Gemma deterministic label prompt 작성
- [ ] teacher 결과 schema validator 구현
- [ ] HyperJev `/v1/decide` mock server 구현
- [ ] Hyper Memory ingestion의 shadow call 연결
- [ ] Parquet dataset manifest 정의
- [ ] 3개 backbone을 5k 샘플로 비교
- [ ] 첫 20k dataset 생성
- [ ] BF16 baseline 학습
- [ ] calibration 및 risk-coverage 보고서 생성
- [ ] Qwen/Gemma JSON baseline과 latency/quality 비교
- [ ] 결과에 따라 0.3B~0.6B POC backbone 확정
