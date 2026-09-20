# HyperJev 정확도 향상 계획

작성일: 2026-09-20  
현재 버전: 1.50.0
대상: `control.skill@1` 및 이후 memory/query typed heads

## 1. 목표와 원칙

목표는 train set을 외우는 100%가 아니라, 새로운 game/robot state에서도 높은
정확도와 안전한 abstention을 동시에 달성하는 것이다.

최종 목표는 다음 네 조건을 모두 만족하는 control model이다.

| 항목 | 최종 기준 |
| --- | --- |
| 독립 human-labeled test accuracy | 99% 이상에 최대한 근접, task별 98% 미만 금지 |
| confidence gate accepted accuracy | 99.5% 이상 |
| safety STOP recall | 100% 목표, 절대 99.9% 미만 금지 |
| control latency | model-only p95 5ms 이내, end-to-end는 별도 측정 |

정확도 숫자에는 반드시 `unique semantic group`, split, human label 여부,
coverage, fallback 수를 함께 기록한다. synthetic target, exact duplicate,
train accuracy만으로 production 승격하지 않는다.

## 2. 현재 baseline

재현 명령:

```bash
uv run python scripts/evaluate_control_quality.py \
  --checkpoint runs/control/control-student-48-300.pt
```

현재 reference `byte/ngram encoder` 결과:

| split | 정확도 | 표본 |
| --- | ---: | ---: |
| train | 100.00% | 32 |
| validation | 37.50% | 8 |
| test | 12.50% | 8 |

Phase A의 독립 evaluator를 추가한 뒤 `reference-token-encoder`를 시험했다.
토큰화와 n-gram feature path를 함께 사용한 결과는 validation `4/8 (50.00%)`,
test `5/8 (62.50%)`였고, baseline보다 각각 12.50%p와 50.00%p 개선됐다.
하지만 synthetic sample 48개, human label 0개이므로 이 숫자는 모델 개선
신호일 뿐 production 정확도 증거가 아니다. evaluator는 아래처럼 계속 99%
미만이면 실패해야 한다.

```bash
uv run python scripts/evaluate_control_quality.py \
  --checkpoint runs/control/control-token-300.pt
```

v1.44.0에는 다음 provenance gate가 추가됐다.

```bash
uv run python scripts/validate_control_dataset.py \
  tests/golden/control_smoke.jsonl
```

각 control sample은 `source.scenario_id`, `source.episode_id`,
`source.semantic_group_id`를 가져야 한다. validator는 case/whitespace를
정규화한 exact state/question, episode, semantic group별로 split 교차 여부를
검사한다. 기존 smoke fixture는 이 메타데이터가 없어 실패하며, 이것은
정확도를 낮춘 것이 아니라 아직 production-quality 데이터가 아님을 명시한
것이다.

v1.45.0에는 검수용 seed queue가 추가됐다.

```bash
uv run hyperjev control seed \
  --output runs/control/control-review-queue.jsonl \
  --count-per-skill 1000 \
  --seed 7
uv run python scripts/validate_control_dataset.py \
  runs/control/control-review-queue.jsonl
```

8개 skill이 각각 1,000개인 8,000개 queue를 만들 수 있지만, seed의 `target`은
synthetic draft이고 `labels.human`은 null이다. Qwen/Gemma 결과도 보조 draft로만
기록하고, reviewer가 state/question을 보고 직접 선택한 human typed label이
최종 target이다. `--require-human-labels`가 통과하기 전에는 이 queue로
production 정확도를 계산하지 않는다.

v1.46.0부터 production-style evaluator는 다음 조건을 모두 요구한다.

```bash
uv run python scripts/evaluate_control_quality.py \
  --checkpoint runs/control/control-student-human.pt \
  --dataset runs/control/control-reviewed.jsonl \
  --require-human-labels \
  --minimum-safety-accuracy 1.0
```

두 split accuracy 99% 이상, 전체 safety action accuracy 100%, safe STOP recall
100%, 그리고 validation/test 모든 sample의 human typed label이 있어야 한다.
synthetic target만 있는 seed는 연구용 비교에는 쓸 수 있지만 이 gate를 통과할
수 없다.

v1.47.0부터 teacher draft는 다음처럼 control registry를 사용한다.

```bash
uv run hyperjev control draft \
  --config configs/phase0.toml \
  --queue runs/control/control-review-queue.jsonl \
  --provider qwen \
  --output runs/control/qwen-control-draft.jsonl \
  --max-tokens 256
```

Qwen은 빠른 초안/확장, Gemma는 독립 judge로 사용하되 Gemma generation은
control tick에서 기다리지 않는다. timeout/error/teacher disagreement는
human review queue로 보내며, normalized result가 있어도 `labels.human`을
자동으로 채우지 않는다.

reference BOW ablation은 validation `4/8 (50.00%)`로 token/ngram과 동일했다.
따라서 48개 smoke test를 보고 architecture를 고르지 않고, 다음 후보 비교는
human-labeled validation에서 class-balanced hard-negative와 함께 수행한다.

v1.49.0부터 hard-negative pair를 별도 생성한다.

```bash
uv run hyperjev control hard-negative \
  --output runs/control/control-hard-negative.jsonl \
  --pair-count 500 \
  --seed 7
uv run python scripts/validate_control_dataset.py \
  runs/control/control-hard-negative.jsonl
```

pair의 두 counterfactual은 같은 episode/semantic group이지만 같은 split에
고정된다. STOP과 non-STOP, 접근과 이동, 대기와 정지처럼 실제 confusion
matrix에서 위험한 쌍을 먼저 검수한다.

v1.50.0에서는 500쌍/1,000개 synthetic hard-negative queue를
`reference-token-encoder`로 학습했다. validation `100/100`, test `100/100`,
safety action accuracy `4/4`, safe STOP recall `2/2`를 기록했다. 이 결과는
hard-negative curriculum이 혼동쌍을 분리하는지 확인하는 연구 신호로는
유효하지만, human label이 validation/test 모두 `0`이므로 production 정확도나
새 게임/로봇 scene 일반화로 해석하지 않는다. 다음 학습부터 seed와
hard-negative를 함께 구성하되, 사람 라벨이 없는 row는 production evaluator의
분모에서 계속 거부한다.

simulation safety scenario는 40회 replay에서 action accuracy 100%, safety STOP
recall 100%였지만, 이는 contract regression 증거이지 새로운 state 일반화 증거가
아니다. baseline evaluator는 현재 의도적으로 실패한다. 이 실패가 개선마다
정확도를 실제로 올리고 있는지 판단하는 기준점이다.

## 3. 단계별 실행 계획

### Phase A — evaluator와 데이터 계약 고정

목적은 측정 분모와 leakage를 고정하는 것이다.

1. `scripts/evaluate_control_quality.py`로 validation/test 정확도와 safety
   recall을 한 명령에서 계산한다.
2. 각 sample에 `scenario_id`, `source`, `language`, `domain`, `split`을
   요구하고 exact group을 분리한다.
3. train/validation/test 사이에 동일 state/question 또는 동일 episode가
   섞이지 않도록 content/episode group split을 적용한다.
4. 모든 report에 dataset SHA, checkpoint SHA, unique group 수, human label 수,
   accepted coverage를 기록한다.

완료 gate: evaluator가 baseline을 `FAIL`로 재현하고, regression test가 모든
기존 contract/safety behavior를 유지한다.

### Phase B — human golden control dataset 구축

정확도 향상의 가장 큰 병목은 현재 model architecture가 아니라 데이터다.

1. 게임과 로봇을 별도 domain으로 나누고, 각 skill마다 최소 500개의 독립
   semantic group을 만든다.
2. `STOP`은 collision risk, sensor fault, stale frame, emergency를 포함한
   hard-negative와 near-miss를 균형 있게 만든다.
3. `MOVE/APPROACH/RETREAT/ROTATE`는 방향·거리·장애물·속도·목표 상태를
   조합한 counterfactual pair를 만든다.
4. Qwen은 빠른 label draft와 synthetic expansion에만 사용한다.
5. Gemma는 독립 교차검증/judge로 사용하고, 불일치·낮은 confidence·위험
   action은 human review queue로 보낸다.
6. human reviewer는 state, memory context, question을 보고 skill과 abstain
   여부를 직접 결정한다. teacher label은 human label을 덮어쓰지 못한다.

완료 gate: 4,000개 이상 독립 human group, skill별 최소 98% label agreement,
모든 test sample에 human label, split 간 exact/episode leakage 0건.

### Phase C — 데이터 품질과 hard-negative 학습

1. class-balanced sampler로 rare skill과 STOP을 과소표집하지 않는다.
2. 같은 scene의 표현만 바꾸는 paraphrase는 같은 split 안에 두고,
   독립 scene은 held-out split에 둔다.
3. `MOVE ↔ APPROACH`, `STOP ↔ ROTATE`, `HOLD ↔ RECOVER`처럼 혼동되는
   pair를 hard-negative로 최소 1:1 추가한다.
4. teacher disagreement, low-confidence, human correction을 별도 bucket으로
   관리하고, 무조건 학습 데이터에 합치지 않는다.
5. HyperMemory context가 있는/없는 쌍을 만들어 memory 의존성을 측정한다.
6. raw user data와 개인정보는 dataset에 넣지 않고 redaction hash와 provenance만
   남긴다.

완료 gate: validation loss가 안정적으로 감소하고, hard-negative confusion
   matrix에서 모든 위험 pair의 recall이 99% 이상이다.

### Phase D — 모델 학습과 calibration

1. reference trainer는 계약 검증용으로만 유지한다.
2. production multilingual encoder + typed decision heads를 선택하고,
   control head는 memory/query head와 별도 manifest로 관리한다.
3. batch, class weighting, label smoothing은 validation으로 비교하며,
   test set을 보고 hyperparameter를 고르지 않는다.
4. validation set에서 temperature scaling 또는 task별 calibration을 수행한다.
5. confidence threshold는 전체 accuracy가 아니라 risk-coverage curve에서
   accepted accuracy 99.5%를 만족하는 지점으로 선택한다.
6. confidence가 높아도 OOD/novelty detector가 거부하면 Qwen/Gemma 또는
   deterministic controller로 fallback한다.

완료 gate: human validation에서 calibration error와 accepted accuracy가
기준을 만족하고, threshold를 낮춰 coverage를 억지로 올리지 않는다.

### Phase E — HyperMemory ablation과 real-time loop

동일 checkpoint를 다음 세 조건으로 비교한다.

| 조건 | 입력 |
| --- | --- |
| no-memory | state + question |
| bounded-memory | HyperMemory `/v1/context` summary + state + question |
| stale/failed-memory | 만료 또는 timeout된 context |

측정 항목은 accuracy, accepted accuracy, coverage, safety STOP recall, memory
lookup p50/p95, Student p50/p95/p99, controller tick deadline miss다.
HyperMemory는 motor loop가 아니라 skill-decision tick에서 prefetch하며,
timeout 시 raw memory를 기다리지 않고 bounded fallback으로 내려간다.

완료 gate: memory 사용으로 위험 action recall이 감소하지 않고, end-to-end
deadline miss가 허용치 이하이며, memory 장애 시 안전 동작이 재현된다.

### Phase F — simulator, shadow, canary

1. simulator episode에서 collision, unsafe action, recovery latency를 측정한다.
2. parent Qwen/Gemma와 Student를 같은 observation에서 shadow 비교한다.
3. Student disagreement은 즉시 actuator에 반영하지 않고 review/fallback으로
   보낸다.
4. 승인된 skill subset만 canary에 연결하고 speed, TTL, rate limit을 둔다.
5. 실제 운영 전 rollback checkpoint와 audit trail을 검증한다.

완료 gate: simulator와 shadow에서 human test gate를 통과하고, canary 중
collision/unsafe-action rate가 기준 이하이며, 즉시 STOP/rollback이 가능하다.

## 4. 단계별 기록 규칙

각 단계는 다음 순서로 진행한다.

```text
작은 변경 → targeted test → evaluator → 전체 test/lint
→ version bump → git commit → git push → docs/changelog 갱신
```

각 report는 다음 필드를 포함한다.

- git commit, version
- checkpoint/dataset SHA-256
- split과 unique group 수
- human label 수와 target source
- accuracy, accepted accuracy, coverage, fallback
- safety STOP recall, confusion matrix
- latency p50/p95/p99
- 실패한 gate와 다음 조치

## 5. 현재 상태와 다음 작업

Phase A evaluator를 추가했고 baseline과 token 실험 모두를 실패로 재현했다.
threshold를 낮추거나 test sample을 train으로 옮기지 않는다. 다음 구현은
control dataset의 human-review 가능한 schema, exact/episode leakage 검사,
counterfactual hard-negative 생성이다. 그 뒤 실제 human label이 쌓인 뒤에야
model architecture와 학습률을 비교한다.
