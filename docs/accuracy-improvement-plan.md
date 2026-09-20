# HyperJev 정확도 향상 계획

작성일: 2026-09-20  
현재 버전: 1.72.0
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

사람 검수가 끝난 queue는 synthetic `target`을 그대로 학습하지 않도록
materialize한다.

```bash
uv run hyperjev control materialize \
  --input runs/control/control-reviewed.jsonl \
  --output runs/control/control-human-target.jsonl
```

이 명령은 각 `labels.human` typed result를 control registry로 다시 검증하고,
`target`을 사람이 선택한 scalar로 교체하며 `provenance.target_source`를
`human_review`로 기록한다. 미검수 row, abstain, 후보에 없는 skill, 확률 합계
오류는 기본 거부한다. 따라서 evaluator만 human label을 참고하고 trainer는
synthetic target을 계속 학습하는 경로를 차단한다.

seed와 hard-negative를 research ablation으로 결합할 때는 다음 명령을 사용한다.

```bash
uv run hyperjev control merge \
  --input runs/control/control-human-target.jsonl \
  --input runs/control/control-hard-reviewed.jsonl \
  --output runs/control/control-human-combined.jsonl \
  --require-human-labels
```

merge는 sample ID 중복과 결합 후 cross-split exact/episode/semantic-group
leakage를 다시 검사한다. 두 입력 중 하나라도 미검수면
`--require-human-labels`에서 실패하므로, synthetic seed와 human golden이
실수로 production training set에 섞이지 않는다.

control 검수는 generic memory registry를 사용하지 않고 control registry를
명시한다.

```bash
uv run hyperjev control review-pack \
  --queue runs/control/control-review-queue.jsonl \
  --draft runs/control/qwen-control-draft.jsonl \
  --output runs/control/control-review-pack.jsonl \
  --allow-raw \
  --prioritize
uv run hyperjev control review-session \
  --review-pack runs/control/control-review-pack.jsonl \
  --queue runs/control/control-review-queue.jsonl \
  --feedback-output runs/control/control-feedback.jsonl \
  --reviewer <reviewer-id> \
  --deduplicate-exact
uv run hyperjev control apply-feedback \
  --queue runs/control/control-review-queue.jsonl \
  --feedback runs/control/control-feedback.jsonl \
  --output runs/control/control-reviewed.jsonl
uv run hyperjev control materialize \
  --input runs/control/control-reviewed.jsonl \
  --output runs/control/control-human-target.jsonl
```

session은 state와 question을 표시하고 `a`(draft 수락), `e`(typed value만
입력), `n/p`(다음/이전), `s`(보류), `q`(저장 후 종료)를 지원한다. `e`에서
control 후보 밖의 값은 거부된다. reviewer가 teacher draft를 그대로 수락해도
그 기록은 human feedback으로 남지만, 실제 운영에서는 위험 pair와
teacher-disagreement를 우선 독립 확인한다.

v1.64.0의 `--prioritize`는 이 원칙을 review pack 순서에 반영한다. 현재 순서는
pair collision, invalid schema, schema repair가 필요한 teacher 결과, typed
confidence가 낮은 결과다. 같은 등급에서는 sample ID를 tie-break로 사용하므로
resume해도 순서가 결정적이다. 정렬은 queue target을 읽지 않으며,
manifest의 `priority_order=uncertain_first`로 사용된 정렬을 추적할 수 있다.
이는 사람의 제한된 검수 시간을 오류 가능성이 높은 row에 먼저 쓰기 위한 운영
개선이지, human label을 대신하는 자동 판정은 아니다.

v1.65.0부터 hard-negative의 `counterfactual_group_id`를 priority group으로
사용한다. pair 한쪽이 낮은 confidence이면 반대편의 confidence가 높아도 두
항목을 함께 앞에 배치한다. review item에는 `kind`, `episode_id`,
`semantic_group_id`, `counterfactual_group_id`, `pair_side`만 복사하고 target과
labels는 복사하지 않는다. 따라서 사람이 두 상태/question을 연속으로 비교할 수
있지만 synthetic 정답을 보거나 pair의 정답을 추론하도록 유도하지 않는다.

v1.66.0에서 이 옵션을 `control review-pack` handler까지 연결하고 실제 Qwen
hard-negative 1,000건으로 검증했다. manifest는 `uncertain_first`, unique
counterfactual group은 500개, pair adjacency violation은 0개였으며 review pack
전체에 target/labels token이 없었다. 앞으로 CLI smoke는 helper 단위 테스트가
아니라 이 manifest와 leakage 결과까지 확인해야 한다.

v1.68.0에서 실제 Qwen confidence를 분석한 결과, HOLD 오답 83건은 모두
confidence `1.0`이었고 APPROACH 오답도 평균 `0.85`였다. 따라서 probability가
높다는 이유만으로 human review에서 제외하면 안 된다. 같은 counterfactual pair의
두 typed 결과가 같아지는 collision을 target 없이 계산해 최우선으로 올렸고,
hard-negative 500 pair 중 167개가 이 조건에 걸렸다. 이 규칙은 calibration의
대체가 아니라 semantic contradiction을 이용한 active-review 우선순위다.

v1.69.0부터 priority manifest에 group/collision 통계를 함께 기록한다. 실제
hard-negative manifest는 `counterfactual_group_count=500`,
`collision_group_count=167`, `collision_item_count=334`이며, reviewer는 이 수치를
보고 먼저 확인할 active-review 범위를 결정할 수 있다. 기본 queue order에서는
통계를 null로 남겨 기존 artifact의 의미를 바꾸지 않는다.

v1.67.0에서 generic `golden review-pack`에도 같은 `--prioritize` flag를
노출했다. control registry와 generic memory registry를 별도 유지하되, human
review operator가 어느 경로를 호출해도 uncertainty-first contract를 사용할 수
있게 CLI surface를 맞췄다.

Qwen/Gemma를 모두 실행한 뒤에는 합의 결과만 silver 후보로 표시한다.

```bash
uv run hyperjev control adjudicate \
  --queue runs/control/control-review-queue.jsonl \
  --qwen-draft runs/control/qwen-control-draft.jsonl \
  --gemma-draft runs/control/gemma-control-draft.jsonl \
  --output runs/control/control-adjudicated.jsonl
```

두 teacher가 typed value까지 일치하면 `status=agreed`와 normalized result를
남긴다. disagreement, schema invalid, abstain은 `normalized_result=null`로
만들고 Qwen/Gemma 결과·latency·error를 comparison에 보존해 review-session에서
human이 직접 value를 입력하게 한다. Gemma가 대답했다고 해서 human label이
자동 생성되지 않는다.

aggregate accuracy만으로 특정 skill의 실패가 가려지지 않도록 evaluator는
validation/test 각각에 skill별 count, correct, accuracy와 confusion matrix를
기록한다. 기본 gate는 각 skill accuracy도 99% 이상이어야 하며, 표본이 0인
skill은 실패한다. 따라서 STOP이 100%여도 MOVE/APPROACH가 부족하면 전체
control gate는 통과하지 않는다.

teacher queue는 장시간 local generation 중에도 resume할 수 있어야 한다.
`control draft --resume`는 manifest와 완료된 record를 먼저 읽고, 중단된
sample만 다시 호출한다. 각 완료 record는 즉시 flush되므로 process 종료나
server slot 대기로 전체 queue가 사라지지 않는다. 다만 queue SHA와 provider가
바뀌면 resume을 거부한다.

학습 split의 target 불균형은 `control train --class-balanced`로 연구 비교할 수
있다. 이는 train split만 deterministic oversampling하고 validation/test는
변경하지 않는다. class balance를 켠 checkpoint도 confidence calibration과
control scenario를 다시 통과해야 하며, 평균 정확도만으로 승격하지 않는다.

v1.70.0에는 counterfactual group의 per-example loss를 높이는 선택형
`--hard-negative-weight`를 추가했다. 기본값 `1.0`은 이전과 같고, `2.0`/`4.0`은
human hard-negative가 materialize된 뒤에만 재시도할 연구 후보로 취급한다. synthetic
combined 1,800건에서 두 weight 모두 validation/test `100%`, STOP recall `100%`였지만
safety action accuracy가 `75%`로 baseline `100%`보다 낮아 현재 production 후보에서
탈락했다. safety를 희생해 aggregate accuracy를 만드는 weight는 사용하지 않는다.

v1.71.0의 Gemma hard-negative probe는 30초 timeout에서 `completed=0`,
`schema_valid=0`, `error_count=1`이었다. 따라서 Gemma는 독립 judge라는 역할은
유지하지만, 현재 label throughput이나 control latency의 동기 경로에 넣지 않는다.
Gemma가 실제 결과를 내는 경우에도 Qwen/Gemma 합의는 silver 후보일 뿐이며,
human label gate를 대체하지 않는다.

v1.72.0에서는 Qwen hard-negative에서 반복된 `APPROACH↔MOVE`와 `HOLD↔STOP`
혼동을 줄이기 위해 보수적인 control fast path를 추가했다. 단일 keyword가 아니라
완전한 compound phrase를 요구하고, 둘 이상의 skill pattern이 동시에 맞으면
모델로 넘긴다. STOP은 별도 명시적 collision rule로 먼저 처리한다. 따라서 이
경로는 일반 자연어 분류기가 아니며, 애매한 state의 정확도를 과장하지 않는다.

| queue | fast-path coverage | fast-path target match | human labels |
| --- | ---: | ---: | ---: |
| hard-negative 1,000 | 1,000/1,000 (100.00%) | 1,000/1,000 (100.00%) | 0/1,000 |
| seed 800 | 750/800 (93.75%) | 750/750 (100.00%) | 0/800 |

실제 `ControlStudentClient`에 hard-negative 1,000행을 넣었을 때 모든 판단이
`safety-rule` 또는 `control-rule`로 처리됐고 synthetic target match는 100%였다.
초기화 이후 in-process 측정의 총 시간은 10.11ms(평균 약 10.11µs/행)였다.
별도의 rule-only 측정은 약 9.97~15.84µs/행 범위였다. 이는 synthetic target과
local CPU 경로의 증거일 뿐 human generalization이나 end-to-end motor deadline
증거가 아니다. 사람 라벨은 여전히 0건이고 production eligibility는 false다.
fast path에 매칭되지 않은 state는 기존 typed Student와 safety fallback을 그대로
사용한다.

v1.52.0의 synthetic combined 연구 실험은 split `180/180 (100%)`였지만, model
only safety action accuracy가 `3/4 (75%)`로 실패했다. 명시적
`obstacle is directly ahead`/즉시 충돌 문구를 model보다 먼저 STOP으로 처리하는
보수적 rule을 추가한 뒤 safety는 `4/4 (100%)`, STOP recall `2/2 (100%)`로
회복됐다. 이는 안전 계약의 방어층 증거이며, rule이 일반 scene 의미를 이해한다는
정확도 증거가 아니다.

v1.58.0에는 고정 confidence threshold별 risk-coverage 측정을 추가했다.

```bash
uv run python scripts/evaluate_control_quality.py \
  --checkpoint /tmp/control-combined-token-100ep.pt \
  --dataset /tmp/hyperjev-control-combined-1800.jsonl \
  --risk-thresholds 0.50 0.90 0.95 0.99
```

report의 `risk_coverage`는 전체 accuracy와 별도로 각 threshold에서 실제로
수락할 표본 수, coverage, accepted accuracy, accepted risk를 기록한다. combined
token checkpoint는 validation/test 모두 네 threshold에서 `180/180`을 수락하고
accepted accuracy `100%`, risk `0%`를 보였지만, 이는 synthetic target과 같은
분포의 confidence 결과다. class-balanced 후보도 같은 risk-coverage 숫자를
냈지만 safety action accuracy는 `3/4 (75%)`였다. 즉 confidence가 높다는
사실만으로 safety correctness나 실제 scene 일반화를 보장하지 않는다.

다음 gate는 risk-coverage가 좋은 영역을 고르는 것이 아니라, 독립 human-labeled
test에서 `accepted risk <= 1%`, 모든 고위험 skill의 STOP recall `100%`,
calibration set의 ECE/NLL 및 시간 예산을 함께 통과시키는 것이다. threshold를
낮춰 coverage를 억지로 높이거나 confidence를 재표현해 통과시키지 않는다.

v1.59.0에는 held-out Student logits에서 task별 temperature를 계산하는
`scripts/calibrate_control_checkpoint.py`를 추가했다.

```bash
uv run python scripts/calibrate_control_checkpoint.py \
  --checkpoint /tmp/control-combined-token-100ep.pt \
  --dataset /tmp/hyperjev-control-combined-1800.jsonl \
  --split validation \
  --output /tmp/control-calibration.json
```

calibration은 `include_logits=True`인 평가 경로에서 task별로만 fit하고,
checkpoint/dataset SHA와 sample 수를 manifest에 기록한다. `argmax` label을
바꾸지 않으며 confidence와 abstention 해석만 위한 artifact다. combined
validation 180건의 synthetic run은 temperature `0.25`가 검색 하한에 도달하고
NLL `0.000007 → 0.000000`이 됐지만 human label은 `0/180`이라
`production_eligible=false`다. temperature가 경계에 붙은 것도 calibration
set이 너무 쉽거나 모델이 과신한다는 신호일 수 있으므로, 범위를 넓히거나
숫자를 채택해 정확도를 주장하지 않는다.

calibration manifest는 runtime에서도 명시적으로 연결해야 한다. `StudentClient`는
manifest의 `checkpoint_sha256`가 현재 checkpoint와 일치하는지 확인한 뒤에만
task별 temperature를 적용한다. `control decide --calibration` 또는
`HYPERJEV_STUDENT_CALIBRATION`으로 경로를 지정하며, 기본값은 calibration을
사용하지 않는다. 따라서 잘못된 checkpoint와 calibration을 조용히 조합하는
실수를 막고, 기존 Student 경로의 argmax와 latency 계약을 보존한다.

Qwen seed draft 800건은 다음 단계의 teacher 운영 증거로 고정했다.

```bash
uv run hyperjev control draft \
  --config configs/phase0.toml \
  --registry registry/control_tasks \
  --queue /tmp/hyperjev-control-seed-800.jsonl \
  --provider qwen \
  --output /tmp/qwen-control-seed-800.jsonl \
  --max-tokens 256 --timeout 30 --resume
```

최종 manifest는 `completed=800`, `schema_valid=800`, synthetic target 일치
`800/800`, task별 100%였다. 2건은 Qwen이 선택값은 맞게 냈지만 확률합을
`0.95`로 반올림해 contract를 위반했으며, ingestion이 모든 후보·비음수·합
범위를 재검증한 뒤에만 정규화하고 `schema_repaired=2`로 기록했다. 원본 teacher
출력이 완벽했다고 처리하지 않은 것이다. Qwen latency는 p50 `2188.502ms`,
p95 `2357.121ms`, max `2596.726ms`이므로 motor loop가 아니라 비동기 draft,
fallback, review queue에서만 사용한다. 여전히 human label은 자동 생성하지 않는다.

teacher 결과는 다음 evaluator로 재현한다.

```bash
uv run python scripts/evaluate_control_teacher_draft.py \
  --queue /tmp/hyperjev-control-seed-800.jsonl \
  --draft /tmp/qwen-control-seed-800.jsonl \
  --output /tmp/qwen-control-seed-800-quality.json
```

이 report는 synthetic target match와 schema/latency만 측정하고 `human_label_gate`
및 `production_ready`를 별도로 남긴다. 따라서 Qwen이 synthetic queue에서
100%를 기록해도 human reviewer가 state/question을 확인하기 전에는 training
target이나 production accuracy로 쓰지 않는다. hard-negative report도 동일한
경계와 per-skill 계산을 사용한다.

hard-negative 500 pair/1,000 sample의 Qwen 결과는 다음과 같다.

| target skill | count | correct | accuracy |
| --- | ---: | ---: | ---: |
| STOP | 250 | 250 | 100.00% |
| MOVE | 167 | 167 | 100.00% |
| INTERACT | 83 | 83 | 100.00% |
| RECOVER | 83 | 83 | 100.00% |
| ROTATE | 83 | 83 | 100.00% |
| RETREAT | 84 | 80 | 95.24% |
| APPROACH | 167 | 83 | 49.70% |
| HOLD | 83 | 0 | 0.00% |

전체는 `829/1000 (82.90%)`, counterfactual pair 양쪽 정답은 `329/500
(65.80%)`이다. 따라서 Qwen 초안을 그대로 silver label로 학습하면 HOLD와
APPROACH 오답을 학습할 위험이 크다. 이 queue는 `control review-pack`으로
human에게 우선 전달하고, 특히 오류 171건과 해당 pair의 반대편 sample을 함께
검수한 뒤에만 materialize한다. STOP recall 100%는 안전 신호지만 다른 skill의
실패를 상쇄하지 못한다.

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
