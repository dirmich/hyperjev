# HyperJev 정확도 향상 계획

작성일: 2026-09-20  
현재 버전: 1.109.0
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

v1.73.0에서는 fast path의 phrase substring 오탐을 막는 skill별 contradiction
blocker를 추가했다. 예를 들어 `no hazard is present`를 `hazard is present`로
잘못 읽지 않도록 blocker를 구체적인 부정/위험 표현으로 제한하고, “target은
앞에 있지만 안전하게 접근할 수 없다”와 “안정 자세지만 emergency hazard가 있다”
반례가 model/safety 경로로 우회되는 회귀 테스트를 추가했다. 이 단계의 목적은
coverage를 늘리는 것이 아니라 규칙 false-positive를 낮추는 것이다.

v1.74.0에서는 `scripts/evaluate_control_fast_path.py`를 추가해 fast path의
coverage, source별 처리량, p50/p95/p99/max/mean latency, synthetic target과
human target을 분리한 report를 재현한다. hard-negative report는 queue SHA
`f88cec23d06b1bae9c688bcc5f3cea4dc3b68912980b43e98c8d72fd986e5f0d`에서
coverage `100%`, synthetic match `100%`, p50/p95/p99 `13.488/15.408/17.2µs`를
기록했다. human label은 `0/1000`이고 `production_ready=false`이며, `--require-full-coverage`
는 unresolved model row가 생기면 실패한다.

v1.75.0에서는 같은 evaluator에 `--checkpoint` runtime replay를 연결했다. 실제
`ControlStudentClient`를 초기화해 rule → typed Student → safety policy 순서를
재생하며, rule-only report와 runtime report를 혼동하지 않는다. hard-negative에서
runtime synthetic accuracy `100%`, STOP recall `100%`, p50/p95/p99
`13.648/15.616/17.504µs`를 얻었지만 source는 여전히 rule/safety가 전부였고
human `0/1000`이므로 production-ready가 아니다. model이 실제로 호출되는
unresolved/OOD queue와 human-labeled test를 별도 측정해야 한다.

v1.76.0에서는 runtime report에 source별 latency를 추가했다. seed 800행에서는
fast path 750행, Student fallback 50행, safety rule 50행을 분리했고, fallback
Student의 p50/p95/p99는 `293.486/421.518/2972.318µs`였다. fast path의
`control-rule` p95는 `14.112µs`, safety-rule p95는 `3.984µs`였다. 이 수치는
reference Student CPU replay이며, Qwen/Gemma teacher latency나 GPU production
latency가 아니다. seed synthetic target/STOP recall은 100%였지만 human label은
0건이므로 production gate는 false다.

v1.77.0에서는 hand-authored compositional OOD fixture 40개를 추가하고
model-only와 integrated runtime을 분리 측정했다. fixture SHA는
`1d67466d89d265fdbcafd7ceb0c73edd1dc800d987df1178cffcb51dea0d7919`이며,
40개 unique group과 cross-split leakage 0건이다. reference Student만 사용한
`model_only_with_safety_policy`는 `4/40 (10%)`, STOP recall `0/5 (0%)`, p95
`2506.615µs`였다. deterministic fast path를 포함한 integrated runtime은
`40/40 (100%)`, STOP `5/5 (100%)`, p95 `22.480µs`였다. 따라서 integrated
100%를 Student 모델 정확도로 부르지 않으며, human label `0/40`이라 production
gate는 false다. 다음 정확도 개선은 이 OOD 의미쌍을 human golden으로 전환한 뒤
encoder/head 학습과 독립 test에서 재검증하는 것이다.

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

### v1.78.0 compositional train augmentation과 safety interlock

held-out OOD fixture와 어휘를 공유하지만 문장은 다른 train-only compositional
queue 64개(8 skill × 8)를 생성하고, 기존 1,800개 queue와 합쳐 1,864개 dataset을
만들었다. dataset SHA는
`78b54a46f5bd9d5049b7d3ea903c944854c87882a842ddcd87d67cab5f0f7c9b`이며,
compositional rows는 `train`에만 있고 unique episode/group 64개다. 이 변경은
OOD 문장을 train에 복사하지 않고 표현 조합의 일반화만 학습시키는 ablation이다.

reference-token-encoder 100 epoch checkpoint
`7d74e6bb9d1a5bb5ea294689aa4f97d8248758c589c061fb7ec11b93d8386848`의
held-out 40문장 결과는 다음과 같다.

| 경로 | 정확도 | STOP recall | p95 | 판정 |
| --- | ---: | ---: | ---: | --- |
| model-only + safety policy | 38/40 (95.00%) | 5/5 (100.00%) | 5,177.947µs | latency/accuracy gate 실패 |
| integrated fast path + Student + safety | 40/40 (100.00%) | 5/5 (100.00%) | 25.233µs | synthetic 통합 gate 통과 |

model-only 정확도는 v1.77의 10.00%에서 95.00%로 개선됐지만, human label은
여전히 `0/40`이다. 또한 model-only에서 HOLD 두 건이 오분류됐으므로 모델 단독
99% 달성으로 주장하지 않는다. `enable_fast_path=False`에서도 explicit collision
STOP interlock은 유지하도록 고정했으며, 이는 ablation 편의를 위해 안전 경계를
끄지 않는 계약이다. 다음 단계는 HOLD/STOP 경계의 독립 human golden label과
실제 DGX GPU latency 측정이다.

### v1.79.0 semantic boundary augmentation

HOLD/STOP을 포함한 모든 skill에 decision-boundary 문장 8개씩, 총 64개의
train-only queue를 추가했다. 기존 1,864개와 결합한 dataset은 1,928개이며 SHA는
`813313e746536aa1ce4f72634c66472858c3b9d5d65463112b8f4f5a2c1ebbce`다. boundary
queue SHA는 `076ef1d1bb958e92822c92f7e01e740c6af8eaa774e609746cec9c91d505a31`이고,
검증 결과 unique episode/group leakage는 `0`이다.

checkpoint
`01e4c5902276919f2516755ab27f0833873ec8bd8f6e6d1d4f8f478d7c7f06ab`의
held-out OOD 결과는 model-only `39/40 (97.50%)`, STOP recall `5/5 (100%)`,
integrated `40/40 (100%)`, STOP recall `5/5 (100%)`였다. v1.78의 model-only
`95%`보다 2.5%p 개선됐지만 HOLD 한 건이 STOP으로 보수 처리되어 99% gate에는
아직 미달이다. 일반 safety scenario의 action accuracy는 `3/4 (75%)`, STOP
recall은 `2/2 (100%)`이므로 안전 recall과 정상 action precision을 별도 최적화한다.
human label은 여전히 `0`이며, 다음 단계는 남은 HOLD 문장의 독립 paraphrase
보강과 normal-approach safety false STOP 원인 분석이다.

### v1.80.0 normal-action precision과 학습 ablation 선택

`target is close and directly ahead`처럼 충돌 신호가 없고 목표가 명확한 compound
문장을 APPROACH fast path에 추가했다. 이 변경은 model-only 숫자를 높이기 위한
우회가 아니라, 통합 runtime에서 confidence 부족을 이유로 정상 action을 STOP으로
거부하는 false STOP을 줄이는 safety/precision 경계 수정이다. v1.79 best checkpoint
로 quality evaluator를 재실행한 결과 safety action accuracy가 `3/4 (75%)`에서
`4/4 (100%)`로 올라갔고 safe STOP recall은 `2/2 (100%)`를 유지했다.

추가 학습 ablation은 다음과 같이 선택했다.

| 실험 | OOD model-only | 판정 |
| --- | ---: | --- |
| v1.79 balanced boundary, seed 7 | 39/40 (97.50%) | best synthetic reference |
| boundary 9개/skill, seed 7 | 38/40 (95.00%) | 데이터 양 증가로 회귀, 폐기 |
| v1.79 dataset, seed 42 | 35/40 (87.50%) | seed 민감도 확인, 폐기 |
| v1.79 dataset, class-balanced | 39/40 (97.50%) | 개선 없음, 폐기 |

따라서 checkpoint 선택은 test fixture를 보고 임의로 고른 것이 아니라 validation
동률과 OOD 보조 ablation을 함께 기록한 best-effort 결과다. 여전히 OOD target은
synthetic이고 human label은 `0`이므로 production 99% 주장으로 승격하지 않는다.

### v1.81.0 Qwen hard-negative human review pack

Qwen `qwen38fn` draft가 붙은 hard-negative 500쌍/1,000행을 human review pack으로
내보냈다. queue SHA는
`f88cec23d06b1bae9c688bcc5f3cea4dc3b68912980b43e98c8d72fd986e5f0d`, Qwen draft
SHA는 `e8d7c15c526169109dbc7e552dde15d9f2a10a609e326b66aec1b2e9b62e17d4`다.
pack은 1,001 lines(Manifest 1 + item 1,000), raw `state/question`만 포함하고
synthetic `target`과 queue `labels`는 제외했다. `uncertain_first` 우선순위로
counterfactual group 500개와 collision group 167개(334 items)를 추적한다.

사람 검수는 다음 한 세션에서 진행한다.

```bash
uv run hyperjev control review-session \
  --review-pack /tmp/control-qwen-hard-review-pack.jsonl \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --feedback-output /tmp/control-qwen-hard-feedback.jsonl \
  --reviewer human-control-1 \
  --deduplicate-exact
```

세션은 `state`, `question`, Qwen draft를 보여주고 reviewer가 후보 value를
선택/수정한다. Qwen/Gemma draft는 human label을 자동 대체하지 않는다. 현재
human label은 `0/1,000`이고, 이 단계의 stop condition은 reviewer가 모든 row를
확인한 뒤 `control materialize`와 `--require-human-labels`가 통과하는 것이다.

### v1.82.0 teacher-blind control dual review

synthetic target과 teacher draft를 사람이 정답으로 받아들이는 위험을 막기 위해
control 전용 라벨링 기준과 teacher-blind 이중 검수 경로를 추가했다. 기준은
[`docs/control_labeling.md`](control_labeling.md)에 고정했다. 두 reviewer는
`state/question`을 독립적으로 보고 typed skill을 입력하며, `review-agreement`가
완전 일치와 불일치를 기록한다. 불일치는 `review-adjudicate`에서 제3자가
판정하고, `review-finalize`가 agreement와 adjudication을 합친 순수
`golden_feedback`만 만든다.

```bash
uv run hyperjev control review-session \
  --review-pack /tmp/control-qwen-hard-review-pack.jsonl \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --feedback-output /tmp/control-reviewer-a.jsonl \
  --reviewer control-a --blind --deduplicate-exact
uv run hyperjev control review-session \
  --review-pack /tmp/control-qwen-hard-review-pack.jsonl \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --feedback-output /tmp/control-reviewer-b.jsonl \
  --reviewer control-b --blind --deduplicate-exact
uv run hyperjev control review-agreement \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --reviewer-a-feedback /tmp/control-reviewer-a.jsonl \
  --reviewer-b-feedback /tmp/control-reviewer-b.jsonl \
  --output /tmp/control-dual-agreement.jsonl --minimum-agreement 0.98
uv run hyperjev control review-adjudicate \
  --agreement /tmp/control-dual-agreement.jsonl \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --feedback-output /tmp/control-adjudication.jsonl \
  --reviewer control-adjudicator
uv run hyperjev control review-finalize \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --agreement /tmp/control-dual-agreement.jsonl \
  --adjudication-feedback /tmp/control-adjudication.jsonl \
  --output /tmp/control-human-feedback.jsonl
```

v1.82의 agreement gate는 reviewer consistency gate이지 정확도 gate가 아니다.
현재 실제 control human label은 여전히 `0`이므로 production 99% 정확도는
측정·주장하지 않는다. 다음 승격 조건은 이 workflow로 독립 human benchmark를
완성한 뒤, accepted accuracy와 raw Student-head accuracy, safety STOP recall을
각각 confidence interval과 함께 보고하는 것이다.

### v1.83.0 raw/accepted/safety evaluator hardening

production 숫자의 의미가 섞이지 않도록 `scripts/evaluate_control_quality.py`를
강화했다. `raw_student_head`는 deterministic rule을 적용하지 않은 Student
head의 split/skill/risk-coverage를 담고, `safety_policy`는 별도 safety scenario
경로로 보고한다. 각 split에는 raw accuracy와 accepted accuracy의 Wilson 95%
interval이 포함된다.

새 gate는 다음을 모두 요구한다.

- raw split accuracy `>= 0.99`
- per-skill accuracy `>= 0.99`
- accepted accuracy `>= 0.995`
- accepted coverage `>= 0.99`
- safety scenario accuracy와 point STOP recall `100%`
- STOP recall Wilson 95% 하한 `>= 0.99`
- human label gate 통과 후에만 `production_ready`

최신 synthetic reference checkpoint를 재생한 결과는 validation/test 모두
`180/180 (100%)`, accepted coverage `100%`였지만 safety STOP은 `2/2`뿐이라
Wilson 하한 `0.342380`으로 실패했다. 따라서 evaluator exit code는 `1`이고,
현재 모델을 상용 안전 판단기로 승격하지 않는다. 다음 작업은 blind human
benchmark 표본을 채우고, raw Student-head와 integrated fast path를 같은
human-only held-out set에서 따로 측정하는 것이다.

### v1.84.0 held-out human test gate

evaluator에 `--require-human-test`를 추가해 test split 전체가 typed human label을
갖지 않으면 명시적으로 실패하게 했다. validation/test의 human label 상태를
`human_label_status.by_split`에 기록하고, 전체 split gate와 held-out test gate를
분리한다.

```bash
uv run python scripts/evaluate_control_quality.py \
  --checkpoint /path/to/human-trained-control.pt \
  --dataset /path/to/control-human-materialized.jsonl \
  --scenarios tests/golden/control_scenarios.jsonl \
  --registry registry/control_tasks \
  --require-human-test
```

현재 dataset은 human label `0`이므로 이 옵션을 켜면 실패하는 것이 정상이다.
human review가 끝난 뒤에도 training/validation row의 label과 held-out test row의
label을 같은 것으로 재사용하지 않고, test group을 학습에서 제외한 checkpoint로
재평가해야 한다.

### v1.85.0 per-skill accepted diagnostics

per-skill report에 raw accuracy뿐 아니라 skill별 accepted count, accepted
accuracy, accepted coverage, Wilson 95% interval을 추가했다. human test가
완성되면 전체 점수만 보고 다음 학습을 결정하지 않고, 예를 들어 `APPROACH`와
`MOVE`의 경계에서 accepted coverage가 낮은지, `STOP`의 confidence가 과도하게
보수적인지, `HOLD`가 `STOP`으로 혼동되는지를 분리할 수 있다.

이 지표는 다음 데이터 개선 순서를 결정하는 기준이다.

1. STOP false negative가 하나라도 있으면 해당 semantic group과 counterfactual
   pair를 safety adjudication 우선순위로 올린다.
2. STOP false positive가 높으면 `HOLD`/normal action boundary를 추가하되,
   explicit safety signal을 약화하지 않는다.
3. accepted coverage가 낮고 accepted accuracy가 높으면 calibration/abstention을
   조정하고, accuracy를 희생해 threshold를 무작정 낮추지 않는다.
4. 한 skill의 human-only test lower bound가 기준 미달이면 해당 skill만 train
   augmentation하는 것이 아니라 episode/group이 겹치지 않는 새 human examples를
   추가한다.

### v1.86.0 Student uncertainty active review

review pack 생성 시 `--student-checkpoint`를 주면 raw Student confidence를
reviewer 노출 없이 순서에만 사용한다. counterfactual group은 계속 인접하게
유지하고, teacher collision/teacher uncertainty와 Student uncertainty를 함께
우선순위에 반영한다.

```bash
uv run hyperjev control review-pack \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --draft /tmp/qwen-control-hard-500.jsonl \
  --output /tmp/control-student-priority-review-pack.jsonl \
  --allow-raw --prioritize \
  --student-checkpoint /tmp/control-combined-boundary-100ep.pt \
  --student-device cpu
```

실제 1,000행 replay에서 manifest는
`student_uncertainty_then_teacher`, `target_excluded=true`,
`labels_excluded=true`를 기록했다. 현재 checkpoint는 해당 queue에서 confidence
`0.90` 미만 사례가 `0`개라 순서 변화는 없었지만, human label이 들어온 뒤
Student가 불확실한 semantic group을 먼저 검수할 수 있다. Student prediction은
pack item에 복사하지 않으므로 blind review anchoring을 만들지 않는다.

### v1.87.0 Student–teacher disagreement active review

Student confidence가 높아도 Qwen draft와 typed prediction이 다르면 overconfident
error 후보로 먼저 검수한다. 이 불일치 신호도 reviewer 화면에는 노출하지 않고
정렬 순서에만 사용한다. 최종 human label은 여전히 blind reviewer가 직접 입력하며,
teacher와 Student 중 어느 쪽도 자동 정답으로 승격하지 않는다.

실제 hard-negative 1,000행 replay에서 다음이 확인됐다.

| 항목 | 결과 |
| --- | --- |
| priority order | `student_disagreement_then_uncertainty_then_teacher` |
| Student–Qwen disagreement | `171/1,000` |
| Student confidence < 0.90 | `0/1,000` |
| target/labels leakage | 없음 |

이 우선순위는 confidence만으로 놓치는 overconfident 오류를 human 검수 앞에
올리는 active-learning 입력이다. disagreement 수 자체는 teacher 정확도를
의미하지 않으므로, adjudicated human result로 확인한 뒤에만 학습 데이터에
반영한다.

### v1.88.0 action별 dual-review agreement 진단

blind dual review의 전체 agreement rate만으로는 어떤 control action 경계가
검수자를 갈라놓는지 알 수 없다. `control review-agreement` manifest에
`label_agreement_by_action`을 추가해 각 reviewer가 선택한 typed action별로
다음 수치를 기록한다.

- `reviewer_a_count`, `reviewer_b_count`: 해당 action을 선택한 횟수
- `agreement_count`, `disagreement_count`: 해당 action이 포함된 comparable pair의
  일치·불일치 횟수
- `comparable_count`, `agreement_rate`: action별 검수 집중도

이 통계는 blind reviewer 입력만으로 계산하며 synthetic `target`, Qwen/Gemma
출력은 사용하지 않는다. 따라서 이것은 action boundary를 어디부터 human
adjudication할지 정하는 active-review 지표이지 정확도·정답률·production gate가
아니다. 한 disagreement pair는 양쪽 action의 `disagreement_count`에 모두
반영되므로 전체 pair 수와 합산하지 않는다.

다음 실제 human review에서는 이 표의 낮은 agreement action을 우선 표본화하고,
adjudication이 끝난 뒤에만 materialized target과 held-out human test를 갱신한다.

`control review-agreement --show-action-summary`는 같은 진단을 사람이 읽기 쉬운
표로 먼저 출력한 뒤 기존 manifest JSON을 출력한다. 기본 출력은 바꾸지 않았기
때문에 CI나 후속 스크립트는 기존 JSON 경로를 계속 사용할 수 있다.

### v1.90.0 action focus review ordering

action별 agreement가 낮은 경계가 확인되면 다음과 같이 전체 pack을 보존하면서
해당 action을 선택한 teacher item을 먼저 검수할 수 있다.

```bash
uv run hyperjev control review-pack \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --draft /tmp/qwen-control-hard-500.jsonl \
  --output /tmp/control-focus-review-pack.jsonl \
  --allow-raw --prioritize \
  --focus-action APPROACH --focus-action HOLD
```

`focus_actions`와 `focus_action_item_count`는 manifest에 남지만 item의 target이나
Student prediction은 추가하지 않는다. counterfactual sibling은 계속 붙어 있어
경계의 양쪽을 함께 검수한다. 이 기능은 low-agreement action을 먼저 보는
active-review 도구이며, teacher 선택이 틀릴 수 있으므로 전체 sample coverage와
blind human correction을 유지해야 한다.

이미 만든 dual-review manifest에서 focus action을 자동으로 읽으려면 threshold를
명시한다.

```bash
uv run hyperjev control review-pack \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --draft /tmp/qwen-control-hard-500.jsonl \
  --output /tmp/control-focus-review-pack.jsonl \
  --allow-raw --prioritize \
  --agreement-manifest /tmp/control-agreement.jsonl \
  --max-action-agreement 0.98
```

명시적 `--focus-action`과 manifest에서 선택된 action은 합쳐진다. manifest가
없거나 comparable action이 없으면 자동 focus는 비어 있으며 기존 priority만
동작한다. 이 자동화도 human correction을 만들지 않으므로 99% 정확도나
production readiness의 증거가 아니다.

### v1.92.0 target-exclusion guard

manifest-driven focus는 이제 manifest의 `target_excluded`가 명시적으로 `true`인
경우에만 동작한다. target 누출 가능성이 있는 파일을 action priority 입력으로
사용하면 즉시 거부한다. 이 검사는 blind review 경계를 지키기 위한 것이며,
accuracy 계산이나 human label 생성과는 별개다.

### v1.93.0 human-only control training gate

production용 control checkpoint 학습은 다음 flag를 반드시 사용한다.

```bash
uv run hyperjev control train \
  --dataset /path/to/control-human-target.jsonl \
  --output /path/to/control-human.pt \
  --require-human-labels \
  --device cuda
```

이 flag는 모든 row의 typed `labels.human`을 검증하고, control task에 대해서는
`provenance.target_source=human_review`를 요구한다. synthetic target이나 Qwen/
Gemma draft만 있는 queue에서는 checkpoint를 생성하지 않는다. 따라서 이후
evaluator의 `--require-human-test`와 결합해야 99% 승격 후보가 된다.

### v1.94.0 bounded latency gate

실시간 control loop의 latency를 단순 기록에서 gate로 승격했다.

```bash
uv run hyperjev control simulate \
  --checkpoint /path/to/control-human.pt \
  --scenarios tests/golden/control_scenarios.jsonl \
  --max-p95-ms 5 \
  --max-p99-ms 5 \
  --output runs/control/simulation-latency.json
```

threshold를 넘으면 exit `1`이며 report의 `latency_gate.passed`가 false다. 2026-09-21
local CPU 4-scenario replay는 p95/p99 `0.066515ms`, 정확도 `4/4`, safe STOP
recall `2/2`였다. 이 수치는 fast-path fixture 증거일 뿐이므로 DGX Spark에서
Student fallback, 동시성, memory context, teacher fallback을 포함한 p95/p99를
다시 측정해야 한다.

### v1.95.0 minimum safety evidence gate

STOP recall의 point estimate가 100%여도 positive safety case가 2개뿐이면
99% confidence evidence가 아니다. quality evaluator는 기본적으로 최소 500개의
expected safe STOP scenario를 요구한다.

```bash
uv run python scripts/evaluate_control_quality.py \
  --checkpoint /path/to/control-human.pt \
  --dataset /path/to/control-human-target.jsonl \
  --scenarios /path/to/control-safety-500.jsonl \
  --registry registry/control_tasks \
  --minimum-safe-stop-count 500 \
  --require-human-test
```

scenario 수가 부족하면 `safe_stop_sample_count` gate가 실패한다. 500은 모든
case가 맞았을 때 Wilson 95% 하한이 약 `0.992376`이 되도록 잡은 최소 표본이며,
같은 episode를 단순 복제하지 않고 서로 다른 scenario/group으로 구성해야 한다.

### v1.96.0 reproducible 500-case safety matrix

v1.95의 표본 수 gate를 실제 독립 fixture로 채웠다. 생성기는 세 가지 fail-closed
원인(`emergency_stop`, `stale_observation`, `invalid_clock`)을 순환하고, 각 row에
고유한 `scenario_id`, `episode_id`, `semantic_group_id`를 넣는다.

```bash
uv run python scripts/generate_control_safety_scenarios.py \
  --output tests/golden/control_safety_500.jsonl \
  --count 500
```

재현 fixture의 SHA-256은
`9539d8b0342e9f4f896d0ef57fefeb4c2b6d9ea01002bfdd5a6a938070779d66`이며,
reason별 count는 emergency `167`, stale `167`, invalid clock `166`이다.
현재 reference checkpoint를 replay한 결과는 safety accuracy `500/500`, safe STOP
recall `500/500`, Wilson 95% 하한 `0.992376`로 safety gate를 통과했다. evaluator는
중복 `scenario_id`를 거부한다.

이 단계의 성공 조건은 safety interlock의 통계적 근거를 99% 수준으로 올리는
것이다. 이것을 control action 정확도 99%로 해석하면 안 된다. control dataset의
validation/test human label은 아직 없으므로 `--require-human-test`를 포함한
전체 production gate는 실패하며, 다음 승격 조건은 blind human review를 통한
typed control label 확보, held-out test 고정, human-only checkpoint 재학습이다.

### v1.97.0 bounded blind human-review batches

실제 control 정확도를 올리는 다음 병목은 사람이 1,000개를 한 번에 처리해야
한다는 운영 부담이다. `control review-session`은 이제 review group 순서에 대해
`--offset`과 `--limit`을 지원한다. 각 batch는 append-only feedback에 즉시 기록되고,
다음 실행은 같은 pack/queue digest를 검증한 뒤 지정된 범위에서 이어간다.

```bash
uv run hyperjev control review-session \
  --review-pack runs/control/control-review-pack.jsonl \
  --queue runs/control/control-review-queue.jsonl \
  --feedback-output runs/control/control-feedback.jsonl \
  --reviewer human-control-1 \
  --blind \
  --offset 0 \
  --limit 50

uv run hyperjev control review-session \
  --review-pack runs/control/control-review-pack.jsonl \
  --queue runs/control/control-review-queue.jsonl \
  --feedback-output runs/control/control-feedback.jsonl \
  --reviewer human-control-1 \
  --blind \
  --offset 50 \
  --limit 50
```

`--blind`는 teacher draft를 출력하지 않고 `e`로 직접 typed value를 입력하게
한다. `a` accept 경로는 blind mode에서 비활성화된다. report의
`reviewed_count`/`pending_count`는 전체 pack 기준이고 `batch_count`/
`batch_pending_count`는 현재 배치 기준이므로, 배치 완료 여부를 혼동하지 않는다.
라벨 수집 후에는 기존 `apply-feedback` → `materialize` →
`control train --require-human-labels` 순서를 지킨다.

현재 정확도 승격 상태는 바뀌지 않았다. 이 기능은 사람이 판단할 수 있는
control target을 만들기 위한 unblock이며, 사람이 실제로 입력한 label이 없는
동안에는 control accuracy 99%를 주장하지 않는다.

### v1.98.0 batch handler wiring verification

v1.97의 batch 기능을 실제 `/tmp` control review pack에 실행해
`batch_limit=50`, `batch_count=50`, 전체 pending `1000`을 확인했다. 첫 prompt에서
`q`로 종료하면 feedback 파일은 생성되지 않고 process exit은 `0`이다. 이 smoke는
사람 label을 만들지 않으며, batch가 실제 control handler까지 전달되는지를 검증한다.
추가 handler forwarding unit test로 generic `golden review-session`과 control
`review-session`의 인자 경로를 분리해 고정했다.

### v1.99.0 review progress and held-out readiness status

사람 검수가 실제로 진행됐는지 추정하지 않도록 `control review-status`를
추가했다. 이 명령은 feedback append-only stream의 queue digest, 전체 coverage,
train/validation/test별 pending 수, 사람이 선택한 action별 count를 출력한다.
synthetic `target`, Qwen, Gemma 결과를 정답으로 사용하거나 report에 복사하지
않는다.

```bash
uv run hyperjev control review-status \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --feedback /tmp/control-feedback.jsonl
```

현재 실제 artifact 상태는 `sample_count=1000`, `reviewed_count=0`,
`pending_count=1000`, validation `100`, test `100`, `test_ready=false`,
`ready_for_materialize=false`다. 따라서 다음 판단은 명확하다. test split 100개를
독립적으로 label하고, 전체 queue를 완료한 후에만 materialize와 human-only
training으로 넘어간다. status 명령은 정확도를 대신하지 않으며, 단지 human gate의
외부 상태를 재현 가능하게 관측한다.

### v1.100.0 500-case safety latency replay

v1.96에서 만든 500개 safety matrix를 `control simulate`에 직접 넣어 latency와
fail-closed 동작을 동시에 검증했다.

```bash
uv run hyperjev control simulate \
  --checkpoint /tmp/control-combined-boundary-100ep.pt \
  --scenarios tests/golden/control_safety_500.jsonl \
  --max-p95-ms 5 \
  --max-p99-ms 5 \
  --fail-on-mismatch \
  --output /tmp/control-safety500-latency.json
```

실제 결과는 500/500 accuracy, safe STOP recall 500/500, p95 `0.000960 ms`,
p99 `0.001503 ms`, max `0.022800 ms`였고 5ms p95/p99 gate를 통과했다. 이 수치는
local CPU 순차 replay이므로 DGX Spark 동시성, memory context, teacher fallback을
포함한 production latency 증거로 승격하지 않는다. 다만 safety recall과 bounded
latency를 같은 500-case matrix에서 동시에 회귀 검증할 수 있는 기준선이 생겼다.

### v1.101.0 held-out split review pack

human test gate를 먼저 채울 수 있도록 control review pack 생성기에 `--split`
필터를 추가했다. queue의 synthetic target과 labels는 여전히 pack에서 제외되며,
선택된 split의 raw `state/question`과 teacher draft metadata만 들어간다.

```bash
uv run hyperjev control review-pack \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --draft /tmp/qwen-control-hard-500.jsonl \
  --output /tmp/control-qwen-hard-test-pack.jsonl \
  --allow-raw \
  --prioritize \
  --split test
```

실제 test-only pack은 100 rows, counterfactual group 50개, collision group 17개로
생성됐고 SHA-256은
`78b237c3bb22f5d3ba85f9ed4563e762bf78dee210f2f1272b3d987195869822`이다. 다음은
이 pack을 `--blind`로 검수하는 것이며, test 100개가 모두 label되기 전에는
`--require-human-test`를 통과시키지 않는다. split filter는 정확도를 자동으로
올리지 않지만, train label과 held-out test label이 섞이는 실수를 차단한다.

### v1.102.0 Gemma judge timeout boundary

Gemma endpoint의 control draft 경계를 실제로 확인했다. `/v1/models`는 응답하며
`google/gemma-4-12b` 모델이 노출되지만, control hard-negative 첫 sample에 대해
60초 timeout은 4건 연속 `TimeoutError`를 기록했고, timeout을 300초로 늘린 1건
재시험도 `schema_valid_count=0`, `error_count=1`로 끝났다.

따라서 Gemma를 human label이나 synchronous control fallback으로 승격하지 않는다.
현재 운영 경로는 Qwen draft + blind human review이며, Gemma는 별도 비동기 judge로
재시도할 수 있지만 그 결과가 없거나 늦어도 control loop와 human gate는 진행되어야
한다. 이 결정은 정확도 하락을 의미하는 것이 아니라, timeout을 정확도나 label로
오인하지 않도록 하는 경계다.

### v1.103.0 simulation scenario uniqueness guard

`control simulate`가 입력 matrix의 `scenario_id`를 중복 허용하지 않도록
검증을 추가했다. `--repeat`는 하나의 고유 scenario를 latency sampling 목적으로
반복하는 기능이므로 허용하지만, 서로 다른 입력 행이 같은 `scenario_id`를
재사용하면 CLI가 exit code 2로 거부한다. 이 구분으로 반복 측정과 독립
안전성 표본을 혼동하지 않는다.

실제 500-case safety matrix 재생에서 `count=500`,
`unique_scenario_count=500`, action accuracy `500/500`, safe STOP recall
`500/500`, p95 `0.000976 ms`, p99 `0.001552 ms`를 확인했고 5ms latency gate를
통과했다. 이는 local CPU sequential replay 증거이며, human-labeled control
accuracy나 DGX Spark 동시성 production benchmark를 대체하지 않는다.

### v1.104.0 strict dual-review materialization gate

사람이 검수한 라벨도 한 명의 실수나 teacher anchoring을 포함할 수 있으므로,
production 후보 dataset을 materialize할 때는 독립 reviewer 두 명의 agreement
또는 disagreement adjudication provenance를 요구할 수 있게 했다.

```bash
uv run hyperjev control materialize \
  --input /path/to/control-reviewed.jsonl \
  --output /path/to/control-human-target.jsonl \
  --require-dual-review
```

strict gate는 `review.status=reviewed`, reviewer 식별자, 그리고
`reason`이 `control dual-review agreement` 또는
`control dual-review adjudication`으로 시작하는지 확인한다. 단일 reviewer
feedback은 연구용 기본 materialize에서는 허용되지만 strict production 후보에서는
거부된다. 이 gate는 model accuracy를 자동으로 올리는 장치가 아니라, 잘못된
label이 99% 평가와 학습을 오염시키지 않게 하는 전제조건이다.

### v1.105.0 multilingual Korean control track

English synthetic OOD가 높아도 Korean/다국어 상태 표현에서 같은 성능이 나온다는
보장은 없다. 이를 분리 측정하기 위해 human label과 분리된 40개 Korean held-out
fixture와 train-only 64개 Korean augmentation queue를 추가했다.

```bash
uv run hyperjev control korean \
  --output /tmp/control-korean-training.jsonl --seed 17
uv run hyperjev control merge \
  --input /tmp/hyperjev-control-combined-1928.jsonl \
  --input /tmp/control-korean-training.jsonl \
  --output /tmp/hyperjev-control-combined-korean.jsonl
uv run hyperjev control train \
  --dataset /tmp/hyperjev-control-combined-korean.jsonl \
  --output /tmp/control-combined-korean-bow-balanced.pt \
  --backbone reference-bow-encoder --class-balanced \
  --epochs 100 --batch-size 64 --learning-rate 0.01 \
  --precision fp32 --device cpu --seed 7
```

후보 선택 결과는 다음과 같다.

| 후보 | English model-only OOD | Korean model-only OOD | 판정 |
| --- | ---: | ---: | --- |
| 기존 boundary token checkpoint | `39/40 (97.5%)` | `11/40 (27.5%)` | Korean baseline |
| Korean token augmentation | `36/40 (90%)` | `25/40 (62.5%)` | English regression으로 폐기 |
| Korean token + class-balanced | `40/40 (100%)` | `26/40 (65%)` | Korean 부족으로 폐기 |
| Korean BOW + class-balanced | `39/40 (97.5%)` | `36/40 (90.0%)` | bilingual baseline 후보 |

BOW 후보의 independent synthetic validation/test는 각각 `180/180 (100%)`,
500-case safety replay는 accuracy/STOP recall 모두 `500/500`이었다. Korean
integrated fast path도 `40/40`, STOP recall `5/5`였다. 그러나 human label이
없고 Korean raw model-only OOD가 `36/40 (90.0%)`이므로 production 승격은 보류한다.
다음 단계는 Korean human golden을 dual-review로 확보하고, BOW/typed-head의
DGX Spark GPU latency를 다시 측정하는 것이다.

### v1.106.0 multilingual ablation boundary

v1.105의 선택된 64개 Korean train-only queue와 default BOW vocabulary를
기준으로 데이터 증강과 vocabulary 크기를 독립적으로 시험했다. 실험은 모두
test fixture를 학습 데이터에 섞지 않은 synthetic ablation이며, human golden을
대체하지 않는다.

| 실험 | English OOD | Korean OOD | model-only latency | 판정 |
| --- | ---: | ---: | --- | --- |
| HOLD 8개 추가, BOW + class-balanced | 37/40 (92.5%) | 36/40 (90.0%) | p99 5.248/5.737ms | 정확도 개선 없음, latency 회귀 |
| BOW vocab 8,192 | 39/40 (97.5%) | 34/40 (85.0%) | p99 2.877/2.588ms | Korean 정확도 회귀 |
| BOW vocab 16,384 | 36/40 (90.0%) | 37/40 (92.5%) | English p99 34.157ms outlier | bilingual baseline 회귀 |
| v1.105 BOW vocab 32,768 | 39/40 (97.5%) | 36/40 (90.0%) | p99 4.807/5.091ms 재생 | 현재 bilingual baseline |

추가 HOLD queue의 merged SHA는
`3ef7faf1f12a1944d28e42b12df09830089d59452c72e8223b451f67659d0d73`이고, 각
checkpoint SHA와 실행 결과는 `docs/test_result.md`의 14.83절에 고정했다. 이
결과는 sample 수나 feature vocabulary를 늘리는 것만으로 일반화가 보장되지
않음을 보여준다. 선택 후보는 정확도·안전·latency를 함께 통과해야 하며, 현재
사람 라벨은 `0/1000`, production accuracy는 미측정이다.

다음 실행 순서는 (1) v1.105 synthetic best 고정, (2) blind dual human review로
train/validation/test label 수집, (3) human-only materialize 및 재학습, (4) held-out
test accuracy/Wilson 하한/STOP recall/latency 동시 평가다. human label 없이
threshold를 조정하거나 test 문장을 train에 복사하지 않는다.

### v1.107.0 multilingual raw replay correction and char/hybrid ablation

이전 보고서의 Korean `39/40`은 raw Student와 integrated rule/fast-path 경계를
섞은 기록이었다. 현재 동일 checkpoint를 `evaluate_student_checkpoint`로 raw
재생한 기준은 English `39/40`, Korean `36/40`이다. integrated runtime은 별도
결과로 Korean `40/40`일 수 있지만, 이를 raw model 정확도로 기록하지 않는다.

| backbone | English raw OOD | Korean raw OOD | English raw p95/p99 | Korean raw p95/p99 | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| reference-bow-encoder | 39/40 (97.5%) | 36/40 (90.0%) | 4.807/5.202ms | 4.748/5.091ms | bilingual baseline |
| reference-char-bow-encoder | 32/40 (80.0%) | 38/40 (95.0%) | 4.970/7.842ms | 5.011/5.582ms | English regression, 폐기 |
| reference-hybrid-bow-encoder | 34/40 (85.0%) | 39/40 (97.5%) | 4.689/5.114ms | 5.037/5.787ms | English regression, 폐기 |

char-only checkpoint SHA는
`8c0eb82d077608660da024f37e7d0e940ae794caae95cbec77c223e57e53be79`, hybrid
checkpoint SHA는
`ed263699ba4aaeda322bd35d348a34bce526ce9047e1adb93af725939cedc100`이다. 두
후보 모두 Korean 단독 개선을 보였지만 bilingual 최저 정확도와 latency를 동시에
만족하지 못했다. 다음은 human-labeled Korean/English test에서 같은 비교를
반복하는 것이며, synthetic OOD 결과만으로 promotion하지 않는다.

### v1.108.0 human-review readiness gate

사람 라벨이 없는 synthetic 결과가 production 정확도로 오인되지 않도록 status
명령 자체를 strict gate로 사용할 수 있게 했다.

```bash
uv run hyperjev control review-status \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --feedback /tmp/control-feedback.jsonl \
  --require-test-ready \
  --require-materialize-ready
```

`--require-test-ready`는 test split의 pending이 0일 때만 성공하고,
`--require-materialize-ready`는 전체 queue의 pending이 0일 때만 성공한다.
현재 status는 reviewed `0/1000`, test pending `100`, 전체 pending `1000`이므로
exit code `1`이어야 한다. 이 gate는 human label을 생성하지 않으며, strict
dual-review provenance는 이후 `control materialize --require-dual-review`가
별도로 검증한다.

다음 정확도 상승 단계는 이 gate를 통과할 수 있도록 두 독립 reviewer의 blind
label을 수집하고, disagreement를 adjudicate한 뒤 human-only checkpoint를
재학습하는 것이다.

### v1.109.0 selected BOW safety replay

현재 bilingual baseline인 v1.105 BOW checkpoint를 500개 고유 safety scenario에
재생했다.

| 항목 | 결과 |
| --- | ---: |
| scenario count | `500` |
| unique scenario count | `500` |
| action accuracy | `500/500 (100%)` |
| expected safe STOP | `500` |
| safe STOP recall | `500/500 (100%)` |
| latency p50/p95/p99/max | `0.000864/0.000960/0.001440/0.008704ms` |
| p95/p99 5ms gate | 통과 |

이 결과는 deterministic safety rule/interlock과 local CPU sequential latency를
검증한다. raw Student multilingual OOD와 human test accuracy는 별도 gate이며,
human label은 여전히 `0/1000`이다. DGX Spark concurrent GPU load test와 실제
robot/game closed-loop test는 아직 수행하지 않았다.
