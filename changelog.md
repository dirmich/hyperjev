# HyperJev changelog

이 파일은 정확도·coverage·fallback 정책의 중간 결과를 기록한다. 숫자는
동일한 dataset/split과 실행 명령으로 재현할 수 있는 경우에만 갱신한다.

## 2026-09-21 — held-out split review pack (v1.101.0)

- `control review-pack --split train|validation|test`를 추가했다.
- 실제 hard-negative queue에서 target-free test-only pack 100개를 생성했다.
- pack SHA-256은 `78b237c3bb22f5d3ba85f9ed4563e762bf78dee210f2f1272b3d987195869822`이며,
  test pending은 review-status와 동일하게 `100`이다.
- split 분리는 human label을 자동 생성하지 않으며, test-first blind review를
  통해 held-out gate를 채우기 위한 운영 기반이다.

## 2026-09-21 — 500-case safety latency replay (v1.100.0)

- 500개 safety matrix를 `control simulate`로 실제 replay했다.
- 결과는 accuracy `500/500`, safe STOP recall `500/500`, p95 `0.000960 ms`,
  p99 `0.001503 ms`, max `0.022800 ms`였다.
- local CPU sequential replay가 5ms p95/p99 gate를 통과했으며, DGX 동시성
  production benchmark와는 구분해 기록했다.

## 2026-09-21 — review progress and held-out readiness status (v1.99.0)

- `control review-status`를 추가해 feedback queue의 전체/분할별 coverage,
  action별 human count, `test_ready`, `ready_for_materialize`를 출력한다.
- 현재 실제 hard-negative queue 1,000개는 `reviewed=0`, `pending=1000`,
  validation pending `100`, test pending `100`이다.
- 이 명령은 synthetic target과 teacher prediction을 읽어 정답으로 만들지 않으며,
  human-label gate의 외부 상태만 검증한다.

## 2026-09-21 — verify control batch handler wiring (v1.98.0)

- v1.97에서 추가한 `--offset`/`--limit`이 control handler에 실제 전달되도록
  연결을 교정했다. generic `golden review-session`은 기존 동작을 유지한다.
- 실제 1,000-item control pack에서 `--blind --offset 0 --limit 50`을 실행해
  report의 `batch_count=50`, `batch_pending_count=50`, 전체 `pending_count=1000`을
  확인했다. `q` 종료에서는 feedback 파일이 생성되지 않는다.
- handler forwarding regression test를 추가했다. 이는 human label이나 정확도를
  생성하지 않는 CLI wiring 검증이다.

## 2026-09-21 — bounded blind human-review batches (v1.97.0)

- `control review-session`에 결정론적 `--offset`/`--limit` 배치를 추가했다.
- `--blind`와 함께 사용하면 Qwen/Gemma draft를 숨긴 채 25~50건 단위로
  사람 label을 입력하고, feedback append-only 파일로 중단 후 재개할 수 있다.
- 결과 report에 전체 progress와 batch progress를 함께 기록해 다음 배치 위치를
  잃지 않는다. synthetic target이나 teacher output을 `labels.human`으로 복사하는
  경로는 추가하지 않았다.
- targeted review-pack tests `23 passed`; 실제 human label 수와 control 정확도는
  여전히 0/미측정이며, 이 기능은 라벨 수집 병목을 줄이는 기반이다.

## 2026-09-21 — reproducible 500-case safety matrix (v1.96.0)

- `scripts/generate_control_safety_scenarios.py`를 추가해 emergency stop,
  stale observation, invalid clock을 포함하는 500개 고유 safety scenario를
  deterministic JSONL fixture로 생성한다.
- evaluator가 모든 scenario의 non-empty unique `scenario_id`를 검증하므로
  같은 episode를 복제해 sample-count gate를 우회할 수 없다.
- `/tmp/control-combined-boundary-100ep.pt`로 500/500 safety accuracy와
  safe STOP recall을 기록했고 Wilson 95% 하한은 `0.992376`이었다.
- 이 결과는 safety interlock 증거이며 control 정확도나 human-label gate를
  대체하지 않는다. 현재 `--require-human-test`는 여전히 test human label
  부재로 실패하고 production-ready는 false다.

## 2026-09-21 — minimum safety STOP evidence gate (v1.95.0)

- control quality evaluator에 `--minimum-safe-stop-count`를 추가하고 기본값을
  `500`으로 설정했다.
- expected safe STOP sample이 부족하면 recall 100%여도 `safe_stop_sample_count`
  failure로 production gate를 통과하지 못한다.
- 작은 fixture의 100% point score를 99% confidence evidence로 오해하지 않도록
  하는 안전 검증 강화다.

## 2026-09-21 — bounded control simulation latency gate (v1.94.0)

- `control simulate`가 p50/p95/p99/max latency를 기록한다.
- `--max-p95-ms`와 `--max-p99-ms`를 지정하면 threshold 초과 시 exit `1`로
  실패해 bounded-latency 조건을 자동 검증한다.
- 현재 4-scenario local CPU replay는 정확도 `4/4`, safe STOP recall `2/2`,
  p95 `0.066515ms`, p99 `0.066515ms`로 5ms gate를 통과했다.
- 이 결과는 4개 fixture와 local CPU 관측치이며 DGX 동시성 production benchmark는
  별도 gate로 남는다.

## 2026-09-21 — human-only control training gate (v1.93.0)

- `control train --require-human-labels`를 연결해 synthetic control target 학습을
  production 경로에서 명시적으로 차단한다.
- control sample은 typed `labels.human`과 `provenance.target_source=human_review`를
  모두 검증하며, 유효하지 않으면 checkpoint를 만들지 않는다.
- synthetic 연구 학습은 기존처럼 flag 없이 가능하지만 production 정확도 근거로
  승격하지 않는다.
- targeted training 테스트 `11 passed, 1 skipped`; 전체 suite는 version bump 후
  다시 실행한다.

## 2026-09-21 — target-exclusion guard for automatic focus (v1.92.0)

- manifest-driven focus가 `target_excluded=true`를 명시한 dual-review manifest만
  받도록 검증한다.
- target이 포함될 수 있는 잘못된 manifest는 action priority 계산 전에 거부한다.
- targeted review-pack 테스트 `21 passed`; 전체 suite와 Ruff를 version bump 후
  다시 실행한다.

## 2026-09-21 — manifest-driven action focus (v1.91.0)

- `review-pack --agreement-manifest`가 dual-review manifest의 action별 agreement
  rate를 읽어 threshold 미만 action을 자동 focus한다.
- 명시적 `--focus-action`과 manifest-driven action은 합집합으로 적용하며, pack
  전체와 sibling adjacency를 유지한다.
- target/teacher를 정답으로 승격하지 않고 review ordering에만 사용한다.
- targeted review-pack 테스트 `21 passed`; 전체 suite와 Ruff를 version bump 후
  다시 실행한다.

## 2026-09-21 — action focus review ordering (v1.90.0)

- `control review-pack --focus-action ACTION --prioritize`를 추가해 낮은
  action agreement 경계를 먼저 배치한다. 여러 action은 flag를 반복한다.
- 전체 review item과 counterfactual sibling adjacency는 유지하며, manifest에
  focus action과 실제 teacher-selected item 수를 기록한다.
- focus는 review 순서일 뿐 teacher를 정답으로 승격하지 않으며, 전체 held-out
  human test와 production gate를 대체하지 않는다.
- targeted review-pack 테스트 `19 passed`; 전체 suite와 Ruff를 version bump 후
  다시 실행한다.

## 2026-09-21 — action agreement CLI 요약 (v1.89.0)

- `control review-agreement --show-action-summary`를 추가해 action별 agreement
  rate와 disagreement 수를 낮은 agreement 순서로 터미널에 출력한다.
- 기존 manifest JSON 출력은 그대로 유지해 자동화 호환성을 보존한다.
- 요약에는 synthetic target과 teacher output을 포함하지 않으며 accuracy가 아닌
  human review 우선순위 진단으로 명시한다.
- targeted review-pack 테스트 `17 passed`; 전체 suite와 Ruff를 version bump 후
  다시 실행한다.

## 2026-09-21 — action별 dual-review agreement 진단 (v1.88.0)

- `control review-agreement` manifest에 `label_agreement_by_action`을 추가해
  reviewer가 선택한 action별 comparable/agreement/disagreement 수와 agreement
  rate를 기록한다.
- 이 지표는 synthetic target이나 teacher를 정답으로 사용하지 않는 blind review
  집중도 진단이며, accuracy나 production gate로 해석하지 않는다.
- 실제 human 라벨이 없는 상태의 production gate는 여전히 false다.
- targeted review-pack 테스트 `15 passed`; 전체 suite와 Ruff를 version bump 후
  다시 실행한다.

## 2026-09-21 — held-out human test gate (v1.84.0)

- `evaluate_control_quality.py --require-human-test`를 추가해 held-out test의
  모든 row가 typed human label을 갖지 않으면 실패시킨다.
- `human_label_status.by_split`, `human_test_gate`, `human_label_gate`를
  분리해 validation synthetic 점수와 human test 점수를 혼동하지 않는다.
- 현재 human label이 `0`인 synthetic dataset에서는 gate가 실패하는 것이
  정상이며, human review 후 test group을 training에서 제외한 checkpoint를
다시 평가해야 한다.

## 2026-09-21 — per-skill accepted diagnostics (v1.85.0)

- control skill별 raw/accepted accuracy, accepted coverage, Wilson interval을
  report한다.
- confusion matrix를 함께 유지해 `HOLD → STOP`, `APPROACH → MOVE` 같은 경계를
  human test에서 바로 추적할 수 있게 했다.
- targeted evaluator tests `5 passed`; human label이 없으므로 현재 지표는
synthetic 연구용이고 production gate는 여전히 false다.

## 2026-09-21 — Student uncertainty active review (v1.86.0)

- control review pack이 optional Student checkpoint의 confidence를 reviewer에게
  노출하지 않고 순서에만 사용한다.
- counterfactual group adjacency와 teacher collision priority를 유지하면서
  Student uncertainty를 함께 정렬한다.
- 실제 1,000행 replay에서 target/labels leakage가 없고 pack item에 Student
  prediction이 복사되지 않음을 확인했다. confidence `<0.90`은 `0`개였다.

## 2026-09-21 — Student–teacher disagreement active review (v1.87.0)

- Student confidence가 높아도 Qwen draft와 typed prediction이 다르면 reviewer
  화면에 노출하지 않고 최우선 review 순서로 올린다.
- hard-negative 1,000행 replay에서 disagreement `171`건을 검출했고,
  target/labels leakage와 prediction 복사를 확인하지 못했다.
- disagreement는 teacher truth가 아니라 overconfident error 후보이며,
  adjudicated human label만 학습 target으로 승격한다.

## 2026-09-21 — raw/accepted/safety evaluator hardening (v1.83.0)

- control quality report에 `raw_student_head`, `safety_policy`,
  `gate_failures`를 추가해 Student 정확도와 rule/safety 경로를 분리했다.
- accepted accuracy와 accepted coverage를 별도 gate로 추가했다.
- raw/safety 정확도와 STOP recall에 Wilson 95% interval을 기록하고, STOP
  recall 하한 기준을 만족하지 못하면 evaluator가 실패하도록 했다.
- 최신 synthetic checkpoint는 validation/test `180/180`과 coverage `100%`지만
  STOP `2/2` 하한 `0.342380`으로 exit `1`이다. human benchmark 전에는
  production ready로 승격하지 않는다.

## 2026-09-21 — teacher-blind control dual review gate (v1.82.0)

- `docs/control_labeling.md`에 STOP 우선순위, 8개 control skill 판정 기준,
  independent review와 표본 계획을 고정했다.
- `control review-session --blind`는 Qwen/Gemma draft를 숨기고 reviewer가
  `STOP`, `HOLD` 같은 typed value를 직접 입력하게 한다.
- `control review-agreement`는 두 feedback stream의 agreement/disagreement와
  target 제외 manifest를 만들며, `review-adjudicate`와 `review-finalize`를
  거치지 않은 불일치는 human target으로 승격되지 않는다.
- targeted review-pack 테스트 `12 passed`; 실제 control human label은 아직
  `0`이므로 synthetic 정확도를 production 99%로 주장하지 않는다.

## 2026-09-21 — Qwen hard-negative human review pack (v1.81.0)

- Qwen `qwen38fn` draft가 있는 hard-negative 500쌍/1,000행을 review pack으로
  만들었다. pack은 1,001 lines이며 queue/draft SHA를 manifest에 고정했다.
- `uncertain_first` 정렬 결과 counterfactual group 500개, collision group
  167개/334 items를 추적한다.
- raw state/question은 local review를 위해 포함하지만 synthetic target과 queue
  labels는 제외했다. reviewer는 `control review-session`에서 typed value를
  직접 선택한다.
- human labels는 `0/1,000`; Qwen/Gemma draft는 human evidence가 아니므로
  production gate는 여전히 false다.
- 검증: review-pack manifest, target/labels leakage scan, existing full suite와
  Ruff (v1.80 evidence).

## 2026-09-21 — normal-action precision fast path (v1.80.0)

- `target is close and directly ahead` compound signal을 APPROACH fast path에
  추가했다. v1.79 checkpoint 재생에서 safety scenario action accuracy는
  `3/4 (75%) → 4/4 (100%)`, safe STOP recall은 `2/2 (100%)`를 유지했다.
- v1.79 best model-only OOD `39/40 (97.5%)`보다 나은 학습 seed는 확인하지
  못했다. 1,936-row 확장은 `95%`, seed 42는 `87.5%`, class-balanced는
  `97.5%`였고 모두 승격하지 않았다.
- integrated OOD는 `40/40 (100%)`, STOP recall `5/5 (100%)`를 유지한다.
  human label `0`이므로 production-ready가 아니다.
- 검증: control/data targeted tests, full suite, Ruff, dataset validator,
  quality evaluator, OOD replay, diff check.

## 2026-09-21 — semantic boundary augmentation (v1.79.0)

- 모든 control skill에 boundary 문장 8개씩, 총 64개를 train-only로 추가했다.
  merged 1,928개 dataset SHA는
  `813313e746536aa1ce4f72634c66472858c3b9d5d65463112b8f4f5a2c1ebbce`다.
- held-out OOD model-only가 `39/40 (97.5%)`, STOP recall `5/5 (100%)`로
  v1.78의 `95%`에서 개선됐다. 남은 오류는 HOLD를 STOP으로 보수 처리한 1건이다.
- integrated runtime은 `40/40 (100%)`, STOP recall `5/5 (100%)`를 유지했다.
  일반 safety scenario action accuracy는 `3/4 (75%)`여서 정상 action precision
  gate는 아직 실패다.
- human labels는 `0`이며, synthetic OOD와 reference CPU checkpoint 결과를
  production 정확도로 승격하지 않는다.
- 검증: targeted control/data tests, dataset validator, OOD replay, quality
  evaluator, Ruff. quality evaluator는 safety action precision gate 때문에 실패를
  반환했으며 이 실패를 다음 단계 입력으로 보존한다.

## 2026-09-21 — compositional augmentation과 safety interlock (v1.78.0)

- held-out OOD fixture를 train에 복사하지 않고, skill별 8개씩 64개의 train-only
  compositional 문장을 생성했다. 기존 queue와 합친 1,864개 dataset의 SHA는
  `78b54a46f5bd9d5049b7d3ea903c944854c87882a842ddcd87d67cab5f0f7c9b`이다.
- reference-token-encoder 100 epoch checkpoint의 validation/test는 모두
  `180/180 (100%)`였고, held-out OOD model-only는 `38/40 (95%)`, STOP recall은
  `5/5 (100%)`였다. v1.77 model-only `10%` 대비 개선됐지만 human label은 없다.
- integrated fast path + Student + safety는 OOD `40/40 (100%)`, STOP recall
  `5/5 (100%)`, p95 `25.233µs`였다. deterministic rule 포함 결과를 model-only
  정확도로 부르지 않는다.
- `enable_fast_path=False`에서도 explicit collision STOP safety interlock을
  유지하도록 수정하고 회귀 테스트를 추가했다.
- 검증: targeted control/data tests `32 passed, 39 subtests`, OOD replay,
  quality evaluator, Ruff. 전체 suite와 push 전 최종 검증을 별도로 수행한다.

## 2026-09-21 — compositional OOD model-only gate (v1.77.0)

- 40개 unique group의 hand-authored compositional OOD fixture를 추가했다.
  validator는 cross-split leakage `0`과 human labels `0/40`을 확인했다.
- reference Student model-only replay는 정확도 `4/40 (10%)`, STOP recall `0/5`였다.
  deterministic fast path를 포함한 integrated runtime은 정확도 `40/40 (100%)`,
  STOP recall `5/5 (100%)`, runtime p95 `22.480µs`였다.
- integrated 100%를 model accuracy로 승격하지 않는다. 이는 규칙/모델/safety
  경계를 분리해 측정하는 synthetic gate이며, 다음 단계는 human golden review다.
- 검증: 전체 suite, OOD validator, model-only/integrated replay, Ruff, diff check.

## 2026-09-21 — source-specific runtime latency (v1.76.0)

- runtime fast-path evaluator에 source별 latency 통계를 추가했다. aggregate만
  보면 Student fallback이 숨겨지는 문제를 막기 위해 `control-rule`, `safety-rule`,
  `hyperjev-control`을 각각 p50/p95/p99/max/mean으로 기록한다.
- seed 800 replay에서 Student fallback 50건의 p50/p95/p99는
  `293.486/421.518/2972.318µs`, control-rule p95는 `14.112µs`, safety-rule
  p95는 `3.984µs`였다.
- synthetic accuracy/STOP recall은 `100%`였지만 human label은 `0/800`이고
  production gate는 false다. 이 수치는 reference CPU Student replay다.
- 검증: 전체 suite, source latency unit test, Ruff, seed runtime replay.

## 2026-09-21 — integrated control runtime replay (v1.75.0)

- fast-path evaluator에 `--checkpoint` runtime replay를 추가해 실제
  `ControlStudentClient`의 rule → Student → safety ordering을 재생한다.
- hard-negative 1,000행에서 source `control-rule=750`, `safety-rule=250`,
  synthetic accuracy `100%`, STOP recall `100%`, runtime p50/p95/p99
  `13.648/15.616/17.504µs`를 확인했다.
- 이번 queue는 known phrase fast path가 전부 처리했으므로 model fallback latency나
  OOD generalization 증거가 아니다. human label `0/1000`, production `false`다.
- 검증: 전체 `133 passed, 1 skipped`, Ruff, diff check, checkpoint-backed replay.

## 2026-09-21 — reproducible control fast-path evaluator (v1.74.0)

- `scripts/evaluate_control_fast_path.py`를 추가해 queue SHA, resolved coverage,
  source count, p50/p95/p99/max/mean latency, synthetic target, human target을
  한 report로 재현한다.
- hard-negative 1,000행 report는 coverage `100%`, synthetic match `100%`,
  latency p50/p95/p99 `13.488/15.408/17.2µs`였다.
- human label `0/1000`, `human_label_gate=false`, `production_ready=false`를
  report에 고정했다. `--require-full-coverage`는 unresolved row를 실패시킨다.
- 검증: 전체 `132 passed, 1 skipped`, Ruff, diff check, 실제 hard-negative
  evaluator 실행.

## 2026-09-21 — control fast-path contradiction blockers (v1.73.0)

- fast path의 substring 오탐을 줄이기 위해 skill별 contradiction blocker를
  추가했다. `no hazard is present`가 `hazard is present`로 잘못 매칭되지 않도록
  blocker를 구체 표현으로 제한했다.
- “target은 앞에 있지만 안전하게 접근할 수 없다”, “안정 자세지만 emergency
  hazard가 있다” 같은 반례는 fast path에서 거부하고 model/safety 경로로 보낸다.
- synthetic coverage/target 결과는 v1.72.0과 동일하며, human label은 `0/1000`이다.
  이 단계는 정확도 상승 증거가 아니라 control false-positive 방어 단계다.
- 검증: control unit `16 passed`, targeted Ruff, diff check.

## 2026-09-21 — compound-phrase control fast path (v1.72.0)

- Qwen hard-negative에서 반복된 `APPROACH↔MOVE`, `HOLD↔STOP` semantic collision을
  보완하기 위해 compound phrase 기반의 보수적 `control-rule` fast path를 추가했다.
- hard-negative 1,000행에서 fast-path coverage와 synthetic target match가 각각
  `100.00%`였고, `ControlStudentClient` 평균 in-process 비용은 약 `10.11µs/행`이었다.
- seed queue에서는 `750/800`행만 규칙으로 결정하고 나머지는 model 경로로 남겼다.
  규칙이 둘 이상 맞는 문장은 적용하지 않으며, explicit STOP은 별도 safety rule로
  먼저 평가한다.
- human label은 `0/1000`이므로 production accuracy나 일반화 정확도는 여전히
  미달이다. 이 단계는 저지연 보조 경로이며 human golden gate를 대체하지 않는다.
- 검증: control unit `16 passed`, Ruff, hard-negative/seed synthetic replay,
  실제 `ControlStudentClient` smoke.

## 2026-09-21 — Gemma hard-negative judge timeout gate (v1.71.0)

- 현재 hard-negative queue의 첫 샘플을 `google/gemma-4-12b`로 30초 probe했다.
- 결과는 `completed=0`, `schema_valid=0`, `error_count=1`,
  `TimeoutError: timed out`이었다.
- Gemma는 control tick이나 1,000건 synchronous labeler로 사용하지 않고, 별도
  async judge/retry worker로 제한한다. 현재 정확도 gate의 human label은 Qwen draft를
  참고한 독립 human review로 생성한다.
- 검증: Gemma probe manifest와 기존 전체 suite `128 passed, 1 skipped` 확인.

## 2026-09-21 — hard-negative weighted-loss ablation (v1.70.0)

- `TrainingConfig.hard_negative_weight`와 `control train --hard-negative-weight`를
  추가했다. `source.counterfactual_group_id`가 있는 sample만 per-example loss를
  가중하며 기본값 `1.0`은 기존 동작과 같다.
- combined 1,800건에서 weight `2.0`과 `4.0`을 각각 100 epoch 실행했다. 두 후보는
  synthetic validation/test `100%`와 STOP recall `100%`였지만 safety action
  accuracy가 `75%`로 baseline `100%`보다 나빴다.
- 따라서 이 기능은 human hard-negative가 materialize된 뒤 재실험할 연구 옵션으로
  남기고 production checkpoint로 승격하지 않았다.
- 검증: trainer 테스트 8개, 전체 suite, Ruff, diff check와 두 checkpoint quality gate.

## 2026-09-21 — review priority statistics manifest (v1.69.0)

- prioritized review pack manifest에 `counterfactual_group_count`,
  `collision_group_count`, `collision_item_count`를 기록한다.
- operator가 JSONL 순서를 다시 분석하지 않아도 active-review 범위와 검수량을
  확인할 수 있다. 기본 queue order에서는 `priority_stats=null`을 유지한다.
- 실제 Qwen hard-negative 결과는 500 groups, 167 collision groups, 334 items다.
- 검증: review 테스트 10개, 실제 CLI manifest, Ruff, diff check 통과.

## 2026-09-21 — pair-collision priority for overconfident teacher errors (v1.68.0)

- confidence만으로 정렬하면 Qwen hard-negative의 HOLD 오답 83건이 confidence
  `1.0`으로 뒤로 밀리는 문제가 확인됐다.
- 같은 counterfactual pair의 두 typed teacher 결과가 동일하면 pair collision으로
  간주해 priority 최상단에 함께 배치한다. 이 규칙은 target을 읽지 않는다.
- 실제 Qwen hard-negative 500 pair에서 collision `167`개가 잡혔고, 모두 pack의
  앞쪽에 pair 단위로 연속 배치됐다.
- 검증: collision priority test 포함 review 테스트 10개, 실데이터 CLI smoke,
  Ruff, diff check 통과.

## 2026-09-21 — generic golden review CLI parity (v1.67.0)

- `golden review-pack`에도 `--prioritize` parser flag를 추가해 control과
  generic review 경로의 public CLI 계약을 일치시켰다.
- v1.66.0에서 handler는 이미 option을 읽고 있었지만 generic parser가 누락된
  것을 CLI surface 점검으로 발견했다.
- 검증: control/golden parser review 테스트 9개, 전체 suite, Ruff, diff check.

## 2026-09-21 — wire pair-aware priority through control CLI (v1.66.0)

- `control review-pack --prioritize`가 실제 `export_review_pack` 호출까지
  priority 옵션을 전달하도록 연결했다.
- v1.65.0 helper/unit 경로는 통과했지만 CLI handler가 옵션을 누락한 것을 실제
  1,000-sample smoke에서 발견해 수정했다.
- 실데이터 검증 결과 `priority_order=uncertain_first`, 500개 pair adjacency
  violation `0`, target/labels leakage `false`다.
- 검증: parser 포함 review 테스트 8개, 실제 CLI pack 생성, Ruff, diff check 통과.

## 2026-09-21 — counterfactual pair-aware review ordering (v1.65.0)

- review pack이 hard-negative의 `counterfactual_group_id`, `pair_side` 등
  target이 아닌 provenance만 보존하도록 했다.
- `--prioritize`를 사용하면 pair의 가장 불확실한 항목을 기준으로 group 전체를
  정렬하고, 같은 pair의 두 항목은 연속해서 검수된다.
- source provenance에는 target/labels를 복사하지 않아 review pack의 target
  leakage 경계를 유지한다.
- 검증: pair 인접성·target 비노출 review 테스트 7개, Ruff, diff check 통과.

## 2026-09-21 — uncertainty-first human review queue (v1.64.0)

- `control review-pack --prioritize`를 추가해 invalid schema, repaired teacher
  output, low-confidence typed result을 queue 앞쪽으로 정렬한다.
- choice는 최대 probability, boolean은 선택값의 probability, score는 interval
  폭으로 confidence를 계산하며 synthetic target은 정렬에 사용하지 않는다.
- 기본값은 기존 queue order를 유지하고, manifest에 `priority_order`를 남긴다.
- 목적은 1,000건을 무작정 순서대로 검수하지 않고 semantic confusion과 teacher
  repair가 발생한 row를 먼저 human label하는 것이다. 이 기능만으로 accuracy가
  상승하거나 human gate가 통과하는 것은 아니며, 현재 human label은 여전히 0이다.
- 검증: review-pack priority 단위 테스트 6개, Ruff, diff check 통과.

## 2026-09-21 — Qwen hard-negative semantic gate (v1.63.0)

- qwen38fn hard-negative 500 pair/1,000 sample draft를 완료했다.
- schema-valid `1000/1000`, synthetic target match `829/1000 (82.90%)`,
  pair 양쪽 정답 `329/500 (65.80%)`였다.
- STOP `250/250 (100%)`는 유지됐지만 APPROACH `83/167 (49.70%)`, HOLD
  `0/83 (0%)`로 semantic confusion이 드러났다. RETREAT은 `80/84 (95.24%)`였다.
- latency p50/p95/p99/max는 `2124.485/2300.150/2376.304/2476.982ms`이며,
  human `0/1000`, `production_ready=false`다. 이 결과는 Qwen pseudo-label을
  그대로 학습하지 말고 HOLD/APPROACH를 human priority로 보내는 gate다.
- 검증: hard queue review pack 생성, quality evaluator report와 pair confusion 계산.

## 2026-09-21 — teacher draft quality evaluator (v1.62.0)

- `scripts/evaluate_control_teacher_draft.py`가 queue/draft SHA, completed/schema
  coverage, synthetic target accuracy, per-skill metrics, repaired count,
  latency p50/p95/p99/max를 재계산한다.
- Qwen seed 800 report는 schema-valid `800/800`, synthetic accuracy `800/800`,
  repair `2`, human `0`, `production_ready=false`였다.
- evaluator는 teacher synthetic 결과와 human production gate를 분리해, teacher
  100%를 human 정확도로 잘못 승격하지 못하게 한다.
- 검증: 전체 suite `121 passed, 1 skipped`, Ruff 통과.

## 2026-09-21 — Qwen control draft completion and bounded repair (v1.61.0)

- qwen38fn 800건 resumable control draft를 완료했다: `800/800 completed`,
  `800/800 schema-valid`, synthetic target match `800/800`, task별 100%.
- 2건의 choice probability 합 `0.95`는 bounded normalization 후에도
  `schema_repaired=2`로 provenance를 남겼다. 외부 typed contract 자체는 느슨하게
  바꾸지 않았다.
- latency는 p50 `2188.502ms`, p95 `2357.121ms`, max `2596.726ms`로 motor
  loop에 사용하지 않고 Qwen 비동기 labeler/fallback으로 제한한다.
- 검증: invalid teacher record 재시도/repair unit을 포함한 전체 suite
  `120 passed, 1 skipped`, Ruff 통과.

## 2026-09-21 — runtime calibration binding (v1.60.0)

- `StudentClient`가 control calibration manifest를 로드해 task별 temperature를
  실제 boolean/choice 확률과 abstain 판단에 적용한다.
- checkpoint SHA-256 mismatch, 잘못된 manifest type, 비정상 temperature는
  runtime 시작 시 거부한다. 기본 경로는 기존과 동일하게 uncalibrated다.
- `control decide --calibration`과 `HYPERJEV_STUDENT_CALIBRATION`/Phase 0
  `model.calibration` 설정을 추가했다.
- 검증: 전체 `118 passed, 1 skipped`, Ruff 통과, matching calibration unit
  test에서 calibrated confidence가 원래 confidence보다 과도하게 커지지 않음을 확인했다.

## 2026-09-21 — held-out control calibration manifest (v1.59.0)

- `scripts/calibrate_control_checkpoint.py`가 held-out Student logits에서
  task별 temperature를 fit하고 checkpoint/dataset SHA, split, human count를
  manifest로 저장한다.
- combined validation 180건은 temperature `0.25`(검색 하한), NLL
  `0.000007 → 0.000000`을 기록했지만 synthetic target이며 human label `0/180`이다.
- calibration은 argmax를 바꾸지 않고 confidence/abstain 해석만 교정한다.
  경계 temperature와 human gate 실패 때문에 `production_eligible=false`로
  유지한다.
- 검증: 전체 `117 passed, 1 skipped`, Ruff 통과, 실제 checkpoint calibration
  실행에서 비생산 상태를 의도대로 exit 1로 확인했다.

## 2026-09-21 — control risk-coverage gate (v1.58.0)

- `threshold_risk_coverage`와 `evaluate_control_quality --risk-thresholds`를
  추가해 confidence threshold별 accepted count, coverage, accepted accuracy,
  accepted risk를 report에 고정했다.
- combined token checkpoint는 validation/test의 `0.50/0.90/0.95/0.99` 모든
  threshold에서 `180/180`, accepted accuracy `100%`, risk `0%`였다.
- class-balanced 후보도 같은 synthetic risk-coverage를 보였지만 safety action
  accuracy는 `3/4 (75%)`였다. 따라서 confidence 숫자만으로 production 정확도를
  주장할 수 없고, 독립 human test와 safety gate가 계속 필수다.
- 검증: 새 risk-coverage unit 2개, targeted metric 5개, evaluator 실 checkpoint
  실행이 통과했다.

## 2026-09-21 — hard-negative token training evidence (v1.50.0)

- v1.49.0의 500 pair/1,000 sample counterfactual queue를
  `reference-token-encoder`로 100 epoch 학습했다.
- validation/test는 각각 `100/100 (100.00%)`, safety action accuracy는
  `4/4 (100.00%)`, safe STOP recall은 `2/2 (100.00%)`였다.
- human label은 `0/200`이라 evaluator의 synthetic gate만 통과했고,
  `production_ready=false`다. 제한된 synthetic pair를 외운 결과와 실제
  scene 일반화를 혼동하지 않도록 다음 단계는 seed+hard-negative 결합과
  human independent test다.

## 2026-09-21 — human target materialization (v1.51.0)

- `hyperjev control materialize`가 reviewed control queue의 typed
  `labels.human`을 검증하고 synthetic `target`을 human target으로 교체한다.
- 원본 queue는 보존하고, materialized row에 `target_source=human_review`와
  이전 target을 기록한다. 미검수/abstain/invalid choice는 기본 거부한다.
- 8개 temporary reviewed row smoke에서 `human_labeled_count=8`과
  `target_source=human_review`를 확인했다. 이는 데이터 경계 검증이지 실제
  일반화 정확도 결과가 아니다.

## 2026-09-21 — combined control queue and explicit STOP safety rule (v1.52.0)

- seed 800개와 hard-negative 1,000개를 `control merge`로 결합하는 재검증
  경로를 추가했다. 결합 dataset은 1,800개, split `1440/180/180`이다.
- combined synthetic checkpoint는 split 정확도 100%였지만 model-only safety
  action accuracy가 75%로 실패했다. 이 실패를 그대로 gate에 남겼다.
- `obstacle is directly ahead`와 즉시 충돌 문구를 model보다 먼저 처리하는
  explicit STOP safety rule 후 safety action accuracy `4/4`, STOP recall
  `2/2`가 됐다. human label은 0이라 production-ready는 아니다.

## 2026-09-21 — control human review CLI (v1.53.0)

- `control review-pack`, `control review-session`, `control apply-feedback`를
  추가해 control registry로 state/question을 검수한다.
- session은 accept, value-only edit, next/previous, skip, save-and-quit을
  지원하며 feedback은 append-only로 남는다. apply 후 materialize해야 human
  target training dataset이 된다.
- 8개 temporary row smoke에서 pack 8 items, reviewed 8/8, feedback 8건,
  apply `ready=true`를 확인했다. 실제 golden accuracy는 아직 산정하지 않았다.

## 2026-09-21 — Qwen/Gemma typed adjudication (v1.54.0)

- `control adjudicate`가 동일 queue의 Qwen/Gemma draft를 schema와 typed value로
  비교한다. 합의만 silver 후보가 되고, disagreement/invalid/abstain은
  normalized result를 비워 human review로 보낸다.
- 8개 synthetic row smoke에서 agreement `7/8 (87.50%)`, disagreement
  `1/8 (12.50%)`였다. 두 teacher 결과와 latency는 review pack에 표시되지만
  human label로 자동 승격되지 않는다.

## 2026-09-21 — per-skill control accuracy gate (v1.55.0)

- control evaluator가 validation/test별 skill count, correct, accuracy와
  confusion matrix를 기록한다.
- 기본 per-skill threshold는 99%이며 sample이 없는 skill도 실패한다. combined
  synthetic checkpoint는 8개 skill 모두 validation/test 100%였지만 human
  label 0이라 production gate는 여전히 별도다.

## 2026-09-21 — resumable teacher draft (v1.56.0)

- `control draft --resume`가 manifest/queue SHA/provider를 확인하고 완료된
  sample을 skip하며, 각 teacher record를 즉시 flush한다.
- 2개 queue를 1개 처리 후 재개하는 unit test에서 호출 2회와 최종 2 row를
  확인했다.
- 800개 Qwen 실측은 llama-server `-np 1`의 다른 Node client 점유로 8분 이상
  대기해 중단했다. output은 없으며, 이는 정확도 실패가 아니라 shared-slot
  운영 문제다.

## 2026-09-21 — class-balanced control training ablation (v1.57.0)

- `control train --class-balanced`가 train target별 deterministic oversampling을
  수행하도록 추가했다. validation/test input은 변경하지 않는다.
- combined synthetic checkpoint는 split/per-skill 100%였지만 safety action
  accuracy `3/4 (75%)`로 실패했다. normal approach confidence가 control
  threshold 아래여서 safety STOP으로 전환된 결과다.
- threshold를 낮춰 통과시키지 않고 후보를 production에서 탈락시켰으며,
  다음은 calibration/risk-coverage 검증이다.

## 2026-09-21 — control counterfactual hard-negative queue (v1.49.0)

- `hyperjev control hard-negative`가 STOP↔RETREAT, MOVE↔APPROACH,
  HOLD↔STOP 등 혼동쌍을 생성한다.
- 한 pair의 두 row는 같은 `episode_id`, `semantic_group_id`, split을 공유하고
  `counterfactual_group_id`와 pair side를 기록한다. validator 결과는
  24 samples/12 groups에서 cross-split leak 0건이었다.
- target은 synthetic seed라 human label을 대체하지 않는다. human review와
  Gemma disagreement 검수를 거친 뒤에만 class-balanced training bucket으로
  편입한다.

## 2026-09-21 — reference BOW encoder ablation (v1.48.0)

- token count를 직접 linear projection하는 `reference-bow-encoder`를 추가해
  token/ngram 후보와 비교했다. typed head와 checkpoint contract는 동일하다.
- 48개 control smoke에서 validation은 `4/8 (50.00%)`로 token/ngram과 같았고,
  개선되지 않았다. test는 validation으로 선택한 뒤에만 확인해야 하므로 이
  단계에서는 승격 후보로 사용하지 않는다.
- 결과는 architecture보다 클래스당 독립 semantic group과 hard-negative가
  부족한 것이 현재 병목임을 보여준다. BOW 경로는 추후 human dataset에서
  다시 비교할 수 있도록 유지한다.

## 2026-09-21 — control Qwen/Gemma draft routing (v1.47.0)

- `hyperjev control draft`가 control registry queue를 Qwen 또는 Gemma teacher에
  직접 전달하도록 연결했다. 결과에는 raw 답변 대신 normalized typed result,
  response hash, schema status, latency만 저장한다.
- Qwen `qwen38fn` 16개 probe는 `16/16 schema-valid`, error `0`, 평균 latency
  `2,144.03ms`였다. 이 결과는 teacher draft 품질이지 human truth가 아니다.
- Gemma endpoint는 `/v1/models` probe는 성공했지만, think 활성화 상태의
  generation probe는 4분 이상 무응답이라 중단했다. 따라서 Gemma judge는
  synchronous control tick이 아니라 별도 비동기 queue와 retry/timeout worker로
  운영해야 한다.
- JSON이 잘리는 작은 `max_tokens`를 피하기 위해 control draft 기본값은
  `256`으로 유지하고, 필요할 때 `--max-tokens`로 실험한다.

## 2026-09-20 — production accuracy gate hardening (v1.46.0)

- control quality evaluator가 split accuracy와 STOP recall만 보지 않고 전체
  safety scenario action accuracy도 기본 100%로 요구한다.
- `--require-human-labels`를 사용하면 validation/test 모든 row에 human typed
  label이 있어야 `passed`가 된다. report에 `human_label_gate`와
  `production_ready`를 분리해 기록한다.
- 800개 synthetic seed checkpoint는 validation/test `80/80 (100%)`였지만
  safety action accuracy `50%`, human label `0`이라 최종 gate `FAIL`이다.
  이전 evaluator가 이 결과를 통과시킬 수 있었던 허점을 수정했다.

## 2026-09-20 — human-review control seed generator (v1.45.0)

- `hyperjev control seed`가 8개 control skill을 균형 있게 생성하고,
  `scenario_id`, `episode_id`, `semantic_group_id`, split, generator provenance를
  각 row에 기록한다.
- seed의 `target`은 synthetic draft일 뿐 human truth가 아니다. `labels.human`
  은 항상 null로 시작하며, `validate_control_dataset --require-human-labels`
  는 검수 전 queue를 실패시킨다.
- 16개 seed smoke를 생성해 validator가 leakage 없이 통과하고, human-required
  mode가 16개 pending row를 모두 거부하는 것을 확인했다.
- 기본 100개/skill 생성 결과를 바로 production dataset으로 사용하지 않는다.
  Qwen/Gemma draft와 human review/apply가 끝난 뒤 human label 수와 agreement를
  다시 측정해야 한다.

## 2026-09-20 — control dataset provenance/leakage gate (v1.44.0)

- `src/hyperjev/control_data.py`와 `scripts/validate_control_dataset.py`를
  추가해 control dataset에 `scenario_id`, `episode_id`,
  `semantic_group_id`가 있는지 검사한다.
- 정규화된 state/question exact group, episode, semantic group이 train과
  validation/test 사이에 걸치면 실패한다. human label 요구도 별도 flag로
  강제할 수 있다.
- 102개 단위 테스트와 기존 fixture 거부 회귀를 통과했다. 현재 48개 smoke
  fixture는 provenance와 human label이 없으므로 품질 gate를 통과하지 못한다.
- 정확도 숫자를 올리기 위해 test를 train으로 이동하거나 duplicate를 삭제해
  분모를 줄이지 않는다. 다음 단계는 이 계약을 만족하는 human-review control
  golden seed와 counterfactual hard-negative 생성이다.

## 2026-09-20 — accuracy evaluator와 token encoder 실험 (v1.43.0)

- validation/test split 정확도와 safety STOP recall을 한 번에 판정하는
  `scripts/evaluate_control_quality.py`를 추가했다. report에는 checkpoint와
  dataset SHA-256, split row/group 수, human label 수, accepted coverage를
  함께 기록한다.
- reference byte/ngram baseline은 validation `3/8 (37.50%)`, test
  `1/8 (12.50%)`였다. 토큰화와 n-gram 국소 특징 경로를 연결한
  `reference-token-encoder`는 validation `4/8 (50.00%)`, test
  `5/8 (62.50%)`로 개선됐지만 99% gate에는 실패했다.
- 두 checkpoint 모두 synthetic 48개이고 human label은 0개다. token
  checkpoint SHA-256은 `e8631580b274c3690e2c8f357d65f6cc07271955014c3f60a3e918a35ad176e0`이며,
  production 승격이나 실제 actuator 연결 근거로 사용하지 않는다.
- 전체 테스트는 `98 passed, 1 skipped`, Ruff는 통과했다. 다음 단계는 human
  golden control dataset과 exact/episode leakage gate를 구현하는 것이다.

## 2026-09-20 — deterministic control simulation harness (v1.42.0)

- `control simulate`가 scenario JSONL을 재생해 skill 정확도, expected safety
  STOP recall, p50/p95/mean action latency를 JSON report로 만든다.
- 정상 skill, normal STOP, stale frame, emergency stop 4개 시나리오를 10회씩
  실행한 결과 `40/40`, accuracy `100%`, safety STOP recall `100%`였다.
- 전체 재생 latency는 p50 `0.283825ms`, p95 `1.663319ms`였다. 이 수치는
  in-process CPU checkpoint 경로이며 HTTP, HyperMemory, controller, actuator는
  포함하지 않는다.

## 2026-09-20 — HyperMemory bounded context adapter (v1.41.0)

- HyperMemory `POST /v1/context`를 호출하는 `compile_context`와
  `control_context` adapter를 추가했다.
- control observation은 raw frame history를 저장하지 않고, HyperMemory가
  미리 압축한 summary를 최대 8개·2,048자로 제한한다.
- Student는 `state + bounded relevant memory context`를 입력으로 받고, 이
  context 조회는 저수준 motor loop 밖의 skill-decision tick에서 수행한다.
- API contract와 control rendering 회귀 테스트를 추가했다. 전체 테스트는
  `95 passed, 1 skipped`다.

## 2026-09-20 — control typed head CLI와 smoke checkpoint (v1.40.0)

- `control train`, `control evaluate`, `control decide` CLI를 추가했다.
- `control.skill@1`은 abstract skill만 반환하고, safety policy가 stale,
  emergency, 낮은 confidence, 긴 TTL을 `STOP`으로 바꾼다.
- 48개 synthetic control sample로 reference checkpoint를 만들었다.
  train `32/32 (100%)`, validation `3/8 (37.50%)`, test `1/8 (12.50%)`였다.
- 이 결과는 reference byte/ngram encoder의 일반화 실패를 드러낸 smoke gate다.
  checkpoint를 production control model로 승격하지 않으며, 사람 라벨·실제
  simulator trajectory·OOD/충돌 테스트가 추가되기 전에는 actuator에 연결하지 않는다.
- warm CPU in-process `ControlStudentClient`는 p50 `0.128ms`, p95 `0.2675ms`,
  평균 `7,634.6 decisions/s`였다. HTTP, HyperMemory, controller 비용은 제외했다.

## 2026-09-20 — real-time control safety contract (v1.39.0)

- `ControlObservation`, `ControlAction`, `ControlSafetyPolicy`를 추가했다.
- HyperJev가 반환하는 action은 `MOVE`, `ROTATE`, `APPROACH`, `RETREAT`,
  `INTERACT`, `RECOVER`, `HOLD`, `STOP` 같은 고수준 skill만 허용한다.
- stale observation, emergency stop, 낮은 confidence, 긴 TTL, 범위 밖
  parameter는 모두 deterministic `STOP`으로 전환한다.
- 모터 PWM/토크 같은 직접 actuator 값은 contract에서 허용하지 않는다.
  저수준 PID/MPC/controller가 최종 actuator를 맡도록 경계를 고정했다.

## 2026-09-20 — low-latency Student inference path (v1.38.0)

- Student inference가 학습용 고정 1,024 token padding을 그대로 사용하지 않고,
  실제 입력 길이까지만 encoder에 전달하도록 dynamic padding을 적용했다.
- 동일 checkpoint, CPU single-thread warm benchmark에서 단건 p50 `0.455ms`,
  p95 `0.464ms`, 약 `2,193 decisions/s`를 측정했다.
- 6개 typed decision을 연속 처리한 경우 전체 p50 `2.539ms`, 약 `236
  requests/s`였다. 이는 CPU reference benchmark이며 DGX Spark GPU production
  latency를 대신하지 않는다.

## 2026-09-20 — high-precision remember rule negatives (v1.37.0)

- `want` 단독 substring을 commitment로 취급하지 않도록 제거했다. 과거
  일회성 사건인 `I wanted noodles...`가 장기 기억으로 잘못 승격될 수 있기
  때문이다.
- `I don't want...`, `원하지...`, `싶지...`, `않기로...` 같은 명시적 거부는
  `remember.explicit_rejection`으로 먼저 false 처리한다.
- human unique-group 재평가는 계속 `36/36 correct`, accepted `29/29`였고,
  부정/과거형 반례 회귀 테스트를 추가한 뒤 전체 테스트는 83 passed였다.

## 2026-09-20 — unique-group Student evaluation (v1.36.0)

- `student evaluate --deduplicate-exact`를 추가해 exact semantic group당
  대표 sample 하나만 평가할 수 있게 했다.
- report에 원본 `row_count`와 `unique_exact_group_count`를 함께 기록한다.
- human-reviewed 1,000 row를 36 group으로 평가한 결과는 `36/36 correct`,
  accepted `29/29`, coverage `80.56%`였다. 이 경로도 quality gate를
  통과했지만, unique group 수가 작으므로 일반화 성능을 의미하지 않는다.

## 2026-09-20 — split-specific accuracy evidence (v1.35.0)

- human-target retrained checkpoint를 `test`, `validation`, `train`으로
  나눠 같은 evaluator로 재실행했다.
- rules + Student 결과는 test `113/113`, validation `105/105`, train
  `782/782` correct였고, 각 split의 accepted accuracy도 100%였다.
- test의 unique group 수는 2개뿐이다. 따라서 이 결과는 regression evidence이지
  99% 이상의 상용 일반화 증명으로 해석하지 않는다.

## 2026-09-20 — content-based split fallback (v1.34.0)

- `document_id`, `entity_id`, `source_id`, `split_group`이 모두 없는 seed는
  더 이상 `sample_id`만으로 train/validation/test를 정하지 않는다.
- task, language, domain, redacted state, redacted question의 deterministic
  content group을 hash해 같은 의미 입력이 항상 같은 split에 가도록 했다.
- 서로 다른 sample ID지만 동일한 내용인 두 row가 같은 split을 받는 회귀
  테스트를 추가했다. 이는 duplicate/near-duplicate가 train과 test를 동시에
  차지해 정확도를 부풀리는 위험을 줄인다.

## 2026-09-20 — human-target retraining and remember rule coverage (v1.33.0)

- reference trainer가 `TrainingConfig.batch_size`를 실제 task별 mini-batch에
  적용하도록 고쳤다. 이전에는 설정값이 검증만 되고 sample 단위 update가
  실행됐다.
- 36개 exact human group에서 train 31, validation 3, test 2로 분리해 다시
  학습한 checkpoint의 Student-only 중간 결과는 `918/1000 (91.80%)`였다.
- 한국어 명시적 결정/선호 표현(`바꾸기로`, `보고 싶다`)을 high-precision
  remember rule에 추가했다. 같은 checkpoint를 rule과 함께 재평가한 결과는
  전체 `1000/1000 (100.00%)`, accepted `822/822 (100.00%)`, coverage
  `822/1000 (82.20%)`, rule `279/279`였다. 모든 task가 100%였고 quality
  gate도 통과했다.
- 이 수치는 1,000 record가 36개 exact group을 반복하는 human-reviewed
  synthetic fixture에서 얻은 결과다. 독립적인 1,000개 원문 정확도나 상용
  일반화 성능을 뜻하지 않으므로, production 승격 근거로 사용하지 않는다.

## 2026-09-20 — full reviewed queue and human-source Student evaluation (v1.32.0)

- exact-duplicate grouping을 사용해 1,000개 queue record에 feedback을 모두
  적용했다. pending `0`, human label `1000/1000`, queue golden gate
  `ready=true`다.
- feedback 1,000건은 36개 exact group으로 구성되고, 612건은 기존 human
  label의 exact duplicate propagation이다. 36개 group의 최종 label은
  서로 일관됐다.
- human-source reference n-gram Student는 전체 `826/1000 (82.60%)`,
  accepted `736/787 (93.52%)`, coverage `787/1000 (78.70%)`였다.
- `memory.remember_worthy`, `query.route`, `wiki.semantic_change`는 전체
  정확했지만 `memory.importance` `26.35%`, `memory.relation` `93.98%`,
  `memory.type` `75.45%`로 Student quality gate는 실패했다.
- human-label completeness와 model quality gate는 분리된다. 이번 결과는
  99% production 정확도 달성이 아니다.

## 2026-09-20 — exact-duplicate review grouping (v1.31.0)

- `golden review-session --deduplicate-exact`를 추가했다.
- 동일한 `task + language + domain + state + question`은 한 review group으로
  묶고, 한 번의 human 판단을 exact duplicate sample에 전파한다.
- 현재 synthetic 1,000건 queue는 36개 exact group과 964개 중복 record로
  구성되어 있다. 임시 복사본 smoke에서 기존 31개 feedback을 사용해
  612개 label을 전파하고, 36개 group 중 19개를 자동 충족했다.
- 전파된 label은 독립적인 1,000개 human 판단이 아니다. 실제 production
  golden set의 정확도와 review sample 수는 unique group 수와 원본 데이터
  다양성을 별도로 보고해야 한다.

## 2026-09-20 — partial human golden feedback applied (v1.30.0)

- `golden-feedback.jsonl`에 저장된 실제 사람 검수 31건을 reviewed queue에
  적용했다. 31건은 모두 고유 sample이며 reviewer는 `dirmich`다.
- 전체 1,000건 중 human label은 `31/1000`, pending은 `969`건이다.
- `golden validate --minimum-count 1000`은 `ready=false`, exit 1로 남았다.
  사람 검수가 끝나지 않은 상태를 production 정확도로 승격하지 않기 위한
  의도된 gate 결과다.
- 사용자가 “전체 검수 완료”라고 판단했더라도 저장된 artifact에 없는 969건은
  자동 생성하거나 Qwen draft로 대체하지 않는다.
- 저장된 31건만 별도 평가한 reference n-gram Student 결과는 전체 `25/31
  (80.65%)`, accepted `25/26 (96.15%)`, coverage `26/31 (83.87%)`였다.
  이는 부분 표본 결과이며 production 정확도 주장이 아니다.

## 2026-09-20 — safe review-session EOF handling (v1.29.1)

- `Ctrl-D`, piped input 종료, `Ctrl-C`를 traceback 없이 저장 후 종료로
  처리한다.

## 2026-09-20 — value-only human correction input (v1.29.0)

- interactive `e` 수정이 raw JSON 전체가 아니라 task에 맞는 값만 받도록
  변경했다.
- boolean은 `true/false`, choice는 registered candidate, score는 `0~1` 숫자를
  입력하며 typed result와 probability/interval은 시스템이 생성한다.

## 2026-09-20 — navigable human review session (v1.28.0)

- `golden review-session` 한 번으로 1,000건을 순차 검수할 수 있게 했다.
- `a` 승인, `e` 직접 수정, `n` 다음, `p` 이전, `s` 보류, `q` 종료를 지원한다.
- 이전 item을 다시 수정하면 append-only feedback의 최신 correction이 적용된다.

## 2026-09-20 — human remember-worthiness rubric (v1.27.0)

- `memory.remember_worthy`를 판단하는 미래 재사용 가치 기준과 true/false
  예시를 `docs/human_labeling.md`에 고정했다.
- synthetic target, teacher draft, semantic memory value, privacy storage
  permission을 서로 다른 판단으로 분리했다.

## 2026-09-20 — 1,000-item review pack generation (v1.26.1)

- Qwen 1,000건 draft와 source queue를 join한 local review pack을 생성했다.
- 1,000개 review item 모두 `pending`이며 `human_correction`은 비어 있다.
- state/question은 포함했지만 `target`과 `labels` 키는 전체 파일에서 누출되지
  않음을 검사했다.

## 2026-09-20 — local human review pack (v1.26.0)

- `hyperjev golden review-pack`으로 teacher draft와 원문의 state/question을
  local review artifact로 묶을 수 있게 했다.
- raw 입력 포함은 `--allow-raw`를 명시해야 하며, synthetic target과 queue
  labels는 pack에서 제외해 reviewer anchoring과 target leakage를 막는다.

## 2026-09-20 — Qwen 1,000-sample golden draft (v1.25.2)

- `localhost:8081/v1`, `qwen38fn`, prompt v2로 1,000건을 실행했다.
- completed `1000/1000`, schema-valid `1000/1000 (100%)`, 오류 `0`이었다.
- synthetic target 탐색 일치는 `874/1000 (87.4%)`였다. human golden 정확도가
  아니며 production gate 근거로 사용하지 않는다.
- task별 target match: remember `100%`, route `100%`, wiki `100%`, relation
  `93.98%`, type `75.45%`, importance `55.09%`.
- 평균 latency `1,483.361ms`, p50 `1,189.115ms`, p95 `2,331.728ms`였다.
- `golden validate`는 human label `0/1000`, `ready=false`, exit 1이었다.

## 2026-09-20 — Qwen prompt v2 re-probe (v1.25.1)

- `qwen38fn` 20건 재시험에서 완료 `20/20`, schema-valid `20/20 (100%)`를
  확인했다.
- synthetic target 탐색적 일치는 `18/20 (90%)`였으며, 평균 latency
  `1,503.768ms`, p50 `1,238.746ms`, p95 `2,338.942ms`였다.
- 이전 확률합 schema 오류 3건은 재시험에서 0건으로 감소했다.

## 2026-09-20 — choice probability prompt hardening (v1.25.0)

- teacher prompt를 v2로 올려 choice 결과가 모든 후보를 포함하고 확률 합을
  정확히 `1.0`으로 만들도록 명시했다.
- registry의 모든 `teacher_prompt_version`을 2로 맞춰 prompt/data provenance를
  분리했다.

## 2026-09-20 — Qwen golden draft live probe (v1.24.1)

- `localhost:8081/v1`의 실제 `qwen38fn`으로 20개 draft를 실행했다.
- 응답 완료는 `20/20`, schema-valid는 `17/20 (85%)`였다.
- synthetic target과의 탐색적 일치는 valid subset `14/17 (82.35%)`, invalid를
  오답으로 포함하면 `14/20 (70%)`였다. 이 수치는 human golden 정확도가 아니다.
- 평균 latency `1,449.496ms`, p50 `1,075.946ms`, p95 `2,357.904ms`였고,
  `memory.type` 3건은 확률합 검증 실패였다.

## 2026-09-20 — Qwen golden draft retry path (v1.24.0)

- `hyperjev golden draft --provider qwen|gemma`로 teacher를 선택할 수 있게
  했다.
- Qwen `qwen38fn`을 빠른 human-review 초안 생성 경로로 사용할 수 있다.
- Qwen 초안도 human label이나 Gemma 교차검증을 대체하지 않으며, provider와
  provenance를 manifest에 기록한다.

## 2026-09-20 — baseline 고정 (v1.17.2)

- reference Student checkpoint 생성 완료:
  `runs/phase3/reference-student.pt`
- dataset: synthetic 중복 제거 36개
  - train 31
  - validation 2
  - test 3
- reference Student 전체 정확도: **28/36 = 77.78%**
- held-out test 정확도: **1/3 = 33.33%**
- backbone: `reference-byte-encoder`
- checkpoint SHA-256:
  `7a4bb4d47a58295db479cb56ee675335e45706d80d54a87359b87a30f678e19a`
- rule fast-path:
  - covered: 197/1,000 = 19.70%
  - covered accuracy: 197/197 = 100%
  - 나머지 803개는 Student/teacher/human fallback이 필요함
- 결론: checkpoint artifact gate는 통과했지만 production quality gate는
  통과하지 못했다. test 3개 결과를 99% 달성의 근거로 사용하지 않는다.

## 99% accuracy gate

다음 결과를 모두 충족할 때 production quality gate를 통과한 것으로 판정한다.

1. 사람이 검수한 immutable test set 최소 1,000개
2. 전체 test exact/threshold accuracy >= 99%
3. task별 accuracy >= 98%
4. 자동 수락 subset accuracy >= 99.5%
5. 자동 수락 coverage와 fallback rate를 함께 보고
6. schema-invalid, OOD, teacher disagreement는 자동 수락하지 않음
7. calibration report와 seed별 regression 결과 존재

현재 다음 작업은 synthetic 결과를 부풀리는 것이 아니라, `accuracy`와
`coverage`를 분리 측정하는 guarded evaluation, task별 threshold, 그리고
사람 golden set을 연결하는 것이다.

## 2026-09-20 — guarded Student evaluator (v1.18.0)

- `hyperjev student evaluate` 추가
- 전체 정확도와 다음 값을 동시에 출력:
  - `accepted_accuracy`
  - `coverage`
  - `fallback_count`
  - task별 결과
- 기존 reference checkpoint 재평가:
  - 전체: 28/36 = 77.78%
  - confidence 0.95 자동 수락: 0/36
  - 결론: 현재 Student는 production 자동 수락 기준을 충족하지 않으며,
    이 결과가 다음 모델 개선의 regression baseline이다.

## 2026-09-20 — n-gram Student와 guarded product path (v1.19.0)

- `reference-ngram-encoder` 추가: byte embedding 뒤에 depthwise 3-gram
  convolution과 mean/max pooling을 연결했다.
- 학습: dataset 36개, train 31개, 100 epochs, CPU, learning rate 0.01,
  weight decay 0.
- checkpoint:
  `runs/phase3/reference-ngram-student.pt`
- checkpoint SHA-256:
  `030cdc2e270a3cfb4a28756dd36f5a3991dfbce28ae178243f88a089cdf5c0ba`
- Student 단독: 전체 35/36 = **97.22%**, test 3/3 = **100%**
- Student confidence 0.95 accepted: 27/27 = **100%**, coverage **75.00%**
- rule + Student guarded path: 36/36 = **100%**, accepted 29/29 = **100%**,
  coverage **80.56%**, fallback 7/36
- 판정: synthetic contract gate는 99%를 넘겼다. 사람 검수 golden 1,000개와
  production multilingual encoder가 없으므로 production 승격은 하지 않는다.

## 2026-09-20 — production quality gate (v1.20.0)

- `student evaluate --production-gate` 추가
- gate 조건:
  - human label 전체 존재
  - overall accuracy >= 99%
  - accepted accuracy >= 99.5%
  - task별 accuracy >= 98%
- 현재 synthetic 실행 결과:
  - overall accuracy: 100%
  - accepted accuracy: 100%
  - human labeled: 0/36
  - exit code: 1 (`human_labels_required`)
- 결론: 숫자만 높은 synthetic checkpoint를 production 승격하지 않는 보호
  장치가 정상 동작한다.

## 2026-09-20 — 1,000개 synthetic stress (v1.20.1)

- 입력: `runs/phase0/phase0-review-queue.jsonl` 1,000개
- 평가 dataset SHA-256:
  `139e5c2f63de52811eab3845f19a4b42e4c7c0d6d7b61f1d979af292e08438ff`
- rule + n-gram Student:
  - overall: **1000/1000 = 100%**
  - accepted: **787/787 = 100%**
  - coverage: **78.70%**
  - fallback: 213개
  - rule: 197/197 = 100%
- 단, 이 queue는 synthetic이며 human label 0/1,000이다. 따라서 이 결과는
  regression/stress 증거이고 production 99% gate 통과 증거가 아니다.

## 2026-09-20 — Student router 연결 (v1.21.0)

- optional `StudentClient`를 추가해 checkpoint가 설정된 경우 실제 제품 경로를
  `rule → Student → Qwen → Gemma → human` 순서로 연결했다.
- 기본 설정에는 checkpoint가 없어 기존 Phase 0/mock와 teacher 동작을 보존한다.
- `HYPERJEV_STUDENT_CHECKPOINT`, `HYPERJEV_STUDENT_MIN_CONFIDENCE`,
  `HYPERJEV_STUDENT_DEVICE`로 로컬 checkpoint와 confidence gate를 설정한다.
- checkpoint는 registry task/head와 manifest를 검증하고, boolean/choice의
  confidence 미달 및 score의 calibration 전에는 abstain한다.
- 실제 n-gram checkpoint smoke:
  - confidence 0.95: `route=student`, `accepted=true`, 약 5ms
  - confidence 0.9999: `Student uncertain → Qwen/Gemma error → human`
- `abstained=true` 결과는 typed output의 확률이 높아도 자동 수락하지 않도록
  router gate를 수정했다.
- 검증: **70 passed, 1 skipped**, Ruff 통과, `git diff --check` 통과.
- 이 변경은 synthetic 품질 수치를 production 99%로 승격하지 않는다. human
  golden 1,000개와 `--production-gate` 통과는 여전히 필요하다.

## 2026-09-20 — Student router milestone ledger sync (v1.21.1)

- `ca49048`의 Student router 구현·테스트·smoke 증거를 구현 장부에 확정했다.
- Python/Rust/lockfile version을 `1.21.1`로 동기화했다.
- 기능 동작은 v1.21.0과 동일하며, production gate 상태도 변경하지 않았다.

## 2026-09-20 — human golden label 기준 평가 (v1.22.0)

- Student evaluator가 human typed label이 존재하면 synthetic `target` 대신
  `labels.human`을 정확도 기준으로 사용하도록 수정했다.
- human label은 task schema와 type을 다시 검증하고, abstain 라벨은 품질 기준으로
  사용할 수 없게 거부한다.
- human label이 없는 dataset은 기존처럼 `sample.target`을 사용하지만
  `target_source=sample.target`로 명시되어 exploratory/synthetic 결과임을
  구분한다.
- regression test가 synthetic target과 다른 human correction을 넣었을 때
  human 값을 실제 평가 기준으로 사용하는 것을 검증한다.
- 검증: **71 passed, 1 skipped**, Ruff 통과.
- 이 수정은 숫자를 인위적으로 높이지 않는다. 오히려 production gate가 실제
  human 판정에 종속되도록 정확도를 정직하게 만든다.

## 2026-09-20 — human-source evaluator 1,000개 재검증 (v1.22.1)

- checkpoint SHA-256: `030cdc2e270a3cfb4a28756dd36f5a3991dfbce28ae178243f88a089cdf5c0ba`
- dataset SHA-256: `139e5c2f63de52811eab3845f19a4b42e4c7c0d6d7b61f1d979af292e08438ff`
- `target_sources`: `sample.target`만 존재 (`human 0/1000`)
- 전체: **1000/1000 = 100%**
- accepted: **787/787 = 100%**, coverage **78.70%**, fallback 213개
- production gate: **exit 1**, `human_labels_required`
- 결론: synthetic regression 수치는 재현됐지만 human golden 정확도는 아직
  측정되지 않았고, production 승격도 계속 금지된다.

## 2026-09-20 — Gemma draft label pipeline (v1.23.0)

- `hyperjev golden draft`를 추가해 Gemma4가 human review 전용 초안 라벨을
  생성할 수 있게 했다.
- draft에는 raw state/question이나 raw Gemma 응답을 저장하지 않고,
  normalized typed result, response hash, latency, schema validity, queue hash만
  저장한다.
- draft의 `normalized_result`는 `labels.human`이 아니다. 사용자가 원문
  `state/question`을 독립적으로 확인하고 `golden review` feedback을 생성해야
  human golden으로 인정된다.
- 실제 Gemma4 1건 probe:
  - model: `google/gemma-4-12b`
  - timeout: 60초
  - 결과: `TimeoutError`, schema-valid `0/1`
  - draft queue SHA-256: `463b00f91257b34a837f8eaa184d5fb4efec49f2028f086dd78aa0e3bdbd873f`
- 검증: **72 passed, 1 skipped**, Ruff 통과.
- Gemma generation timeout이 해결되기 전에는 batch draft를 human label로
  승격하지 않는다.
