# HyperJev changelog

이 파일은 정확도·coverage·fallback 정책의 중간 결과를 기록한다. 숫자는
동일한 dataset/split과 실행 명령으로 재현할 수 있는 경우에만 갱신한다.

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
