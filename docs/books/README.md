# HyperJev를 처음부터 끝까지 만드는 책

이 디렉터리는 HyperJev를 빈 저장소에서 시작해 DGX Spark의 실제 서비스로
가져가기까지의 과정을 책으로 엮기 위한 원고 저장소다. 구현이 진행될 때마다
설계 이유, 실행 명령, 테스트 증거, 커밋을 이 원고에 누적한다. 최종 출판본은
이 파일을 목차로 사용하고 각 장의 `상태`와 `남은 gate`를 최신 값으로 갱신해
만든다.

## 독자가 얻게 될 것

- `docs/prd.md`를 구현 가능한 계약과 단계별 gate로 분해하는 방법
- Qwen3 8B와 Gemma 4를 teacher로 배치하는 local-first 개발 환경
- rule → optional Student → Qwen → Gemma → human fallback의 typed decision router
- encoder + typed decision heads Student와 별도 연구 트랙 HyperJev-D의 경계
- privacy-safe dataset factory, calibration, golden review, shadow serving,
  registry, drift monitoring을 한 흐름으로 연결하는 방법
- DGX Spark에서 실제로 통과시켜야 하는 검증과 아직 외부 자원이 필요한 검증

## 원문과 구현의 우선순위

1. 제품 계약은 [`docs/prd.md`](../prd.md)가 기준이다.
2. 실제 실행 계약은 `registry/tasks/`, `configs/phase0.toml`,
   `src/hyperjev/contracts.py`가 기준이다.
3. 이 책은 왜 그렇게 구현했는지와 어떤 증거를 얻었는지를 설명하는
   학습 자료다. 책과 코드가 어긋나면 코드를 고친 뒤 이 원고를 갱신한다.

## 장 목록

| 장 | 파일 | 내용 | 현재 상태 |
| --- | --- | --- | --- |
| 1 | [범위와 문제](01-scope-and-problem.md) | HyperJev의 역할과 경계 | 초안 완료 |
| 2 | [PRD 해부](02-prd-analysis.md) | 요구사항, 출력 계약, gate | 초안 완료 |
| 3 | [Phase 0 기반](03-phase0-foundation.md) | 환경, 구조, 첫 테스트 | 초안 완료 |
| 4 | [Teacher와 Golden](04-teachers-and-golden.md) | Qwen/Gemma baseline과 human review | 초안 완료 |
| 5 | [Router에서 Serving까지](05-router-to-serving.md) | Phase 1~7의 누적 설계 | 초안 완료 |
| 6 | [Dataset, Student, Training](06-dataset-student-training.md) | 학습 데이터와 typed-head 준비 | 진행 중 |
| 7 | [검증과 운영](07-verification-and-operations.md) | 테스트, DGX, 관측성, 남은 gate | 진행 중 |
| 8 | [구현 일지](08-implementation-ledger.md) | 단계별 version/commit/push 증거 | 계속 갱신 |
| 9 | [최종 gate 체크리스트](09-final-gates.md) | 재현 명령과 출시 전 판정 | 초안 완료 |

## 현재 스냅샷

현재 `main`은 origin에 push될 1.137.0까지 진행되어 있다. Python 테스트와 skip
수는 최신 전체 suite 실행 결과를 기준으로 기록하며, Ruff 검사가 통과한
상태이다. dataset validator와 training plan,
reference train CLI, reference Student checkpoint 생성, n-gram Student 비교와
guarded 99% 평가, optional Student router 연결, parent LLM 대 HyperJev
경계의 성능 시험 기록까지 완료됐다. training plan·calibration·checkpoint를
registry manifest로 묶는 contract와 30-sample confidence/coverage 분석도
추가됐다. memory/query checkpoint는 synthetic 36개 group의
reference-ngram-encoder이고 control checkpoint는 별도 48개 smoke fixture다.
둘 다 human golden이 없거나 일반화 gate를 통과하지 않았으므로 production
모델로 승격하지 않았다. hard-negative synthetic checkpoint는 validation/test
100%였지만 human label 0건이라 production-ready가 아니다. 실제
사람 검수 golden set, Gemma live baseline 전수 실행, production multilingual
PyTorch checkpoint 학습, Student GPU inference benchmark, Rust toolchain compile은
이 책에서 성공했다고 가장하지 않고 외부 의존성 gate로 표시한다. 상세 결과는
[`docs/test_result.md`](../test_result.md)에 있다.

v1.122.0에서는 Qwen control draft에서 반복된 명확한 문장을 compound fast path로
추가하고, blocker 문장 회귀 테스트를 확장했다. 또한 99% point accuracy 주장을
Wilson 95% 하한과 최소 381개 독립 held-out group으로 제한했다. 현재 human label은
여전히 0건이므로 synthetic replay는 production accuracy가 아니다.

v1.123.0에서는 forced STOP만 있던 safety 평가에 non-trigger near-miss를 추가하고,
false abstaining STOP rate를 별도 gate로 기록했다. 따라서 safety recall과 false
STOP precision을 함께 확인할 수 있다.

v1.124.0에서는 human held-out 검수를 위한 explicit split quota를 추가했다. seed
17의 4,000-row queue는 test 384개와 validation 384개, unique group 4,000개를
재현하며, 아직 human label이 없으므로 production evidence는 아니다.

v1.125.0에서는 test 384개에 대한 Qwen draft와 target-excluded blind review pack을
생성했다. Qwen synthetic 비교는 95.0521%였지만 human label은 0건이며, 책의
production accuracy로 해석하지 않는다.

v1.126.0에서는 control task에만 semantic boundary를 주입하는 prompt v3를
실험했다. 동일한 held-out 384개에서 Qwen synthetic match가 365/384 (95.0521%)에서
384/384 (100%)로, schema-valid가 378/384에서 384/384로 올라갔다. 이 수치는
human label이 아닌 teacher/reference 비교이며, blind review 전에는 production
accuracy나 checkpoint 승격으로 해석하지 않는다.

v1.127.0에서는 이 prompt v3 draft의 hash에 바인딩된 target-excluded blind review
pack을 새로 만들었다. reviewed `0/384`, `test_ready=false`이므로 다음 단계는
사람 A/B의 독립 검수와 disagreement adjudication이다.

v1.128.0에서는 4,000-row bilingual queue로 Student checkpoint를 학습해
validation/test `384/384`와 Wilson 하한 `0.990095`를 확인했다. safety 1,000건도
STOP recall `100%`, false STOP `0%`, p99 `0.032688ms`였다. 그러나 human label은
`0/4000`이므로 이 checkpoint는 production 모델이 아니다.

v1.129.0에서는 Gemma fast alias를 held-out 384개에 독립 실행해 `383/384`
synthetic match와 Qwen/Gemma agreement `383/384`를 확인했다. disagreement 한 건은
사람 adjudication 전까지 승격하지 않으며, 실제 model/prompt provenance가 manifest에
기록되도록 수정했다.

v1.130.0에서는 adjudication 결과를 export한 blind review pack에도 두 teacher의
실제 model, prompt version, comparison을 보존했다. 유일한 disagreement가 첫
review item으로 남아 사람 판단을 기다린다.

v1.131.0에서는 4,000-row bilingual seed와 1,000-row hard-negative queue를
병합하고 hard-negative weight 2/4를 비교했다. weight 2 후보는 merged
validation/test `484/484`, hard queue `1000/1000`, English OOD `87.5%`, Korean
OOD `85.0%`, STOP recall `100%`를 기록했다. weight 4의 English OOD 회귀 때문에
weight 2를 다음 human review용 후보로 선택했지만, human label `0/5000`이므로
production 모델로 승격하지 않았다.

v1.132.0에서는 같은 checkpoint를 integrated fast path와 Student-only로 나눠
측정했다. integrated fixture는 rule이 모두 해결해 `40/40` 및 `1000/1000`, p95
약 `34~38µs`였지만 Student-only OOD는 English `87.5%`, Korean `85.0%`였다.
책에서는 이 둘을 분리해 기록하며, rule coverage를 encoder 일반화 정확도로
포장하지 않는다.

v1.133.0에서는 hybrid word+character BOW를 폐기했다. merged validation/test는
100%였지만 Student-only hard `99.7%`, English OOD `62.5%`, Korean OOD `50.0%`로
weight-2 BOW보다 악화됐다. 따라서 독립 OOD 회귀를 통과하지 못한 backbone은
production 후보가 될 수 없다는 원칙을 추가했다.

v1.134.0에서는 Korean INTERACT 경계를 train-only로 보강해 Student-only
English/Korean OOD를 `40/40`으로 만들었지만, CUDA model-only p95가 약
`10.0~10.9ms`, p99에 `503~546ms` outlier가 있어 latency gate를 통과하지 못했다.
정확도와 bounded latency를 별도 gate로 운영하는 이유를 이 실험으로 남겼다.

v1.135.0에서는 warmup 10회와 CUDA synchronize를 evaluator 계약으로 고정했다.
500ms급 비동기 outlier는 사라졌지만 Student-only English/Korean OOD p95는 약
10.4ms로 5ms gate를 넘었다. 다음 장은 vocab/hidden 축소와 preallocation을
정확도 회귀 없이 비교한다.

v1.136.0에서는 16k/128, 32k/128, 16k/256 축소 모델을 모두 폐기했다. 일부
English p95는 좋아졌지만 Korean OOD가 95~97.5%로 내려가거나 hard latency가
약 9.4~9.7ms였다. 정확도 기준선은 32k/256 targeted BOW로 유지하고, 다음은
runtime allocation 최적화다.

v1.137.0에서는 32k/256 표현을 유지한 채 dense vocab count tensor를 만들지 않는
sparse BOW projection을 적용했다. dense 수식과의 동치 unit test를 통과했고,
같은 checkpoint로 hard/English/Korean Student-only OOD는 모두 100%, CUDA
synchronized p95는 `0.159/0.182/0.183ms`가 됐다. human label `0/5192`는 그대로라
production accuracy로 승격하지 않았으며, 다음 gate는 human novel-state review다.

v1.103.0에서는 control simulation 입력의 중복 `scenario_id`를 차단했다. 이로써
`--repeat` latency sampling과 독립 safety scenario 수를 구분하고, 중복 행으로
정확도·STOP recall을 부풀리는 실수를 회귀 테스트로 막는다. 500-case 재생은
고유 500개, 정확도 100%, safe STOP recall 100%, p99 0.001552ms였지만 여전히
local CPU evidence이며 human control accuracy 증거는 아니다.

v1.104.0에서는 production 후보 materialize에 선택적 dual-review provenance
gate를 추가했다. agreement 또는 adjudication이 없는 single-review label은
strict 모드에서 거부되며, 기본 연구 workflow는 호환성을 유지한다. 이 단계는
정답을 자동 생성하지 않고, 사람 라벨의 신뢰성을 평가·학습 단계로 전달하기
위한 품질 경계다.

v1.105.0에서는 Korean held-out OOD와 train-only augmentation을 추가했다. BOW
class-balanced candidate가 English/Korean raw model-only에서 97.5%/90.0%를 보였고,
Korean integrated fast path는 100%였지만 human label은 0건이다. 따라서 이 장은
다국어 synthetic 개선과 실제 human production gate를 분리해 기록한다.

v1.106.0에서는 Korean HOLD 증강과 BOW vocabulary 8,192/16,384 ablation을
비교했다. 정확도는 v1.105의 32,768 BOW 후보를 넘지 못했고, 일부 후보는 latency
또는 Korean OOD 정확도가 악화되어 폐기했다. 실패한 실험도 checkpoint 선정
근거와 함께 기록하며, human label `0/1000` 상태에서는 production 승격을
보류한다.

v1.107.0에서는 raw Student와 integrated rule/fast-path의 측정 경계를 다시
분리했다. char-BOW와 hybrid-BOW는 Korean raw OOD를 각각 95.0%와 97.5%까지
올렸지만 English가 80.0%와 85.0%로 회귀해 폐기했다. 이 실패를 기록한 이유는
한국어 단일 지표를 올리는 것이 다국어 상용 정확도 향상과 같지 않기 때문이다.

v1.108.0에서는 `review-status`에 test-ready와 전체 materialize-ready strict
gate를 추가했다. 현재 human label이 0/1000이면 명령이 실패하는 것이 정상이며,
이 실패는 모델 실패가 아니라 사람 golden 없이 production 승격하지 않도록 하는
품질 계약이다.

v1.109.0에서는 선택된 BOW checkpoint를 500개 고유 safety scenario에 다시
재생해 accuracy와 safe STOP recall 모두 100%, p99 0.001440ms를 확인했다. 이
장에서는 이 safety replay를 human-labeled multilingual control accuracy와
분리해 기록한다.

v1.110.0에서는 model registry promotion에 quality report binding을 추가했다.
strict 승격은 human label/test gate, checkpoint/dataset hash, STOP recall과
Wilson 하한을 모두 다시 확인한다. 이는 사람이 검수한 결과를 자동으로 만들지
않으므로 현재 human label `0/1000` 상태에서 promotion이 실패하는 것이 정상이다.

v1.111.0에서는 control review seed의 prompt version을 2로 올렸다. 800개 seed
row의 state가 모두 고유하고 영어/한국어 질문이 8종으로 분산되도록 해 같은
질문의 기계적 반복을 줄였다. 이는 review와 학습 입력의 다양성 개선이지,
human label이나 production 정확도 증거는 아니다.

v1.112.0에서는 이 800개를 Qwen `qwen38fn`으로 전수 draft했다. valid row 기준
synthetic 일치율은 96.95%였지만 human label은 0건이다. APPROACH/MOVE,
RETREAT/MOVE, INTERACT/APPROACH 혼동을 포함한 target-excluded blind review
pack을 만들었고, 이후 두 사람의 검수와 adjudication을 거쳐야 한다.

v1.113.0에서는 v2 queue를 Student에 바로 학습시키는 실험을 수행했지만
English/Korean OOD가 77.5%/67.5%로 회귀해 폐기했다. safety 500-case는 통과했으므로
안전 경로와 semantic accuracy를 별도 gate로 기록하는 원칙을 다시 확인했다.

v1.114.0에서는 질문을 제거한 `reference-control-bow-encoder`로 동일한 synthetic
dataset을 다시 학습해 BOW 질문 노이즈 가설을 분리했다. v1.113.0보다 English
`31/40 → 35/40`, Korean `27/40 → 29/40`으로 회복했지만 v1.105 기준선
`39/40`, `36/40`에는 못 미쳐 후보를 폐기했다. state-only가 질문 표현을 제거해도
human label `0/1800` 상태에서는 production 정확도나 promotion을 주장할 수 없다.

v1.115.0에서는 malformed quality report가 evaluator traceback으로 끝나지 않고
명시적 gate failure로 거부되도록 hardening했다. 이는 정확도 점수를 바꾸지 않으며,
human label 없는 후보를 production-ready로 잘못 표시하지 않는 운영 경계다.

v1.116.0에서는 state/question vocabulary를 분리한 segmented BOW를 실험했지만
English `34/40`, Korean `28/40`으로 기준선에 미달해 폐기했다. 이 결과를 통해
표현 충돌을 줄이는 것만으로는 99% 목표에 도달하지 못하며, human adjudication을
통한 혼동쌍 데이터가 우선이라는 결론을 기록한다.

v1.117.0 상태 점검에서 v2 queue는 `0/800` human reviewed, held-out test는
`0/80`으로 확인됐다. 이 책은 synthetic 점수와 human production accuracy를
분리하며, 다음 장의 실제 검수·adjudication·human-only training이 완료되기 전에는
99% 달성을 주장하지 않는다.

v1.118.0에서는 Korean OOD의 명확한 제어 문장 10개를 복합 phrase fast path로
라우팅하고 explicit Korean STOP signal을 보강했다. integrated Korean OOD와
combined replay가 각각 `40/40`, `1800/1800`으로 resolved 되었고 p99는 각각
`66.880us`, `34.624us`였다. 이는 raw model-only 정확도가 아니라 안전한 typed
rule coverage와 latency 개선이며, human label `0/800` 상태에서 상용 정확도를
의미하지 않는다.

v1.119.0에서는 새 Korean phrase fast path에 blocker regression을 추가했다.
차단된 회전·접근·후퇴·상호작용·복구 문장은 deterministic rule에서 defer되어
false-positive를 줄인다. 이는 human label 없는 synthetic/integrated 결과를
상용 정확도로 승격시키지 않는 기존 gate와 함께 동작한다.

v1.120.0에서는 Gemma의 reasoning control을 환경변수로 주입할 수 있게 하고,
부분 benchmark의 sample denominator를 실제 run 범위로 고쳤다. candidate Gemma
alias는 reasoning을 끄면 6/6 typed draft를 반환했지만 human label이나 production
judge 승격 근거는 아니다.

v1.121.0에서는 candidate Gemma를 control STOP slice 20건으로 검증한 결과
`13/20`에 그쳐 폐기했고, partial teacher draft의 full-queue accuracy를 `null`로
보수화했다. 이는 schema-valid와 실제 판단 정확도를 분리하는 검증 원칙이다.

v1.58.0에서는 control evaluator에 threshold별 risk-coverage를 추가했다.
synthetic combined checkpoint가 confidence `0.50/0.90/0.95/0.99`에서 모두
100% accepted accuracy를 보였어도 class-balanced 후보의 safety action은
75%였으므로, confidence와 정확도·안전성을 같은 숫자로 취급하지 않는 원칙을
책의 검증 장에 반영했다.

v1.59.0에서는 held-out logits에서 task별 temperature를 계산하는 control
calibration manifest를 추가했다. 실제 synthetic validation 180건에서 NLL은
줄었지만 temperature가 검색 하한에 붙고 human label이 0건이어서 calibration
artifact를 production 품질 증거로 승격하지 않았다.

v1.60.0에서는 calibration manifest를 Student runtime에 연결했다. task별
temperature를 확률/abstain에 적용하되 checkpoint SHA가 맞지 않으면 거부해,
calibration 파일만 바꾸어 품질을 위장하거나 다른 모델에 잘못 적용하는 경로를
막았다.

v1.61.0에서는 Qwen 800건 control draft를 완료했다. synthetic target은
800/800이었지만 2건은 확률합 repair가 필요했고, p95 2.36초라 실시간 motor
loop가 아니라 비동기 teacher 경로로 분리했다. teacher의 100% 결과는 human
golden 정확도 증거가 아니다.

v1.62.0에서는 teacher draft를 반복 측정하는 evaluator를 추가했다. queue와
draft hash를 고정하고 synthetic match, schema repair, latency, human gate를
분리하므로 Qwen 초안의 100%를 production 정확도로 오해하지 않는다.

v1.63.0 hard-negative 실험에서 Qwen은 STOP에는 100%였지만 HOLD 0%,
APPROACH 49.70%였다. 이 결과를 통해 teacher가 만든 counterfactual label도
사람 검수 전에는 학습 target이 될 수 없고, 오류 pair를 우선 검수해야 한다는
운영 규칙을 책에 추가했다.

v1.64.0에서는 `control review-pack --prioritize`를 추가했다. review pack은
invalid schema, repair가 필요한 teacher 결과, 낮은 typed confidence 순으로
정렬하고 manifest에 `uncertain_first`를 기록한다. 이 정렬은 synthetic target을
보지 않으므로 target leakage가 없으며, 사람이 먼저 확인할 위험 row를 줄여준다.
다만 아직 human label은 0건이어서 이 기능은 검수 throughput 개선이지 정확도
증거가 아니다.

v1.65.0에서는 hard-negative pair의 provenance를 review item에 안전하게 남기고,
priority 정렬 시 counterfactual group 전체를 연속 배치했다. 따라서 한쪽만 보고
다음 500개 뒤에 반대쪽을 보는 검수 순서를 피할 수 있다. target과 labels는 여전히
pack에서 제외되므로, 이 기능은 human 검수 품질과 일관성을 높이는 장치이지
pseudo-label 생성기가 아니다.

v1.66.0에서는 실제 `control review-pack` CLI handler가 `--prioritize`를
`export_review_pack`까지 전달하는지 고쳤다. Qwen hard-negative 1,000건을 실제로
재생성해 `uncertain_first`, 500 pair, adjacency violation 0, target/labels
누출 false를 확인했다. 이 smoke가 helper만 통과하고 CLI가 queue order를 내는
회귀를 잡았으므로 이후 단계의 운영 검증 기준에 포함한다.

v1.67.0에서는 generic `golden review-pack` parser에도 `--prioritize`를 추가해
control과 memory review 도구의 CLI 계약을 일치시켰다. handler만 수정하고
parser를 놓치는 회귀를 막기 위해 두 parser flag를 각각 테스트한다.

v1.68.0에서는 confidence만 믿을 수 없는 teacher 오류를 잡기 위해
counterfactual pair collision을 최우선 검수로 보냈다. 실제 Qwen hard-negative에서
HOLD 오답 confidence가 1.0이었지만 pair collision 167개가 검출되어 앞쪽에
배치됐다. 이는 target을 노출하지 않고도 semantic contradiction을 이용하는
active-review 개선이다.

v1.69.0에서는 review manifest에 active-review 범위를 구조화했다. 실제 Qwen
hard-negative는 500 counterfactual groups, 167 collision groups, 334 collision
items로 기록되므로, 운영자가 JSONL을 재분석하지 않고도 우선 검수량을 알 수 있다.

v1.70.0에서는 hard-negative provenance를 loss 가중치로 반영하는 연구 옵션을
추가했다. weight 2.0/4.0은 synthetic split accuracy 100%를 유지했지만 safety
action accuracy를 75%로 떨어뜨려 탈락했다. 이 실패를 기록한 이유는 human
hard-negative가 들어온 뒤 같은 실험을 재현하되, STOP 안전 gate를 절대 양보하지
않기 위해서다.

v1.71.0에서는 Gemma4 hard-negative 1건 probe가 30초 timeout으로 실패한 결과를
기록했다. Gemma는 control tick의 parent가 아니라 async judge로 격리하고, 실제
golden target은 human reviewer가 확정해야 한다는 운영 경계를 재검증했다.

v1.72.0에서는 hard-negative의 `APPROACH↔MOVE`, `HOLD↔STOP` 혼동을 보완하는
compound-phrase control fast path를 추가했다. hard-negative 1,000행은 synthetic
target 기준 100%를 약 10.11µs/행 평균으로 처리했고, seed 800행은 750행을
규칙으로 확정했다. 이는 human label이 0건인 synthetic regression 결과이므로
production accuracy로 해석하지 않는다. 모호한 입력은 model/fallback으로
남기고 explicit STOP은 safety rule로 먼저 처리한다.

v1.73.0에서는 부정문·위험문이 compound phrase를 포함해도 fast path가 오작동하지
않도록 skill별 contradiction blocker를 추가했다. 정상 HOLD의 `no hazard`가
차단되지 않는 회귀와, 접근 불가/비상 위험 반례의 fast path 우회 테스트를 모두
통과했다. 이는 coverage 확장이 아니라 안전 false-positive를 줄이는 단계다.

v1.74.0에서는 fast path 전용 evaluator를 저장소에 추가했다. queue SHA와
source별 coverage, p50/p95/p99 latency, synthetic/human target을 분리해 같은
명령으로 재현할 수 있다. hard-negative는 100% coverage와 100% synthetic match를
보였지만 human `0/1000`, production `false`를 유지한다.

v1.75.0에서는 evaluator에 실제 checkpoint runtime replay를 추가했다. 같은
hard-negative를 `ControlStudentClient`로 재생해 rule/safety source, synthetic
accuracy, STOP recall, runtime latency를 별도로 확인한다. 이번 queue는
100%였지만 모든 row가 known fast path였고 human `0/1000`이므로, 실제 일반화
정확도와 model fallback latency gate는 여전히 남아 있다.

v1.76.0에서는 runtime latency를 source별로 나눴다. seed queue의 local
reference Student fallback은 p50/p95/p99 `293.486/421.518/2972.318µs`였고,
control-rule p95는 `14.112µs`였다. 이 차이는 teacher가 아니라 local Student
fallback 비용이며, human golden과 GPU/e2e gate가 없으므로 production 수치가
아니다.

v1.77.0에서는 40개 compositional OOD fixture에서 model-only 정확도 `10%`,
STOP recall `0%`와 integrated runtime 정확도 `100%`, STOP recall `100%`를
나란히 측정했다. 이 차이는 fast path가 모델 오류를 가리는 것이 아니라 안전
경계의 일부라는 사실을 보여준다. fixture는 synthetic/human `0/40`이므로
production 정확도 증거가 아니다.

v1.78.0에서는 OOD 자체를 train에 복사하지 않은 64개 compositional train 문장을
추가해 model-only 정확도를 `95%`까지 올렸고, integrated runtime은 `100%`를
유지했다. model-only STOP recall은 명시적 collision safety interlock을 항상 켜도록
고정해 `100%`가 됐다. 다만 human label `0/40`이고 model-only HOLD 오분류가 남아
있으므로 상용 99% gate나 모델 정확도 100%로 승격하지 않는다.

v1.79.0에서는 모든 skill의 semantic boundary 문장을 균형 있게 보강해 model-only
OOD 정확도를 `97.5%`까지 높였다. integrated runtime은 `100%`를 유지했지만 HOLD
한 건이 STOP으로 보수 처리됐고, 독립 safety scenario의 정상 action accuracy는
`75%`였다. STOP recall `100%`와 정상 action precision을 따로 최적화해야 하며,
human label `0` 때문에 상용 gate는 계속 닫혀 있다.

v1.80.0에서는 명확한 정상 APPROACH compound signal을 fast path에 추가해 safety
scenario action accuracy를 `100%`로 복구하고 STOP recall `100%`를 유지했다. 추가
boundary/seed/class-balanced 학습은 v1.79 best `97.5%`를 넘지 못해 승격하지
않았다. 이 단계에서도 human label이 없으므로 합성 OOD와 통합 rule 경로의 수치를
상용 모델 정확도로 부르지 않는다.

v1.81.0에서는 Qwen hard-negative 1,000행을 human review pack으로 준비했다.
pack은 synthetic target을 숨기고 state/question과 Qwen draft만 보여주며, 500개
counterfactual group과 167개 collision group을 uncertain-first로 정렬한다. 이제
정확도를 더 올리는 가장 중요한 입력은 새 synthetic 문장이 아니라 reviewer의
typed human label이다.

v1.82.0에서는 이 reviewer label을 독립적인 근거로 만들기 위해
[`control_labeling.md`](../control_labeling.md)의 기준과 teacher-blind dual review
workflow를 추가했다. 두 사람이 같은 state/question을 독립 판정하고,
agreement는 자동 합치되 disagreement는 제3자 adjudication을 거쳐야 최종
feedback이 된다. 이 장은 synthetic target이나 Qwen/Gemma 답을 human truth로
간주하지 않는 방법, 그리고 실제 human test가 생기기 전에는 99% 정확도를
주장하지 않는 이유를 기록한다.

v1.83.0에서는 evaluator도 같은 원칙을 강제한다. raw Student-head의 정확도,
confidence threshold를 통과한 accepted coverage, safety policy의 STOP recall을
서로 다른 필드로 보고하고, Wilson 95% 하한을 gate에 넣었다. synthetic에서
`2/2 STOP`이 나와도 하한이 `0.342380`이면 승격되지 않는다. 이 기록은 좋은
점수와 충분한 증거가 서로 다른 조건임을 보여준다.

v1.84.0에서는 held-out test 전체에 human label이 있는지 별도 gate로 검사한다.
`human_label_gate`와 `human_test_gate`를 나누었기 때문에 validation의 synthetic
점수와 독립 test의 human 점수를 섞어 production 승인을 내릴 수 없다.

v1.85.0에서는 skill별 accepted accuracy와 coverage를 기록한다. 이 정보가
있어야 “전체 99%” 뒤에 숨은 `HOLD → STOP` 보수 오류나 `APPROACH → MOVE`
경계 오류를 찾아 새 human group을 추가할 수 있다.

v1.86.0에서는 Student confidence를 human reviewer에게 숨긴 채 review 순서에만
사용한다. 이 active-review 방식은 불확실한 semantic group을 먼저 검수하면서도
모델 예측으로 human label이 오염되는 anchoring을 막는다.

v1.87.0에서는 confidence가 높아도 Student와 Qwen이 다르면 먼저 검수한다.
이 disagreement는 정답이 아니라 overconfident error 후보이며, adjudicated human
label이 확정되기 전에는 학습 target으로 사용하지 않는다.

v1.40.0에서는 실시간 적용을 위한 `control.skill@1` typed head와 safety-bounded
`control decide` 경로를 추가했다. 48개 synthetic smoke checkpoint의 train은
100%였지만 validation 37.5%, test 12.5%였으므로 production actuator에 연결하지
않는다. control latency p50 0.128ms는 in-process CPU model-only 측정이며,
simulator와 저수준 controller를 포함한 end-to-end 결과가 아니다.

v1.41.0에서는 HyperMemory `POST /v1/context`를 control skill tick에 연결할 수
있는 adapter를 추가했다. context는 최대 8개 summary와 2,048자로 잘라 모델에
전달하며, 매 frame의 raw transcript를 motor loop에 넣지 않는다.

v1.42.0에서는 `control simulate`로 정상 action과 safety STOP 시나리오를 같은
JSONL에서 재생하고 latency/recall을 기록한다. 이 harness의 100% 결과는
contract regression 증거이지, 새로운 상태에 대한 model 일반화 정확도는 아니다.

v1.43.0에서는 validation/test 정확도와 safety STOP recall을 함께 판정하는
독립 evaluator를 추가하고, dataset/checkpoint SHA와 split/group/human-label
메타데이터를 report에 고정했다. `reference-token-encoder` 실험은 validation
50.00%, test 62.50%로 baseline보다 좋아졌지만 99% gate에는 실패했으며,
human golden 데이터가 없는 synthetic 결과로 production 승격하지 않는다.

v1.44.0에서는 control dataset에 scenario/episode/semantic-group provenance를
요구하고, normalized exact input과 episode/group이 split 사이에 섞이면
실패시키는 validator를 추가했다. 기존 smoke fixture는 이 계약을 만족하지
않으므로 human golden seed를 만들기 전까지 품질 dataset으로 취급하지 않는다.

v1.45.0에서는 `hyperjev control seed`로 8개 skill을 균형 있게 만들고, 기존
human review/apply 흐름에 넣을 수 있는 pending queue를 추가했다. seed target은
synthetic draft이고 human label은 비어 있으므로, validator의 human-required
gate를 통과하기 전에는 정확도 학습/출시 데이터로 사용하지 않는다.

v1.46.0에서는 evaluator가 safety 전체 action accuracy 100%와 human label gate를
함께 확인하도록 강화했다. 따라서 synthetic seed의 split 100% 결과가 있어도
safety action 50% 또는 human label 0개이면 production-ready가 아니다.

v1.47.0에서는 control queue를 Qwen/Gemma draft 경로에 직접 연결했다. Qwen
16개는 schema-valid 16/16이었지만 Gemma는 think 활성화 generation이 매우
느려 비동기 judge queue가 필요하다는 운영 결과를 남겼다. 두 teacher 모두
human label을 대체하지 않는다.

v1.48.0에서는 BOW count encoder를 ablation했다. validation 50%로 token/ngram과
같아 승격하지 않았으며, control accuracy 상승의 우선순위를 human semantic
group과 hard-negative 데이터로 확정했다.

v1.49.0에서는 혼동쌍 counterfactual hard-negative queue를 추가했다. pair는
같은 episode/group과 split에 고정해 leakage를 막고, human label 전에는
synthetic seed로만 취급한다.

각 구현 단계는 다음 순서를 따른다.

```text
요구사항 확인 → 작은 변경 → targeted test → 전체 test/lint
→ version bump → commit → push → 원고 갱신
```
