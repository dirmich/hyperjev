# 6장. Dataset, Student, Calibration, Training

## 6.1 Dataset이 training input이 되려면

teacher output을 받았다는 것만으로 학습 가능한 데이터가 되지 않는다.
`src/hyperjev/training.py`의 training dataset validator는 다음을 확인한다.

- canonical sample required fields와 registry task version
- target의 boolean/choice/score 타입과 candidate 범위
- `provenance.split`이 train/validation/test 중 하나인지
- `privacy_raw_inputs_stored`가 false인지
- optional `soft_target`의 타입, candidate coverage, probability 합
- dataset 전체의 SHA-256과 task/split count

이 검증을 거친 뒤 `train plan`은 checkpoint를 만들지 않고 실행 manifest를
생성한다. plan에는 Student head manifest, epoch/batch/lr/weight decay/seed,
precision, torch availability, `checkpoint.status=not_created`가 들어간다.

```bash
uv run hyperjev train plan \
  --dataset runs/phase2/dataset.jsonl \
  --output runs/phase3/training-plan.json \
  --model-id hyperjev-0.3b-poc \
  --epochs 3 \
  --batch-size 32 \
  --precision bf16
```

이 명령의 목적은 “학습했다”고 말하는 것이 아니라 DGX training job이 어떤
데이터와 head inventory를 사용해야 하는지 재현 가능한 입력으로 고정하는
것이다.

PyTorch가 설치된 DGX에서는 같은 검증된 dataset으로 reference trainer를
실행할 수 있다.

```bash
uv run hyperjev train run \
  --dataset runs/phase2/dataset.jsonl \
  --output runs/phase3/reference-student.pt \
  --model-id hyperjev-reference \
  --epochs 3 \
  --device auto
```

이 trainer는 production tokenizer가 아니라 deterministic byte-hash encoder를
사용하는 contract smoke path다. 결과는 `reference_student_checkpoint`와
checkpoint SHA로 보고되고, torch가 없는 호스트에서는 명시적인 dependency
error를 낸다. 따라서 이 경로의 성공을 production multilingual Student의
성공으로 해석하지 않는다.

## 6.2 Encoder + typed heads

Student의 공통 표현은 encoder 하나를 공유하고 task별 head가 typed output을
만든다.

```text
state + question
       ↓
shared multilingual encoder
       ↓
boolean heads   choice heads   score heads
       ↓             ↓             ↓
value/prob     selected/probs   value/interval
```

새 task는 자유형 prompt를 추가하는 대신 registry version, head spec, contract
test, evaluation fixture를 함께 추가해야 한다. 이 규칙이 multi-task model의
head와 API contract가 어긋나는 문제를 줄인다.

## 6.3 Calibration

`src/hyperjev/calibration.py`는 numerical framework 없이 held-out logits에 대한
deterministic temperature fitting과 probability conversion을 제공한다. 평가는
정확도 하나로 끝내지 않는다.

- boolean: accuracy, precision, recall, F1, Brier, ECE, adaptive ECE,
  NLL, AUROC, AUPRC, risk-coverage
- choice: accuracy, top-2 accuracy, macro-F1, multiclass Brier, NLL,
  adaptive ECE, risk-coverage
- score: MAE, RMSE, Spearman correlation, interval-90 coverage/width

confidence는 route와 fallback 비용에도 쓰이므로 calibration은 장식용 report가
아니다. 잘못된 confidence로 auto-accept하면 Hyper Memory가 오염될 수 있다.

CLI는 held-out logits/labels JSON을 받아 temperature와 before/after NLL,
sample count, input hash를 포함한 `calibration_manifest`를 만든다.

```bash
uv run hyperjev calibrate \
  --input heldout-logits.json \
  --output runs/phase3/calibration.json
```

## 6.4 실제 학습과 이 책의 정직한 경계

reference trainer를 실제로 실행해 checkpoint를 만들 수는 있지만, 이것은
production multilingual encoder가 아닌 deterministic byte-hash encoder를
사용하는 contract smoke path다. synthetic queue에서 중복을 제거한 36개
샘플로 생성한 reference 결과는 다음과 같다.

| 항목 | 결과 |
| --- | --- |
| checkpoint | `runs/phase3/reference-student.pt` |
| split | train 31, validation 2, test 3 |
| 전체 exact/threshold accuracy | 28/36 = 77.78% |
| held-out test accuracy | 1/3 = 33.33% |
| checkpoint SHA-256 | `7a4bb4d47a58295db479cb56ee675335e45706d80d54a87359b87a30f678e19a` |

평가용 `reference-ngram-encoder`도 추가했다. 100 epoch 학습 결과 Student
단독은 전체 97.22%, test 100%였고, confidence 0.95 이상 자동 수락 subset은
27/27 = 100%였다. rule fast-path를 합친 guarded synthetic path는 36/36 =
100%였지만 coverage는 80.56%에 불과하다. 이 수치는 사람 golden 일반화가
아니므로 production checkpoint 승격 근거가 아니다.

따라서 artifact 생성 gate는 통과했지만 quality gate는 통과하지 못했다. 실제
학습은 사람 검수 golden과 production multilingual tokenizer/backbone, optimizer
loop, checkpoint storage가 준비된 뒤 다시 실행해야 한다.

평가 시에는 `labels.human`이 존재하면 synthetic `target`을 정답으로 사용하지
않는다. human typed result를 task schema로 다시 검증하고, report에
`target_source=human`을 남긴다. human label이 없는 dataset의 수치는
`target_source=sample.target`인 exploratory 결과로만 취급한다.

```text
training plan
  → DGX tokenizer/backbone selection
  → typed-head loss + multi-task sampler
  → validation/calibration
  → checkpoint hash + model registry
  → shadow/canary promotion
```

checkpoint 파일이 없는데 model registry를 active로 만들지 않는다. 이 원칙은
개발 속도보다 재현성과 운영 안전성을 우선하는 HyperJev의 핵심이다.

## 6.6 Student를 실제 router에 연결하기

checkpoint를 생성하는 것과 제품 경로에서 사용하는 것은 별도 단계다.
`StudentClient`는 manifest의 backbone/vocab/sequence 설정으로 모델을 복원하고,
router는 다음 순서를 사용한다.

```text
rule → Student → Qwen → Gemma → human
```

기본 설정은 backward compatible하게 Student를 비활성화한다. 개발자가 검증된
로컬 checkpoint를 연결할 때는 다음과 같이 실행한다.

```bash
HYPERJEV_STUDENT_CHECKPOINT=runs/phase3/reference-ngram-student.pt \
HYPERJEV_STUDENT_MIN_CONFIDENCE=0.95 \
uv run hyperjev decide --request runs/phase3/student-smoke-request.json
```

Student가 confidence gate를 넘으면 parent LLM 호출을 줄인다. gate를 넘지
못하면 결과를 버리지 않고 Qwen fallback으로 전달하며, teacher도 실패하면
Gemma와 human review로 계속 내려간다. `abstained=true`는 확률이 높아 보여도
수락하지 않는다. 따라서 coverage를 높이기 위해 threshold를 무리하게 낮추는
것보다 accepted accuracy와 fallback 비용을 함께 기록해야 한다.

## 6.5 Training plan에서 registry manifest로

checkpoint가 실제로 생성되고 calibration manifest가 준비되면 두 artifact를
git provenance와 함께 registry manifest로 묶는다.

## 6.7 Mixed hard-negative curriculum을 선택하는 법

seed queue만으로 validation/test가 100%여도 기존 OOD에서 어떤 혼동쌍이 남는지
알 수 없다. 그래서 4,000-row bilingual seed와 1,000-row counterfactual
hard-negative queue를 provenance-aware merge한 5,000-row dataset을 사용했다.
학습 split은 `4032/484/484`이고 exact group은 `5000`, semantic group은 `4500`이다.

같은 데이터에 hard-negative loss weight `2`와 `4`를 각각 100 epoch 학습했다.
weight 4는 신규 hard queue 1,000개를 맞혔지만 기존 English OOD가 `33/40`으로
하락했다. weight 2는 hard queue `1000/1000`, 기존 English OOD `35/40`, Korean
OOD `34/40`으로 seed-only 후보의 OOD 기준을 유지했다. 따라서 단일 aggregate
score가 아니라 `기존 OOD 회귀 없음 + hard queue 개선 + STOP recall 유지`를
선택 규칙으로 삼아 weight 2를 다음 review 후보로 보류 선택했다.

이 실험의 전체 정확도는 여전히 synthetic target 기준이다. human label은
`0/5000`이고 quality evaluator는 `--require-human-test`에서 실패한다. 그러므로
hard-negative curriculum은 혼동쌍을 줄이는 학습 방법의 증거이지, 곧바로
production accuracy나 99% 보증이 아니다. 다음 학습은 human adjudication으로
확정된 correction만 target으로 materialize하고, 사람 오류가 확인된 pair만
추가해야 한다.

```bash
uv run hyperjev model manifest \
  --training-plan runs/phase3/training-plan.json \
  --calibration runs/phase3/calibration.json \
  --checkpoint runs/phase3/reference-student.pt \
  --git-commit "$(git rev-parse HEAD)" \
  --output runs/phase6/model-manifest.json
```

이 명령은 checkpoint SHA-256을 다시 계산하고, training plan의 dataset hash와
typed head/task version, calibration version, precision을 manifest에 고정한다.
따라서 파일 이름이나 최신 파일 선택만으로 모델을 식별하지 않는다. 생성된
manifest를 registry에 등록한 뒤에도 lifecycle은 `trained → evaluated →
calibrated → candidate → canary → active` 순서의 명시적 전이를 따른다.
reference checkpoint는 `trained` 상태로 registry manifest에 등록했지만,
test 3개 중 1개만 맞았으므로 `evaluated`, `candidate`, `canary`, `active`로
승격하지 않았다. calibration도 held-out 3개에 한정되어 production gate가
아니다. production checkpoint 등록과 promotion은 여전히 별도 품질 gate다.

## 6.7 control 전용 typed head

실시간 게임/로봇 실험은 memory/query task와 다른 위험 모델을 가지므로
`registry/control_tasks/control.skill.yaml`에 `control.skill@1`을 별도 정의했다.
이 head의 출력은 직접 actuator 값이 아니라 abstract skill이다. control
checkpoint가 memory checkpoint와 registry를 공유하지 않게 해 task head의
candidate 순서나 manifest가 섞이는 사고를 막는다.

현재 CLI는 다음 세 단계로 재현된다.

```text
control train → control evaluate → control decide
```

`control evaluate`는 train accuracy만 보고하지 않고 validation/test를 별도
출력한다. 48개 synthetic sample의 reference checkpoint는 train `32/32`,
validation `3/8`, test `1/8`이었다. 이것은 작은 dataset과 reference
byte/ngram encoder가 새로운 표현을 이해하지 못한다는 증거이며, model
confidence도 calibration되지 않았음을 보여준다. 따라서 control model은
human-labelled state/action data, simulator trajectory, collision/unsafe-action
metric, OOD detector를 추가한 뒤에야 다음 lifecycle 단계로 이동한다.

runtime에서 safety policy는 observation age 100ms, confidence 0.90, action TTL
100ms를 기본으로 삼고 위반 시 `STOP`을 만든다. 이 방어층은 model accuracy를
대체하지 않는다. 가장 안전한 구조는 HyperJev가 skill을 고르고, HyperMemory가
짧은 relevant context/episode summary를 제공하며, deterministic controller가
최종 움직임을 제한하는 세 층 구조다.

HyperMemory adapter는 `/v1/context`의 compact result를 받아 observation에
주입한다. 이 호출은 고주기 sensor loop와 분리하고, memory timeout이 model
input을 무한히 기다리게 하지 않는다. summary는 prompt에 넣기 전에 개수와
문자 수를 제한하므로, 기억이 많아질수록 control latency가 무제한 증가하지
않는다. 저장/검색 실패는 안전한 motion을 계속 생성하기 위한 신호가 아니라
fallback 또는 hold/stop 사유로 기록한다.
