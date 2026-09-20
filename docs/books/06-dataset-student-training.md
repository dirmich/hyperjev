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

## 6.5 Training plan에서 registry manifest로

checkpoint가 실제로 생성되고 calibration manifest가 준비되면 두 artifact를
git provenance와 함께 registry manifest로 묶는다.

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
