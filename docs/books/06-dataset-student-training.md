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

- boolean: accuracy, precision, recall, F1, Brier, ECE
- choice: accuracy, top-2 accuracy, macro-F1, multiclass Brier
- score: MAE, RMSE, interval-90 coverage/width

confidence는 route와 fallback 비용에도 쓰이므로 calibration은 장식용 report가
아니다. 잘못된 confidence로 auto-accept하면 Hyper Memory가 오염될 수 있다.

## 6.4 실제 학습과 이 책의 정직한 경계

현재 구현은 dataset validator와 training plan까지다. 실제 학습은 DGX image에
PyTorch, tokenizer/backbone, optimizer loop, checkpoint storage가 준비된 뒤
다음 단계로 연결한다.

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
