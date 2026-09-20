# HyperJev changelog

이 파일은 정확도·coverage·fallback 정책의 중간 결과를 기록한다. 숫자는
동일한 dataset/split과 실행 명령으로 재현할 수 있는 경우에만 갱신한다.

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
