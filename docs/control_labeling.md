# HyperJev control 라벨링 기준

이 문서는 `control.skill@1`의 독립 human golden label을 만들기 위한 판정
기준이다. 대상은 로봇·게임 상태의 `state`와 질문이며, synthetic `target`,
Qwen draft, Gemma draft는 reviewer에게 공개하지 않는다.

## 1. 기본 원칙

1. 두 reviewer는 같은 입력을 독립적으로 보고 typed value 하나를 선택한다.
2. reviewer 화면에는 `state`, `question`, 후보 skill, 필요한 provenance만
   보인다. synthetic target과 teacher prediction은 숨긴다.
3. 상태에 즉시 충돌·비상 위험이 있으면 다른 의도가 있어도 `STOP`을 선택한다.
4. 상태가 충분하지 않아 안전한 행동을 결정할 수 없으면 임의의 이동 skill을
   선택하지 말고, 프로젝트의 안전 정책에 따라 `STOP`을 선택한다. 이 경우
   reason에는 정보 부족 또는 위험 신호를 기록한다.
5. 두 typed result가 완전히 같을 때만 agreement로 처리한다. disagreement는
   자동으로 어느 reviewer가 맞다고 결정하지 않고 제3자 adjudicator가 판정한다.
6. 최종 feedback은 `control review-finalize`로 만든 뒤에만 학습 target으로
   materialize한다.

## 2. Skill 판정표

| skill | 선택 기준 | 선택하지 않는 경우 |
| --- | --- | --- |
| `STOP` | 충돌 임박, 비상 정지, 장애물과의 접촉 위험, 제동 거리 부족, 안전 경계 침범 | 단순히 움직임을 잠시 기다리는 안정 상태 |
| `HOLD` | 현재 자세·위치가 안정적이고 새 명령이나 즉시 위험이 없어 현 상태를 유지 | 충돌 위험, 명확한 이동 목표, 복구가 필요한 고장 |
| `MOVE` | 특정 목표에 접근하는 것이 아니라 안전하게 전진·이동할 열린 경로가 있음 | 방향 전환이 필요하거나 목표가 명시적으로 가까이 고정됨 |
| `ROTATE` | 위치를 크게 바꾸기 전에 heading·방향을 바꾸는 것이 주된 다음 행동 | 곧장 전진하거나 후퇴하는 것이 주된 행동 |
| `APPROACH` | 특정 목표·물체·웨이포인트와의 거리를 줄이는 것이 주된 행동이고 안전함 | 목표가 손에 닿아 상호작용할 수 있거나 위험 때문에 멀어져야 함 |
| `RETREAT` | 위험·장애물·접근하는 물체에서 안전한 공간 쪽으로 거리를 늘림 | 즉시 충돌이면 `STOP`, 단순 방향 전환이면 `ROTATE` |
| `INTERACT` | 버튼·스위치·핸들·도구 등 대상이 정렬되고 실제로 조작 가능한 거리임 | 대상까지 먼저 이동해야 하거나 대상이 불명확함 |
| `RECOVER` | 위치 추정 상실, 균형 오류, stuck 상태 등 정상 제어를 회복하는 것이 주된 행동 | 충돌이 임박하면 복구보다 먼저 `STOP` |

### STOP 우선순위

다음 신호가 하나라도 명시되면 `STOP`이 우선이다.

- collision, impact, contact imminent, emergency stop, braking distance 부족
- 장애물이 안전 경계 안에 있음
- 현재 경로에서 멈추지 않으면 접촉할 가능성이 있음
- 센서가 즉시 위험을 보고하거나 상태가 안전 여부를 판정할 수 없음

`RETREAT`가 더 자연스러워 보이더라도 이미 접촉이 임박한 상태면 `STOP`이다.
반대로 `HOLD`는 안정적이고 위험이 없으며 단지 다음 명령을 기다리는 상태에만
사용한다. “pause”, “wait”, “stay”라는 단어만으로 `STOP`을 선택하지 않는다.

## 3. 검수 절차

### 독립 1차 검수

```bash
uv run hyperjev control review-session \
  --review-pack /tmp/control-qwen-hard-review-pack.jsonl \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --feedback-output /tmp/control-reviewer-a.jsonl \
  --reviewer control-a --blind --deduplicate-exact
```

Reviewer B는 별도의 feedback 파일과 reviewer ID로 같은 명령을 실행한다.
`--blind`에서는 teacher draft를 표시하지 않고 `e`로 typed value를 직접
입력해야 한다. 값은 JSON 전체가 아니라 `STOP`, `HOLD` 같은 후보 문자열이다.

### 합의와 adjudication

```bash
uv run hyperjev control review-agreement \
  --queue /tmp/hyperjev-control-hard-500.jsonl \
  --reviewer-a-feedback /tmp/control-reviewer-a.jsonl \
  --reviewer-b-feedback /tmp/control-reviewer-b.jsonl \
  --output /tmp/control-dual-agreement.jsonl \
  --minimum-agreement 0.98

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

agreement rate가 기준을 넘더라도 그것만으로 정확도 99%를 의미하지 않는다.
agreement는 reviewer consistency gate이고, accuracy는 별도로 보류한 독립
test set에서 계산한다. 불일치 항목은 adjudication 없이는 finalization되지
않으며, 누락 reviewer가 있으면 agreement gate가 실패한다.

## 4. 데이터 분할과 표본 계획

- 동일 episode, semantic group, counterfactual pair는 하나의 split에만 둔다.
- reviewer가 본 test item은 training queue로 되돌리지 않는다.
- train/validation은 모델 개발용이고, human test set은 최종 보고용으로 잠근다.
- 각 skill에 충분한 독립 group을 확보하고, STOP은 다른 skill보다 더 큰
  safety sample을 별도로 수집한다.
- 0건 실패를 근거로 safety recall을 주장할 때는 표본 수와 one-sided
  confidence bound를 함께 기록한다. 예를 들어 95% 신뢰수준에서 99.9%
  하한을 목표로 하면 약 3,000개의 독립적인 STOP 사례가 필요하다.

최종 보고서에는 sample/group 수, reviewer agreement, adjudication 수,
human-labeled coverage, per-skill confusion matrix, confidence interval,
accepted coverage, fallback/STOP 수, 그리고 model-only와 integrated 경로를
분리해서 기록한다.

