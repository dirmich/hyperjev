# Human Golden Labeling Guide

이 문서는 `memory.remember_worthy@1`의 사람 검수 기준이다. reviewer는
synthetic `target`이나 teacher draft를 정답으로 보지 않고, `state`와
`question`만 읽어 독립적으로 판단한다.

## 무엇을 판단하는가

질문은 “이 문장을 지금 저장할 것인가?”가 아니라 “나중에 Hyper Memory에서
검색·개인화·의사결정에 다시 사용할 장기 기억으로 남길 가치가 있는가?”이다.
사실인지 여부, 문장이 그럴듯한지 여부, 민감한 데이터 저장 허용 여부와는
별개의 판단이다. 개인정보·secret은 semantic value가 있어도 privacy policy가
허용하지 않으면 저장하지 않는다.

## `true` 기준

다음 중 하나 이상이면 `true`를 우선한다.

- 앞으로 반복해서 적용할 결정, 정책, 약속, 변경 사항
- 사용자의 지속적인 선호나 작업 방식
- 프로젝트·사람·목표·제약에 관한 안정적인 사실
- 미래 대화나 작업에서 검색될 가능성이 높고, 저장하지 않으면 다시 묻거나
  재추론해야 하는 정보

예:

| state | label | 이유 |
| --- | --- | --- |
| `다음 분기부터 배포 승인 절차를 바꾸기로 했다.` | true | 미래의 작업 흐름에 반복 적용되는 결정 |
| `나는 답변을 받을 때 코드 예제를 먼저 보고 싶다.` | true | 지속적인 사용자 선호 |
| `프로젝트의 기본 배포 브랜치는 main이다.` | true | 반복 참조되는 프로젝트 사실 |

## `false` 기준

다음이면 `false`를 우선한다.

- 한 번뿐인 일상 사건이나 현재 시점에만 유효한 상태
- 미래 검색·개인화·행동 변경에 거의 도움이 되지 않는 정보
- 인사, filler, 일반 상식처럼 사용자 또는 프로젝트에 특화되지 않은 내용

예:

| state | label | 이유 |
| --- | --- | --- |
| `오늘 점심으로 비빔밥을 먹었다.` | false | 일회성 사건이며 재사용 가치가 낮음 |
| `오늘 회의실을 한 번 예약했다.` | false | 일시적인 작업 상태 |

## 애매한 경우

기억 누락(false negative)의 비용이 과잉 저장(false positive)보다 크다는 PRD
비용행렬을 적용하되, 애매함을 무조건 `true`로 바꾸지는 않는다. 미래에
재사용될 구체적 근거가 있으면 `true`, 근거가 없으면 `false`로 판정하고,
정말 합의할 수 없으면 `abstained: true`와 reason을 남겨 별도 adjudication
대상으로 보낸다.

`probability`는 label의 확신도다. 사람이 원문을 확인하고 확정한 hard label은
보통 `1.0`을 사용하고, 경계 사례는 `0.7~0.9`와 구체적인 reason을 남긴다.

## 검수 절차

1. review pack에서 `state`, `question`, Qwen draft를 읽는다.
2. synthetic target은 보지 않는다. review pack에는 의도적으로 포함되지 않는다.
3. 위 기준으로 `true`/`false`를 독립 결정한다.
4. draft가 맞으면 그대로 승인하고, 틀리면 typed correction을 입력한다.
5. 판단 근거를 `--reason`에 짧게 남긴다.

첫 synthetic item의 state가 `다음 분기부터 배포 승인 절차를 바꾸기로 했다.`라면,
위 rubric상 `true`가 자연스럽지만, 실제 golden label은 reviewer가 원문을 확인해
직접 확정해야 한다.
