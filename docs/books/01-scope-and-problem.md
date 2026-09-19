# 1장. HyperJev의 범위와 문제

## 1.1 출발점

HyperJev는 Hyper Memory가 매번 무거운 언어 모델에 전체 판단을 맡기지 않도록
하는 local-first typed decision layer다. 입력은 대화나 문서에서 추출된 상태와
질문이고, 출력은 자연어가 아니라 `boolean`, `choice`, `score`처럼 명시된
decision contract다.

이 프로젝트의 개발 환경은 DGX Spark를 중심으로 한다. 현재 teacher 역할은
다음처럼 고정한다.

| 모델 | 책임 | 사용하지 않는 책임 |
| --- | --- | --- |
| Qwen3 8B | 데이터 생성, labeler, fallback | 최종 독립 judge로 단독 신뢰하지 않음 |
| Gemma 4 | 교차 검증, judge | 주 데이터 생성기의 유일한 출처가 아님 |
| HyperJev Student | 제품 주력 inference | teacher prompt를 그대로 복사하지 않음 |
| HyperJev-D | diffusion 비교 연구 | 제품 baseline 경로에 섞지 않음 |

핵심 경계는 “teacher를 빠르게 붙였다”와 “Student 모델이 완성됐다”를 같은
말로 부르지 않는 것이다. Router와 dataset factory가 동작해도 encoder + typed
heads checkpoint가 없으면 Student inference 완료가 아니다.

## 1.2 무엇을 해결하는가

Hyper Memory에는 다음과 같은 작은 판단이 반복된다.

- 이 문장을 장기 기억으로 남길 것인가
- 기억은 fact, preference, decision 중 무엇인가
- 기억의 중요도는 어느 정도인가
- query는 profile, temporal, exact 중 어떤 경로로 갈 것인가
- 새 기억이 기존 기억을 contradict, extend, duplicate 하는가
- 문서 변경이 사실의 의미를 바꾸었는가

이 판단은 모두 서로 다른 출력 타입과 오류 비용을 가진다. 따라서 단일 자유형
텍스트 생성 모델의 응답을 후처리하는 대신 task registry에서 출력 타입,
candidate, threshold, fallback 정책을 버전 관리한다.

## 1.3 비목표

이 책의 구현에서 하지 않는 일도 명확히 남긴다.

- HyperJev-D를 제품 baseline으로 조기 통합하지 않는다.
- teacher의 chain-of-thought나 raw prompt를 dataset과 log에 저장하지 않는다.
- synthetic target을 사람 검수 golden label이라고 부르지 않는다.
- PyTorch가 없는 개발 호스트에서 checkpoint가 생성됐다고 주장하지 않는다.
- Rust compiler가 없는 호스트에서 Rust workspace compile 성공을 주장하지 않는다.

## 1.4 성공의 정의

성공은 “코드가 많다”가 아니라 아래의 경계를 모두 증거로 설명할 수 있는
상태다.

1. registry와 typed contract가 잘못된 입력을 거부한다.
2. Qwen/Gemma 연결과 timeout, response format, fallback이 재현 가능하다.
3. dataset이 privacy gate, redaction, agreement, grouped split을 통과한다.
4. Student head inventory가 registry에서 자동으로 나온다.
5. calibration과 평가 metric이 결과의 신뢰도까지 측정한다.
6. serving, cache, shadow, model registry, drift의 상태 전이가 추적된다.
7. human golden gate와 실제 DGX checkpoint gate는 통과 여부를 정직하게
   표시한다.

