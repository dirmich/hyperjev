# 8장. 구현 일지: version, commit, push

이 장은 작업을 재현하는 독자를 위해 각 단계의 작은 결과와 Git 증거를
기록한다. 모든 단계는 테스트 후 version bump, commit, `origin main` push의
순서로 진행했다.

| version | commit | 단계 |
| --- | --- | --- |
| 0.1.0 | `34f1352` | package, config, registry, 기본 구조 |
| 0.2.0 | `363a611` | typed contracts와 sample validation |
| 0.3.0 | `01d1e13` | teacher probe와 baseline runner |
| 0.4.0 | `daabccd` | container/runbook Phase 0 환경 |
| 0.5.0 | `00e5fa8` | evaluation과 golden gate |
| 0.6.0 | `ac7137b` | teacher generation timeout 확장 |
| 0.7.0 | `f4af87a` | mock API와 teacher format profile |
| 0.8.0 | `24f7f00` | deterministic golden queue |
| 0.9.0 | `eec6551` | rule teacher router와 provenance |
| 1.0.0 | `adac673` | API와 Hyper Memory integration |
| 1.1.0 | `4c2d4a5` | agreement-gated dataset factory |
| 1.2.0 | `371dad4` | typed Student contract와 calibration |
| 1.3.0 | `165e9b6` | batch serving과 shadow logging |
| 1.4.0 | `09e9931` | privacy-safe decision cache |
| 1.5.0 | `38e2156` | model promotion registry |
| 1.6.0 | `d35fb81` | v1 API registry endpoints |
| 1.7.0 | `dee8436` | drift monitoring metrics |
| 1.7.1 | `6966b13` | drift operation documentation |
| 1.8.0 | `582a696` | append-only golden review workflow |
| 1.9.0 | `058d856` | typed evaluation quality/calibration metrics |

## 다음 기록 규칙

다음 단계가 완료되면 이 표에 version과 short SHA를 추가하고, 해당 장의
`현재 증거`도 같이 갱신한다. 특히 다음 네 가지는 별도 기록이 필요하다.

1. training plan의 실제 DGX 실행
2. 사람 golden 1,000개와 manifest hash
3. Student checkpoint/calibration/promotion
4. Rust/API production compile과 load test

## 재현 명령

```bash
git clone https://github.com/dirmich/hyperjev.git
cd hyperjev
uv sync --frozen --extra dev
uv run pytest -q
git log --oneline --decorate --max-count=20
```

책의 각 장은 이 명령으로 확인 가능한 코드와 실행 명령을 기준으로 쓰며,
모델 endpoint에 대한 현재 접근 권한이나 DGX local state를 공개 repository에
복사하지 않는다.

