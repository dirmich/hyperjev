# 7장. 검증, DGX 운영, 그리고 남은 gate

## 7.1 반복 검증 명령

코드 변경 후 최소 검증은 아래 세 가지다.

```bash
uv run ruff check src tests
uv run pytest -q
git diff --check
```

CLI와 mock server를 바꾼 단계에서는 다음 smoke도 추가한다.

```bash
uv run hyperjev --version
uv run hyperjev registry validate
uv run hyperjev benchmark --dry-run --limit 1
uv run hyperjev golden --help
```

현재 마지막 확정 증거는 1.9.0 기준 Ruff 통과와 Python 테스트 55개 통과다.
Golden review와 metric unit test가 포함되어 있다.

## 7.2 DGX Spark runbook

```bash
uv sync --frozen --extra dev
uv run hyperjev doctor
uv run hyperjev teacher check
uv run hyperjev benchmark --limit 1
```

`doctor`는 ARM64, Docker, `nvidia-smi`, disk, registry, teacher probe를 보고한다.
실제 generation은 300초 read timeout을 사용하고, connectivity probe는 3초로
분리한다. `HYPERJEV_QWEN_BASE_URL`, `HYPERJEV_GEMMA_BASE_URL`, model alias는
환경 변수로 덮어쓸 수 있다.

컨테이너 검사는 기존 localhost Qwen에 접근할 수 있도록 host networking을
사용한다.

```bash
docker compose --profile phase0 run --rm phase0-check
```

## 7.3 완료와 미완료를 구분하는 표

| 항목 | 현재 증거 | 판정 |
| --- | --- | --- |
| Python package/CLI | uv build, version command | 완료 |
| registry/contracts | unit/contract tests | 완료 |
| Qwen smoke | DGX one-sample live run | smoke 완료, 품질 gate 미완료 |
| Gemma baseline | config/probe path | full live evidence 필요 |
| synthetic queue | deterministic 1,000 queue | human golden 아님 |
| human golden | review/apply workflow | 1,000개 검수 필요 |
| dataset factory | unit tests와 privacy/agreement code | 실데이터 run 필요 |
| Student model | manifest/optional torch builder | checkpoint 필요 |
| calibration/metrics | deterministic code/tests | held-out model data 필요 |
| Rust crates | workspace skeleton | host cargo 부재, DGX/toolchain compile 필요 |
| cache/registry/drift | local tests/metrics | production load gate 필요 |

이 표의 `필요`를 코드가 있다고 바꾸어 쓰지 않는 것이 책의 중요한 원칙이다.

## 7.4 실패를 기록하는 방법

실패는 숨기지 않고 원인 층을 분리한다.

- schema failure: typed parser/registry 계약 문제
- timeout: endpoint/GPU queue 또는 timeout 설정 문제
- disagreement: teacher label/confidence 불일치, review로 보냄
- privacy exclusion: training에 포함시키지 않는 정상 결과
- dependency unavailable: torch/cargo 같은 실행환경 준비 문제
- quality gate failure: code bug가 아니라 데이터/모델 품질 미달일 수 있음

