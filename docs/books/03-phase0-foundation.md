# 3장. Phase 0 기반을 만드는 과정

## 3.1 빈 저장소에서 시작한 규칙

작업 디렉터리는 `/home/dirmich/work/0.ai/hyperjev`이고 공개 remote는
`https://github.com/dirmich/hyperjev.git`이다. `.omx/`는 런타임 상태이므로
반드시 `.gitignore`에 둔다. 모델 endpoint 주소와 alias는 config에 둘 수 있지만
credential과 raw prompt는 저장하지 않는다.

첫 환경 확인 명령은 다음과 같다.

```bash
uv sync --extra dev
uv run hyperjev --version
uv run pytest -q
uv run ruff check src tests
uv run hyperjev doctor
uv run hyperjev registry validate
```

호스트에 `nvidia-smi`가 있는지와 `cargo`가 있는지는 별도로 기록한다.
Python contract 검증이 성공해도 CUDA kernel이나 Rust compile이 성공했다는
뜻은 아니다.

## 3.2 기본 프로젝트 구조

```text
configs/phase0.toml       Qwen/Gemma/Hyper Memory endpoint와 privacy 설정
registry/tasks/            versioned task registry
src/hyperjev/              contract, router, dataset, serving, CLI
crates/                    Rust production path의 workspace skeleton
python/                    training/calibration/evaluation/export 작업 경계
tests/                     unit, contract, golden, integration fixtures
docs/                      PRD, architecture, runbook, book manuscript
runs/                      실행 산출물; 민감한 출력은 저장하지 않음
```

기본 Python package는 Python 3.10 이상을 지원하고 `uv.lock`으로 dependency
해석을 고정한다. PyYAML과 tomli 외에는 초기 계약 검증에 새 dependency를
추가하지 않았다. PyTorch는 Student training image의 선택적 dependency로
경계를 만들었다.

## 3.3 첫 구현의 작은 단위

Phase 0은 한 번에 서버, 모델, 학습을 모두 만들지 않는다.

1. package/registry/config/CLI skeleton
2. typed request/response와 sample validation
3. teacher client와 dry-run/live baseline runner
4. mock API와 golden validation
5. router와 Hyper Memory ingestion gate
6. dataset factory와 Student manifest
7. serving, cache, shadow, registry, monitoring
8. golden review, metrics, training plan

각 단위는 변경 후 다음 증거를 남긴다.

```bash
uv run ruff check src tests
uv run pytest -q
git diff --check
git status --short --branch
git commit -m "..."
git push origin main
```

## 3.4 timeout을 짧게 두지 않은 이유

`doctor`의 3초 timeout은 `/models` connectivity probe용이다. Qwen/Gemma의
실제 generation은 thinking과 GPU queue 때문에 더 오래 걸릴 수 있으므로
config 기본 read timeout은 300초로 두고, benchmark의 `--timeout`은 명시적
실험에서만 override한다. timeout이 짧아서 실패한 것을 모델 품질 실패로
오해하지 않도록 probe와 generation 경로를 분리했다.

## 3.5 Phase 0의 진짜 완료 조건

Phase 0 코드가 존재하는 것과 Phase 0 product gate를 통과하는 것은 다르다.
현재 코드로 자동 검증 가능한 것은 registry, contract, mock/router, privacy,
metrics, CLI workflow다. 다음은 사람 또는 DGX runtime에서 추가해야 한다.

- 사람 1,000개 golden review
- Qwen/Gemma live baseline의 충분한 sample run
- 실제 encoder Student 학습과 checkpoint
- GPU latency, memory, TensorRT/ONNX export

