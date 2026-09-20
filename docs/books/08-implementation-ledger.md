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
| 1.10.0 | `4bd6b53` | validated training plan CLI와 프로젝트 책 원고 |
| 1.11.0 | `ea6a55a` | optional reference training run과 checkpoint hash |
| 1.11.1 | `511326e` | reference training milestone 문서 기록 |
| 1.12.0 | `13287a1` | PRD quality/calibration metric coverage |
| 1.13.0 | `3fd213a` | Gemma generation timeout을 900초로 확장 |
| 1.13.1 | `2064058` | Gemma timeout 결과와 운영 gate 문서화 |
| 1.13.2 | `f8a2333` | 최종 release gate checklist 장 추가 |
| 1.13.3 | `ac30dc9` | 구현 일지와 책 원고 ledger 정리 |
| 1.14.0 | `87a77d5` | calibration manifest CLI |
| 1.14.1 | `8b2d067` | parent LLM/rule/mock 성능 결과 문서화 |
| 1.15.0 | `0d32acd` | training/calibration/checkpoint 기반 model manifest |
| 1.15.1 | `72d6a77` | model manifest milestone 원고 기록 |
| 1.16.0 | `a975d77` | 30-sample confidence/coverage와 성능 차이 분석 |
| 1.17.0 | `b5073d8` | reference Student checkpoint path, dotted task head, rule accuracy fix |
| 1.17.1 | `21687f8` | reference Student 정확도, calibration, rule coverage 결과 문서화 |
| 1.17.2 | `ce0430d` | Student checkpoint milestone 장부 및 version 정합성 |
| 1.17.3 | `52db7f0` | 99% accuracy gate와 baseline changelog 고정 |
| 1.18.0 | `f4d4319` | guarded Student evaluator와 accepted accuracy/coverage 분리 |
| 1.19.0 | 진행 중 | n-gram Student와 rule+Student guarded 100% synthetic path |
| 1.20.0 | `6913c8e` | human label을 요구하는 production quality gate |
| 1.20.1 | `99169d2` | 1,000개 synthetic stress 결과 기록 |
| 1.21.0 | `ca49048` | optional Student router 연결과 abstain fallback guard |
| 1.21.1 | `55c640a` | Student router 장부와 version 정합성 기록 |
| 1.22.0 | `9366aa7` | human golden label 기준 정확도 평가 정정 |
| 1.22.1 | `04ffea7` | 1,000개 재평가와 synthetic source 증거 기록 |
| 1.23.0 | 진행 중 | Gemma draft label pipeline과 human review 경계 |
| 1.24.0 | 진행 중 | Qwen/Gemma 선택형 golden draft와 Qwen 재시도 경로 |
| 1.24.1 | 진행 중 | qwen38fn 실제 20-sample draft probe와 schema/latency 결과 |
| 1.25.0 | 진행 중 | choice probability 합 prompt hardening과 prompt v2 provenance |
| 1.25.1 | 진행 중 | Qwen prompt v2 20-sample 재시험 결과 |
| 1.25.2 | 진행 중 | qwen38fn 1,000-sample draft 전체 실행과 task별 결과 |
| 1.26.0 | 진행 중 | explicit raw local human review pack과 target leakage guard |
| 1.26.1 | 진행 중 | 1,000-item Qwen review pack 생성 및 leakage scan |
| 1.27.0 | 진행 중 | human remember-worthiness rubric과 검수 기준 문서 |
| 1.28.0 | 진행 중 | next/previous navigable human review session |
| 1.29.0 | 진행 중 | task-aware value-only correction input |
| 1.29.1 | 진행 중 | review-session EOF/pipe safe quit |
| 1.30.0 | 진행 중 | 실제 human feedback 31건 적용, partial Student 평가, 969건 pending gate 기록 |
| 1.31.0 | 진행 중 | exact duplicate review group과 label propagation |
| 1.32.0 | 진행 중 | 1,000건 feedback 적용과 human-source Student 전체 평가 |
| 1.33.0 | 진행 중 | 실제 mini-batch 학습과 human-target 재학습, remember 규칙 보강 |
| 1.34.0 | 진행 중 | source group 부재 시 내용 기반 split으로 leakage 방지 |
| 1.35.0 | 진행 중 | human-target checkpoint의 train/validation/test split 평가 기록 |
| 1.36.0 | 진행 중 | Student unique exact-group 평가 CLI와 group 수 report |

## 다음 기록 규칙

다음 단계가 완료되면 이 표에 version과 short SHA를 추가하고, 해당 장의
`현재 증거`도 같이 갱신한다. 특히 다음 네 가지는 별도 기록이 필요하다.

1. training plan의 실제 DGX 실행
2. 사람 golden 1,000개와 manifest hash
3. Student checkpoint/calibration/promotion (reference checkpoint는 생성됐지만 production 승격 전)
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
