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
| 1.37.0 | 진행 중 | remember 부정/과거형 반례와 high-precision rule 회귀 테스트 |
| 1.38.0 | 진행 중 | Student dynamic padding으로 JEv형 low-latency inference 경로 최적화 |
| 1.39.0 | 진행 중 | 실시간 control observation/action 계약과 deterministic safety shield |
| 1.40.0 | 진행 중 | control typed head CLI, smoke checkpoint, split별 일반화 gate |
| 1.41.0 | 진행 중 | HyperMemory `/v1/context` bounded control context adapter |
| 1.42.0 | 진행 중 | deterministic control scenario replay와 latency/safety report |
| 1.43.0 | 진행 중 | accuracy evaluator, split metadata, reference token encoder 실험 |
| 1.44.0 | 진행 중 | control dataset provenance와 cross-split leakage validator |
| 1.45.0 | 진행 중 | 균형 잡힌 human-review control seed generator와 CLI |
| 1.46.0 | 진행 중 | safety action accuracy와 human-label production gate 강화 |
| 1.47.0 | 진행 중 | control Qwen/Gemma draft routing과 teacher latency/schema 기록 |
| 1.48.0 | 진행 중 | reference BOW encoder ablation과 validation 기반 후보 탈락 |
| 1.49.0 | 진행 중 | control counterfactual hard-negative queue와 same-split pair gate |
| 1.50.0 | 진행 중 | hard-negative token training의 100% synthetic 결과와 human gate 기록 |
| 1.51.0 | 진행 중 | human typed label을 학습 target으로 materialize하는 control CLI |
| 1.52.0 | 진행 중 | seed+hard-negative merge와 explicit STOP safety rule 검증 |
| 1.53.0 | 진행 중 | control registry 기반 human review pack/session/apply CLI |
| 1.54.0 | 진행 중 | Qwen/Gemma typed adjudication과 disagreement human gate |
| 1.55.0 | 진행 중 | per-skill accuracy/recall과 confusion matrix production gate |
| 1.56.0 | 진행 중 | long-running Qwen/Gemma draft resume와 shared-slot 기록 |
| 1.57.0 | 진행 중 | class-balanced training ablation과 calibration 실패 gate |
| 1.58.0 | 진행 중 | control threshold별 risk-coverage report와 safety 분리 gate |
| 1.59.0 | 진행 중 | held-out control logits calibration manifest와 human gate |
| 1.60.0 | 진행 중 | runtime calibration binding과 checkpoint hash guard |
| 1.61.0 | 진행 중 | Qwen 800 control draft와 bounded probability repair |
| 1.62.0 | 진행 중 | teacher draft quality evaluator와 synthetic/human gate 분리 |
| 1.63.0 | 진행 중 | Qwen hard-negative semantic gate와 human priority pack |
| 1.64.0 | 진행 중 | uncertainty-first review pack 정렬과 target leakage 없는 priority manifest |
| 1.65.0 | 진행 중 | counterfactual pair provenance 보존과 pair-aware review ordering |
| 1.66.0 | 진행 중 | control CLI priority wiring과 실제 1,000건 pair/leakage smoke 검증 |
| 1.67.0 | 진행 중 | generic golden review CLI에도 priority flag를 노출해 경로 정합성 확보 |
| 1.68.0 | 진행 중 | overconfident teacher 오류를 잡는 counterfactual pair-collision priority |
| 1.69.0 | 진행 중 | review manifest에 counterfactual/collision priority 통계 기록 |
| 1.70.0 | 진행 중 | hard-negative weighted loss ablation과 safety gate 탈락 기록 |
| 1.71.0 | 진행 중 | Gemma hard-negative timeout probe와 async judge 격리 gate |
| 1.72.0 | 진행 중 | compound-phrase control fast path와 synthetic hard-negative 100% 경로 |
| 1.73.0 | 진행 중 | skill별 contradiction blocker와 fast-path negative regression |
| 1.74.0 | 진행 중 | reproducible fast-path evaluator와 queue SHA/latency/human gate report |
| 1.75.0 | 진행 중 | checkpoint-backed integrated runtime replay와 safety/source report |
| 1.76.0 | 진행 중 | runtime source별 fallback latency report와 seed replay |
| 1.77.0 | 진행 중 | compositional OOD fixture와 model-only/integrated 정확도 분리 gate |
| 1.78.0 | 진행 중 | train-only compositional augmentation, STOP safety interlock, OOD 95% model-only / 100% integrated |
| 1.79.0 | 진행 중 | balanced semantic-boundary augmentation, OOD 97.5% model-only / 100% integrated, safety precision failure 기록 |
| 1.80.0 | 진행 중 | normal APPROACH compound fast path, safety action 100% 유지, 추가 학습 ablation 비교/폐기 기록 |
| 1.81.0 | 진행 중 | Qwen hard-negative 1,000행 human review pack, target 숨김/uncertain-first 검증 |
| 1.82.0 | 진행 중 | control 전용 라벨 기준, teacher-blind dual review, disagreement adjudication/finalize gate |
| 1.83.0 | 진행 중 | raw Student-head/accepted/safety 분리 evaluator, Wilson confidence gate, coverage gate |
| 1.84.0 | 진행 중 | held-out test human label gate와 split별 human status report |
| 1.85.0 | 진행 중 | per-skill accepted accuracy/coverage/confidence interval과 confusion 진단 |
| 1.86.0 | 진행 중 | Student uncertainty active review ordering, prediction/target leakage 방지 |
| 1.87.0 | 진행 중 | Student–Qwen disagreement 우선순위로 overconfident 오류 수집 |
| 1.88.0 | 진행 중 | blind dual review manifest에 action별 agreement/disagreement 진단 추가 |
| 1.89.0 | 진행 중 | review-agreement에 낮은 agreement 우선 human-readable summary 추가 |
| 1.90.0 | 진행 중 | low-agreement action을 먼저 배치하는 focus review ordering 추가 |
| 1.91.0 | 진행 중 | dual-review manifest에서 low-agreement focus action 자동 추출 |
| 1.92.0 | 진행 중 | automatic focus 입력에 target-exclusion contract guard 추가 |
| 1.93.0 | 진행 중 | control train human-only target/source gate와 checkpoint 미생성 검증 |
| 1.94.0 | 진행 중 | control simulation p95/p99/max latency report와 threshold gate 추가 |
| 1.95.0 | 진행 중 | 최소 500 expected safe STOP evidence gate와 Wilson 표본 기준 추가 |
| 1.96.0 | 진행 중 | 고유 episode/group을 포함한 재현 가능한 500-case safety matrix와 scenario ID 중복 방지 추가 |
| 1.97.0 | 진행 중 | blind human review의 결정론적 offset/limit batch와 전체·배치 progress report 추가 |
| 1.98.0 | 진행 중 | control review-session batch handler forwarding 교정과 실제 1,000-row smoke 검증 |
| 1.99.0 | 진행 중 | review-status로 전체/분할 human coverage와 held-out test readiness 관측 추가 |
| 1.100.0 | 진행 중 | 500-case safety matrix의 실제 control simulate latency replay와 5ms p95/p99 gate 검증 |
| 1.101.0 | 진행 중 | control review pack의 held-out train/validation/test split filter와 test-only pack 검증 |
| 1.102.0 | 진행 중 | Gemma control draft의 60s/300s timeout boundary 측정과 비동기 judge 격리 결정 |
| 1.103.0 | 진행 중 | control simulation 중복 scenario ID 차단과 500-case 고유 safety replay 검증 |
| 1.104.0 | 진행 중 | production materialize의 선택적 strict dual-review provenance gate |
| 1.105.0 | 진행 중 | Korean OOD/augmentation, BOW 후보 비교, multilingual fast path 검증 |
| 1.106.0 | 진행 중 | Korean HOLD/vocabulary ablation 비교, latency·정확도 gate로 후보 폐기 |
| 1.107.0 | 진행 중 | raw/integrated 측정 경계 정정, char/hybrid BOW ablation 및 후보 폐기 |
| 1.108.0 | 진행 중 | review-status test/materialize readiness strict exit-code gate |
| 1.109.0 | 진행 중 | 선택 BOW checkpoint의 500-case 고유 safety replay와 latency gate |
| 1.110.0 | 진행 중 | model promotion의 human/quality report와 checkpoint·dataset hash binding |
| 1.111.0 | 진행 중 | control review seed prompt 다양화와 800-row unique prompt 재생성 |
| 1.112.0 | 진행 중 | Qwen v2 800-row draft, per-skill 오류 분석, target-excluded review pack |
| 1.113.0 | 진행 중 | v2 diversity + hard-negative BOW ablation 측정 및 accuracy 회귀 후보 폐기 |
| 1.114.0 | 진행 중 | state-only control BOW ablation으로 질문 노이즈 분리, 기준선 미달 후보 폐기 |
| 1.115.0 | 진행 중 | malformed quality report를 traceback 없이 명시적 gate failure로 거부 |
| 1.116.0 | 진행 중 | segmented state/question BOW ablation 측정 및 기준선 미달 후보 폐기 |
| 1.117.0 | 진행 중 | v2 source queue human-label gate 상태와 다음 검수 stop condition 기록 |
| 1.118.0 | 진행 중 | Korean integrated control fast path 40/40·1800/1800 및 latency 재검증 |
| 1.119.0 | 진행 중 | Korean fast path blocker 5종 defer regression 고정 |
| 1.120.0 | 진행 중 | Gemma reasoning option·candidate probe 및 partial-run evaluator denominator 교정 |
| 1.121.0 | 진행 중 | Gemma control candidate 13/20 rejection 및 partial draft accuracy gate 교정 |
| 1.122.0 | 진행 중 | Qwen 반복 control phrase fast path 추가, blocker 회귀 확장, Wilson 하한·381 group 99% gate |

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
