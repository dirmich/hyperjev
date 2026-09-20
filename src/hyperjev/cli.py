"""Command line entry point for Phase 0 operations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .baseline import run_benchmark
from .calibration import fit_temperature
from .config import ConfigError, load_config
from .contracts import DecisionRequest
from .control import ControlObservation, ControlSafetyPolicy, ControlStudentClient
from .control_data import (
    generate_control_boundary_training_queue,
    generate_control_compositional_training_queue,
    generate_control_hard_negative_queue,
    generate_control_korean_training_queue,
    generate_control_review_queue,
    materialize_control_human_dataset,
    merge_control_datasets,
)
from .dataset_factory import build_dataset, validate_dataset
from .doctor import system_checks
from .evaluation import evaluate_run, validate_golden_set
from .golden import append_golden_feedback, apply_golden_feedback, generate_review_queue
from .golden_draft import adjudicate_teacher_drafts, generate_teacher_draft
from .mock_server import serve
from .model_registry import STATUSES, ModelRegistry, build_model_manifest
from .registry import RegistryError, TaskRegistry
from .review_pack import (
    compare_control_reviews,
    control_review_status,
    export_review_pack,
    finalize_control_reviews,
    format_control_review_action_summary,
    load_control_review_focus_actions,
    run_adjudication_session,
    run_review_session,
)
from .routing import DecisionRouter
from .student import StudentConfig, student_manifest
from .student_inference import evaluate_student_checkpoint, write_student_evaluation
from .teachers import probe_teacher
from .training import (
    TrainingConfig,
    TrainingDependencyError,
    run_reference_training,
    write_training_plan,
)


class CalibrationInputError(ValueError):
    """Raised when the calibration CLI input envelope is malformed."""


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="path to a Phase 0 TOML config")


def _load(config_path: str | None):
    config = load_config(config_path)
    registry = TaskRegistry.load(config.registry_path)
    return config, registry


def _registry_validate(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    print(json.dumps({"registry": str(config.registry_path), "count": len(registry), "tasks": registry.ids()}))
    return 0


def _teacher_check(args: argparse.Namespace) -> int:
    config, _ = _load(args.config)
    probes = [probe_teacher(settings, timeout_s=args.timeout) for settings in config.teachers.values()]
    print(json.dumps({"teachers": [probe.to_dict() for probe in probes]}, ensure_ascii=False))
    return 0 if all(probe.ok for probe in probes) else 1


def _doctor(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    probes = [probe_teacher(settings, timeout_s=args.timeout) for settings in config.teachers.values()]
    checks = system_checks(config.root)
    report = {
        "version": __version__,
        "phase": config.phase,
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "docker": shutil.which("docker") is not None,
        "system_checks": [check.to_dict() for check in checks],
        "registry_count": len(registry),
        "teachers": [probe.to_dict() for probe in probes],
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0 if all(probe.ok for probe in probes) and all(check.ok for check in checks) else 1


def _benchmark(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    providers = tuple(args.provider) if args.provider else ("qwen", "gemma")
    run = run_benchmark(
        config,
        registry,
        limit=args.limit,
        providers=providers,
        dry_run=args.dry_run,
        timeout_s=args.timeout,
        output_path=args.output,
    )
    print(json.dumps({"output": str(run.output_path), "manifest": run.manifest}, ensure_ascii=False))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    report = evaluate_run(args.run, args.samples or config.smoke_set_path, registry)
    serialized = json.dumps(report, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    print(serialized)
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CalibrationInputError("calibration input must be a JSON object")
    logits = payload.get("logits")
    labels = payload.get("labels")
    if not isinstance(logits, list) or not isinstance(labels, list):
        raise CalibrationInputError("calibration input requires logits and labels lists")
    result = fit_temperature(
        logits,
        labels,
        minimum=args.minimum,
        maximum=args.maximum,
        steps=args.steps,
    )
    input_digest = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    manifest = {
        "record_type": "calibration_manifest",
        "calibration_version": f"cal-{input_digest[:12]}",
        "input_sha256": input_digest,
        "result": result.to_dict(),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


def _golden_validate(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    report = validate_golden_set(
        args.samples or config.smoke_set_path,
        registry,
        minimum_count=args.minimum_count,
        require_human=not args.allow_unreviewed,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ready"] else 1


def _golden_generate(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    output = args.output or str(config.runs_path / "phase0-review-queue.jsonl")
    report = generate_review_queue(output, registry, count=args.count, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _golden_draft(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    provider = args.provider
    queue = args.queue or config.runs_path / "phase0-review-queue.jsonl"
    output = args.output or config.runs_path / f"{provider}-golden-draft.jsonl"
    report = generate_teacher_draft(
        config,
        registry,
        queue,
        output,
        provider=provider,
        limit=args.limit,
        timeout_s=args.timeout,
        resume=args.resume,
    )
    print(json.dumps(report["manifest"], ensure_ascii=False))
    return 0 if report["manifest"]["error_count"] == 0 else 1


def _golden_review(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    queue = args.queue or str(config.runs_path / "phase0-review-queue.jsonl")
    feedback = args.feedback_output or str(config.runs_path / "golden-feedback.jsonl")
    correction_source = args.correction
    if args.correction_file:
        correction_source = Path(args.correction_file).read_text(encoding="utf-8")
    correction = json.loads(correction_source)
    report = append_golden_feedback(
        queue,
        feedback,
        registry,
        sample_id=args.sample_id,
        correction=correction,
        reviewer=args.reviewer,
        reason=args.reason,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _golden_review_pack(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    queue = args.queue or config.runs_path / "phase0-review-queue.jsonl"
    output = args.output or config.runs_path / "qwen-golden-review-pack.jsonl"
    report = export_review_pack(
        queue,
        args.draft,
        output,
        registry,
        include_raw=args.allow_raw,
        prioritize=args.prioritize,
    )
    print(json.dumps(report["manifest"], ensure_ascii=False))
    return 0


def _golden_review_session(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    queue = args.queue or config.runs_path / "phase0-review-queue.jsonl"
    feedback = args.feedback_output or config.runs_path / "golden-feedback.jsonl"
    report = run_review_session(
        args.review_pack,
        queue,
        feedback,
        registry,
        reviewer=args.reviewer,
        deduplicate_exact=args.deduplicate_exact,
        blind_teacher=args.blind,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _golden_apply_feedback(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    queue = Path(args.queue or config.runs_path / "phase0-review-queue.jsonl")
    feedback = args.feedback or config.runs_path / "golden-feedback.jsonl"
    output = Path(args.output or queue.with_name(f"{queue.stem}-reviewed{queue.suffix}"))
    report = apply_golden_feedback(queue, feedback, output, registry)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _serve(args: argparse.Namespace) -> int:
    config, _ = _load(args.config)
    serve(config, host=args.host, port=args.port, mode=args.mode)
    return 0


def _decide(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    request = DecisionRequest.from_dict(payload)
    outcome = DecisionRouter(config, registry).decide(request)
    response = outcome.to_dict() if request.options.return_evidence else outcome.response.to_dict()
    print(json.dumps(response, ensure_ascii=False))
    return 0


def _dataset_build(args: argparse.Namespace) -> int:
    config, registry = _load(args.config)
    report = build_dataset(
        config,
        registry,
        args.seed,
        args.output,
        review_path=args.review_output,
        limit=args.limit,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False))
    return 0


def _dataset_validate(args: argparse.Namespace) -> int:
    _, registry = _load(args.config)
    report = validate_dataset(args.samples, registry)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ready"] else 1


def _student_manifest(args: argparse.Namespace) -> int:
    _, registry = _load(args.config)
    manifest = student_manifest(
        registry,
        StudentConfig(
            model_id=args.model_id,
            backbone=args.backbone,
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_sequence_length,
            precision=args.precision,
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


def _student_evaluate(args: argparse.Namespace) -> int:
    _, registry = _load(args.config)
    report = evaluate_student_checkpoint(
        args.checkpoint,
        args.dataset,
        registry,
        split=args.split,
        minimum_confidence=args.minimum_confidence,
        allow_score=args.allow_score,
        with_rules=args.with_rules,
        score_tolerance=args.score_tolerance,
        minimum_accuracy=args.minimum_accuracy,
        minimum_accepted_accuracy=args.minimum_accepted_accuracy,
        minimum_task_accuracy=args.minimum_task_accuracy,
        deduplicate_exact=args.deduplicate_exact,
        device=args.device,
    )
    if args.output:
        write_student_evaluation(report, args.output)
    print(json.dumps(report, ensure_ascii=False))
    return 1 if args.production_gate and not report["quality_gate"]["ready"] else 0


def _control_train(args: argparse.Namespace) -> int:
    registry = TaskRegistry.load(args.registry)
    report = run_reference_training(
        args.dataset,
        args.output,
        registry,
        student=StudentConfig(
            model_id=args.model_id,
            backbone=args.backbone,
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_sequence_length,
            precision=args.precision,
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            precision=args.precision,
            class_balance=args.class_balanced,
            hard_negative_weight=args.hard_negative_weight,
        ),
        device=args.device,
        require_human_labels=args.require_human_labels,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _control_seed(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    registry = TaskRegistry.load(args.registry)
    output = args.output or config.runs_path / "control-review-queue.jsonl"
    report = generate_control_review_queue(
        output,
        registry,
        count_per_skill=args.count_per_skill,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _control_draft(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    registry = TaskRegistry.load(args.registry)
    queue = Path(args.queue)
    output = Path(args.output) if args.output else queue.with_name(f"{args.provider}-control-draft.jsonl")
    report = generate_teacher_draft(
        config,
        registry,
        queue,
        output,
        provider=args.provider,
        limit=args.limit,
        timeout_s=args.timeout,
        max_tokens=args.max_tokens,
        resume=args.resume,
    )
    print(json.dumps(report["manifest"], ensure_ascii=False))
    return 0 if report["manifest"]["error_count"] == 0 else 1


def _control_hard_negative(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    registry = TaskRegistry.load(args.registry)
    output = args.output or config.runs_path / "control-hard-negative.jsonl"
    report = generate_control_hard_negative_queue(
        output,
        registry,
        pair_count=args.pair_count,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _control_compositional(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    registry = TaskRegistry.load(args.registry)
    output = args.output or config.runs_path / "control-compositional-training.jsonl"
    report = generate_control_compositional_training_queue(output, registry, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _control_boundary(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    registry = TaskRegistry.load(args.registry)
    output = args.output or config.runs_path / "control-boundary-training.jsonl"
    report = generate_control_boundary_training_queue(output, registry, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _control_korean(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    registry = TaskRegistry.load(args.registry)
    output = args.output or config.runs_path / "control-korean-training.jsonl"
    report = generate_control_korean_training_queue(output, registry, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _control_materialize(args: argparse.Namespace) -> int:
    registry = TaskRegistry.load(args.registry)
    report = materialize_control_human_dataset(
        args.input,
        args.output,
        registry,
        require_dual_review=args.require_dual_review,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _control_merge(args: argparse.Namespace) -> int:
    registry = TaskRegistry.load(args.registry)
    report = merge_control_datasets(
        args.input,
        args.output,
        registry,
        require_human_labels=args.require_human_labels,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _control_review_pack(args: argparse.Namespace) -> int:
    registry = TaskRegistry.load(args.registry)
    focus_actions = set(args.focus_action or [])
    if args.agreement_manifest:
        focus_actions.update(
            load_control_review_focus_actions(
                args.agreement_manifest,
                maximum_agreement=args.max_action_agreement,
            )
        )
    report = export_review_pack(
        args.queue,
        args.draft,
        args.output,
        registry,
        include_raw=args.allow_raw,
        prioritize=args.prioritize,
        student_checkpoint=args.student_checkpoint,
        student_device=args.student_device,
        focus_actions=focus_actions or None,
        review_split=args.split,
    )
    print(json.dumps(report["manifest"], ensure_ascii=False))
    return 0


def _control_review_status(args: argparse.Namespace) -> int:
    registry = TaskRegistry.load(args.registry)
    report = control_review_status(args.queue, args.feedback, registry)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _control_review_session(args: argparse.Namespace) -> int:
    registry = TaskRegistry.load(args.registry)
    report = run_review_session(
        args.review_pack,
        args.queue,
        args.feedback_output,
        registry,
        reviewer=args.reviewer,
        deduplicate_exact=args.deduplicate_exact,
        blind_teacher=args.blind,
        batch_offset=args.offset,
        batch_limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _control_review_agreement(args: argparse.Namespace) -> int:
    registry = TaskRegistry.load(args.registry)
    report = compare_control_reviews(
        args.queue,
        args.reviewer_a_feedback,
        args.reviewer_b_feedback,
        args.output,
        registry,
        minimum_agreement=args.minimum_agreement,
    )
    if args.show_action_summary:
        print(format_control_review_action_summary(report["manifest"]))
    print(json.dumps(report["manifest"], ensure_ascii=False))
    return 0 if report["manifest"]["agreement_gate"] else 1


def _control_review_adjudicate(args: argparse.Namespace) -> int:
    registry = TaskRegistry.load(args.registry)
    report = run_adjudication_session(
        args.agreement,
        args.queue,
        args.feedback_output,
        registry,
        reviewer=args.reviewer,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _control_review_finalize(args: argparse.Namespace) -> int:
    registry = TaskRegistry.load(args.registry)
    report = finalize_control_reviews(
        args.queue,
        args.agreement,
        args.adjudication_feedback,
        args.output,
        registry,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ready"] else 1


def _control_apply_feedback(args: argparse.Namespace) -> int:
    registry = TaskRegistry.load(args.registry)
    report = apply_golden_feedback(args.queue, args.feedback, args.output, registry)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _control_adjudicate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    registry = TaskRegistry.load(args.registry)
    report = adjudicate_teacher_drafts(
        config,
        registry,
        args.queue,
        args.qwen_draft,
        args.gemma_draft,
        args.output,
    )
    print(json.dumps(report["manifest"], ensure_ascii=False))
    return 0


def _control_decide(args: argparse.Namespace) -> int:
    registry = TaskRegistry.load(args.registry)
    observation = ControlObservation.from_dict(json.loads(Path(args.observation).read_text(encoding="utf-8")))
    client = ControlStudentClient(
        args.checkpoint,
        registry,
        policy=ControlSafetyPolicy(
            minimum_confidence=args.minimum_confidence,
            max_observation_age_ms=args.max_observation_age_ms,
            max_action_ttl_ms=args.max_action_ttl_ms,
        ),
        calibration_path=args.calibration,
        device=args.device,
    )
    action = client.decide(observation, now_ms=args.now_ms)
    print(json.dumps(action.to_dict(), ensure_ascii=False))
    return 0


def _control_evaluate(args: argparse.Namespace) -> int:
    registry = TaskRegistry.load(args.registry)
    report = evaluate_student_checkpoint(
        args.checkpoint,
        args.dataset,
        registry,
        split=args.split,
        minimum_confidence=args.minimum_confidence,
        minimum_accuracy=args.minimum_accuracy,
        minimum_accepted_accuracy=args.minimum_accepted_accuracy,
        minimum_task_accuracy=args.minimum_task_accuracy,
        deduplicate_exact=args.deduplicate_exact,
        device=args.device,
    )
    if args.output:
        write_student_evaluation(report, args.output)
    print(json.dumps(report, ensure_ascii=False))
    return 1 if args.production_gate and not report["quality_gate"]["ready"] else 0


def _control_simulate(args: argparse.Namespace) -> int:
    if args.repeat < 1:
        raise ValueError("control simulation repeat must be positive")
    for name, value in (("max_p95_ms", args.max_p95_ms), ("max_p99_ms", args.max_p99_ms)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when provided")
    registry = TaskRegistry.load(args.registry)
    client = ControlStudentClient(
        args.checkpoint,
        registry,
        policy=ControlSafetyPolicy(
            minimum_confidence=args.minimum_confidence,
            max_observation_age_ms=args.max_observation_age_ms,
            max_action_ttl_ms=args.max_action_ttl_ms,
        ),
        device=args.device,
    )
    rows = []
    scenario_ids: set[str] = set()
    for line_number, line in enumerate(Path(args.scenarios).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise TypeError(f"{args.scenarios}:{line_number}: scenario must be an object")
        scenario_id = str(raw.get("scenario_id", ""))
        if not scenario_id:
            raise ValueError(f"{args.scenarios}:{line_number}: scenario_id is required")
        if scenario_id in scenario_ids:
            raise ValueError(f"{args.scenarios}:{line_number}: scenario_id must be unique")
        scenario_ids.add(scenario_id)
        observation = ControlObservation.from_dict(raw.get("observation", {}))
        now_ms = float(raw["now_ms"])
        expected_skill = str(raw["expected_skill"])
        expected_reason = raw.get("expected_reason")
        expected_safe_stop = bool(raw.get("expected_safe_stop", expected_skill == "STOP"))
        for _ in range(args.repeat):
            started = time.perf_counter()
            action = client.decide(observation, now_ms=now_ms)
            elapsed_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "expected_skill": expected_skill,
                    "skill": action.skill,
                    "expected_reason": expected_reason,
                    "reason": action.reason,
                    "expected_safe_stop": expected_safe_stop,
                    "safe_stop": action.skill == "STOP" and action.abstained,
                    "confidence": round(action.confidence, 6),
                    "correct": action.skill == expected_skill
                    and (expected_reason is None or action.reason == expected_reason),
                    "elapsed_ms": elapsed_ms,
                }
            )
    if not rows:
        raise ValueError("control scenario file is empty")
    latencies = sorted(row["elapsed_ms"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    expected_stops = [row for row in rows if row["expected_safe_stop"]]
    observed_stops = sum(row["safe_stop"] for row in expected_stops)
    p95_ms = latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)]
    p99_ms = latencies[max(0, math.ceil(len(latencies) * 0.99) - 1)]
    latency_gate = {
        "max_p95_ms": args.max_p95_ms,
        "max_p99_ms": args.max_p99_ms,
        "p95_passed": args.max_p95_ms is None or p95_ms <= args.max_p95_ms,
        "p99_passed": args.max_p99_ms is None or p99_ms <= args.max_p99_ms,
    }
    latency_gate["passed"] = latency_gate["p95_passed"] and latency_gate["p99_passed"]
    report = {
        "record_type": "control_simulation",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "scenarios": str(Path(args.scenarios).resolve()),
        "repeat": args.repeat,
        "count": len(rows),
        "unique_scenario_count": len(scenario_ids),
        "correct": correct,
        "accuracy": round(correct / len(rows), 6),
        "expected_safe_stop_count": len(expected_stops),
        "safe_stop_recall": round(observed_stops / len(expected_stops), 6) if expected_stops else None,
        "latency_ms": {
            "p50": round(latencies[len(latencies) // 2], 6),
            "p95": round(p95_ms, 6),
            "p99": round(p99_ms, 6),
            "max": round(max(latencies), 6),
            "mean": round(sum(latencies) / len(latencies), 6),
        },
        "latency_gate": latency_gate,
        "results": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if args.fail_on_mismatch and correct != len(rows):
        return 1
    return 0 if latency_gate["passed"] else 1


def _training_plan(args: argparse.Namespace) -> int:
    _, registry = _load(args.config)
    plan = write_training_plan(
        args.dataset,
        args.output,
        registry,
        student=StudentConfig(
            model_id=args.model_id,
            backbone=args.backbone,
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_sequence_length,
            precision=args.precision,
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            precision=args.precision,
            hard_negative_weight=args.hard_negative_weight,
        ),
    )
    print(json.dumps(plan, ensure_ascii=False))
    return 0


def _training_run(args: argparse.Namespace) -> int:
    _, registry = _load(args.config)
    report = run_reference_training(
        args.dataset,
        args.output,
        registry,
        student=StudentConfig(
            model_id=args.model_id,
            backbone=args.backbone,
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_sequence_length,
            precision=args.precision,
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            precision=args.precision,
            hard_negative_weight=args.hard_negative_weight,
        ),
        device=args.device,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _model_registry(args: argparse.Namespace) -> ModelRegistry:
    config, _ = _load(args.config)
    return ModelRegistry(args.registry or config.model_registry_path)


def _model_register(args: argparse.Namespace) -> int:
    record = _model_registry(args).register(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    print(json.dumps(record, ensure_ascii=False))
    return 0


def _model_manifest(args: argparse.Namespace) -> int:
    training_plan = json.loads(Path(args.training_plan).read_text(encoding="utf-8"))
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    manifest = build_model_manifest(
        training_plan,
        calibration,
        args.checkpoint,
        git_commit=args.git_commit,
        runtime=args.runtime,
        status=args.status,
        model_id=args.model_id,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


def _model_promote(args: argparse.Namespace) -> int:
    record = _model_registry(args).transition(args.model_id, args.to, reason=args.reason)
    print(json.dumps(record, ensure_ascii=False))
    return 0


def _model_rollback(args: argparse.Namespace) -> int:
    record = _model_registry(args).rollback(args.model_id, reason=args.reason)
    print(json.dumps(record, ensure_ascii=False))
    return 0


def _model_list(args: argparse.Namespace) -> int:
    print(json.dumps(list(_model_registry(args).list()), ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hyperjev")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    _config_argument(doctor)
    doctor.add_argument("--timeout", type=float, default=3.0)
    doctor.set_defaults(handler=_doctor)

    registry = subparsers.add_parser("registry")
    registry_subparsers = registry.add_subparsers(dest="registry_command", required=True)
    validate = registry_subparsers.add_parser("validate")
    _config_argument(validate)
    validate.set_defaults(handler=_registry_validate)

    teacher = subparsers.add_parser("teacher")
    teacher_subparsers = teacher.add_subparsers(dest="teacher_command", required=True)
    check = teacher_subparsers.add_parser("check")
    _config_argument(check)
    check.add_argument("--timeout", type=float, default=3.0)
    check.set_defaults(handler=_teacher_check)

    benchmark = subparsers.add_parser("benchmark")
    _config_argument(benchmark)
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--provider", action="append", choices=("qwen", "gemma"))
    benchmark.add_argument("--dry-run", action="store_true")
    benchmark.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="teacher read timeout in seconds; defaults to each teacher's config (Qwen 300s, Gemma 900s)",
    )
    benchmark.add_argument("--output")
    benchmark.set_defaults(handler=_benchmark)

    evaluate = subparsers.add_parser("evaluate")
    _config_argument(evaluate)
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--samples")
    evaluate.add_argument("--output")
    evaluate.set_defaults(handler=_evaluate)

    calibrate = subparsers.add_parser("calibrate")
    _config_argument(calibrate)
    calibrate.add_argument("--input", required=True, help="JSON object containing held-out logits and labels")
    calibrate.add_argument("--output", help="calibration manifest JSON path")
    calibrate.add_argument("--minimum", type=float, default=0.25)
    calibrate.add_argument("--maximum", type=float, default=4.0)
    calibrate.add_argument("--steps", type=int, default=76)
    calibrate.set_defaults(handler=_calibrate)

    golden = subparsers.add_parser("golden")
    golden_subparsers = golden.add_subparsers(dest="golden_command", required=True)
    golden_validate = golden_subparsers.add_parser("validate")
    _config_argument(golden_validate)
    golden_validate.add_argument("--samples")
    golden_validate.add_argument("--minimum-count", type=int, default=1000)
    golden_validate.add_argument("--allow-unreviewed", action="store_true")
    golden_validate.set_defaults(handler=_golden_validate)
    golden_generate = golden_subparsers.add_parser("generate")
    _config_argument(golden_generate)
    golden_generate.add_argument("--output")
    golden_generate.add_argument("--count", type=int, default=1000)
    golden_generate.add_argument("--seed", type=int, default=0)
    golden_generate.set_defaults(handler=_golden_generate)
    golden_draft = golden_subparsers.add_parser("draft")
    _config_argument(golden_draft)
    golden_draft.add_argument("--provider", choices=("qwen", "gemma"), default="gemma")
    golden_draft.add_argument("--queue")
    golden_draft.add_argument("--output")
    golden_draft.add_argument("--limit", type=int)
    golden_draft.add_argument("--timeout", type=float)
    golden_draft.add_argument("--resume", action="store_true")
    golden_draft.set_defaults(handler=_golden_draft)
    golden_review = golden_subparsers.add_parser("review")
    _config_argument(golden_review)
    golden_review.add_argument("--queue")
    golden_review.add_argument("--feedback-output")
    golden_review.add_argument("--sample-id", required=True)
    correction = golden_review.add_mutually_exclusive_group(required=True)
    correction.add_argument("--correction", help="typed decision result as a JSON object")
    correction.add_argument("--correction-file", help="file containing a typed decision result JSON object")
    golden_review.add_argument("--reviewer", required=True)
    golden_review.add_argument("--reason", default="")
    golden_review.set_defaults(handler=_golden_review)
    golden_pack = golden_subparsers.add_parser("review-pack")
    _config_argument(golden_pack)
    golden_pack.add_argument("--queue")
    golden_pack.add_argument("--draft", required=True)
    golden_pack.add_argument("--output")
    golden_pack.add_argument(
        "--allow-raw",
        action="store_true",
        required=True,
        help="explicitly allow raw state/question in this local review artifact",
    )
    golden_pack.add_argument(
        "--prioritize", action="store_true", help="place invalid/repaired/low-confidence items first"
    )
    golden_pack.set_defaults(handler=_golden_review_pack)
    golden_session = golden_subparsers.add_parser("review-session")
    _config_argument(golden_session)
    golden_session.add_argument("--review-pack", required=True)
    golden_session.add_argument("--queue")
    golden_session.add_argument("--feedback-output")
    golden_session.add_argument("--reviewer", required=True)
    golden_session.add_argument(
        "--deduplicate-exact",
        action="store_true",
        help="review one item per exact task/language/domain/state/question group and propagate its label",
    )
    golden_session.add_argument(
        "--blind",
        action="store_true",
        help="hide teacher output and require an independently entered value",
    )
    golden_session.set_defaults(handler=_golden_review_session)
    golden_apply = golden_subparsers.add_parser("apply-feedback")
    _config_argument(golden_apply)
    golden_apply.add_argument("--queue")
    golden_apply.add_argument("--feedback")
    golden_apply.add_argument("--output")
    golden_apply.set_defaults(handler=_golden_apply_feedback)

    serve_parser = subparsers.add_parser("serve")
    _config_argument(serve_parser)
    serve_parser.add_argument("--host")
    serve_parser.add_argument("--port", type=int)
    serve_parser.add_argument("--mode", choices=("mock", "router"), default="mock")
    serve_parser.set_defaults(handler=_serve)

    decide = subparsers.add_parser("decide")
    _config_argument(decide)
    decide.add_argument("--request", required=True, help="JSON file containing a /v1/decide request")
    decide.set_defaults(handler=_decide)

    dataset = subparsers.add_parser("dataset")
    dataset_subparsers = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_build = dataset_subparsers.add_parser("build")
    _config_argument(dataset_build)
    dataset_build.add_argument("--seed", required=True, help="seed JSONL path")
    dataset_build.add_argument("--output", required=True, help="normalized dataset JSONL path")
    dataset_build.add_argument("--review-output")
    dataset_build.add_argument("--limit", type=int)
    dataset_build.set_defaults(handler=_dataset_build)
    dataset_validate = dataset_subparsers.add_parser("validate")
    _config_argument(dataset_validate)
    dataset_validate.add_argument("--samples", required=True, help="normalized dataset JSONL path")
    dataset_validate.set_defaults(handler=_dataset_validate)

    student = subparsers.add_parser("student")
    student_subparsers = student.add_subparsers(dest="student_command", required=True)
    student_manifest_parser = student_subparsers.add_parser("manifest")
    _config_argument(student_manifest_parser)
    student_manifest_parser.add_argument("--model-id", default="hyperjev-student-dev")
    student_manifest_parser.add_argument("--backbone", default="small-multilingual-encoder")
    student_manifest_parser.add_argument("--hidden-size", type=int, default=256)
    student_manifest_parser.add_argument("--vocab-size", type=int, default=32768)
    student_manifest_parser.add_argument("--max-sequence-length", type=int, default=1024)
    student_manifest_parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    student_manifest_parser.set_defaults(handler=_student_manifest)
    student_evaluate_parser = student_subparsers.add_parser("evaluate")
    _config_argument(student_evaluate_parser)
    student_evaluate_parser.add_argument("--checkpoint", required=True)
    student_evaluate_parser.add_argument("--dataset", required=True)
    student_evaluate_parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    student_evaluate_parser.add_argument("--minimum-confidence", type=float, default=0.95)
    student_evaluate_parser.add_argument("--allow-score", action="store_true")
    student_evaluate_parser.add_argument("--with-rules", action="store_true")
    student_evaluate_parser.add_argument("--score-tolerance", type=float, default=0.10)
    student_evaluate_parser.add_argument("--minimum-accuracy", type=float, default=0.99)
    student_evaluate_parser.add_argument("--minimum-accepted-accuracy", type=float, default=0.995)
    student_evaluate_parser.add_argument("--minimum-task-accuracy", type=float, default=0.98)
    student_evaluate_parser.add_argument(
        "--deduplicate-exact",
        action="store_true",
        help="evaluate one representative per exact task/language/domain/state/question group",
    )
    student_evaluate_parser.add_argument(
        "--production-gate",
        action="store_true",
        help="return exit code 1 when human labels or quality thresholds are missing",
    )
    student_evaluate_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    student_evaluate_parser.add_argument("--output")
    student_evaluate_parser.set_defaults(handler=_student_evaluate)

    control = subparsers.add_parser("control")
    control_subparsers = control.add_subparsers(dest="control_command", required=True)
    control_seed = control_subparsers.add_parser("seed")
    _config_argument(control_seed)
    control_seed.add_argument("--registry", default="registry/control_tasks")
    control_seed.add_argument("--output")
    control_seed.add_argument("--count-per-skill", type=int, default=100)
    control_seed.add_argument("--seed", type=int, default=7)
    control_seed.set_defaults(handler=_control_seed)
    control_draft = control_subparsers.add_parser("draft")
    _config_argument(control_draft)
    control_draft.add_argument("--registry", default="registry/control_tasks")
    control_draft.add_argument("--queue", required=True)
    control_draft.add_argument("--output")
    control_draft.add_argument("--provider", choices=("qwen", "gemma"), default="qwen")
    control_draft.add_argument("--limit", type=int)
    control_draft.add_argument("--timeout", type=float)
    control_draft.add_argument("--max-tokens", type=int, default=256)
    control_draft.add_argument("--resume", action="store_true")
    control_draft.set_defaults(handler=_control_draft)
    control_hard_negative = control_subparsers.add_parser("hard-negative")
    _config_argument(control_hard_negative)
    control_hard_negative.add_argument("--registry", default="registry/control_tasks")
    control_hard_negative.add_argument("--output")
    control_hard_negative.add_argument("--pair-count", type=int, default=100)
    control_hard_negative.add_argument("--seed", type=int, default=7)
    control_hard_negative.set_defaults(handler=_control_hard_negative)
    control_compositional = control_subparsers.add_parser("compositional")
    _config_argument(control_compositional)
    control_compositional.add_argument("--registry", default="registry/control_tasks")
    control_compositional.add_argument("--output")
    control_compositional.add_argument("--seed", type=int, default=7)
    control_compositional.set_defaults(handler=_control_compositional)
    control_korean = control_subparsers.add_parser("korean")
    _config_argument(control_korean)
    control_korean.add_argument("--registry", default="registry/control_tasks")
    control_korean.add_argument("--output")
    control_korean.add_argument("--seed", type=int, default=7)
    control_korean.set_defaults(handler=_control_korean)
    control_boundary = control_subparsers.add_parser("boundary")
    _config_argument(control_boundary)
    control_boundary.add_argument("--registry", default="registry/control_tasks")
    control_boundary.add_argument("--output")
    control_boundary.add_argument("--seed", type=int, default=7)
    control_boundary.set_defaults(handler=_control_boundary)
    control_materialize = control_subparsers.add_parser("materialize")
    control_materialize.add_argument("--registry", default="registry/control_tasks")
    control_materialize.add_argument("--input", required=True, help="human-reviewed control JSONL")
    control_materialize.add_argument("--output", required=True, help="human-target training JSONL")
    control_materialize.add_argument(
        "--require-dual-review",
        action="store_true",
        help="require control dual-review agreement or adjudication provenance",
    )
    control_materialize.set_defaults(handler=_control_materialize)
    control_merge = control_subparsers.add_parser("merge")
    control_merge.add_argument("--registry", default="registry/control_tasks")
    control_merge.add_argument(
        "--input", action="append", required=True, help="control JSONL; repeat at least twice"
    )
    control_merge.add_argument("--output", required=True)
    control_merge.add_argument("--require-human-labels", action="store_true")
    control_merge.set_defaults(handler=_control_merge)
    control_review_pack = control_subparsers.add_parser("review-pack")
    control_review_pack.add_argument("--registry", default="registry/control_tasks")
    control_review_pack.add_argument("--queue", required=True)
    control_review_pack.add_argument("--draft", required=True)
    control_review_pack.add_argument("--output", required=True)
    control_review_pack.add_argument(
        "--allow-raw", action="store_true", required=True, help="include state/question for local review"
    )
    control_review_pack.add_argument(
        "--prioritize", action="store_true", help="place invalid/repaired/low-confidence items first"
    )
    control_review_pack.add_argument(
        "--student-checkpoint",
        help="use raw Student confidence only to order blind review items",
    )
    control_review_pack.add_argument(
        "--student-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="device used for optional Student uncertainty ranking",
    )
    control_review_pack.add_argument(
        "--focus-action",
        action="append",
        metavar="ACTION",
        help="prioritize teacher-selected action; repeat for multiple actions",
    )
    control_review_pack.add_argument(
        "--agreement-manifest",
        help="focus actions whose dual-review agreement is below the threshold",
    )
    control_review_pack.add_argument(
        "--max-action-agreement",
        type=float,
        default=0.98,
        help="maximum action agreement rate for manifest-driven focus (default: 0.98)",
    )
    control_review_pack.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        help="export only one provenance split for focused human review",
    )
    control_review_pack.set_defaults(handler=_control_review_pack)
    control_review_status_parser = control_subparsers.add_parser("review-status")
    control_review_status_parser.add_argument("--registry", default="registry/control_tasks")
    control_review_status_parser.add_argument("--queue", required=True)
    control_review_status_parser.add_argument("--feedback", required=True)
    control_review_status_parser.set_defaults(handler=_control_review_status)
    control_review_session = control_subparsers.add_parser("review-session")
    control_review_session.add_argument("--registry", default="registry/control_tasks")
    control_review_session.add_argument("--review-pack", required=True)
    control_review_session.add_argument("--queue", required=True)
    control_review_session.add_argument("--feedback-output", required=True)
    control_review_session.add_argument("--reviewer", required=True)
    control_review_session.add_argument("--deduplicate-exact", action="store_true")
    control_review_session.add_argument(
        "--offset",
        type=int,
        default=0,
        help="zero-based review-group offset for a deterministic batch",
    )
    control_review_session.add_argument(
        "--limit",
        type=int,
        help="maximum number of review groups in this batch",
    )
    control_review_session.add_argument(
        "--blind",
        action="store_true",
        help="hide teacher output and require an independently entered value",
    )
    control_review_session.set_defaults(handler=_control_review_session)
    control_review_agreement = control_subparsers.add_parser("review-agreement")
    control_review_agreement.add_argument("--registry", default="registry/control_tasks")
    control_review_agreement.add_argument("--queue", required=True)
    control_review_agreement.add_argument("--reviewer-a-feedback", required=True)
    control_review_agreement.add_argument("--reviewer-b-feedback", required=True)
    control_review_agreement.add_argument("--output", required=True)
    control_review_agreement.add_argument("--minimum-agreement", type=float, default=0.98)
    control_review_agreement.add_argument(
        "--show-action-summary",
        action="store_true",
        help="print a human-readable action agreement summary before the manifest",
    )
    control_review_agreement.set_defaults(handler=_control_review_agreement)
    control_review_adjudicate = control_subparsers.add_parser("review-adjudicate")
    control_review_adjudicate.add_argument("--registry", default="registry/control_tasks")
    control_review_adjudicate.add_argument("--agreement", required=True)
    control_review_adjudicate.add_argument("--queue", required=True)
    control_review_adjudicate.add_argument("--feedback-output", required=True)
    control_review_adjudicate.add_argument("--reviewer", required=True)
    control_review_adjudicate.set_defaults(handler=_control_review_adjudicate)
    control_review_finalize = control_subparsers.add_parser("review-finalize")
    control_review_finalize.add_argument("--registry", default="registry/control_tasks")
    control_review_finalize.add_argument("--queue", required=True)
    control_review_finalize.add_argument("--agreement", required=True)
    control_review_finalize.add_argument("--adjudication-feedback", required=True)
    control_review_finalize.add_argument("--output", required=True)
    control_review_finalize.set_defaults(handler=_control_review_finalize)
    control_apply_feedback = control_subparsers.add_parser("apply-feedback")
    control_apply_feedback.add_argument("--registry", default="registry/control_tasks")
    control_apply_feedback.add_argument("--queue", required=True)
    control_apply_feedback.add_argument("--feedback", required=True)
    control_apply_feedback.add_argument("--output", required=True)
    control_apply_feedback.set_defaults(handler=_control_apply_feedback)
    control_adjudicate = control_subparsers.add_parser("adjudicate")
    _config_argument(control_adjudicate)
    control_adjudicate.add_argument("--registry", default="registry/control_tasks")
    control_adjudicate.add_argument("--queue", required=True)
    control_adjudicate.add_argument("--qwen-draft", required=True)
    control_adjudicate.add_argument("--gemma-draft", required=True)
    control_adjudicate.add_argument("--output", required=True)
    control_adjudicate.set_defaults(handler=_control_adjudicate)
    control_train = control_subparsers.add_parser("train")
    control_train.add_argument("--registry", default="registry/control_tasks")
    control_train.add_argument("--dataset", required=True)
    control_train.add_argument("--output", required=True)
    control_train.add_argument("--model-id", default="hyperjev-control-dev")
    control_train.add_argument("--backbone", default="reference-ngram-encoder")
    control_train.add_argument("--hidden-size", type=int, default=128)
    control_train.add_argument("--vocab-size", type=int, default=32768)
    control_train.add_argument("--max-sequence-length", type=int, default=256)
    control_train.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    control_train.add_argument("--epochs", type=int, default=30)
    control_train.add_argument("--batch-size", type=int, default=8)
    control_train.add_argument("--learning-rate", type=float, default=0.01)
    control_train.add_argument("--weight-decay", type=float, default=0.0)
    control_train.add_argument("--seed", type=int, default=7)
    control_train.add_argument("--class-balanced", action="store_true")
    control_train.add_argument(
        "--hard-negative-weight",
        type=float,
        default=1.0,
        help="multiply loss for samples with source.counterfactual_group_id",
    )
    control_train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    control_train.add_argument(
        "--require-human-labels",
        action="store_true",
        help="reject synthetic control targets; require validated human_review labels",
    )
    control_train.set_defaults(handler=_control_train)
    control_decide = control_subparsers.add_parser("decide")
    control_decide.add_argument("--registry", default="registry/control_tasks")
    control_decide.add_argument("--checkpoint", required=True)
    control_decide.add_argument("--observation", required=True)
    control_decide.add_argument("--now-ms", type=float, required=True)
    control_decide.add_argument("--minimum-confidence", type=float, default=0.90)
    control_decide.add_argument("--max-observation-age-ms", type=float, default=100.0)
    control_decide.add_argument("--max-action-ttl-ms", type=int, default=100)
    control_decide.add_argument("--calibration")
    control_decide.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    control_decide.set_defaults(handler=_control_decide)
    control_evaluate = control_subparsers.add_parser("evaluate")
    control_evaluate.add_argument("--registry", default="registry/control_tasks")
    control_evaluate.add_argument("--checkpoint", required=True)
    control_evaluate.add_argument("--dataset", required=True)
    control_evaluate.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    control_evaluate.add_argument("--minimum-confidence", type=float, default=0.90)
    control_evaluate.add_argument("--minimum-accuracy", type=float, default=0.99)
    control_evaluate.add_argument("--minimum-accepted-accuracy", type=float, default=0.995)
    control_evaluate.add_argument("--minimum-task-accuracy", type=float, default=0.98)
    control_evaluate.add_argument(
        "--deduplicate-exact",
        action="store_true",
        help="evaluate one representative per exact task/language/domain/state/question group",
    )
    control_evaluate.add_argument(
        "--production-gate",
        action="store_true",
        help="return exit code 1 when human labels or quality thresholds are missing",
    )
    control_evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    control_evaluate.add_argument("--output")
    control_evaluate.set_defaults(handler=_control_evaluate)
    control_simulate = control_subparsers.add_parser("simulate")
    control_simulate.add_argument("--registry", default="registry/control_tasks")
    control_simulate.add_argument("--checkpoint", required=True)
    control_simulate.add_argument("--scenarios", required=True)
    control_simulate.add_argument("--repeat", type=int, default=1)
    control_simulate.add_argument("--minimum-confidence", type=float, default=0.90)
    control_simulate.add_argument("--max-observation-age-ms", type=float, default=100.0)
    control_simulate.add_argument("--max-action-ttl-ms", type=int, default=100)
    control_simulate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    control_simulate.add_argument("--output")
    control_simulate.add_argument("--fail-on-mismatch", action="store_true")
    control_simulate.add_argument("--max-p95-ms", type=float)
    control_simulate.add_argument("--max-p99-ms", type=float)
    control_simulate.set_defaults(handler=_control_simulate)

    training = subparsers.add_parser("train")
    training_subparsers = training.add_subparsers(dest="training_command", required=True)
    training_plan = training_subparsers.add_parser("plan")
    _config_argument(training_plan)
    training_plan.add_argument("--dataset", required=True, help="normalized dataset JSONL path")
    training_plan.add_argument("--output", required=True, help="training plan JSON path")
    training_plan.add_argument("--model-id", default="hyperjev-student-dev")
    training_plan.add_argument("--backbone", default="small-multilingual-encoder")
    training_plan.add_argument("--hidden-size", type=int, default=256)
    training_plan.add_argument("--vocab-size", type=int, default=32768)
    training_plan.add_argument("--max-sequence-length", type=int, default=1024)
    training_plan.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    training_plan.add_argument("--epochs", type=int, default=3)
    training_plan.add_argument("--batch-size", type=int, default=32)
    training_plan.add_argument("--learning-rate", type=float, default=2e-4)
    training_plan.add_argument("--weight-decay", type=float, default=0.01)
    training_plan.add_argument("--seed", type=int, default=7)
    training_plan.add_argument("--gradient-accumulation-steps", type=int, default=1)
    training_plan.add_argument("--hard-negative-weight", type=float, default=1.0)
    training_plan.set_defaults(handler=_training_plan)
    training_run = training_subparsers.add_parser("run")
    _config_argument(training_run)
    training_run.add_argument("--dataset", required=True, help="normalized dataset JSONL path")
    training_run.add_argument("--output", required=True, help="reference checkpoint path")
    training_run.add_argument("--model-id", default="hyperjev-student-dev")
    training_run.add_argument("--backbone", default="reference-byte-encoder")
    training_run.add_argument("--hidden-size", type=int, default=256)
    training_run.add_argument("--vocab-size", type=int, default=32768)
    training_run.add_argument("--max-sequence-length", type=int, default=1024)
    training_run.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    training_run.add_argument("--epochs", type=int, default=3)
    training_run.add_argument("--batch-size", type=int, default=32)
    training_run.add_argument("--learning-rate", type=float, default=2e-4)
    training_run.add_argument("--weight-decay", type=float, default=0.01)
    training_run.add_argument("--seed", type=int, default=7)
    training_run.add_argument("--gradient-accumulation-steps", type=int, default=1)
    training_run.add_argument("--hard-negative-weight", type=float, default=1.0)
    training_run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    training_run.set_defaults(handler=_training_run)

    model = subparsers.add_parser("model")
    model_subparsers = model.add_subparsers(dest="model_command", required=True)
    model_manifest = model_subparsers.add_parser("manifest")
    model_manifest.add_argument("--training-plan", required=True)
    model_manifest.add_argument("--calibration", required=True)
    model_manifest.add_argument("--checkpoint", required=True)
    model_manifest.add_argument("--git-commit", required=True)
    model_manifest.add_argument("--runtime", default="pytorch")
    model_manifest.add_argument("--status", choices=STATUSES, default="trained")
    model_manifest.add_argument("--model-id")
    model_manifest.add_argument("--output")
    model_manifest.set_defaults(handler=_model_manifest)
    model_register = model_subparsers.add_parser("register")
    _config_argument(model_register)
    model_register.add_argument("--registry")
    model_register.add_argument("--manifest", required=True)
    model_register.set_defaults(handler=_model_register)
    model_promote = model_subparsers.add_parser("promote")
    _config_argument(model_promote)
    model_promote.add_argument("--registry")
    model_promote.add_argument("model_id")
    model_promote.add_argument("--to", required=True, choices=("evaluated", "calibrated", "candidate", "canary", "active", "retired"))
    model_promote.add_argument("--reason", default="")
    model_promote.set_defaults(handler=_model_promote)
    model_rollback = model_subparsers.add_parser("rollback")
    _config_argument(model_rollback)
    model_rollback.add_argument("--registry")
    model_rollback.add_argument("model_id")
    model_rollback.add_argument("--reason", default="rollback")
    model_rollback.set_defaults(handler=_model_rollback)
    model_list = model_subparsers.add_parser("list")
    _config_argument(model_list)
    model_list.add_argument("--registry")
    model_list.set_defaults(handler=_model_list)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ConfigError, RegistryError, TrainingDependencyError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
