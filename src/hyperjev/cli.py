"""Command line entry point for Phase 0 operations."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .baseline import run_benchmark
from .config import ConfigError, load_config
from .contracts import DecisionRequest
from .dataset_factory import build_dataset, validate_dataset
from .doctor import system_checks
from .evaluation import evaluate_run, validate_golden_set
from .golden import generate_review_queue
from .mock_server import serve
from .registry import RegistryError, TaskRegistry
from .routing import DecisionRouter
from .student import StudentConfig, student_manifest
from .teachers import probe_teacher


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
        help="teacher read timeout in seconds; defaults to each teacher's config (300s)",
    )
    benchmark.add_argument("--output")
    benchmark.set_defaults(handler=_benchmark)

    evaluate = subparsers.add_parser("evaluate")
    _config_argument(evaluate)
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--samples")
    evaluate.add_argument("--output")
    evaluate.set_defaults(handler=_evaluate)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ConfigError, RegistryError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
