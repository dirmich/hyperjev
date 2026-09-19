"""Command line entry point for Phase 0 operations."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from collections.abc import Sequence

from . import __version__
from .baseline import run_benchmark
from .config import ConfigError, load_config
from .doctor import system_checks
from .evaluation import evaluate_run, validate_golden_set
from .registry import RegistryError, TaskRegistry
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
