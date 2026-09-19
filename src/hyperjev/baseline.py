"""Reproducible Phase 0 teacher baseline runner."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Phase0Config
from .contracts import ContractError, parse_decision_result, validate_result_for_task
from .prompts import PROMPT_VERSION, messages_for_sample
from .registry import TaskRegistry
from .samples import load_jsonl
from .teachers import TeacherClient, response_hash


def _extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        raise ContractError("teacher output does not contain a JSON object")
    value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ContractError("teacher output JSON must be an object")
    return value


@dataclass(frozen=True)
class BenchmarkRun:
    output_path: Path
    manifest: dict[str, Any]
    records: tuple[dict[str, Any], ...]


def _environment_manifest(config: Phase0Config) -> dict[str, Any]:
    return {
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "model_track": config.model_track,
        "diffusion_track": config.diffusion_track,
        "privacy_store_raw_inputs": False,
    }


def run_benchmark(
    config: Phase0Config,
    registry: TaskRegistry,
    *,
    limit: int | None = None,
    providers: tuple[str, ...] = ("qwen", "gemma"),
    dry_run: bool = False,
    timeout_s: float = 30.0,
    output_path: str | Path | None = None,
) -> BenchmarkRun:
    """Run the same canonical samples through selected teachers.

    Raw state and teacher output are deliberately not written to the result
    file. The response hash and token/latency metadata keep runs comparable
    without turning the repository into a data store.
    """

    samples = load_jsonl(config.smoke_set_path, registry)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:limit]
    unknown = set(providers).difference(config.teachers)
    if unknown:
        raise ValueError(f"unknown teacher providers: {sorted(unknown)}")
    if not samples:
        raise ValueError("benchmark has no samples")
    dataset_hash = __import__("hashlib").sha256(config.smoke_set_path.read_bytes()).hexdigest()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "record_type": "manifest",
        "run_id": run_id,
        "dataset_path": str(config.smoke_set_path),
        "dataset_hash": dataset_hash,
        "sample_count": len(samples),
        "providers": list(providers),
        "dry_run": dry_run,
        "prompt_version": PROMPT_VERSION,
        "request_parameters": {"temperature": 0, "top_p": 1, "seed": 0, "max_tokens": 256},
        "environment": _environment_manifest(config),
    }
    records: list[dict[str, Any]] = []
    for provider in providers:
        settings = config.teachers[provider]
        task_client = None if dry_run else TeacherClient(settings, timeout_s=timeout_s)
        for sample in samples:
            task = registry.get(sample.task_id, sample.task_version)
            record: dict[str, Any] = {
                "record_type": "sample",
                "run_id": run_id,
                "sample_id": sample.sample_id,
                "task_id": sample.task_id,
                "task_version": sample.task_version,
                "provider": provider,
                "model": settings.model,
                "prompt_version": PROMPT_VERSION,
                "status": "dry_run" if dry_run else "pending",
                "latency_ms": None,
                "response_sha256": None,
                "response_type": None,
                "schema_valid": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "error": None,
            }
            if dry_run:
                records.append(record)
                continue
            try:
                completion = task_client.complete(messages_for_sample(provider, sample, task))  # type: ignore[union-attr]
                record["status"] = "completed"
                record["latency_ms"] = round(completion.elapsed_ms, 3)
                record["response_sha256"] = response_hash(completion.content)
                record["model"] = completion.model
                record["prompt_tokens"] = completion.prompt_tokens
                record["completion_tokens"] = completion.completion_tokens
                record["total_tokens"] = completion.total_tokens
                try:
                    parsed = parse_decision_result(_extract_json(completion.content))
                    validate_result_for_task(task, parsed)
                    record["response_type"] = parsed.type
                    record["schema_valid"] = True
                except (ContractError, json.JSONDecodeError) as exc:
                    record["schema_valid"] = False
                    record["error"] = str(exc)
            except (OSError, TypeError, ValueError) as exc:  # per-sample evidence
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)

    selected_path = Path(output_path) if output_path else config.runs_path / f"teacher-baseline-{run_id}.jsonl"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    with selected_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return BenchmarkRun(output_path=selected_path, manifest=manifest, records=tuple(records))
