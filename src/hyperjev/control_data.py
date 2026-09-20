"""Quality and leakage checks for human-labelled control datasets."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .registry import TaskRegistry
from .samples import CanonicalSample, load_jsonl

CONTROL_SPLITS = frozenset({"train", "validation", "test"})
_WHITESPACE = re.compile(r"\s+")


def _normalise_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value.casefold()).strip()


def _exact_key(sample: CanonicalSample) -> tuple[str, int, str, str, str, str]:
    return (
        sample.task_id,
        sample.task_version,
        sample.language,
        sample.domain,
        _normalise_text(sample.state),
        _normalise_text(sample.question),
    )


def _metadata_value(sample: CanonicalSample, name: str) -> str:
    value = sample.source.get(name)
    if value is None:
        value = sample.provenance.get(name)
    return str(value).strip() if value is not None else ""


def _cross_split_groups(
    groups: Mapping[str, list[tuple[str, str]]],
) -> list[dict[str, Any]]:
    leaks: list[dict[str, Any]] = []
    for key, entries in sorted(groups.items()):
        splits = sorted({split for split, _sample_id in entries})
        if len(splits) < 2:
            continue
        leaks.append(
            {
                "key": key,
                "splits": splits,
                "sample_ids": sorted(sample_id for _split, sample_id in entries),
            }
        )
    return leaks


def validate_control_dataset(
    path: str | Path,
    registry: TaskRegistry,
    *,
    require_human_labels: bool = False,
) -> dict[str, Any]:
    """Return a deterministic quality report for a control JSONL dataset.

    The validator is intentionally stricter than the generic canonical sample
    loader. A control sample must retain scenario, episode, and semantic-group
    provenance so that split leakage can be detected before training.
    """

    samples = load_jsonl(path, registry)
    errors: list[dict[str, str]] = []
    split_counts: dict[str, int] = defaultdict(int)
    exact_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    episode_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    semantic_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    human_labeled_count = 0

    for sample in samples:
        split = str(sample.provenance.get("split", ""))
        if split not in CONTROL_SPLITS:
            errors.append(
                {"sample_id": sample.sample_id, "reason": "invalid_split", "value": split}
            )
        else:
            split_counts[split] += 1

        if sample.task_id != "control.skill":
            errors.append(
                {
                    "sample_id": sample.sample_id,
                    "reason": "unexpected_task",
                    "value": sample.task_id,
                }
            )

        for field in ("scenario_id", "episode_id", "semantic_group_id"):
            if not _metadata_value(sample, field):
                errors.append(
                    {
                        "sample_id": sample.sample_id,
                        "reason": f"missing_{field}",
                        "value": "",
                    }
                )

        if sample.labels.get("human") is not None:
            human_labeled_count += 1
        elif require_human_labels:
            errors.append(
                {
                    "sample_id": sample.sample_id,
                    "reason": "human_label_required",
                    "value": "null",
                }
            )

        exact_groups[repr(_exact_key(sample))].append((split, sample.sample_id))
        episode_id = _metadata_value(sample, "episode_id")
        semantic_group_id = _metadata_value(sample, "semantic_group_id")
        if episode_id:
            episode_groups[episode_id].append((split, sample.sample_id))
        if semantic_group_id:
            semantic_groups[semantic_group_id].append((split, sample.sample_id))

    exact_leaks = _cross_split_groups(exact_groups)
    episode_leaks = _cross_split_groups(episode_groups)
    semantic_group_leaks = _cross_split_groups(semantic_groups)
    errors.extend(
        {
            "sample_id": ",".join(leak["sample_ids"]),
            "reason": reason,
            "value": leak["key"],
        }
        for reason, leaks in (
            ("cross_split_exact_leak", exact_leaks),
            ("cross_split_episode_leak", episode_leaks),
            ("cross_split_semantic_group_leak", semantic_group_leaks),
        )
        for leak in leaks
    )

    return {
        "record_type": "control_dataset_quality",
        "dataset": str(Path(path).resolve()),
        "sample_count": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "unique_exact_group_count": len(exact_groups),
        "unique_episode_count": len(episode_groups),
        "unique_semantic_group_count": len(semantic_groups),
        "human_labeled_count": human_labeled_count,
        "require_human_labels": require_human_labels,
        "leaks": {
            "exact": exact_leaks,
            "episode": episode_leaks,
            "semantic_group": semantic_group_leaks,
        },
        "errors": errors,
        "passed": not errors,
    }
