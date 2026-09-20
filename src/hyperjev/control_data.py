"""Quality and leakage checks for human-labelled control datasets."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .registry import TaskRegistry
from .samples import CanonicalSample, load_jsonl

CONTROL_SPLITS = frozenset({"train", "validation", "test"})
CONTROL_SKILLS = (
    "STOP",
    "HOLD",
    "MOVE",
    "ROTATE",
    "APPROACH",
    "RETREAT",
    "INTERACT",
    "RECOVER",
)
_WHITESPACE = re.compile(r"\s+")

_CONTROL_SEED_TEMPLATES = {
    "STOP": {
        "en": ("obstacle is inside the safety zone", "sensor reports immediate collision risk"),
        "ko": ("장애물이 안전 구역 안에 있다", "센서가 즉시 충돌 위험을 보고한다"),
    },
    "HOLD": {
        "en": ("the pose is stable and no new command arrived", "wait safely for the next sensor update"),
        "ko": ("자세가 안정적이고 새 명령이 없다", "다음 센서 업데이트까지 안전하게 기다린다"),
    },
    "MOVE": {
        "en": ("the corridor is clear for forward motion", "free space is available on the route"),
        "ko": ("통로가 비어 있어 앞으로 움직일 수 있다", "경로에 자유 공간이 있다"),
    },
    "ROTATE": {
        "en": ("the heading is wrong and the turn space is clear", "turn toward the next waypoint"),
        "ko": ("방향이 틀렸고 회전 공간이 비어 있다", "다음 웨이포인트 방향으로 회전한다"),
    },
    "APPROACH": {
        "en": ("the target is ahead and can be approached safely", "the reachable target is getting closer"),
        "ko": ("목표가 앞에 있고 안전하게 접근할 수 있다", "접근 가능한 목표가 가까워지고 있다"),
    },
    "RETREAT": {
        "en": ("move away because the hazard is approaching", "safe space behind is available"),
        "ko": ("위험이 다가오므로 멀어져야 한다", "뒤쪽에 안전한 공간이 있다"),
    },
    "INTERACT": {
        "en": ("the aligned object is within reach", "the nearby switch is ready to activate"),
        "ko": ("정렬된 물체가 손이 닿는 거리에 있다", "근처 스위치를 작동할 준비가 됐다"),
    },
    "RECOVER": {
        "en": ("localization is lost and recovery is needed", "the controller reports a balance fault"),
        "ko": ("위치를 잃어 복구가 필요하다", "제어기가 균형 오류를 보고한다"),
    },
}

_CONTROL_HARD_NEGATIVE_PAIRS = (
    (
        "STOP",
        "RETREAT",
        {
            "en": (
                ("collision risk is immediate and stopping is mandatory", "STOP"),
                ("the hazard is approaching but the clear path behind is safe", "RETREAT"),
            ),
            "ko": (
                ("충돌 위험이 즉시 발생해 정지가 필수다", "STOP"),
                ("위험이 다가오지만 뒤의 빈 경로는 안전하다", "RETREAT"),
            ),
        },
    ),
    (
        "MOVE",
        "APPROACH",
        {
            "en": (
                ("the corridor is clear and continue forward without a target lock", "MOVE"),
                ("the reachable target is locked directly ahead", "APPROACH"),
            ),
            "ko": (
                ("통로가 비었고 특정 목표 고정 없이 앞으로 계속 간다", "MOVE"),
                ("접근 가능한 목표가 바로 앞에 고정됐다", "APPROACH"),
            ),
        },
    ),
    (
        "HOLD",
        "STOP",
        {
            "en": (
                ("pause safely while the pose is stable and no hazard is present", "HOLD"),
                ("emergency hazard is detected inside the collision zone", "STOP"),
            ),
            "ko": (
                ("위험이 없고 자세가 안정적이므로 안전하게 잠시 멈춘다", "HOLD"),
                ("충돌 구역 안에서 비상 위험이 감지됐다", "STOP"),
            ),
        },
    ),
    (
        "ROTATE",
        "MOVE",
        {
            "en": (
                ("heading is wrong and a clear turn is required at the junction", "ROTATE"),
                ("heading is aligned and the open route is straight ahead", "MOVE"),
            ),
            "ko": (
                ("방향이 틀려 교차로에서 빈 공간으로 회전해야 한다", "ROTATE"),
                ("방향이 맞고 열린 경로가 곧장 앞에 있다", "MOVE"),
            ),
        },
    ),
    (
        "INTERACT",
        "APPROACH",
        {
            "en": (
                ("the button is aligned and within hand reach", "INTERACT"),
                ("the object is still far and must be approached first", "APPROACH"),
            ),
            "ko": (
                ("버튼이 정렬됐고 손이 닿는 거리에 있다", "INTERACT"),
                ("물체가 아직 멀어 먼저 접근해야 한다", "APPROACH"),
            ),
        },
    ),
    (
        "RECOVER",
        "STOP",
        {
            "en": (
                ("localization is lost but no collision risk is present", "RECOVER"),
                ("the sensor reports an emergency collision risk", "STOP"),
            ),
            "ko": (
                ("충돌 위험은 없지만 위치 추정이 끊겨 복구가 필요하다", "RECOVER"),
                ("센서가 비상 충돌 위험을 보고한다", "STOP"),
            ),
        },
    ),
)


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


def generate_control_review_queue(
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    count_per_skill: int = 100,
    seed: int = 7,
) -> dict[str, Any]:
    """Create a balanced, provenance-rich control queue awaiting human review."""

    if count_per_skill < 1:
        raise ValueError("count_per_skill must be positive")
    registry.get("control.skill", 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_count = len(CONTROL_SKILLS) * count_per_skill
    with output.open("w", encoding="utf-8") as handle:
        index = 0
        for skill_index, skill in enumerate(CONTROL_SKILLS):
            for variant in range(count_per_skill):
                index += 1
                language = "ko" if (variant + skill_index + seed) % 2 else "en"
                templates = _CONTROL_SEED_TEMPLATES[skill][language]
                base_state = templates[variant % len(templates)]
                state = (
                    f"{base_state}; scenario variant {variant:04d}"
                    if language == "en"
                    else f"{base_state}; 시나리오 변형 {variant:04d}"
                )
                split_bucket = (variant + skill_index * 3 + seed) % 10
                split = "train" if split_bucket < 8 else "validation" if split_bucket == 8 else "test"
                sample = {
                    "sample_id": f"control-review-{index:05d}",
                    "task_id": "control.skill",
                    "task_version": 1,
                    "state": state,
                    "question": (
                        "select next safe high-level control skill"
                        if language == "en"
                        else "다음 안전한 고수준 제어 skill을 선택하라"
                    ),
                    "target": skill,
                    "language": language,
                    "domain": "control-review-seed",
                    "source": {
                        "kind": "synthetic-review-seed",
                        "scenario_id": f"control-scenario-{index:05d}",
                        "episode_id": f"control-episode-{index:05d}",
                        "semantic_group_id": f"control-group-{index:05d}",
                        "seed": seed,
                    },
                    "labels": {"qwen": None, "gemma": None, "human": None},
                    "review": {"status": "pending", "reviewer": None},
                    "provenance": {
                        "prompt_version": 1,
                        "generator": "control-review-seed-v1",
                        "split": split,
                        "privacy_raw_inputs_stored": False,
                        "target_source": "synthetic_seed_only",
                    },
                }
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "record_type": "control_review_queue",
        "output_path": str(output.resolve()),
        "sample_count": sample_count,
        "count_per_skill": count_per_skill,
        "skill_counts": {skill: count_per_skill for skill in CONTROL_SKILLS},
        "seed": seed,
        "human_labeled": False,
    }


def generate_control_hard_negative_queue(
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    pair_count: int = 100,
    seed: int = 7,
) -> dict[str, Any]:
    """Create same-split counterfactual pairs for commonly confused skills."""

    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    registry.get("control.skill", 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for pair_index in range(pair_count):
            _first_skill, _second_skill, localized = _CONTROL_HARD_NEGATIVE_PAIRS[
                pair_index % len(_CONTROL_HARD_NEGATIVE_PAIRS)
            ]
            language = "ko" if (pair_index + seed) % 2 else "en"
            split_bucket = (pair_index + seed) % 10
            split = "train" if split_bucket < 8 else "validation" if split_bucket == 8 else "test"
            episode_id = f"control-hard-episode-{pair_index:05d}"
            semantic_group_id = f"control-hard-group-{pair_index:05d}"
            for side, (state, target) in enumerate(localized[language], start=1):
                sample_index = pair_index * 2 + side
                sample = {
                    "sample_id": f"control-hard-{sample_index:05d}",
                    "task_id": "control.skill",
                    "task_version": 1,
                    "state": f"{state}; counterfactual variant {pair_index:04d}",
                    "question": (
                        "select next safe high-level control skill"
                        if language == "en"
                        else "다음 안전한 고수준 제어 skill을 선택하라"
                    ),
                    "target": target,
                    "language": language,
                    "domain": "control-hard-negative",
                    "source": {
                        "kind": "synthetic-hard-negative",
                        "scenario_id": f"control-hard-scenario-{sample_index:05d}",
                        "episode_id": episode_id,
                        "semantic_group_id": semantic_group_id,
                        "counterfactual_group_id": semantic_group_id,
                        "pair_side": side,
                        "seed": seed,
                    },
                    "labels": {"qwen": None, "gemma": None, "human": None},
                    "review": {"status": "pending", "reviewer": None},
                    "provenance": {
                        "prompt_version": 1,
                        "generator": "control-hard-negative-v1",
                        "split": split,
                        "privacy_raw_inputs_stored": False,
                        "target_source": "synthetic_seed_only",
                    },
                }
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "record_type": "control_hard_negative_queue",
        "output_path": str(output.resolve()),
        "pair_count": pair_count,
        "sample_count": pair_count * 2,
        "seed": seed,
        "human_labeled": False,
    }


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
