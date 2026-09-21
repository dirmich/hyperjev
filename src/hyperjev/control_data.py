"""Quality and leakage checks for human-labelled control datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .contracts import (
    BooleanDecision,
    ChoiceDecision,
    ScoreDecision,
    parse_decision_result,
    validate_result_for_task,
)
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

_CONTROL_COMPOSITIONAL_TRAIN_TEMPLATES = {
    "STOP": (
        "front range sensor detects a collision within braking distance",
        "braking is required before the obstacle is reached",
        "the vehicle is about to hit a barrier",
        "the safety perimeter contains an immediate impact threat",
        "halt now because the path ends at a wall",
        "contact is imminent on the current heading",
        "the obstacle has entered the emergency buffer",
        "no safe stopping margin remains ahead",
    ),
    "HOLD": (
        "remain still while the stable pose is monitored",
        "keep position until the next command is available",
        "wait without moving for the sensor refresh",
        "maintain the current stance in the quiet scene",
        "pause at the waypoint with no hazard nearby",
        "the robot is stable and should not advance yet",
        "hold position while the target is temporarily unavailable",
        "stay in place until perception updates",
    ),
    "MOVE": (
        "advance through an unobstructed hallway",
        "continue forward along an open route",
        "travel toward the waypoint on a clear path",
        "forward navigation is safe in free space",
        "proceed straight through the corridor",
        "the route ahead permits normal motion",
        "move onward where the floor is clear",
        "an open lane is available for forward travel",
    ),
    "ROTATE": (
        "reorient toward the east corridor",
        "turn left to align with the waypoint",
        "change heading at the junction",
        "the next route requires a ninety degree turn",
        "rotate in place to face the goal",
        "the robot must pivot before continuing",
        "adjust orientation toward the next branch",
        "turn in the clear space to correct heading",
    ),
    "APPROACH": (
        "close the gap to the visible marker",
        "move nearer to the selected object",
        "the destination is visible but still distant",
        "reduce distance to the goal safely",
        "go toward the locked target",
        "the object can be reached by moving closer",
        "shorten the remaining distance to the waypoint",
        "advance toward the identified target",
    ),
    "RETREAT": (
        "back up from the moving obstacle",
        "increase distance from the approaching hazard",
        "reverse into the safe rear area",
        "withdraw from the blocked front",
        "move backward to escape the danger",
        "retire toward the clear space behind",
        "back away while the hazard closes in",
        "leave the forward danger zone using the rear path",
    ),
    "INTERACT": (
        "press the illuminated button",
        "grasp the aligned handle",
        "activate the switch beside the robot",
        "touch the reachable object",
        "pick up the selected item",
        "open the latch that is within reach",
        "use the nearby control panel",
        "take hold of the highlighted tool",
    ),
    "RECOVER": (
        "reinitialize after pose estimation failure",
        "restore balance after a fall",
        "the controller lost localization",
        "reset the navigation fault",
        "recover from the unstable state",
        "reacquire position after tracking was lost",
        "stabilize the robot after a control error",
        "restart recovery for the failed motion controller",
    ),
}

_CONTROL_BOUNDARY_TRAIN_TEMPLATES = {
    "STOP": (
        "braking distance is unsafe before the next waypoint",
        "impact risk leaves no time to continue forward",
        "the front boundary has become unsafe for motion",
        "the vehicle must halt before reaching the barrier",
        "collision margin is exhausted on the current route",
        "a sudden obstruction removes the safe travel margin",
        "continuing would enter the forbidden contact zone",
        "the safety monitor requires an immediate halt",
    ),
    "HOLD": (
        "keep this pose while awaiting a fresh instruction",
        "remain motionless until the next command arrives",
        "stay at the waypoint while perception refreshes",
        "do not advance while the scene remains calm",
        "wait in place because no hazard requires movement",
        "preserve the current orientation until a decision arrives",
        "pause the robot during a stable sensor interval",
        "hold position while the target is not yet actionable",
    ),
    "MOVE": (
        "advance along the unobstructed travel lane",
        "continue ahead while the route remains open",
        "normal forward travel is available beyond the waypoint",
        "follow the open corridor toward the next checkpoint",
        "the navigation lane permits steady forward motion",
        "proceed across the clear section of the route",
        "move ahead through the available free space",
        "the aligned route is open for continued travel",
    ),
    "ROTATE": (
        "pivot to align with the newly selected corridor",
        "turn the platform toward the next navigation branch",
        "the heading must change before the route continues",
        "reorient at the junction toward the planned waypoint",
        "rotate until the destination direction is aligned",
        "correct the heading without advancing the robot",
        "face the alternate corridor at the next intersection",
        "a direction change is required at the route split",
    ),
    "APPROACH": (
        "move closer to the visible destination marker",
        "shorten the gap separating the robot from its target",
        "the selected object is reachable after closing distance",
        "advance nearer to the identified waypoint",
        "the goal remains ahead and requires a closer position",
        "reduce the remaining distance to the tracked object",
        "continue toward the target until it is within reach",
        "the robot should draw nearer to the locked destination",
    ),
    "RETREAT": (
        "back away to widen the gap from the hazard",
        "reverse toward the clear area behind the robot",
        "the front risk requires increasing separation",
        "withdraw along the safe rear route",
        "move away from the obstacle approaching ahead",
        "leave the blocked area by reversing carefully",
        "retreat until the danger is outside the near zone",
        "the rear path is the safer direction from this threat",
    ),
    "INTERACT": (
        "operate the reachable control on the panel",
        "grip the handle that is aligned with the end effector",
        "activate the selected device within reach",
        "press the available control button",
        "touch the marked object at the interaction point",
        "pick up the tool positioned for grasping",
        "engage the nearby latch with the robot hand",
        "use the accessible switch beside the platform",
    ),
    "RECOVER": (
        "restore the pose after the estimator lost tracking",
        "rebuild localization following the navigation fault",
        "regain balance before normal control resumes",
        "restart the controller after the failed maneuver",
        "recover the robot from the unstable orientation",
        "reinitialize perception after position confidence collapsed",
        "stabilize the platform after the motion error",
        "resume from the fault state with a recovery action",
    ),
}

_CONTROL_KOREAN_COMPOSITIONAL_TRAIN_TEMPLATES = {
    "STOP": (
        "전방 센서가 충돌 직전의 위험을 감지했다",
        "장애물이 제동 거리 안에 있어 즉시 멈춰야 한다",
        "앞의 벽에 닿기 전에 주행을 중단한다",
        "안전 여유가 사라져 계속 진행할 수 없다",
        "현재 경로에 즉각적인 접촉 위험이 있다",
        "비상 충돌 신호가 들어와 차량을 세운다",
        "앞쪽 위험 때문에 브레이크를 작동한다",
        "충돌을 피하려면 지금 정지해야 한다",
    ),
    "HOLD": (
        "안정된 자세를 유지하며 새 명령을 기다린다",
        "센서가 갱신될 때까지 현재 위치에 머문다",
        "주변에 위험이 없어 잠시 움직이지 않는다",
        "다음 지시가 올 때까지 웨이포인트에서 대기한다",
        "목표가 아직 행동 가능하지 않아 위치를 지킨다",
        "조용한 장면에서 현재 자세를 보존한다",
        "전진하지 않고 다음 관측을 기다린다",
        "안전한 상태이므로 로봇을 그대로 둔다",
    ),
    "MOVE": (
        "막힘 없는 복도를 따라 앞으로 전진한다",
        "열린 경로를 따라 다음 지점으로 이동한다",
        "앞의 자유 공간에서 정상 주행을 계속한다",
        "장애물 없는 통로를 곧장 통과한다",
        "주행 차선이 비어 있어 계속 나아간다",
        "다음 웨이포인트까지 직선으로 진행한다",
        "바닥이 열린 구간을 지나 앞으로 간다",
        "안전한 빈 공간을 따라 전진할 수 있다",
    ),
    "ROTATE": (
        "다음 복도를 향해 로봇의 방향을 돌린다",
        "교차로에서 웨이포인트 쪽으로 회전한다",
        "계속 주행하기 전에 헤딩을 바꾼다",
        "목표 방향에 맞도록 제자리에서 선회한다",
        "다음 갈림길을 바라보도록 자세를 조정한다",
        "직진하지 않고 방향만 먼저 교정한다",
        "오른쪽 통로에 맞춰 플랫폼을 돌린다",
        "새 경로를 향해 회전할 공간이 충분하다",
    ),
    "APPROACH": (
        "눈앞의 표지판까지 거리를 줄인다",
        "선택된 물체에 가까워지도록 이동한다",
        "보이는 목적지가 아직 멀어 접근을 계속한다",
        "안전하게 목표와의 간격을 줄인다",
        "고정된 대상 쪽으로 가까이 간다",
        "도달할 물체를 향해 한 걸음 더 나아간다",
        "식별된 웨이포인트까지 남은 거리를 줄인다",
        "앞에 있는 목표에 천천히 접근한다",
    ),
    "RETREAT": (
        "다가오는 장애물에서 멀어지도록 후진한다",
        "위험과의 거리를 늘리기 위해 뒤로 간다",
        "뒤쪽의 안전 구역으로 물러난다",
        "막힌 앞쪽을 피해 후방으로 이동한다",
        "위험에서 벗어나기 위해 뒤로 물러난다",
        "앞의 위협이 가까워져 후퇴한다",
        "비어 있는 뒤쪽 경로를 이용해 멀어진다",
        "전방 위험 구역을 빠져나오며 후진한다",
    ),
    "INTERACT": (
        "손이 닿는 조명 버튼을 누른다",
        "정렬된 손잡이를 잡는다",
        "옆의 스위치를 눌러 장치를 활성화한다",
        "접근 가능한 물체를 만진다",
        "선택된 물건을 집어 든다",
        "가까운 걸쇠를 열기 위해 조작한다",
        "팔이 닿는 제어판을 사용한다",
        "표시된 도구를 잡아 올린다",
    ),
    "RECOVER": (
        "자세 추정이 실패해 시스템을 재초기화한다",
        "넘어진 뒤 로봇의 균형을 회복한다",
        "위치 추적이 끊겨 복구 절차를 시작한다",
        "탐색 오류를 재설정한다",
        "불안정한 상태에서 로봇을 안정화한다",
        "위치를 잃은 뒤 다시 좌표를 확보한다",
        "동작 제어 오류에서 복귀한다",
        "실패한 동작을 정리하고 복구를 재개한다",
    ),
}

_CONTROL_REVIEW_QUESTION_TEMPLATES = {
    "en": (
        "select next safe high-level control skill",
        "choose the safest control action for this state",
        "which control skill should execute next",
        "decide the next controller action from this state",
    ),
    "ko": (
        "다음 안전한 고수준 제어 skill을 선택하라",
        "이 상태에서 가장 안전한 제어 동작을 골라라",
        "다음에 실행할 제어 skill은 무엇인가",
        "현재 상태에서 다음 제어 행동을 결정하라",
    ),
}


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
    test_count_per_skill: int | None = None,
    validation_count_per_skill: int | None = None,
) -> dict[str, Any]:
    """Create a balanced, provenance-rich control queue awaiting human review."""

    if count_per_skill < 1:
        raise ValueError("count_per_skill must be positive")
    for name, value in (
        ("test_count_per_skill", test_count_per_skill),
        ("validation_count_per_skill", validation_count_per_skill),
    ):
        if value is not None and (value < 0 or value > count_per_skill):
            raise ValueError(f"{name} must be between zero and count_per_skill")
    reserved_per_skill = (test_count_per_skill or 0) + (validation_count_per_skill or 0)
    if reserved_per_skill > count_per_skill:
        raise ValueError("test and validation counts cannot exceed count_per_skill together")
    registry.get("control.skill", 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_count = len(CONTROL_SKILLS) * count_per_skill
    unique_prompts: set[tuple[str, str, str]] = set()
    unique_questions: set[str] = set()
    with output.open("w", encoding="utf-8") as handle:
        index = 0
        for skill_index, skill in enumerate(CONTROL_SKILLS):
            for variant in range(count_per_skill):
                index += 1
                language = "ko" if (variant + skill_index + seed) % 2 else "en"
                if language == "en":
                    templates = (
                        _CONTROL_SEED_TEMPLATES[skill][language]
                        + _CONTROL_COMPOSITIONAL_TRAIN_TEMPLATES[skill]
                        + _CONTROL_BOUNDARY_TRAIN_TEMPLATES[skill]
                    )
                else:
                    templates = (
                        _CONTROL_SEED_TEMPLATES[skill][language]
                        + _CONTROL_KOREAN_COMPOSITIONAL_TRAIN_TEMPLATES[skill]
                    )
                base_state = templates[variant % len(templates)]
                state = (
                    f"{base_state}; scenario variant {variant:04d}"
                    if language == "en"
                    else f"{base_state}; 시나리오 변형 {variant:04d}"
                )
                question = _CONTROL_REVIEW_QUESTION_TEMPLATES[language][variant % 4]
                unique_prompts.add((language, _normalise_text(state), _normalise_text(question)))
                unique_questions.add(_normalise_text(question))
                if test_count_per_skill is not None or validation_count_per_skill is not None:
                    test_quota = test_count_per_skill or 0
                    validation_quota = validation_count_per_skill or 0
                    if test_quota and variant < test_quota:
                        split = "test"
                    elif (
                        validation_quota
                        and variant < test_quota + validation_quota
                    ):
                        split = "validation"
                    else:
                        split = "train"
                else:
                    split_bucket = (variant + skill_index * 3 + seed) % 10
                    split = "train" if split_bucket < 8 else "validation" if split_bucket == 8 else "test"
                sample = {
                    "sample_id": f"control-review-{index:05d}",
                    "task_id": "control.skill",
                    "task_version": 1,
                    "state": state,
                    "question": question,
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
                        "prompt_version": 2,
                        "generator": "control-review-seed-v2",
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
        "prompt_version": 2,
        "unique_prompt_count": len(unique_prompts),
        "unique_question_count": len(unique_questions),
        "requested_test_count_per_skill": test_count_per_skill,
        "requested_validation_count_per_skill": validation_count_per_skill,
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


def generate_control_compositional_training_queue(
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    seed: int = 7,
) -> dict[str, Any]:
    """Create a train-only queue of compositional control paraphrases.

    The held-out OOD fixture is intentionally not generated here. These rows are
    synthetic research data and remain in the train split until human review.
    """

    registry.get("control.skill", 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    index = 0
    with output.open("w", encoding="utf-8") as handle:
        for skill_index, skill in enumerate(CONTROL_SKILLS):
            for variant, base_state in enumerate(_CONTROL_COMPOSITIONAL_TRAIN_TEMPLATES[skill]):
                index += 1
                sample = {
                    "sample_id": f"control-compositional-train-{index:05d}",
                    "task_id": "control.skill",
                    "task_version": 1,
                    "state": f"{base_state}; compositional training variant {variant:04d}",
                    "question": "select next safe high-level control skill",
                    "target": skill,
                    "language": "en",
                    "domain": "control-compositional-training",
                    "source": {
                        "kind": "synthetic-compositional-training",
                        "scenario_id": f"control-compositional-scenario-{index:05d}",
                        "episode_id": f"control-compositional-episode-{index:05d}",
                        "semantic_group_id": f"control-compositional-group-{index:05d}",
                        "seed": seed,
                        "skill_index": skill_index,
                    },
                    "labels": {"qwen": None, "gemma": None, "human": None},
                    "review": {"status": "pending", "reviewer": None},
                    "provenance": {
                        "prompt_version": 1,
                        "generator": "control-compositional-training-v1",
                        "split": "train",
                        "privacy_raw_inputs_stored": False,
                        "target_source": "synthetic_compositional_only",
                    },
                }
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "record_type": "control_compositional_training_queue",
        "output_path": str(output.resolve()),
        "sample_count": index,
        "skill_counts": {skill: len(_CONTROL_COMPOSITIONAL_TRAIN_TEMPLATES[skill]) for skill in CONTROL_SKILLS},
        "seed": seed,
        "split": "train",
        "human_labeled": False,
    }


def generate_control_boundary_training_queue(
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    seed: int = 7,
) -> dict[str, Any]:
    """Create balanced train-only examples for semantic decision boundaries."""

    registry.get("control.skill", 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    index = 0
    with output.open("w", encoding="utf-8") as handle:
        for skill_index, skill in enumerate(CONTROL_SKILLS):
            for variant, base_state in enumerate(_CONTROL_BOUNDARY_TRAIN_TEMPLATES[skill]):
                index += 1
                sample = {
                    "sample_id": f"control-boundary-train-{index:05d}",
                    "task_id": "control.skill",
                    "task_version": 1,
                    "state": f"{base_state}; boundary training variant {variant:04d}",
                    "question": "select next safe high-level control skill",
                    "target": skill,
                    "language": "en",
                    "domain": "control-boundary-training",
                    "source": {
                        "kind": "synthetic-boundary-training",
                        "scenario_id": f"control-boundary-scenario-{index:05d}",
                        "episode_id": f"control-boundary-episode-{index:05d}",
                        "semantic_group_id": f"control-boundary-group-{index:05d}",
                        "seed": seed,
                        "skill_index": skill_index,
                    },
                    "labels": {"qwen": None, "gemma": None, "human": None},
                    "review": {"status": "pending", "reviewer": None},
                    "provenance": {
                        "prompt_version": 1,
                        "generator": "control-boundary-training-v1",
                        "split": "train",
                        "privacy_raw_inputs_stored": False,
                        "target_source": "synthetic_boundary_only",
                    },
                }
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "record_type": "control_boundary_training_queue",
        "output_path": str(output.resolve()),
        "sample_count": index,
        "skill_counts": {skill: len(_CONTROL_BOUNDARY_TRAIN_TEMPLATES[skill]) for skill in CONTROL_SKILLS},
        "seed": seed,
        "split": "train",
        "human_labeled": False,
    }


def generate_control_korean_training_queue(
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    seed: int = 7,
) -> dict[str, Any]:
    """Create balanced Korean train-only paraphrases for multilingual control."""

    registry.get("control.skill", 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    index = 0
    with output.open("w", encoding="utf-8") as handle:
        for skill_index, skill in enumerate(CONTROL_SKILLS):
            for variant, base_state in enumerate(_CONTROL_KOREAN_COMPOSITIONAL_TRAIN_TEMPLATES[skill]):
                index += 1
                sample = {
                    "sample_id": f"control-korean-train-{index:05d}",
                    "task_id": "control.skill",
                    "task_version": 1,
                    "state": f"{base_state}; 한국어 학습 변형 {variant:04d}",
                    "question": "다음 안전한 고수준 제어 skill을 선택하라",
                    "target": skill,
                    "language": "ko",
                    "domain": "control-korean-compositional-training",
                    "source": {
                        "kind": "synthetic-korean-compositional-training",
                        "scenario_id": f"control-korean-scenario-{index:05d}",
                        "episode_id": f"control-korean-episode-{index:05d}",
                        "semantic_group_id": f"control-korean-group-{index:05d}",
                        "seed": seed,
                        "skill_index": skill_index,
                    },
                    "labels": {"qwen": None, "gemma": None, "human": None},
                    "review": {"status": "pending", "reviewer": None},
                    "provenance": {
                        "prompt_version": 1,
                        "generator": "control-korean-compositional-training-v1",
                        "split": "train",
                        "privacy_raw_inputs_stored": False,
                        "target_source": "synthetic_korean_compositional_only",
                    },
                }
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "record_type": "control_korean_training_queue",
        "output_path": str(output.resolve()),
        "sample_count": index,
        "skill_counts": {
            skill: len(_CONTROL_KOREAN_COMPOSITIONAL_TRAIN_TEMPLATES[skill])
            for skill in CONTROL_SKILLS
        },
        "seed": seed,
        "split": "train",
        "language": "ko",
        "human_labeled": False,
    }


def _human_target(sample: CanonicalSample, registry: TaskRegistry) -> Any:
    """Return a validated scalar target from one typed human correction."""

    raw = sample.labels.get("human")
    if not isinstance(raw, dict):
        raise TypeError(f"{sample.sample_id}: human label is required")
    task = registry.get(sample.task_id, sample.task_version)
    try:
        result = parse_decision_result(raw)
        validate_result_for_task(task, result)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{sample.sample_id}: invalid human label: {exc}") from exc
    if result.abstained:
        raise ValueError(f"{sample.sample_id}: human label must not abstain")
    if isinstance(result, BooleanDecision):
        return result.value
    if isinstance(result, ChoiceDecision):
        return result.selected
    if isinstance(result, ScoreDecision):
        return result.value
    raise ValueError(f"{sample.sample_id}: unsupported human label type")


def materialize_control_human_dataset(
    input_path: str | Path,
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    require_human_labels: bool = True,
    require_dual_review: bool = False,
) -> dict[str, Any]:
    """Copy a control queue while replacing synthetic targets with typed human labels.

    The source queue remains immutable. This creates the only dataset form that
    the control training command should use for a production accuracy claim.
    """

    source = Path(input_path)
    output = Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("materialized output must differ from the source queue")
    samples = load_jsonl(source, registry)
    validation = validate_control_dataset(source, registry, require_human_labels=require_human_labels)
    if not validation["passed"]:
        raise ValueError(f"control dataset is not ready for materialization: {validation['errors'][:3]}")
    raw_records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dual_reviewed_count = 0
    if require_dual_review:
        dual_review_errors: list[str] = []
        for raw in raw_records:
            sample_id = str(raw.get("sample_id", ""))
            review = raw.get("review")
            if not isinstance(review, dict):
                dual_review_errors.append(f"{sample_id}: review metadata is missing")
                continue
            reason = str(review.get("reason", ""))
            reviewer = str(review.get("reviewer", ""))
            if review.get("status") != "reviewed":
                dual_review_errors.append(f"{sample_id}: review status is not reviewed")
            elif not reviewer.strip():
                dual_review_errors.append(f"{sample_id}: reviewer is empty")
            elif not reason.startswith("control dual-review "):
                dual_review_errors.append(
                    f"{sample_id}: review reason is not dual-review provenance"
                )
            else:
                dual_reviewed_count += 1
        if dual_review_errors:
            raise ValueError(
                "control dataset is not ready for dual-review materialization: "
                f"{dual_review_errors[:3]}"
            )
    by_id = {sample.sample_id: sample for sample in samples}
    materialized: list[dict[str, Any]] = []
    for raw in raw_records:
        sample = by_id[str(raw["sample_id"])]
        record = deepcopy(raw)
        record["target"] = _human_target(sample, registry)
        provenance = dict(record.get("provenance", {}))
        provenance["target_source"] = "human_review"
        provenance["materialized_from_target"] = raw.get("target")
        record["provenance"] = provenance
        materialized.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in materialized:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    output_validation = validate_control_dataset(output, registry, require_human_labels=require_human_labels)
    if not output_validation["passed"]:
        raise ValueError(f"materialized control dataset failed validation: {output_validation['errors'][:3]}")
    return {
        "record_type": "control_human_materialized_dataset",
        "input_path": str(source.resolve()),
        "output_path": str(output.resolve()),
        "sample_count": len(materialized),
        "human_labeled_count": output_validation["human_labeled_count"],
        "dual_review_required": require_dual_review,
        "dual_reviewed_count": dual_reviewed_count if require_dual_review else None,
        "target_source": "human_review",
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def merge_control_datasets(
    input_paths: list[str | Path],
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    require_human_labels: bool = False,
) -> dict[str, Any]:
    """Merge control queues and re-run the combined leakage gate."""

    if len(input_paths) < 2:
        raise ValueError("at least two control datasets are required")
    output = Path(output_path)
    sources = [Path(path) for path in input_paths]
    if any(source.resolve() == output.resolve() for source in sources):
        raise ValueError("merged output must differ from every source dataset")
    merged_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    source_reports: list[dict[str, Any]] = []
    for source in sources:
        report = validate_control_dataset(source, registry, require_human_labels=require_human_labels)
        if not report["passed"]:
            raise ValueError(f"control dataset is not ready for merge: {report['errors'][:3]}")
        source_reports.append(report)
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id across control datasets: {sample_id}")
            seen_ids.add(sample_id)
            merged_records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in merged_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    combined = validate_control_dataset(output, registry, require_human_labels=require_human_labels)
    if not combined["passed"]:
        raise ValueError(f"merged control dataset failed validation: {combined['errors'][:3]}")
    return {
        "record_type": "control_merged_dataset",
        "output_path": str(output.resolve()),
        "input_paths": [str(source.resolve()) for source in sources],
        "input_count": len(sources),
        "sample_count": len(merged_records),
        "split_counts": combined["split_counts"],
        "human_labeled_count": combined["human_labeled_count"],
        "require_human_labels": require_human_labels,
        "input_sha256": {
            str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in sources
        },
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_reports": source_reports,
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
