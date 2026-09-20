"""Small, high-precision rules used before teacher calls."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import BooleanDecision, DecisionResult
from .registry import TaskDefinition


@dataclass(frozen=True)
class RuleMatch:
    """A deterministic decision with an audit-friendly explanation."""

    rule_id: str
    result: DecisionResult
    reason: str


_COMMITMENT_TERMS = (
    "결정",
    "하기로",
    "선호",
    "좋아해",
    "선택",
    "싶다",
    "원한다",
    "기로",
    "prefer",
    "want",
    "would like",
    "we decided",
    "will ",
    "must ",
    "remember",
)
_GREETING_TERMS = ("안녕", "반가", "고마워", "감사", "hello", "hi", "thanks", "thank you")
_STYLE_TERMS = ("제목", "title", "스타일", "style", "서식", "format", "공백", "spacing", "문법")
_CHANGE_PATTERN = re.compile(
    r"(?:에서\s+.+?\s*로\s*(?:변경|바꾸|바뀌|수정)|\b(?:changed|change|updated|update)\b)",
    re.IGNORECASE,
)


def match_rule(task: TaskDefinition, *, state: str, question: str) -> RuleMatch | None:
    """Return only conservative rules whose false-positive cost is bounded."""

    # The public request currently carries a stable question id rather than
    # free-form question prose. Do not let ids such as ``remember`` trigger a
    # semantic rule by themselves; rules must be grounded in the state.
    text = state.strip()
    lowered = text.lower()
    if task.id == "memory.remember_worthy":
        if any(term in lowered for term in _GREETING_TERMS) and not any(
            term in lowered for term in _COMMITMENT_TERMS
        ) and len(text) <= 120:
            return RuleMatch(
                rule_id="remember.greeting_only",
                result=BooleanDecision(value=False, probability=0.99),
                reason="greeting_or_thanks_without_durable_commitment",
            )
        if any(term in lowered for term in _COMMITMENT_TERMS):
            return RuleMatch(
                rule_id="remember.explicit_commitment",
                result=BooleanDecision(value=True, probability=0.97),
                reason="explicit_commitment_or_preference_signal",
            )
    if task.id == "wiki.semantic_change":
        if any(term in lowered for term in _STYLE_TERMS):
            return RuleMatch(
                rule_id="wiki.style_only",
                result=BooleanDecision(value=False, probability=0.97),
                reason="style_or_format_only_signal",
            )
        if _CHANGE_PATTERN.search(text):
            return RuleMatch(
                rule_id="wiki.factual_change",
                result=BooleanDecision(value=True, probability=0.95),
                reason="explicit_factual_change_signal",
            )
    return None
