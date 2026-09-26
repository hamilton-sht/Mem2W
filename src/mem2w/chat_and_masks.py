"""Token-level labels for action and recall samples.

The returned labels are unshifted. A causal LM shifts exactly once inside its
loss function; callers must not shift these labels in the data pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


IGNORE_INDEX = -100


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    role: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid token span: {self}")


def labels_from_spans(input_ids: Sequence[int], spans: Iterable[TokenSpan], *, ignore_index: int = IGNORE_INDEX) -> list[int]:
    """Mark only assistant/action or recall completion spans for supervision."""

    labels = [ignore_index] * len(input_ids)
    for span in spans:
        if span.end > len(input_ids):
            raise ValueError(f"span exceeds sequence length: {span.end} > {len(input_ids)}")
        if span.role not in {"assistant", "recall_completion"}:
            raise ValueError(f"non-supervised role passed to labels_from_spans: {span.role}")
        labels[span.start : span.end] = input_ids[span.start : span.end]
    return labels


def labels_from_role_mask(input_ids: Sequence[int], supervised: Sequence[bool], *, ignore_index: int = IGNORE_INDEX) -> list[int]:
    if len(input_ids) != len(supervised):
        raise ValueError("input_ids and supervised mask must have the same length")
    return [token if flag else ignore_index for token, flag in zip(input_ids, supervised)]


def assert_role_mask(mask: Sequence[bool], roles: Sequence[str]) -> None:
    if len(mask) != len(roles):
        raise AssertionError("role mask length differs from token role sequence")
    invalid = {role for role, flag in zip(roles, mask) if flag and role not in {"assistant", "recall_completion"}}
    if invalid:
        raise AssertionError(f"supervision leaked into non-target roles: {sorted(invalid)}")


def assert_no_target_in_prompt(prompt_messages: Sequence[dict[str, str]], target: str) -> None:
    """Guard recall/action construction against putting the target in its prompt."""

    if any(message.get("content") == target for message in prompt_messages):
        raise AssertionError("target completion appears verbatim in prompt messages")
