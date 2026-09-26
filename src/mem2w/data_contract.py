"""Typed JSONL contracts for teacher episodes and derived SFT samples.

The functions in this module deliberately remove teacher memory messages by
structure and metadata. They never perform a global string replacement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class DataContractError(ValueError):
    """Raised for malformed or potentially leaking teacher data."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Message":
        role = value.get("role")
        content = value.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise DataContractError("message requires string role and content")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise DataContractError("message.metadata must be an object")
        return cls(role=role, content=content, metadata=dict(metadata))

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True)
class ActionSample:
    sample_id: str
    source_episode_id: str
    memory_snapshot_id: str
    prompt_messages: tuple[Message, ...]
    supervised_message_indices: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sample_type": "action",
            "source_episode_id": self.source_episode_id,
            "memory_snapshot_id": self.memory_snapshot_id,
            "messages": [message.as_dict() for message in self.prompt_messages],
            "supervised_message_indices": list(self.supervised_message_indices),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RecallSample:
    sample_id: str
    source_episode_id: str
    source_retrieval_event_id: str
    memory_snapshot_id: str
    prompt_messages: tuple[Message, ...]
    completion: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sample_type": "recall",
            "source_episode_id": self.source_episode_id,
            "source_retrieval_event_id": self.source_retrieval_event_id,
            "memory_snapshot_id": self.memory_snapshot_id,
            "messages": [message.as_dict() for message in self.prompt_messages],
            "completion": self.completion,
            "metadata": dict(self.metadata),
        }


def validate_episode(episode: Mapping[str, Any]) -> None:
    required = ("episode_id", "split", "memory_snapshot_id", "messages", "retrieval_events", "outcome")
    missing = [key for key in required if key not in episode]
    if missing:
        raise DataContractError(f"episode is missing fields: {', '.join(missing)}")
    if not isinstance(episode["messages"], list) or not episode["messages"]:
        raise DataContractError("episode.messages must be a non-empty list")
    if not isinstance(episode["retrieval_events"], list):
        raise DataContractError("episode.retrieval_events must be a list")
    if not isinstance(episode["outcome"], Mapping):
        raise DataContractError("episode.outcome must be an object")
    for raw in episode["messages"]:
        Message.from_dict(raw)
    for event in episode["retrieval_events"]:
        for key in ("event_id", "query_text", "k_requested", "k_returned", "injected_context_text"):
            if key not in event:
                raise DataContractError(f"retrieval event is missing {key}")
        if not isinstance(event["injected_context_text"], str):
            raise DataContractError("injected_context_text must be the exact serialized payload string")


def _is_teacher_memory(message: Message) -> bool:
    return (
        message.metadata.get("source") == "teacher_memory"
        or message.metadata.get("is_teacher_memory") is True
        or message.role == "memory"
    )


def build_action_sample(episode: Mapping[str, Any]) -> ActionSample:
    """Build a full-trajectory action sample with teacher memory structurally removed."""

    validate_episode(episode)
    messages = tuple(Message.from_dict(raw) for raw in episode["messages"])
    kept = tuple(message for message in messages if not _is_teacher_memory(message))
    supervised = tuple(index for index, message in enumerate(kept) if message.role == "assistant")
    if not supervised:
        raise DataContractError("action episode has no assistant action messages")

    payloads = [event["injected_context_text"] for event in episode["retrieval_events"]]
    leaked = [payload for payload in payloads if any(payload == message.content for message in kept)]
    if leaked:
        raise DataContractError("teacher retrieval payload remains as an action prompt message")
    return ActionSample(
        sample_id=f"{episode['episode_id']}:action",
        source_episode_id=str(episode["episode_id"]),
        memory_snapshot_id=str(episode["memory_snapshot_id"]),
        prompt_messages=kept,
        supervised_message_indices=supervised,
        metadata={
            "split": episode["split"],
            "reward": episode["outcome"].get("reward"),
            "success": episode["outcome"].get("success"),
            "termination_reason": episode["outcome"].get("termination_reason"),
        },
    )


def build_recall_samples(episode: Mapping[str, Any], system_prompt: str) -> list[RecallSample]:
    """Build one recall sample per real retrieval event, including empty payloads."""

    validate_episode(episode)
    result: list[RecallSample] = []
    for event in episode["retrieval_events"]:
        prompt = (
            Message(role="system", content=system_prompt),
            Message(
                role="user",
                content=(
                    f"当前检索查询：\n{event['query_text']}\n\n"
                    f"请从内部记忆中召回与该查询相关的至多 {event['k_requested']} 条历史经验。"
                    "\n按规定的记忆格式输出；没有相关记忆时输出空列表。"
                ),
            ),
        )
        result.append(
            RecallSample(
                sample_id=f"{episode['episode_id']}:{event['event_id']}:recall",
                source_episode_id=str(episode["episode_id"]),
                source_retrieval_event_id=str(event["event_id"]),
                memory_snapshot_id=str(episode["memory_snapshot_id"]),
                prompt_messages=prompt,
                completion=event["injected_context_text"],
                metadata={
                    "split": episode["split"],
                    "k_requested": event["k_requested"],
                    "k_returned": event["k_returned"],
                    "payload_sha256": hashlib.sha256(
                        event["injected_context_text"].encode("utf-8")
                    ).hexdigest(),
                },
            )
        )
    return result


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataContractError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise DataContractError(f"JSONL line {line_number} is not an object")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
