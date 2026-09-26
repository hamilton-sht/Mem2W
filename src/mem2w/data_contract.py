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
    # Preserve structured tool_calls/tool_call_id and any provider-specific
    # fields.  Dropping these fields would turn an executable trajectory into
    # plain text and make the derived action sample semantically different.
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Message":
        role = value.get("role")
        content = value.get("content", "")
        if content is None and ("tool_calls" in value or role in {"tool_call", "tool_response"}):
            content = ""
        if not isinstance(role, str) or not isinstance(content, str):
            raise DataContractError("message requires string role and content (null is allowed for structured tool calls)")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise DataContractError("message.metadata must be an object")
        extra = {key: value[key] for key in ("tool_calls", "tool_call_id", "name") if key in value}
        return cls(role=role, content=content, metadata=dict(metadata), extra=extra)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        result.update(dict(self.extra))
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
    required = ("episode_id", "split", "memory_snapshot_id", "outcome")
    missing = [key for key in required if key not in episode]
    if missing:
        raise DataContractError(f"episode is missing fields: {', '.join(missing)}")
    if not any(key in episode for key in ("messages", "output", "trajectory")):
        raise DataContractError("episode is missing structured messages (messages/output/trajectory)")
    if not any(key in episode for key in ("retrieval_events", "retrieval_records", "retrievals")):
        raise DataContractError("episode is missing retrieval_events (or retrieval_records/retrievals)")
    if not isinstance(episode["episode_id"], (str, int)) or not str(episode["episode_id"]).strip():
        raise DataContractError("episode.episode_id must be a non-empty string or integer")
    if not isinstance(episode["split"], str) or not episode["split"].strip():
        raise DataContractError("episode.split must be a non-empty string")
    if not isinstance(episode["memory_snapshot_id"], (str, int)) or not str(episode["memory_snapshot_id"]).strip():
        raise DataContractError("episode.memory_snapshot_id must be provided")
    raw_messages = _episode_messages(episode)
    if not isinstance(raw_messages, list) or not raw_messages:
        raise DataContractError("episode.messages (or structured output/trajectory.messages) must be a non-empty list")
    events = _episode_retrieval_events(episode)
    if not isinstance(events, list):
        raise DataContractError("episode.retrieval_events must be a list")
    if not isinstance(episode["outcome"], Mapping):
        raise DataContractError("episode.outcome must be an object")
    for raw in raw_messages:
        Message.from_dict(raw)
    for event in events:
        if not isinstance(event, Mapping):
            raise DataContractError("retrieval event must be an object")
        for key in ("event_id", "query_text", "k_requested", "k_returned", "injected_context_text"):
            if key not in event:
                raise DataContractError(f"retrieval event is missing {key}")
        if not str(event["event_id"]).strip() or not str(event["query_text"]).strip():
            raise DataContractError("retrieval event_id and query_text must be non-empty")
        if isinstance(event["k_requested"], bool) or not isinstance(event["k_requested"], int) or event["k_requested"] < 0:
            raise DataContractError("retrieval k_requested must be a non-negative integer")
        if isinstance(event["k_returned"], bool) or not isinstance(event["k_returned"], int) or event["k_returned"] < 0:
            raise DataContractError("retrieval k_returned must be a non-negative integer")
        if event["k_returned"] > event["k_requested"]:
            raise DataContractError("retrieval k_returned cannot exceed k_requested")
        if not isinstance(event["injected_context_text"], str):
            raise DataContractError("injected_context_text must be the exact serialized payload string")
        if not event["injected_context_text"].strip() and event.get("recall_missing_payload") is not True:
            raise DataContractError("empty retrieval payload is ambiguous; encode the fixed no-hit result explicitly")


def _episode_messages(episode: Mapping[str, Any]) -> Any:
    raw_messages = episode.get("messages")
    if raw_messages is not None:
        return raw_messages
    for container_name in ("output", "trajectory"):
        container = episode.get(container_name)
        if isinstance(container, Mapping) and "messages" in container:
            return container["messages"]
        if isinstance(container, list):
            return container
    return None


def _episode_retrieval_events(episode: Mapping[str, Any]) -> Any:
    events = episode.get("retrieval_events")
    if events is None:
        events = episode.get("retrieval_records")
    if events is None:
        events = episode.get("retrievals")
    return [] if events is None else events


def _is_teacher_memory(message: Message) -> bool:
    return (
        message.metadata.get("source") in {"teacher_memory", "retrieved_memory", "memory_retrieval"}
        or message.metadata.get("is_teacher_memory") is True
        or message.metadata.get("is_retrieved_memory") is True
        or message.role == "memory"
    )


def build_action_sample(episode: Mapping[str, Any]) -> ActionSample:
    """Build a full-trajectory action sample with teacher memory structurally removed."""

    validate_episode(episode)
    messages = tuple(Message.from_dict(raw) for raw in _episode_messages(episode))
    kept = tuple(message for message in messages if not _is_teacher_memory(message))
    supervised = tuple(
        index for index, message in enumerate(kept) if message.role in {"assistant", "tool_call"}
    )
    if not supervised:
        raise DataContractError("action episode has no assistant action messages")

    payloads = [
        event["injected_context_text"]
        for event in _episode_retrieval_events(episode)
        if event.get("injected_context_text")
    ]
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
            **_research_audit_metadata(episode),
        },
    )


def build_recall_samples(episode: Mapping[str, Any], system_prompt: str) -> list[RecallSample]:
    """Build one recall sample per real retrieval event, including empty payloads."""

    validate_episode(episode)
    result: list[RecallSample] = []
    for event in _episode_retrieval_events(episode):
        if event.get("recall_missing_payload") is True:
            continue
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
                    "selected_memory_ids": event.get("selected_memory_ids"),
                    "payload_sha256": hashlib.sha256(
                        event["injected_context_text"].encode("utf-8")
                    ).hexdigest(),
                    **_research_audit_metadata(episode),
                },
            )
        )
    return result


def _research_audit_metadata(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve MemRL provenance without putting diagnostics in model input."""

    outcome = episode.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    retrieved_ids = episode.get("_memq_retrieved_ids", episode.get("retrieved_memory_ids"))
    values = {
        "task_id": episode.get("task_id"),
        "task_family": episode.get("task_family"),
        "epoch": episode.get("epoch"),
        "task_description": episode.get("_memq_task_description", episode.get("task_description")),
        "partial_credit": episode.get("partial_credit", outcome.get("partial_credit")),
        "partial_credit_diagnostic": episode.get("partial_credit_diagnostic", outcome.get("partial_credit_diagnostic")),
        "diagnostics": episode.get("diagnostics", outcome.get("diagnostics")),
        "reflection_diagnostics": episode.get("reflection_diagnostics", outcome.get("reflection_diagnostics")),
        "source_algorithm": episode.get("source_algorithm", "memrl"),
        "retrieved_memory_ids": list(retrieved_ids) if isinstance(retrieved_ids, list) else retrieved_ids,
        "q_value": episode.get("q_value"),
        "q_visits": episode.get("q_visits"),
    }
    return {key: value for key, value in values.items() if value is not None}


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
