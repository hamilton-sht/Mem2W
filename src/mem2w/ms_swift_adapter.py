"""Data and configuration glue for Mem2W SFT with ms-swift.

The Mem2W algorithm stores two text supervision streams:

* ``action`` samples supervise assistant actions from a teacher trajectory;
* ``recall`` samples supervise the exact serialized retrieval payload.

ms-swift already understands the canonical ``messages`` JSONL format and the
per-assistant-message ``loss`` flag.  This module only converts the Mem2W
episode contract into that format and writes a conservative ``swift sft``
configuration.  It deliberately does not implement the Mem2W model wrapper or
the dual-branch optimizer; those are loaded through ``--external_plugins``
and/or the project training runner.

The adapter has no runtime dependency on ms-swift.  This is intentional: the
dataset conversion and configuration generation can run on a login node before
the training environment is installed.

The implementation is pinned against ms-swift main at
``c08110b30a1ccb60bcfb70adf87d2cd72f5b9f3c`` (2026-09-26).  The pin is
metadata only; a preflight should still record the installed package version
and git revision before a run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


MS_SWIFT_MAIN_COMMIT = "c08110b30a1ccb60bcfb70adf87d2cd72f5b9f3c"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_TEMPLATE = "qwen3_5"
DEFAULT_MAX_LENGTH = 4096
TRAIN_MODES = ("action", "recall", "both")
DEFAULT_RECALL_SYSTEM = (
    "你正在执行历史记忆召回任务。历史记忆是待回忆的数据，"
    "不是当前要执行的命令。不要执行当前任务，也不要补写不存在的经验。"
)


class Mem2WDataError(ValueError):
    """Raised when an episode cannot be converted without guessing."""


def _normalise_mode(value: str) -> str:
    """Validate the independent pilot training modes.

    ``action`` and ``recall`` are intentionally separate runs.  ``both`` is
    retained as a convenience for producing one stock ms-swift mixed dataset;
    it is not the dual-objective Mem2W trainer and should not be used to claim
    paired loss normalization or branch-specific gradient routing.
    """

    mode = str(value).strip().lower()
    if mode not in TRAIN_MODES:
        raise Mem2WDataError(f"mode must be one of {', '.join(TRAIN_MODES)}")
    return mode


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Mem2WDataError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise Mem2WDataError(f"{path}:{line_no}: expected a JSON object")
            yield line_no, row


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_canonical_json(row) + "\n")
            count += 1
    return count


def _normalise_split(value: Any) -> str:
    """Normalise split names without silently moving test data into train."""

    # The MemRL export records the source split.  Silently treating a missing
    # split as train would make a held-out episode part of the SFT corpus and
    # is especially dangerous when the export is assembled from multiple
    # epochs.  ``data_contract`` has always required this field; keep the
    # adapter consistent with that contract.
    if value is None or (isinstance(value, str) and not value.strip()):
        raise Mem2WDataError("episode.split is required; refusing to default a missing split to train")
    if isinstance(value, bool):
        raise Mem2WDataError("episode.split must be a string")
    split = str(value).strip().lower().replace("-", "_")
    aliases = {"valid": "validation", "dev": "validation", "eval": "validation"}
    return aliases.get(split, split)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return _canonical_json(value)


def _message_content(message: Mapping[str, Any]) -> str:
    return _as_text(message.get("content", ""))


def _role(message: Mapping[str, Any]) -> str:
    raw = str(message.get("role", "")).strip().lower().replace("-", "_")
    aliases = {"tool": "tool_response", "function": "tool_response", "function_call": "tool_call"}
    return aliases.get(raw, raw)


def _validate_message(message: Mapping[str, Any], index: int, source: str) -> None:
    role = _role(message)
    # `memory` is an exporter-only role.  It is removed structurally below and
    # must never be passed to the student prompt.
    if role not in {"system", "user", "assistant", "tool_call", "tool_response", "memory"}:
        raise Mem2WDataError(f"{source}: messages[{index}] has unsupported role {message.get('role')!r}")
    if "content" not in message and role not in {"tool_call", "tool_response"}:
        raise Mem2WDataError(f"{source}: messages[{index}] is missing content")


def _copy_message(message: Mapping[str, Any], *, loss: Optional[bool] = None) -> Dict[str, Any]:
    copied = copy.deepcopy(dict(message))
    # Keep ms-swift's canonical role spelling.  `tool` is accepted by
    # MessagesPreprocessor, but writing tool_response avoids relying on its
    # alias table and keeps the manifest unambiguous.
    copied["role"] = _role(copied)
    if loss is not None and copied["role"] in {"assistant", "tool_call"}:
        copied["loss"] = loss
    return copied


def _retrieval_payloads(episode: Mapping[str, Any]) -> List[str]:
    payloads: List[str] = []
    for event in _episode_retrieval_events(episode):
        if not isinstance(event, Mapping):
            continue
        payload = event.get("injected_context_text")
        if payload is not None:
            payloads.append(_as_text(payload))
    return payloads


def _episode_retrieval_events(episode: Mapping[str, Any]) -> Any:
    """Return the normalized retrieval-event list from common runner aliases."""

    events = episode.get("retrieval_events")
    if events is None:
        # ``retrieval_records`` is the name used by some MemRL exports.  It is
        # an alias only: the event objects still need query/payload fields and
        # are validated by ``_recall_messages``.
        events = episode.get("retrieval_records")
    if events is None:
        events = episode.get("retrievals")
    return events or []


def _prompt_contains_payload(messages: Sequence[Mapping[str, Any]], payloads: Sequence[str]) -> Optional[str]:
    """Return a leaked payload if it appears in a non-assistant prompt span."""

    for message in messages:
        if _role(message) in {"assistant", "tool_call"}:
            continue
        content = _message_content(message)
        for payload in payloads:
            if payload and payload in content:
                return payload
    return None


def _task_prefix(episode: Mapping[str, Any]) -> List[Dict[str, Any]]:
    task = episode.get("task")
    if not isinstance(task, Mapping):
        task = {}
    # The MemRL runner calls this field ``_memq_task_description``.  The
    # normalized contract uses ``task.query``; accepting both lets the
    # converter consume a frozen runner export without fabricating a task.
    query = task.get("query")
    if query is None:
        query = task.get("initial_query")
    if query is None:
        query = episode.get("_memq_task_description")
    if query is None:
        query = episode.get("task_description")
    if query is None:
        raise Mem2WDataError(
            "episode.task.query or _memq_task_description is required when messages do not contain a user turn"
        )
    content = _as_text(query)
    initial = task.get("initial_observation")
    if initial not in (None, ""):
        content = f"{content}\n\n初始观察：\n{_as_text(initial)}"
    system = task.get("system") or task.get("system_prompt")
    if system is None:
        system = episode.get("system_prompt")
    result: List[Dict[str, Any]] = []
    if system not in (None, ""):
        result.append({"role": "system", "content": _as_text(system)})
    result.append({"role": "user", "content": content})
    return result


def _action_messages(episode: Mapping[str, Any], source: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_messages = episode.get("messages")
    # Some runner exports wrap the serialized trajectory under ``output`` or
    # ``trajectory``.  We only accept structured message lists here; a free
    # form serialized trajectory cannot be converted without guessing role
    # boundaries and is rejected below with a useful error.
    if raw_messages is None:
        for container_name in ("output", "trajectory"):
            container = episode.get(container_name)
            if isinstance(container, Mapping) and isinstance(container.get("messages"), list):
                raw_messages = container["messages"]
                break
            if isinstance(container, list):
                raw_messages = container
                break
    if raw_messages is None:
        raise Mem2WDataError(
            f"{source}: structured episode.messages (or output/trajectory.messages) is required; "
            "a serialized trajectory string cannot be split into action labels safely"
        )
    if not isinstance(raw_messages, list) or not raw_messages:
        raise Mem2WDataError(f"{source}: episode.messages must be a non-empty list")
    for index, message in enumerate(raw_messages):
        if not isinstance(message, Mapping):
            raise Mem2WDataError(f"{source}: messages[{index}] must be an object")
        _validate_message(message, index, source)

    # A teacher exporter may store only the alternating action/observation
    # suffix.  In that case reconstruct the visible task prefix from `task`.
    has_user = any(_role(message) == "user" for message in raw_messages)
    messages: List[Dict[str, Any]] = []
    if not has_user:
        messages.extend(_task_prefix(episode))
    else:
        first_user = next(i for i, message in enumerate(raw_messages) if _role(message) == "user")
        if first_user > 0 and _role(raw_messages[0]) not in {"system"}:
            messages.extend(_task_prefix(episode))

    payloads = _retrieval_payloads(episode)
    for index, message in enumerate(raw_messages):
        role = _role(message)
        metadata = message.get("metadata", {})
        is_teacher_memory = (
            role == "memory"
            or isinstance(metadata, Mapping)
            and (
                metadata.get("source") in {"teacher_memory", "retrieved_memory", "memory_retrieval"}
                or metadata.get("is_teacher_memory") is True
                or metadata.get("is_retrieved_memory") is True
            )
        )
        if is_teacher_memory:
            # Removing by role/metadata is intentional.  A global text replace
            # could delete legitimate task text that happens to repeat a memory.
            continue
        copied = _copy_message(message)
        role = copied["role"]
        # Only actions are supervised.  tool_call is transformed to an
        # assistant span by ms-swift's agent template, so mark it explicitly.
        if role in {"assistant", "tool_call"}:
            copied["loss"] = True
        elif role in {"system", "user", "tool_response"}:
            copied["loss"] = False
        messages.append(copied)

    leaked = _prompt_contains_payload(messages, payloads)
    if leaked is not None:
        raise Mem2WDataError(
            f"{source}: teacher retrieval payload appears in an action prompt; "
            "remove the injected memory from messages before conversion"
        )

    extra: Dict[str, Any] = {
        "sample_type": "action",
        "source_episode_id": episode.get("episode_id"),
        "memory_snapshot_id": episode.get("memory_snapshot_id"),
        "success": (episode.get("outcome") or {}).get("success") if isinstance(episode.get("outcome"), Mapping) else None,
        "reward": (episode.get("outcome") or {}).get("reward") if isinstance(episode.get("outcome"), Mapping) else None,
        "termination_reason": (episode.get("outcome") or {}).get("termination_reason") if isinstance(episode.get("outcome"), Mapping) else None,
        "split": _normalise_split(episode.get("split")),
        "prompt_version": (episode.get("teacher") or {}).get("prompt_version") if isinstance(episode.get("teacher"), Mapping) else None,
    }
    extra.update(_research_audit_metadata(episode))
    return messages, extra


def _recall_prompt(query: str, k_requested: int) -> str:
    if k_requested < 0:
        raise Mem2WDataError("k_requested must be non-negative")
    return (
        f"当前检索查询：\n{query}\n\n"
        f"请从内部记忆中召回与该查询相关的至多 {k_requested} 条历史经验。\n"
        "按规定的记忆格式输出；没有相关记忆时输出空列表。"
    )


def _recall_messages(episode: Mapping[str, Any], event: Mapping[str, Any], source: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    query = event.get("query_text")
    payload = event.get("injected_context_text")
    if query is None or not _as_text(query).strip():
        raise Mem2WDataError(f"{source}: retrieval event is missing query_text")
    if payload is None:
        raise Mem2WDataError(f"{source}: retrieval event is missing injected_context_text")
    k_requested = event.get("k_requested")
    if isinstance(k_requested, bool) or not isinstance(k_requested, int):
        raise Mem2WDataError(f"{source}: k_requested must be an integer")
    if k_requested < 0:
        raise Mem2WDataError(f"{source}: k_requested must be non-negative")
    k_returned = event.get("k_returned")
    if k_returned is not None:
        if isinstance(k_returned, bool) or not isinstance(k_returned, int) or k_returned < 0:
            raise Mem2WDataError(f"{source}: k_returned must be a non-negative integer")
        if k_returned > k_requested:
            raise Mem2WDataError(f"{source}: k_returned cannot exceed k_requested")
    selected_ids = event.get("selected_memory_ids")
    if selected_ids is not None and not isinstance(selected_ids, list):
        raise Mem2WDataError(f"{source}: selected_memory_ids must be a list when present")
    payload_text = _as_text(payload)
    if not payload_text.strip():
        raise Mem2WDataError(f"{source}: recall payload cannot be empty; encode no-hit as {{\"memories\":[]}}")

    messages = [
        {"role": "system", "content": DEFAULT_RECALL_SYSTEM, "loss": False},
        {"role": "user", "content": _recall_prompt(_as_text(query), k_requested), "loss": False},
        {"role": "assistant", "content": payload_text, "loss": True},
    ]
    extra = {
        "sample_type": "recall",
        "source_episode_id": episode.get("episode_id"),
        "source_retrieval_event_id": event.get("event_id"),
        "memory_snapshot_id": episode.get("memory_snapshot_id"),
        "retrieval_k_requested": k_requested,
        "retrieval_k_returned": k_returned,
        "selected_memory_ids": list(selected_ids) if selected_ids is not None else None,
        "payload_sha256": hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
        "split": _normalise_split(episode.get("split")),
        "prompt_version": (episode.get("teacher") or {}).get("prompt_version") if isinstance(episode.get("teacher"), Mapping) else None,
    }
    extra.update(_research_audit_metadata(episode))
    return messages, extra


def _research_audit_metadata(episode: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy provenance fields from the MemRL runner without duplicating payloads.

    The 0916 MemRL baseline stores useful diagnostics next to the trajectory:
    task/epoch identity, exact/partial reward, retrieval IDs and scalar-Q
    bookkeeping.  These fields are audit metadata only; they are deliberately
    not converted into prompt messages or loss labels.  Keeping them in the
    derived JSONL makes split filtering and post-run attribution possible.
    """

    outcome = episode.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    diagnostics = episode.get("diagnostics")
    if diagnostics is None:
        diagnostics = outcome.get("diagnostics")
    partial_credit = episode.get("partial_credit", outcome.get("partial_credit"))
    partial_diagnostic = episode.get("partial_credit_diagnostic", outcome.get("partial_credit_diagnostic"))
    retrieved_ids = episode.get("_memq_retrieved_ids")
    if retrieved_ids is None:
        retrieved_ids = episode.get("retrieved_memory_ids")
    result: Dict[str, Any] = {
        "task_id": episode.get("task_id"),
        "task_family": episode.get("task_family"),
        "epoch": episode.get("epoch"),
        "task_description": episode.get("_memq_task_description", episode.get("task_description")),
        "partial_credit": partial_credit,
        "partial_credit_diagnostic": partial_diagnostic,
        "diagnostics": diagnostics,
        "reflection_diagnostics": episode.get("reflection_diagnostics", outcome.get("reflection_diagnostics")),
        "source_algorithm": episode.get("source_algorithm", "memrl"),
        "retrieved_memory_ids": list(retrieved_ids) if isinstance(retrieved_ids, list) else retrieved_ids,
        "q_value": episode.get("q_value"),
        "q_visits": episode.get("q_visits"),
    }
    # Avoid writing a dozen null columns into every row.  ``source_algorithm``
    # is retained when present (or defaults to the research run's baseline).
    return {key: value for key, value in result.items() if value is not None}


def _sample_row(messages: List[Dict[str, Any]], metadata: Mapping[str, Any]) -> Dict[str, Any]:
    row = {"messages": messages}
    row.update(metadata)
    row["channel"] = str(metadata["sample_type"])
    # Keep a hash over the exact message content and metadata used by the
    # converter.  This gives the run manifest a stable audit handle without
    # storing a second copy of potentially sensitive payloads.
    row["raw_content_hash"] = _sha256_json({"messages": messages, **metadata})
    # Per-example thinking mode is supported by ms-swift>=4.3.0 and avoids
    # silently inheriting the template's default for Qwen3.5.
    row["chat_template_kwargs"] = {"enable_thinking": False}
    return row


def convert_episodes(input_path: os.PathLike[str] | str, output_dir: os.PathLike[str] | str) -> Dict[str, Any]:
    """Convert Mem2W episode JSONL into ms-swift action/recall JSONL files.

    Files are emitted as ``action_<split>.jsonl`` and
    ``recall_<split>.jsonl``.  The returned manifest is also written to
    ``manifest.json`` and includes source/output hashes and row counts.
    """

    source_path = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source_path.is_file():
        raise Mem2WDataError(f"input JSONL does not exist: {source_path}")
    destination.mkdir(parents=True, exist_ok=True)

    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    source_rows = 0
    recall_missing_payload = 0
    for line_no, episode in _read_jsonl(source_path):
        source_rows += 1
        source = f"{source_path}:{line_no}"
        episode_id = episode.get("episode_id")
        if not episode_id:
            raise Mem2WDataError(f"{source}: episode_id is required")
        action_messages, action_meta = _action_messages(episode, source)
        action_meta["source_episode_id"] = str(episode_id)
        split = _normalise_split(episode.get("split"))
        buckets.setdefault(("action", split), []).append(_sample_row(action_messages, action_meta))

        events = _episode_retrieval_events(episode)
        if not isinstance(events, list):
            raise Mem2WDataError(f"{source}: retrieval_events must be a list")
        for event_index, event in enumerate(events):
            if not isinstance(event, Mapping):
                raise Mem2WDataError(f"{source}: retrieval_events[{event_index}] must be an object")
            event_source = f"{source} retrieval_events[{event_index}]"
            event_id = event.get("event_id")
            if event_id is None or not str(event_id).strip():
                raise Mem2WDataError(f"{event_source}: event_id is required for retrieval provenance")
            # Raw AutomationBench imports intentionally keep an action-ready
            # episode when actor prompt capture was incomplete.  Such an event
            # is not a valid recall target; skip only the recall row and retain
            # the action row.  Never substitute retrieval_records for payload.
            if event.get("recall_missing_payload") is True or not str(event.get("injected_context_text") or "").strip():
                if event.get("recall_missing_payload") is True:
                    recall_missing_payload += 1
                    continue
                raise Mem2WDataError(f"{event_source}: retrieval event has an empty injected_context_text")
            recall_messages, recall_meta = _recall_messages(episode, event, event_source)
            recall_meta["source_episode_id"] = str(episode_id)
            recall_meta["source_retrieval_event_id"] = str(event_id)
            buckets.setdefault(("recall", split), []).append(_sample_row(recall_messages, recall_meta))

    files: Dict[str, Dict[str, Any]] = {}
    for (sample_type, split), rows in sorted(buckets.items()):
        path = destination / f"{sample_type}_{split}.jsonl"
        count = _write_jsonl(path, rows)
        files[path.name] = {"count": count, "sha256": _sha256_file(path), "sample_type": sample_type, "split": split}

    manifest = {
        "format": "mem2w-ms-swift-v1",
        "ms_swift_main_commit": MS_SWIFT_MAIN_COMMIT,
        "source": {"path": str(source_path), "sha256": _sha256_file(source_path), "episodes": source_rows},
        "files": files,
        "qa": {"recall_missing_payload": recall_missing_payload},
        "constraints": {
            "action_prompt_excludes_teacher_payload": True,
            "recall_target_is_exact_injected_context_text": True,
            "assistant_loss_flag": True,
            "thinking_enabled": False,
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


@dataclass
class SwiftSftConfig:
    """The subset of ms-swift SftArguments used by the Mem2W pilot."""

    model: str = DEFAULT_MODEL
    model_revision: Optional[str] = None
    template: str = DEFAULT_TEMPLATE
    dataset: List[str] = field(default_factory=list)
    val_dataset: List[str] = field(default_factory=list)
    output_dir: str = "artifacts/ms_swift_sft"
    tuner_type: str = "full"
    torch_dtype: str = "bfloat16"
    bf16: bool = True
    max_length: int = DEFAULT_MAX_LENGTH
    # ms-swift's CLI accepts ``delete`` and maps it to the template's
    # allocation-free ``raise`` behavior.  ``raise`` is a template-internal
    # value and is rejected by SftArguments when supplied in JSON.
    truncation_strategy: str = "delete"
    packing: bool = False
    gradient_checkpointing: bool = False
    enable_thinking: bool = False
    loss_scale: str = "default"
    is_binary_loss_scale: bool = True
    remove_unused_columns: bool = False
    enable_channel_loss: bool = True
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    seed: int = 42
    report_to: List[str] = field(default_factory=list)
    external_plugins: List[str] = field(default_factory=list)


def build_swift_config(
    *,
    action_data: Sequence[os.PathLike[str] | str],
    recall_data: Sequence[os.PathLike[str] | str],
    action_val_data: Sequence[os.PathLike[str] | str] = (),
    recall_val_data: Sequence[os.PathLike[str] | str] = (),
    output_dir: os.PathLike[str] | str = "artifacts/ms_swift_sft",
    model: str = DEFAULT_MODEL,
    model_revision: Optional[str] = None,
    external_plugins: Sequence[os.PathLike[str] | str] = (),
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = 42,
    mode: str = "both",
) -> Dict[str, Any]:
    """Build a valid ms-swift JSON config for one pilot training mode.

    ``action`` and ``recall`` select one dataset family and are the supported
    first training paths.  ``both`` remains available for diagnostics and
    mixed-data smoke tests.  The generated JSON checks the ms-swift
    dataset/template path; the project ``mem2w-sft-train`` runner is the
    canonical memory-only path because stock ``tuner_type=full`` may reopen
    frozen parameters.  Neither path implements the later W/C schedule.
    """

    mode = _normalise_mode(mode)
    action_paths = [str(Path(path).expanduser().resolve()) for path in action_data]
    recall_paths = [str(Path(path).expanduser().resolve()) for path in recall_data]
    action_val_paths = [str(Path(path).expanduser().resolve()) for path in action_val_data]
    recall_val_paths = [str(Path(path).expanduser().resolve()) for path in recall_val_data]
    if mode == "action":
        train_paths = action_paths
        val_paths = action_val_paths
    elif mode == "recall":
        train_paths = recall_paths
        val_paths = recall_val_paths
    else:
        train_paths = [*action_paths, *recall_paths]
        val_paths = [*action_val_paths, *recall_val_paths]
    if not train_paths:
        raise Mem2WDataError(f"mode={mode!r} requires at least one matching dataset path")
    if max_length <= 0:
        raise Mem2WDataError("max_length must be positive")
    cfg = SwiftSftConfig(
        model=model,
        model_revision=model_revision,
        dataset=train_paths,
        val_dataset=val_paths,
        output_dir=str(Path(output_dir).expanduser().resolve()),
        max_length=max_length,
        seed=seed,
        external_plugins=[str(Path(path).expanduser().resolve()) for path in external_plugins],
    )
    result = asdict(cfg)
    # ms-swift's argument parser treats an omitted model_revision differently
    # from a null JSON value; omit it unless the caller explicitly locked one.
    if model_revision is None:
        result.pop("model_revision", None)
    if not result["val_dataset"]:
        result.pop("val_dataset")
    if not result["external_plugins"]:
        result.pop("external_plugins")
    return result


def write_swift_config(path: os.PathLike[str] | str, config: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() not in {".json", ".jsonl"}:
        raise Mem2WDataError("ms-swift config output must use .json (swift sft accepts JSON configs)")
    destination.write_text(json.dumps(dict(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mem2W ↔ ms-swift dataset/config adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="convert episode JSONL into action/recall ms-swift JSONL")
    convert.add_argument("--input", required=True, type=Path, help="Mem2W episode JSONL")
    convert.add_argument("--output-dir", required=True, type=Path)

    config = subparsers.add_parser("config", help="write a swift sft JSON configuration")
    config.add_argument("--action-data", action="append", default=[], type=Path)
    config.add_argument("--recall-data", action="append", default=[], type=Path)
    config.add_argument("--action-val-data", action="append", default=[], type=Path)
    config.add_argument("--recall-val-data", action="append", default=[], type=Path)
    config.add_argument("--output", required=True, type=Path)
    config.add_argument("--output-dir", default="artifacts/ms_swift_sft")
    config.add_argument(
        "--mode",
        choices=TRAIN_MODES,
        default="both",
        help="pilot branch to train; action/recall are independent runs, both mixes files",
    )
    config.add_argument("--model", default=DEFAULT_MODEL)
    config.add_argument("--model-revision")
    config.add_argument("--external-plugin", action="append", default=[], type=Path)
    config.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    config.add_argument("--seed", type=int, default=42)

    prepare = subparsers.add_parser(
        "prepare", help="convert episodes and write a ready-to-run swift sft config"
    )
    prepare.add_argument("--input", required=True, type=Path, help="Mem2W episode JSONL")
    prepare.add_argument("--data-dir", required=True, type=Path)
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--output-dir", default="artifacts/ms_swift_sft")
    prepare.add_argument(
        "--mode",
        choices=TRAIN_MODES,
        default="both",
        help="pilot branch to train; action/recall are independent runs, both mixes files",
    )
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--model-revision")
    prepare.add_argument("--external-plugin", action="append", default=[], type=Path)
    prepare.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    prepare.add_argument("--seed", type=int, default=42)

    command = subparsers.add_parser("command", help="print the command for an existing ms-swift config")
    command.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "convert":
            manifest = convert_episodes(args.input, args.output_dir)
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0
        if args.command == "prepare":
            manifest = convert_episodes(args.input, args.data_dir)
            data_dir = Path(args.data_dir).expanduser().resolve()
            train_action = data_dir / "action_train.jsonl"
            train_recall = data_dir / "recall_train.jsonl"
            if args.mode == "action" and not train_action.is_file():
                raise Mem2WDataError("prepare --mode action requires action_train.jsonl")
            if args.mode == "recall" and not train_recall.is_file():
                raise Mem2WDataError("prepare --mode recall requires recall_train.jsonl")
            if args.mode == "both" and (not train_action.is_file() or not train_recall.is_file()):
                raise Mem2WDataError(
                    "prepare --mode both requires action_train.jsonl and recall_train.jsonl; "
                    "use --mode action or --mode recall for a single branch"
                )
            action_val = data_dir / "action_validation.jsonl"
            recall_val = data_dir / "recall_validation.jsonl"
            config = build_swift_config(
                action_data=[train_action],
                recall_data=[train_recall],
                action_val_data=[action_val] if action_val.is_file() else [],
                recall_val_data=[recall_val] if recall_val.is_file() else [],
                output_dir=args.output_dir,
                model=args.model,
                model_revision=args.model_revision,
                external_plugins=args.external_plugin,
                max_length=args.max_length,
                seed=args.seed,
                mode=args.mode,
            )
            destination = write_swift_config(args.config, config)
            print(json.dumps({"manifest": manifest, "config": str(destination)}, ensure_ascii=False, indent=2))
            print("Run with: swift sft " + str(destination))
            return 0
        if args.command == "command":
            config_path = Path(args.config).expanduser().resolve()
            if not config_path.is_file():
                raise Mem2WDataError(f"config does not exist: {config_path}")
            print("swift sft " + str(config_path))
            return 0
        config = build_swift_config(
            action_data=args.action_data,
            recall_data=args.recall_data,
            action_val_data=args.action_val_data,
            recall_val_data=args.recall_val_data,
            output_dir=args.output_dir,
            model=args.model,
            model_revision=args.model_revision,
            external_plugins=args.external_plugin,
            max_length=args.max_length,
            seed=args.seed,
            mode=args.mode,
        )
        destination = write_swift_config(args.output, config)
        print(destination)
        print("Run with: swift sft " + str(destination))
        return 0
    except Mem2WDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
