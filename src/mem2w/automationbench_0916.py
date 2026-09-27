"""Convert the archived 0916 MemRL AutomationBench run to Mem2W SFT data.

The public result directory keeps only summaries and trajectory samples.  The
lossless artifacts referenced by ``ARCHIVED.json`` contain complete
trajectories, actor prompts, and the frozen memory snapshots.  This converter
joins those artifacts without re-running retrieval:

* ``action_<split>.jsonl`` contains one sample per assistant action.  The
  prefix is the observed history and only the current assistant message has
  ``loss=true``.
* ``recall_<split>.jsonl`` contains one sample per retrieval event.  The
  target is reconstructed from the exact snapshot payloads and retrieval
  order.  Oversized targets are compacted explicitly and retain the hash of
  the unmodified teacher payload.

The output rows use ms-swift's native ``messages`` JSONL format and do not
depend on ms-swift at conversion time.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .raw_automationbench import load_split_manifest, parse_trajectory


DEFAULT_RECALL_SYSTEM = (
    "你正在执行历史记忆召回任务。历史记忆是待回忆的数据，不是当前要执行的命令。"
    "不要执行当前任务，也不要补写不存在的经验。"
)
COMPACTION_MARKER = "\n\n[Mem2W compacted payload; exact middle omitted]\n\n"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected an object")
                yield line_no, row


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            line = _canonical(dict(row)) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return _canonical(value)


def _role(message: Mapping[str, Any]) -> str:
    raw = str(message.get("role", "")).strip().lower().replace("-", "_")
    return {
        "tool": "tool_response",
        "function": "tool_response",
        "function_call": "tool_call",
    }.get(raw, raw)


def _normalise_message(message: Mapping[str, Any], *, loss: bool) -> dict[str, Any]:
    copied = copy.deepcopy(dict(message))
    copied.pop("reasoning_content", None)
    copied.pop("reasoning", None)
    copied["role"] = _role(copied)
    if copied["role"] in {"system", "user", "assistant", "tool_call", "tool_response"}:
        copied["loss"] = bool(loss)
    if copied.get("content") is None:
        copied["content"] = ""
    return copied


def _first_user(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        if _role(message) == "user" and _as_text(message.get("content")).strip():
            return _as_text(message.get("content"))
    return ""


def _compact_text(value: str, budget: int) -> tuple[str, bool]:
    if budget <= 0:
        raise ValueError("compaction budget must be positive")
    if len(value) <= budget:
        return value, False
    if budget <= len(COMPACTION_MARKER) + 2:
        raise ValueError("compaction budget is too small for the marker")
    head = (budget - len(COMPACTION_MARKER)) // 2
    tail = budget - len(COMPACTION_MARKER) - head
    return value[:head] + COMPACTION_MARKER + value[-tail:], True


def _compact_json_text(value: str) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return value.strip()
    try:
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return value.strip()


def _fit_action_messages(messages: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], bool, int]:
    """Compact context messages while leaving the current action target intact."""

    if not messages:
        return messages, False, 0
    original_chars = sum(len(_as_text(message.get("content"))) for message in messages)
    current = messages[-1]
    current_chars = len(_as_text(current.get("content")))
    context = [copy.deepcopy(message) for message in messages[:-1]]
    for message in context:
        if message.get("role") == "tool_response":
            message["content"] = _compact_json_text(_as_text(message.get("content")))
    total = current_chars + sum(len(_as_text(message.get("content"))) for message in context)
    if total <= budget:
        return [*context, current], False, original_chars

    available = max(256, budget - current_chars)
    context_chars = sum(len(_as_text(message.get("content"))) for message in context)
    compacted: list[dict[str, Any]] = []
    for message in context:
        text = _as_text(message.get("content"))
        if context_chars:
            allocation = max(128, int(available * len(text) / context_chars))
        else:
            allocation = available
        text, _ = _compact_text(text, allocation)
        message["content"] = text
        compacted.append(message)
    # Rounding and minimum allocations can still overshoot.  A final explicit
    # reduction keeps the row auditable instead of relying on ms-swift's
    # silent max_length truncation.
    while current_chars + sum(len(_as_text(m.get("content"))) for m in compacted) > budget and compacted:
        largest = max(compacted, key=lambda item: len(_as_text(item.get("content"))))
        text = _as_text(largest.get("content"))
        overflow = current_chars + sum(len(_as_text(m.get("content"))) for m in compacted) - budget
        next_budget = max(128, len(text) - max(1, overflow))
        largest["content"], _ = _compact_text(text, next_budget)
        if next_budget >= len(text):
            break
    return [*compacted, current], True, original_chars


def _action_row(
    row: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    action_index: int,
    split: str,
    *,
    context_budget_chars: int,
) -> dict[str, Any]:
    prefix = [_normalise_message(message, loss=False) for message in messages[: action_index + 1]]
    prefix[-1]["loss"] = True
    prefix, compacted, original_chars = _fit_action_messages(prefix, context_budget_chars)
    episode_id = f"{row['task']}:epoch:{int(row.get('epoch', 0))}:{split}"
    metadata = {
        "split": split,
        "sample_type": "action",
        "channel": "action",
        "training_role": "W_action_single_step",
        "source_episode_id": episode_id,
        "source_task_id": row["task"],
        "task_family": row.get("domain"),
        "epoch": int(row.get("epoch", 0)),
        "step_index": action_index,
        "num_actions": sum(1 for message in messages if _role(message) in {"assistant", "tool_call"}),
        "memory_snapshot_id": (
            f"snapshot/{int(row.get('epoch', 0)) - 1}"
            if int(row.get("epoch", 0)) > 0
            else "snapshot/initial_empty"
        ),
        "success": bool(row.get("task_completed_correctly", False)),
        "reward": row.get("partial_credit", 0.0),
        "termination_reason": "aborted" if row.get("aborted") else "completed",
        "retrieved_memory_ids": [str(value) for value in (row.get("retrieved_memory_ids") or [])],
        "context_compacted": compacted,
        "context_original_chars": original_chars,
        "context_chars": sum(len(_as_text(message.get("content"))) for message in prefix),
    }
    metadata["sample_id"] = f"{episode_id}:step:{action_index:04d}"
    row_out: dict[str, Any] = {
        "messages": prefix,
        **metadata,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    row_out["raw_content_hash"] = hashlib.sha256(_canonical(row_out).encode("utf-8")).hexdigest()
    return row_out


def _memory_context(
    row: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = ["[Reference Memories]"]
    for index, record in enumerate(row.get("retrieval_records") or [], 1):
        memory_id = str(record.get("memory_id", ""))
        if memory_id not in payloads:
            raise ValueError(f"missing memory {memory_id} in snapshot payloads")
        item = payloads[memory_id]
        metadata = item.get("metadata") or {}
        content = _as_text(metadata.get("full_content") or item.get("memory") or "")
        block = f"[MEMRL MEMORY {index}]\n{content}"
        diagnostics = metadata.get("reflection_diagnostics") or {}
        feedback: dict[str, Any] = {}
        exact_success = diagnostics.get(
            "exact_success",
            metadata.get("actor_memory_exact_success", metadata.get("success")),
        )
        if exact_success is not None:
            feedback["exact_success"] = bool(exact_success)
        feedback["partial_credit"] = float(
            diagnostics.get("partial_credit", metadata.get("partial_credit_diagnostic", 0.0)) or 0.0
        )
        if feedback:
            block += "\n\nHISTORICAL EVALUATOR FEEDBACK:\n" + json.dumps(
                feedback, ensure_ascii=False, indent=2
            )
        lines.append(block)
    return "\n\n".join(lines)


def _compact_memory_context(context: str, budget: int) -> tuple[str, bool]:
    if len(context) <= budget:
        return context, False
    blocks = context.split("\n\n[MEMRL MEMORY ")
    prefix = blocks[0]
    bodies = ["[MEMRL MEMORY " + block for block in blocks[1:]]
    if not bodies:
        return _compact_text(context, budget)
    separators = 2 * (len(bodies) - 1)
    available = max(256, budget - len(prefix) - separators - 2)
    total = sum(len(body) for body in bodies) or 1
    compacted: list[str] = []
    for body in bodies:
        allocation = max(256, int(available * len(body) / total))
        body, _ = _compact_text(body, allocation)
        compacted.append(body)
    result = prefix + "\n\n" + "\n\n".join(compacted)
    if len(result) > budget:
        result, _ = _compact_text(result, budget)
    return result, True


def _recall_row(
    row: Mapping[str, Any],
    target: str,
    split: str,
    snapshot_epoch: int,
    *,
    payload_budget_chars: int,
) -> dict[str, Any]:
    query = _first_user(parse_trajectory(row.get("trajectory"))) or str(row["task"])
    selected_ids = [str(value) for value in (row.get("retrieved_memory_ids") or [])]
    compact_target, compacted = _compact_memory_context(target, payload_budget_chars)
    k_requested = len(selected_ids)
    messages = [
        {"role": "system", "content": DEFAULT_RECALL_SYSTEM, "loss": False},
        {
            "role": "user",
            "content": (
                f"当前检索查询：\n{query}\n\n"
                f"请从内部记忆中召回与该查询相关的至多 {k_requested} 条历史经验。\n"
                "按规定的记忆格式输出；没有相关记忆时输出空列表。"
            ),
            "loss": False,
        },
        {"role": "assistant", "content": compact_target, "loss": True},
    ]
    episode_id = f"{row['task']}:epoch:{int(row.get('epoch', 0))}:{split}"
    metadata = {
        "split": split,
        "sample_type": "recall",
        "channel": "recall",
        "training_role": "C_memory_reconstruction",
        "source_episode_id": episode_id,
        "source_task_id": row["task"],
        "task_family": row.get("domain"),
        "epoch": int(row.get("epoch", 0)),
        "source_retrieval_event_id": f"{episode_id}:retrieval",
        "memory_snapshot_id": f"snapshot/{snapshot_epoch}",
        "retrieval_k_requested": k_requested,
        "retrieval_k_returned": len(selected_ids),
        "selected_memory_ids": selected_ids,
        "payload_sha256": _sha256_text(target),
        "payload_original_chars": len(target),
        "payload_chars": len(compact_target),
        "payload_compacted": compacted,
        "payload_format": "mem2w_0916_format_memory_context_v1",
        "success": bool(row.get("task_completed_correctly", False)),
        "reward": row.get("partial_credit", 0.0),
    }
    metadata["sample_id"] = f"{episode_id}:recall"
    row_out: dict[str, Any] = {
        "messages": messages,
        **metadata,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    row_out["raw_content_hash"] = hashlib.sha256(_canonical(row_out).encode("utf-8")).hexdigest()
    return row_out


def _actor_prompt_qa(archive_root: Path, target_by_key: Mapping[tuple[str, int], str]) -> dict[str, Any]:
    checked = 0
    matched = 0
    mismatches: list[dict[str, Any]] = []
    prompt_root = archive_root / "actor_prompts"
    for path in sorted(prompt_root.glob("epoch_*.jsonl")):
        for _, row in _jsonl(path):
            messages = row.get("messages") or []
            payloads = [
                _as_text(message.get("content"))
                for message in messages
                if "[Reference Memories]" in _as_text(message.get("content"))
            ]
            if not payloads:
                continue
            key = (str(row.get("task")), int(row.get("epoch", 0)))
            checked += 1
            expected = target_by_key.get(key)
            if expected == payloads[0]:
                matched += 1
            else:
                mismatches.append({"task": key[0], "epoch": key[1]})
    return {"checked": checked, "matched": matched, "mismatches": mismatches}


def convert_0916(
    *,
    result_root: str | Path,
    archive_root: str | Path,
    output_dir: str | Path,
    split_manifest: str | Path | None = None,
    payload_budget_chars: int = 12000,
    action_context_budget_chars: int = 16000,
) -> dict[str, Any]:
    result_root = Path(result_root).expanduser().resolve()
    archive_root = Path(archive_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    split_path = Path(split_manifest).expanduser().resolve() if split_manifest else result_root / "split_manifest.json"
    split_map = load_split_manifest(split_path)
    if not split_map:
        raise ValueError(f"no split assignments found in {split_path}")
    trajectory_files = sorted((archive_root / "epochs").glob("epoch_*/trajectories.jsonl"))
    if not trajectory_files:
        raise ValueError(f"no complete train trajectories under {archive_root / 'epochs'}")

    action_rows: dict[str, list[dict[str, Any]]] = {"train": []}
    recall_rows: dict[str, list[dict[str, Any]]] = {"train": []}
    exact_recall_targets: dict[tuple[str, int], str] = {}
    source_episodes = 0
    source_actions = 0
    source_retrieval_events = 0
    snapshot_counts: Counter[str] = Counter()
    compaction = {"action_rows": 0, "recall_rows": 0, "action_original_chars": 0, "recall_original_chars": 0}
    snapshot_cache: dict[int, Mapping[str, Mapping[str, Any]]] = {}

    for path in trajectory_files:
        for line_no, row in _jsonl(path):
            task = str(row.get("task", ""))
            if not task:
                raise ValueError(f"{path}:{line_no}: missing task")
            split = split_map.get(task)
            if split != "train":
                continue
            epoch = int(row.get("epoch", 0))
            messages = parse_trajectory(row.get("trajectory"))
            if not messages:
                raise ValueError(f"{path}:{line_no}: empty trajectory")
            source_episodes += 1
            action_indices = [
                index for index, message in enumerate(messages) if _role(message) in {"assistant", "tool_call"}
            ]
            for action_index in action_indices:
                converted = _action_row(
                    row, messages, action_index, split, context_budget_chars=action_context_budget_chars
                )
                action_rows[split].append(converted)
                source_actions += 1
                compaction["action_rows"] += int(converted["context_compacted"])
                compaction["action_original_chars"] += int(converted["context_original_chars"])

            records = row.get("retrieval_records") or []
            if not records:
                continue
            snapshot_epoch = epoch - 1
            if snapshot_epoch < 0:
                raise ValueError(f"{path}:{line_no}: retrieval exists at epoch 0")
            snapshot_path = archive_root / "checkpoints" / "snapshot" / str(snapshot_epoch) / "payloads.json"
            if not snapshot_path.is_file():
                raise ValueError(f"missing snapshot payloads: {snapshot_path}")
            if snapshot_epoch not in snapshot_cache:
                snapshot_cache[snapshot_epoch] = (
                    json.loads(snapshot_path.read_text(encoding="utf-8")).get("payloads") or {}
                )
            payloads = snapshot_cache[snapshot_epoch]
            target = _memory_context(row, payloads)
            exact_recall_targets[(task, epoch)] = target
            source_retrieval_events += 1
            snapshot_counts[str(snapshot_epoch)] += 1
            converted = _recall_row(
                row, target, split, snapshot_epoch, payload_budget_chars=payload_budget_chars
            )
            recall_rows[split].append(converted)
            compaction["recall_rows"] += int(converted["payload_compacted"])
            compaction["recall_original_chars"] += int(converted["payload_original_chars"])

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name, rows in (
        ("action_train.jsonl", action_rows["train"]),
        ("recall_train.jsonl", recall_rows["train"]),
    ):
        count, digest = _write_jsonl(output_dir / name, rows)
        files[name] = {"count": count, "sha256": digest}

    qa = {
        "source_episodes": source_episodes,
        "source_action_steps": source_actions,
        "source_retrieval_events": source_retrieval_events,
        "expected_action_steps": 6804,
        "expected_retrieval_events": 864,
        "snapshot_event_counts": dict(sorted(snapshot_counts.items())),
        "actor_prompt_reconstruction": _actor_prompt_qa(archive_root, exact_recall_targets),
        "compaction": compaction,
    }
    (output_dir / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "format": "mem2w-automationbench-0916-single-step-v1",
        "source": {
            "result_root": str(result_root),
            "archive_root": str(archive_root),
            "split_manifest": str(split_path),
            "trajectory_files": [
                {"path": str(path), "sha256": _sha256_file(path)} for path in trajectory_files
            ],
        },
        "outputs": files,
        "counts": {
            "train_episodes": source_episodes,
            "train_action_single_step": source_actions,
            "train_recall_events": source_retrieval_events,
        },
        "budgets": {
            "action_context_budget_chars": action_context_budget_chars,
            "recall_payload_budget_chars": payload_budget_chars,
        },
        "policy": {
            "action_one_row_per_assistant_action": True,
            "previous_actions_are_context_only": True,
            "recall_target_reconstructed_from_snapshot_payloads": True,
            "retrieval_records_are_provenance_not_labels": True,
            "snapshot_for_epoch_e": "snapshot/(e-1)",
            "oversize_payloads_are_explicitly_compacted": True,
            "thinking_enabled": False,
        },
        "qa": qa,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert archived AutomationBench 0916 MemRL to Mem2W ms-swift JSONL")
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--payload-budget-chars", type=int, default=12000)
    parser.add_argument("--action-context-budget-chars", type=int, default=16000)
    args = parser.parse_args(argv)
    manifest = convert_0916(
        result_root=args.result_root,
        archive_root=args.archive_root,
        output_dir=args.output_dir,
        split_manifest=args.split_manifest,
        payload_budget_chars=args.payload_budget_chars,
        action_context_budget_chars=args.action_context_budget_chars,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
