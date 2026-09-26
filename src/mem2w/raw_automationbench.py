"""Import raw AutomationBench MemRL rollouts into the Mem2W episode contract.

The runner's ``trajectories.jsonl`` is an evaluation log, not an SFT dataset:
``trajectory`` is JSON-lines text and retrieval provenance is stored as IDs and
scores.  This importer performs only lossless normalization.  In particular,
it never reconstructs a recall target from ``retrieval_records``.  A recall
target is available only when a joined actor-prompt contains the exact
``[Reference Memories]``/``[MEMRL MEMORY]`` payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MEMORY_MARKERS = ("[Reference Memories]", "[MEMRL MEMORY")
_EPOCH_RE = re.compile(r"epoch[_-]?(\d+)", re.IGNORECASE)
DEFAULT_PAYLOAD_BUDGET_CHARS = 12000


class RawImportError(ValueError):
    """Raised for malformed raw runner input."""


@dataclass(frozen=True)
class ImportOptions:
    trajectories: Path
    output_dir: Path
    actor_prompts: Path | None = None
    split_manifest: Path | None = None
    snapshot_root: Path | None = None
    # Lossless is the default.  Compaction is an explicit, auditable change
    # to the recall target for old exports whose actor prompt is too large for
    # the model context window.
    compact_payload: bool = False
    payload_budget_chars: int = DEFAULT_PAYLOAD_BUDGET_CHARS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RawImportError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise RawImportError(f"{path}:{line_no}: expected an object")
            yield line_no, value


def _read_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return [row for _, row in _jsonl(path)]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RawImportError(f"{path}: invalid JSON: {exc}") from exc


def _trajectory_files(path: Path) -> list[Path]:
    """Expand one trajectory file or an ``epochs`` directory deterministically."""

    path = path.expanduser()
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("epoch_*/trajectories.jsonl"))
        if not files and path.name.startswith("epoch_"):
            candidate = path / "trajectories.jsonl"
            if candidate.is_file():
                files = [candidate]
        if not files:
            raise RawImportError(f"{path}: no epoch_*/trajectories.jsonl files found")
        return files
    raise RawImportError(f"trajectory input does not exist: {path}")


def _task_id(row: Mapping[str, Any]) -> str:
    for key in ("task_id", "task", "example_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    nested = row.get("metadata")
    if isinstance(nested, Mapping):
        for key in ("task_id", "task", "example_id", "id"):
            value = nested.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return ""


def _epoch(row: Mapping[str, Any], path: Path | None = None) -> int:
    value = row.get("epoch")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise RawImportError(f"epoch must be an integer, got {value!r}") from exc
    if path is not None:
        # For a run directory, the epoch is in the parent directory name
        # (``epoch_4/trajectories.jsonl``), not in the file name itself.
        for candidate in (path.name, path.parent.name, path.parent.parent.name):
            match = _EPOCH_RE.search(candidate)
            if match:
                return int(match.group(1))
    return 0


def _normalise_split(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RawImportError("split manifest contains an empty split")
    value = value.strip().lower().replace("-", "_")
    return {"valid": "validation", "dev": "validation", "eval": "validation"}.get(value, value)


def load_split_manifest(path: Path | None) -> dict[str, str]:
    """Load task-id -> split from JSON, JSONL, or a nested ``splits`` map."""

    if path is None:
        return {}
    value = _read_json_or_jsonl(path)
    if isinstance(value, Mapping) and isinstance(value.get("splits"), Mapping):
        value = value["splits"]
    result: dict[str, str] = {}
    if isinstance(value, Mapping):
        # AutomationBench's frozen manifest uses explicit task-id lists rather
        # than a task -> split map.  Handle it before the generic mapping path
        # (which would otherwise mistake metadata such as ``seed`` for a task).
        train_ids = value.get("selected_train_task_ids")
        eval_ids = value.get("selected_eval_task_ids")
        if train_ids is not None or eval_ids is not None:
            if train_ids is not None and not isinstance(train_ids, list):
                raise RawImportError("selected_train_task_ids must be a list")
            if eval_ids is not None and not isinstance(eval_ids, list):
                raise RawImportError("selected_eval_task_ids must be a list")
            for task in train_ids or []:
                result[str(task)] = "train"
            for task in eval_ids or []:
                task = str(task)
                if task in result:
                    raise RawImportError(f"task appears in both train and evaluation split: {task}")
                result[task] = "validation"
            return result
        # Compact export form: {"train": [task_id, ...], "validation": [...]}
        compact_names = {"train", "validation", "test", "eval", "dev"}
        if any(name in value for name in compact_names):
            for split_name, tasks in value.items():
                if split_name not in compact_names:
                    continue
                if not isinstance(tasks, list):
                    raise RawImportError(f"split {split_name!r} must contain a task-id list")
                normalized = _normalise_split(split_name)
                for task in tasks:
                    task = str(task)
                    if task in result and result[task] != normalized:
                        raise RawImportError(f"task appears in multiple splits: {task}")
                    result[task] = normalized
            return result
        for key, split in value.items():
            if isinstance(split, Mapping):
                split = split.get("split")
            result[str(key)] = _normalise_split(split)
        return result
    if isinstance(value, list):
        for row in value:
            if not isinstance(row, Mapping):
                continue
            task = _task_id(row)
            if not task:
                continue
            split = row.get("split")
            if split is None:
                split = row.get("dataset_split")
            result[task] = _normalise_split(split)
        return result
    raise RawImportError(f"{path}: expected a split map or list of task records")


def _strip_reasoning(value: Any) -> Any:
    """Remove provider reasoning fields recursively, preserving tool structure."""

    if isinstance(value, Mapping):
        return {
            str(key): _strip_reasoning(item)
            for key, item in value.items()
            if key not in {"reasoning_content", "reasoning"}
        }
    if isinstance(value, list):
        return [_strip_reasoning(item) for item in value]
    return value


def parse_trajectory(value: Any) -> list[dict[str, Any]]:
    """Parse runner JSON-lines trajectory and retain structured tool fields."""

    if isinstance(value, list):
        rows = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            rows = []
            for line_no, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RawImportError(f"trajectory line {line_no}: invalid JSON: {exc}") from exc
        else:
            rows = parsed if isinstance(parsed, list) else [parsed]
    else:
        raise RawImportError("trajectory must be a JSON-lines string or list")
    messages: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RawImportError(f"trajectory[{index}] must be an object")
        row = dict(_strip_reasoning(row))
        role = row.get("role")
        if not isinstance(role, str) or not role.strip():
            raise RawImportError(f"trajectory[{index}] is missing role")
        # ``serialize_trajectory`` emits these keys even when null.  Keep
        # tool_calls and tool_call_id exactly; only reasoning is removed.
        messages.append(row)
    return messages


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, Mapping) and item.get("text") is not None:
                chunks.append(str(item["text"]))
        return "".join(chunks)
    return str(value)


def _first_user(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        if str(message.get("role", "")).lower() == "user":
            text = _message_text(message.get("content"))
            if text.strip():
                return text
    return ""


def _contains_memory_marker(text: str) -> bool:
    return any(marker in text for marker in MEMORY_MARKERS)


def _prompt_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Find a joined actor prompt message list without interpreting free text."""

    for key in ("messages", "prompt", "actor_prompt", "input"):
        value = row.get(key)
        if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
            return [dict(_strip_reasoning(item)) for item in value]
        if isinstance(value, Mapping) and isinstance(value.get("messages"), list):
            return [dict(_strip_reasoning(item)) for item in value["messages"] if isinstance(item, Mapping)]
    return []


def load_actor_prompts(root: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if root is None:
        return {}
    files = sorted(root.glob("epoch_*.jsonl")) if root.is_dir() else [root]
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for path in files:
        for _, row in _jsonl(path):
            task = _task_id(row)
            if not task:
                continue
            epoch = _epoch(row, path)
            index[(task, epoch)] = row
    return index


def _memory_payload(prompt_row: Mapping[str, Any] | None) -> tuple[str | None, list[dict[str, Any]]]:
    if prompt_row is None:
        return None, []
    messages = _prompt_messages(prompt_row)
    candidates = []
    for message in messages:
        text = _message_text(message.get("content"))
        if _contains_memory_marker(text):
            candidates.append(text)
    # A prompt may be a raw string rather than chat messages.  It is still
    # exact if one of the explicit markers is present.
    if not candidates:
        for key in ("actor_prompt", "prompt", "input"):
            value = prompt_row.get(key)
            if isinstance(value, str) and _contains_memory_marker(value):
                candidates.append(value)
    return (candidates[0] if candidates else None), messages


def _compact_payload(payload: str, budget: int) -> tuple[str, bool]:
    """Return a deterministic, visibly marked head/tail compacted payload."""

    if budget <= 0:
        raise RawImportError("payload_budget_chars must be positive")
    if len(payload) <= budget:
        return payload, False
    marker = "\n\n[Mem2W compacted recall payload; exact middle omitted]\n\n"
    if budget <= len(marker) + 2:
        raise RawImportError("payload_budget_chars is too small for the compaction marker")
    head = (budget - len(marker)) // 2
    tail = budget - len(marker) - head
    return payload[:head] + marker + payload[-tail:], True


def _annotate_memory_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        copied = dict(message)
        text = _message_text(copied.get("content"))
        if _contains_memory_marker(text):
            metadata = dict(copied.get("metadata") or {})
            metadata.update({"source": "teacher_memory", "is_teacher_memory": True})
            copied["metadata"] = metadata
        result.append(copied)
    return result


def _snapshot_info(root: Path | None, epoch: int) -> dict[str, Any]:
    info: dict[str, Any] = {"memory_snapshot_id": f"snapshot/{epoch}"}
    if root is None:
        return info
    candidates = [root / f"snapshot/{epoch}", root / f"epoch_{epoch}", root / str(epoch)]
    for candidate in candidates:
        if candidate.exists():
            info["snapshot_path"] = str(candidate)
            if candidate.is_file():
                info["snapshot_sha256"] = _sha256_file(candidate)
            break
    return info


def _canonical_episode(
    row: Mapping[str, Any],
    *,
    line_no: int,
    source_path: Path | None,
    split_map: Mapping[str, str],
    actor_index: Mapping[tuple[str, int], Mapping[str, Any]],
    snapshot_root: Path | None,
    compact_payload: bool,
    payload_budget_chars: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = _task_id(row)
    if not task_id:
        raise RawImportError("task/task_id is missing")
    epoch = _epoch(row, source_path)
    # When a split manifest is supplied it is authoritative.  A row-level
    # split must not silently override a fixed task assignment.
    split = split_map.get(task_id) if split_map else None
    if not split_map and row.get("split") is not None:
        split = _normalise_split(row.get("split"))
    if split is None:
        raise RawImportError(f"no split_manifest entry for task_id={task_id}")
    messages = parse_trajectory(row.get("trajectory"))
    if not messages:
        raise RawImportError("trajectory is empty")
    if not any(str(message.get("role", "")).lower() in {"assistant", "tool_call"} for message in messages):
        raise RawImportError("trajectory has no assistant/tool action")
    prompt_row = actor_index.get((task_id, epoch))
    payload, prompt_messages = _memory_payload(prompt_row)
    # The trajectory itself usually contains the prompt; actor prompt files
    # are joined only to recover exact memory text and annotate its message.
    messages = _annotate_memory_messages(messages)
    query = _first_user(messages) or _message_text(row.get("task_description")) or task_id
    retrieved_ids = [str(value) for value in (row.get("retrieved_memory_ids") or [])]
    retrieval_records = row.get("retrieval_records") or []
    has_retrieval = bool(retrieved_ids or retrieval_records or payload is not None)
    missing_payload = has_retrieval and payload is None
    episode_id = f"{task_id}:epoch:{epoch}:{split}"
    event = None
    payload_stats: dict[str, Any] = {}
    if has_retrieval:
        target_payload = payload
        compacted = False
        if target_payload is not None:
            payload_stats = {
                "payload_original_chars": len(target_payload),
                "payload_original_sha256": hashlib.sha256(target_payload.encode("utf-8")).hexdigest(),
            }
            if compact_payload:
                target_payload, compacted = _compact_payload(target_payload, payload_budget_chars)
            payload_stats.update(
                {
                    "payload_chars": len(target_payload),
                    "payload_sha256": hashlib.sha256(target_payload.encode("utf-8")).hexdigest(),
                    "payload_compacted": compacted,
                    "payload_format": "mem2w_actor_prompt_compact_v1" if compacted else "mem2w_actor_prompt_v1",
                }
            )
        event = {
            "event_id": f"{episode_id}:retrieval",
            "before_action_index": 0,
            "query_text": query,
            "k_requested": len(retrieved_ids),
            "k_returned": len(retrieved_ids),
            "selected_memory_ids": retrieved_ids,
            # Do not replace this with retrieval_records.  Missing text is
            # explicitly marked and is never emitted as a recall target.
            "injected_context_text": target_payload or "",
            "recall_missing_payload": missing_payload,
            "retrieval_records": retrieval_records,
            **payload_stats,
        }
    outcome = {
        "reward": row.get("partial_credit", 0.0),
        "success": bool(row.get("task_completed_correctly", False)),
        "termination_reason": "aborted" if row.get("aborted") else "completed",
    }
    episode: dict[str, Any] = {
        "episode_id": episode_id,
        "task_id": task_id,
        "epoch": epoch,
        "split": split,
        "task_family": row.get("domain"),
        "memory_snapshot_id": f"snapshot/{epoch}",
        "task": {"query": query},
        "_memq_task_description": query,
        "_memq_retrieved_ids": retrieved_ids,
        "messages": messages,
        "retrieval_events": [event] if event is not None else [],
        "retrieval_provenance": {
            "retrieved_memory_ids": retrieved_ids,
            "retrieval_records": retrieval_records,
            "has_retrieval": has_retrieval,
            "recall_missing_payload": missing_payload,
        },
        "outcome": outcome,
        "partial_credit": row.get("partial_credit"),
        # The trajectory log exposes partial_credit; a separate
        # partial_credit_diagnostic is only trusted when the exporter wrote it.
        # Do not relabel the reward as a diagnostic field.
        "partial_credit_diagnostic": row.get("partial_credit_diagnostic"),
        "diagnostics": {"usage": row.get("usage"), "perf": row.get("perf"), "steps": row.get("steps")},
        "source_algorithm": "memrl",
        "source": {
            "trajectory_path": str(source_path) if source_path else None,
            "trajectory_line": line_no,
            "task_contract_sha256": row.get("task_contract_sha256"),
        },
    }
    episode.update(_snapshot_info(snapshot_root, epoch))
    qa = {
        "task_id": task_id,
        "epoch": epoch,
        "split": split,
        "recall_missing_payload": missing_payload,
        "joined_actor_prompt": prompt_row is not None,
        "retrieval_present": has_retrieval,
        "retrieval_record_count": len(retrieval_records),
        **payload_stats,
    }
    return episode, qa


def import_automationbench(options: ImportOptions) -> dict[str, Any]:
    split_map = load_split_manifest(options.split_manifest)
    actor_index = load_actor_prompts(options.actor_prompts)
    trajectory_files = _trajectory_files(options.trajectories)
    output = options.output_dir
    output.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for trajectory_file in trajectory_files:
        for line_no, row in _jsonl(trajectory_file):
            task = _task_id(row)
            try:
                episode, qa = _canonical_episode(
                    row,
                    line_no=line_no,
                    source_path=trajectory_file,
                    split_map=split_map,
                    actor_index=actor_index,
                    snapshot_root=options.snapshot_root,
                    compact_payload=options.compact_payload,
                    payload_budget_chars=options.payload_budget_chars,
                )
            except RawImportError as exc:
                reason = str(exc)
                reason_counts[reason] += 1
                rejected.append(
                    {
                        "path": str(trajectory_file),
                        "line": line_no,
                        "task_id": task,
                        "epoch": row.get("epoch"),
                        "reason": reason,
                    }
                )
                continue
            episodes.append(episode)
            qa_rows.append(qa)

    episodes_path = output / "episodes.jsonl"
    rejected_path = output / "rejected.jsonl"
    qa_path = output / "qa.json"
    with episodes_path.open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode, ensure_ascii=False, sort_keys=True) + "\n")
    with rejected_path.open("w", encoding="utf-8") as stream:
        for row in rejected:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    split_counts = Counter(str(row["split"]) for row in qa_rows)
    qa = {
        "accepted": len(episodes),
        "rejected": len(rejected),
        "action_ready": len(episodes),
        "recall_ready": sum(
            bool(row["retrieval_present"] and not row["recall_missing_payload"])
            for row in qa_rows
        ),
        "recall_missing_payload": sum(row["recall_missing_payload"] for row in qa_rows),
        "prompt_missing_payload": sum(row["recall_missing_payload"] for row in qa_rows),
        "retrieval_present": sum(row["retrieval_present"] for row in qa_rows),
        "overlong_payload_events": sum(
            1 for row in qa_rows if row.get("payload_original_chars", 0) > options.payload_budget_chars
        ),
        "payload_budget_chars": options.payload_budget_chars,
        "compact_payload": options.compact_payload,
        "joined_actor_prompt": sum(row["joined_actor_prompt"] for row in qa_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "rejection_reasons": dict(reason_counts),
        "rows": qa_rows,
    }
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "format": "mem2w-automationbench-import-v1",
        "source": {
            "path": str(options.trajectories.resolve()),
            "files": [
                {"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in trajectory_files
            ],
        },
        "actor_prompts": str(options.actor_prompts.resolve()) if options.actor_prompts else None,
        "split_manifest": str(options.split_manifest.resolve()) if options.split_manifest else None,
        "snapshot_root": str(options.snapshot_root.resolve()) if options.snapshot_root else None,
        "episodes": len(episodes),
        "rejected": len(rejected),
        "recall_missing_payload": qa["recall_missing_payload"],
        "prompt_missing_payload": qa["prompt_missing_payload"],
        "outputs": {"episodes": str(episodes_path), "rejected": str(rejected_path), "qa": str(qa_path)},
        "policy": {
            "retrieval_records_are_not_recall_payload": True,
            "reasoning_content_removed": True,
            "missing_recall_payload_keeps_action_episode": True,
            "compact_payload": options.compact_payload,
            "payload_budget_chars": options.payload_budget_chars,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import raw AutomationBench trajectories for Mem2W")
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--actor-prompts", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument(
        "--compact-payload",
        action="store_true",
        help="explicitly compact overlong exact actor prompts with a marked head/tail target",
    )
    parser.add_argument(
        "--payload-budget-chars",
        type=int,
        default=DEFAULT_PAYLOAD_BUDGET_CHARS,
        help=f"target character budget when --compact-payload is set (default: {DEFAULT_PAYLOAD_BUDGET_CHARS})",
    )
    args = parser.parse_args(argv)
    manifest = import_automationbench(
        ImportOptions(
            trajectories=args.trajectories,
            output_dir=args.output_dir,
            actor_prompts=args.actor_prompts,
            split_manifest=args.split_manifest,
            snapshot_root=args.snapshot_root,
            compact_payload=args.compact_payload,
            payload_budget_chars=args.payload_budget_chars,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
