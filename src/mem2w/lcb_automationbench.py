"""Convert LiveCodeBench execution records to the current episode contract.

LiveCodeBench is a single-turn benchmark, while the AutomationBench importer
expects one JSON object per episode with a JSON-lines ``trajectory`` and an
explicit retrieval event.  This module bridges the two formats while keeping
the actor-visible Composer guidance exact.  Q keys remain provenance fields;
the reconstructed guidance is the retrieval payload, and events with no
guidance are marked ``recall_missing_payload`` and remain usable for action
training only.

The converter writes both the AutomationBench-compatible raw trajectory stream
and the Mem2W canonical episode stream.  Private test cases are never copied
to either output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LCB_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)

LCB_MEMORY_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests.\n"
    "If reference memories are provided, they show solution approaches from "
    "past similar coding problems. Use them to learn patterns and avoid past "
    "mistakes, but always reason about the current problem independently."
)

FORMATTING_WITH_STARTER = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)

FORMATTING_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


class LCBConversionError(ValueError):
    """Raised when an LCB source cannot be converted without guessing."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LCBConversionError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise LCBConversionError(f"{path}:{line_no}: expected a JSON object")
            yield line_no, row


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else "" if value is None else str(value)


def _format_problem(problem: Mapping[str, Any]) -> str:
    content = _as_text(problem.get("question_content"))
    prompt = f"### Question:\n{content}\n\n"
    starter = _as_text(problem.get("starter_code")).strip()
    if starter:
        prompt += f"### Format: {FORMATTING_WITH_STARTER}\n"
        prompt += f"```python\n{starter}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    return prompt + "### Answer: (use the provided format with backticks)\n\n"


def _problem_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line_no, row in _jsonl(path):
        question_id = str(row.get("question_id", "")).strip()
        if not question_id:
            raise LCBConversionError(f"{path}:{line_no}: missing question_id")
        if question_id in result:
            raise LCBConversionError(f"{path}:{line_no}: duplicate question_id={question_id}")
        result[question_id] = row
    if not result:
        raise LCBConversionError(f"no problems found in {path}")
    return result


def _split_index(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise LCBConversionError(f"{path}: split manifest must be an object")
    result: dict[str, str] = {}
    for key, split in (("train", "train"), ("test", "validation")):
        values = raw.get(key, [])
        if not isinstance(values, list):
            raise LCBConversionError(f"{path}: {key} must be a list")
        for value in values:
            task_id = str(value)
            if task_id in result:
                raise LCBConversionError(f"{path}: task appears in multiple splits: {task_id}")
            result[task_id] = split
    if not result:
        raise LCBConversionError(f"{path}: split manifest has no train/test IDs")
    return result


def _epoch_from_path(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", str(path))
    if not match:
        raise LCBConversionError(f"cannot infer epoch from {path}")
    return int(match.group(1))


def _problem_messages(
    problem: Mapping[str, Any],
    predicted_code: str,
    guidance: Any = None,
    has_retrieval: bool = False,
) -> list[dict[str, Any]]:
    code = _as_text(predicted_code)
    if code and not code.lstrip().startswith("```"):
        code = f"```python\n{code.rstrip()}\n```"
    messages: list[dict[str, Any]] = [{
        "role": "system",
        "content": LCB_MEMORY_SYSTEM_PROMPT if has_retrieval or _as_text(guidance).strip() else LCB_SYSTEM_PROMPT,
    }]
    guidance_text = _as_text(guidance).strip()
    if guidance_text:
        messages.append({
            "role": "user",
            "content": (
                "Use the following guidance as transferable algorithmic advice. "
                "Do not copy code blindly; solve the current problem independently.\n\n"
                + guidance_text
            ),
            "metadata": {"source": "teacher_memory", "is_teacher_memory": True},
        })
    messages.extend([
        {"role": "user", "content": _format_problem(problem)},
        {"role": "assistant", "content": code},
    ])
    return messages


def _observation(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("observation_summary")
    if isinstance(value, Mapping):
        return dict(value)
    # Validation files use the compact evaluator schema directly rather than
    # nesting it under ``observation_summary``.
    return {
        key: row.get(key)
        for key in ("predicted_code", "num_passed", "num_total", "error")
        if key in row
    }


def _setting_q_keys(row: Mapping[str, Any]) -> list[str]:
    values = row.get("used_q_keys") or row.get("q_keys") or []
    if not isinstance(values, list):
        raise LCBConversionError("used_q_keys must be a list")
    return [str(value) for value in values if str(value).strip()]


def _reward(row: Mapping[str, Any], observation: Mapping[str, Any]) -> float:
    raw = row.get("reward")
    if raw is None:
        passed, total = observation.get("num_passed"), observation.get("num_total")
        if isinstance(passed, (int, float)) and isinstance(total, (int, float)) and total:
            return float(passed) / float(total)
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise LCBConversionError(f"invalid reward: {raw!r}") from exc


def _synthetic_task_id(task_id: str, setting_id: str, run_id: str, epoch: int) -> str:
    setting = setting_id.strip() or "evaluation"
    run = run_id.strip() or "run"
    return f"lcb.{task_id}::epoch:{epoch}::setting:{setting}::run:{run}"


def _retrieval_event(
    episode_id: str,
    task_description: str,
    q_keys: Sequence[str],
    injected_context_text: str,
    *,
    before_action_index: int = 2,
) -> dict[str, Any] | None:
    if not q_keys:
        return None
    records = [
        {"memory_id": key, "q_key": key, "payload_available": False}
        for key in q_keys
    ]
    return {
        "event_id": f"{episode_id}:retrieval",
        "before_action_index": before_action_index,
        "query_text": task_description,
        "k_requested": len(q_keys),
        "k_returned": len(q_keys),
        "selected_memory_ids": list(q_keys),
        "injected_context_text": injected_context_text,
        "recall_missing_payload": not bool(injected_context_text.strip()),
        "retrieval_records": records,
        "payload_format": "lcb_actor_guidance_v1" if injected_context_text.strip() else None,
    }


def _raw_row(
    row: Mapping[str, Any],
    problem: Mapping[str, Any],
    *,
    task_id: str,
    split: str,
    epoch: int,
    setting_id: str,
    run_id: str,
    include_retrieval: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observation = _observation(row)
    task_description = _as_text(problem.get("question_content"))
    q_keys = _setting_q_keys(row) if include_retrieval else []
    unique_task = _synthetic_task_id(task_id, setting_id, run_id, epoch)
    messages = _problem_messages(
        problem,
        observation.get("predicted_code", ""),
        row.get("guidance"),
        bool(q_keys),
    )
    episode_id = f"{unique_task}:epoch:{epoch}:{split}"
    guidance_text = _as_text(row.get("guidance")).strip()
    guidance_payload = (
        "Use the following guidance as transferable algorithmic advice. "
        "Do not copy code blindly; solve the current problem independently.\n\n"
        + guidance_text
        if guidance_text
        else ""
    )
    event = _retrieval_event(episode_id, task_description, q_keys, guidance_payload)
    exact_success = bool(row.get("exact_success", row.get("success", False)))
    reward = _reward(row, observation)
    diagnostics = {
        "num_passed": observation.get("num_passed"),
        "num_total": observation.get("num_total"),
        "error": observation.get("error"),
    }
    trajectory = "\n".join(_canonical(message) for message in messages)
    raw = {
        "task": unique_task,
        "task_id": task_id,
        "setting_id": setting_id or None,
        "run_id": run_id or None,
        "critic_id": row.get("critic_id"),
        "domain": "lcb",
        "epoch": epoch,
        "split": split,
        "task_description": task_description,
        "guidance": row.get("guidance"),
        "used_q_keys": q_keys,
        "retrieved_memory_ids": q_keys,
        "retrieval_records": event.get("retrieval_records", []) if event else [],
        "task_completed_correctly": exact_success,
        "partial_credit": reward,
        "partial_credit_diagnostic": diagnostics,
        "observation_summary": observation,
        "trajectory": trajectory,
        "source_protocol": "lcb_memrl_tg_execution_v1",
    }
    canonical = {
        "episode_id": episode_id,
        "task_id": task_id,
        "task_family": "lcb",
        "epoch": epoch,
        "split": split,
        "memory_snapshot_id": f"snapshot/{epoch - 1}" if epoch > 0 else "snapshot/initial_empty",
        "task": {"query": task_description},
        "_memq_task_description": task_description,
        "_memq_retrieved_ids": q_keys,
        "messages": messages,
        "retrieval_events": [event] if event else [],
        "retrieval_provenance": {
            "retrieved_memory_ids": q_keys,
            "retrieval_records": event.get("retrieval_records", []) if event else [],
            "has_retrieval": bool(event),
            "recall_missing_payload": bool(event and event.get("recall_missing_payload")),
        },
        "outcome": {
            "reward": reward,
            "success": exact_success,
            "termination_reason": "completed" if observation.get("error") is None else "error",
            "partial_credit": reward,
            "partial_credit_diagnostic": diagnostics,
        },
        "partial_credit": reward,
        "partial_credit_diagnostic": diagnostics,
        "diagnostics": diagnostics,
        "source_algorithm": "memrl_tg",
        "source": {
            "source_protocol": "lcb_memrl_tg_execution_v1",
            "task_id": task_id,
            "setting_id": setting_id or None,
            "run_id": run_id or None,
            "critic_id": row.get("critic_id"),
            "guidance": row.get("guidance"),
        },
    }
    return raw, canonical


def _execution_files(
    run_root: Path,
    include_eval: bool,
    epochs: Sequence[int] | None = None,
) -> list[tuple[Path, str]]:
    base = run_root / "memrl_tg"
    selected_epochs = set(epochs) if epochs is not None else None
    files: list[tuple[Path, str]] = [
        (path, "train")
        for path in sorted((base / "epochs").glob("epoch_*/executions.jsonl"))
        if selected_epochs is None or _epoch_from_path(path) in selected_epochs
    ]
    if include_eval:
        files.extend(
            (path, "validation")
            for path in sorted((base / "eval").glob("epoch_*.jsonl"))
            if selected_epochs is None or _epoch_from_path(path) in selected_epochs
        )
    if not files:
        suffix = "" if selected_epochs is None else f" for epochs={sorted(selected_epochs)}"
        raise LCBConversionError(f"no executions found below {base}{suffix}")
    return files


def convert_lcb(
    *,
    problems: str | Path,
    run_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    include_eval: bool = True,
    epochs: Sequence[int] | None = None,
) -> dict[str, Any]:
    problem_path = Path(problems).expanduser().resolve()
    run_path = Path(run_root).expanduser().resolve()
    split_path = Path(split_manifest).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    problems_by_id = _problem_index(problem_path)
    splits = _split_index(split_path)
    raw_rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    retrieval_count = 0
    missing_payload_count = 0
    seen_episode_ids: set[str] = set()
    for path, file_split in _execution_files(run_path, include_eval, epochs):
        source_files.append({"path": str(path), "sha256": _sha256_file(path), "split": file_split})
        epoch = _epoch_from_path(path)
        for line_no, row in _jsonl(path):
            task_id = str(row.get("task_id") or row.get("sample_index") or "").strip()
            if not task_id:
                rejected.append({"path": str(path), "line": line_no, "reason": "missing task_id"})
                continue
            if task_id not in problems_by_id:
                rejected.append({"path": str(path), "line": line_no, "task_id": task_id, "reason": "missing problem"})
                continue
            if task_id not in splits:
                rejected.append({"path": str(path), "line": line_no, "task_id": task_id, "reason": "missing split assignment"})
                continue
            split = file_split if file_split == "validation" else splits[task_id]
            setting_id = str(row.get("setting_id") or "evaluation")
            run_id = str(row.get("run_id") or "")
            try:
                raw, episode = _raw_row(
                    row,
                    problems_by_id[task_id],
                    task_id=task_id,
                    split=split,
                    epoch=epoch,
                    setting_id=setting_id,
                    run_id=run_id,
                    include_retrieval=(split == "train"),
                )
            except LCBConversionError as exc:
                rejected.append({"path": str(path), "line": line_no, "task_id": task_id, "reason": str(exc)})
                continue
            episode_id = str(episode["episode_id"])
            if episode_id in seen_episode_ids:
                rejected.append({
                    "path": str(path),
                    "line": line_no,
                    "task_id": task_id,
                    "reason": f"duplicate episode_id={episode_id}",
                })
                continue
            seen_episode_ids.add(episode_id)
            raw_rows.append(raw)
            episodes.append(episode)
            split_counts[split] += 1
            retrieval_count += len(episode["retrieval_events"])
            missing_payload_count += sum(
                bool(event.get("recall_missing_payload"))
                for event in episode["retrieval_events"]
            )
    output_path.mkdir(parents=True, exist_ok=True)
    raw_file = output_path / "trajectories.jsonl"
    episodes_file = output_path / "episodes.jsonl"
    rejected_file = output_path / "rejected.jsonl"
    raw_count, raw_sha = _write_jsonl(raw_file, raw_rows)
    episode_count, episode_sha = _write_jsonl(episodes_file, episodes)
    rejected_count, rejected_sha = _write_jsonl(rejected_file, rejected)
    qa = {
        "source_problems": len(problems_by_id),
        "source_execution_files": len(source_files),
        "accepted": episode_count,
        "rejected": rejected_count,
        "split_counts": dict(sorted(split_counts.items())),
        "retrieval_events": retrieval_count,
        "recall_missing_payload": missing_payload_count,
        "private_test_cases_copied": False,
        "includes_validation": include_eval,
        "rejection_reasons": dict(Counter(item["reason"] for item in rejected)),
    }
    (output_path / "qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "format": "mem2w-automationbench-lcb-v1",
        "source": {
            "problems": {"path": str(problem_path), "sha256": _sha256_file(problem_path)},
            "run_root": str(run_path),
            "split_manifest": {"path": str(split_path), "sha256": _sha256_file(split_path)},
            "execution_files": source_files,
        },
        "outputs": {
            "trajectories": {"path": str(raw_file), "count": raw_count, "sha256": raw_sha},
            "episodes": {"path": str(episodes_file), "count": episode_count, "sha256": episode_sha},
            "rejected": {"path": str(rejected_file), "count": rejected_count, "sha256": rejected_sha},
            "qa": str(output_path / "qa.json"),
        },
        "qa": qa,
        "selected_epochs": sorted(set(epochs)) if epochs is not None else None,
        "policy": {
            "automationbench_compatible_trajectory": True,
            "single_turn_lcb_trajectory": True,
            "q_keys_are_provenance_only": True,
            "retrieval_payload_reconstructed_from_guidance": True,
            "missing_memory_payload_keeps_action_episode": True,
            "private_test_cases_excluded": True,
            "validation_retrieval_events_excluded": True,
            "predicted_code_wrapped_in_python_fence": True,
        },
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert LiveCodeBench execution records to AutomationBench/Mem2W episodes"
    )
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-eval", action="store_true", help="exclude validation execution files")
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        help="restrict conversion to the specified epoch numbers (for example: --epochs 10)",
    )
    args = parser.parse_args(argv)
    manifest = convert_lcb(
        problems=args.problems,
        run_root=args.run_root,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        include_eval=not args.no_eval,
        epochs=args.epochs,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
