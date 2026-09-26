"""Audits Mem2W JSONL against the native ms-swift ``qwen3_5`` template.

The converter intentionally has no ms-swift dependency so it can be used on a
login node.  This module keeps that property for its structural audit and only
imports ``swift`` when the optional token-level audit is requested.  The latter
loads a processor, encodes each sample with ``qwen3_5`` and checks the actual
``labels`` emitted by ms-swift rather than trusting the per-message ``loss``
flags in the JSONL.

Typical GPU/login-node invocation (from a native ms-swift checkout)::

    PYTHONPATH=/path/to/ms-swift \
      python -m mem2w.template_audit \
      --dataset data/ms_swift/action_train.jsonl \
      --dataset data/ms_swift/recall_train.jsonl \
      --model /mnt/public/model/Qwen3.5-4B \
      --report artifacts/template_audit.json

The model is used for its processor/tokenizer only; no model weights are
loaded.  The output report records package/model metadata and every failed row
with a short reason.  Structural checks can be run without torch or
transformers by omitting ``--model``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence


class TemplateAuditError(ValueError):
    """A dataset row violates the Mem2W/ms-swift data contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _message_text(message: Mapping[str, Any]) -> str:
    value = message.get("content", "")
    if isinstance(value, str):
        return value
    return _canonical_json(value)


def _role(message: Mapping[str, Any]) -> str:
    role = str(message.get("role", "")).strip().lower().replace("-", "_")
    return {"tool": "tool_response", "function": "tool_response", "function_call": "tool_call"}.get(role, role)


def _parse_object(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _row_hash(row: Mapping[str, Any]) -> str:
    """Reproduce :func:`ms_swift_adapter._sample_row`'s content hash."""

    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"messages", "channel", "raw_content_hash", "chat_template_kwargs"}
    }
    return _sha256_text(_canonical_json({"messages": row.get("messages"), **metadata}))


def _payloads_by_episode(source_paths: Sequence[os.PathLike[str] | str]) -> dict[str, list[str]]:
    """Read exact retrieval payloads from canonical episode JSONL files."""

    result: dict[str, list[str]] = {}
    for source_path in source_paths:
        path = Path(source_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    episode = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TemplateAuditError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(episode, Mapping):
                    raise TemplateAuditError(f"{path}:{line_no}: episode must be an object")
                episode_id = episode.get("episode_id")
                if episode_id is None:
                    raise TemplateAuditError(f"{path}:{line_no}: episode_id is required")
                events = episode.get("retrieval_events")
                if events is None:
                    events = episode.get("retrieval_records")
                if events is None:
                    events = episode.get("retrievals")
                if not isinstance(events or [], list):
                    raise TemplateAuditError(f"{path}:{line_no}: retrieval events must be a list")
                payloads = []
                for event in events or []:
                    if isinstance(event, Mapping) and event.get("injected_context_text"):
                        payloads.append(str(event["injected_context_text"]))
                result.setdefault(str(episode_id), []).extend(payloads)
    return result


def _validate_tool_message(message: Mapping[str, Any], index: int, errors: list[str]) -> None:
    role = _role(message)
    if role == "tool_call":
        obj = _parse_object(message.get("content"))
        if obj is None or not str(obj.get("name", "")).strip() or not isinstance(obj.get("arguments"), Mapping):
            errors.append(f"messages[{index}] tool_call content must be an object with name and object arguments")
        if "tool_call_id" in message and not str(message.get("tool_call_id") or "").strip():
            errors.append(f"messages[{index}] tool_call_id is empty")
    elif role == "tool_response" and "tool_call_id" in message:
        if not str(message.get("tool_call_id") or "").strip():
            errors.append(f"messages[{index}] tool_call_id is empty")
    if role == "assistant" and message.get("tool_calls") is not None:
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            errors.append(f"messages[{index}] assistant.tool_calls must be a non-empty list")
            return
        for call_index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                errors.append(f"messages[{index}].tool_calls[{call_index}] must be an object")
                continue
            function = call.get("function", call)
            if not isinstance(function, Mapping) or not str(function.get("name", "")).strip():
                errors.append(f"messages[{index}].tool_calls[{call_index}] is missing function.name")
            if "arguments" not in (function if isinstance(function, Mapping) else {}):
                errors.append(f"messages[{index}].tool_calls[{call_index}] is missing arguments")
            if "id" in call and not str(call.get("id") or "").strip():
                errors.append(f"messages[{index}].tool_calls[{call_index}].id is empty")


def audit_structure(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_episode_paths: Sequence[os.PathLike[str] | str] = (),
) -> dict[str, Any]:
    """Audit derived JSONL without importing torch/ms-swift.

    Besides checking the explicit ``loss`` flags, this verifies provenance
    hashes and (when canonical episodes are supplied) that teacher payloads do
    not occur in action prompt messages.  It intentionally treats metadata as
    non-training input: only ``messages`` and ``chat_template_kwargs`` are
    passed to the template encoder.
    """

    payloads = _payloads_by_episode(source_episode_paths) if source_episode_paths else {}
    report: dict[str, Any] = {"rows": 0, "passed": 0, "failed": 0, "errors": [], "by_sample_type": {}}
    for row_index, row in enumerate(rows):
        report["rows"] += 1
        errors: list[str] = []
        if not isinstance(row, Mapping):
            errors.append("row must be an object")
            sample_type = "unknown"
        else:
            sample_type = str(row.get("sample_type", row.get("channel", "unknown")))
            messages = row.get("messages")
            if sample_type not in {"action", "recall"}:
                errors.append("sample_type/channel must be action or recall")
            if not isinstance(messages, list) or not messages:
                errors.append("messages must be a non-empty list")
                messages = []
            for index, message in enumerate(messages):
                if not isinstance(message, Mapping):
                    errors.append(f"messages[{index}] must be an object")
                    continue
                role = _role(message)
                if role not in {"system", "user", "assistant", "tool_call", "tool_response"}:
                    errors.append(f"messages[{index}] has unsupported role {message.get('role')!r}")
                if role in {"assistant", "tool_call"} and message.get("loss") is not True:
                    errors.append(f"messages[{index}] target role {role} must set loss=true")
                if role in {"system", "user", "tool_response"} and message.get("loss") is True:
                    errors.append(f"messages[{index}] prompt role {role} must not set loss=true")
                _validate_tool_message(message, index, errors)
            kwargs = row.get("chat_template_kwargs")
            if not isinstance(kwargs, Mapping) or kwargs.get("enable_thinking") is not False:
                errors.append("chat_template_kwargs.enable_thinking must be false")
            stored_hash = row.get("raw_content_hash")
            if stored_hash and stored_hash != _row_hash(row):
                errors.append("raw_content_hash does not match messages/metadata")
            if sample_type == "action":
                if not any(_role(message) in {"assistant", "tool_call"} for message in messages if isinstance(message, Mapping)):
                    errors.append("action sample has no assistant/tool_call target")
                episode_id = str(row.get("source_episode_id", ""))
                for payload in payloads.get(episode_id, []):
                    if payload and any(payload in _message_text(message) for message in messages if isinstance(message, Mapping) and _role(message) not in {"assistant", "tool_call"}):
                        errors.append("teacher retrieval payload appears in an action prompt")
                        break
            elif sample_type == "recall":
                assistant_targets = [message for message in messages if isinstance(message, Mapping) and _role(message) == "assistant"]
                if len(assistant_targets) != 1 or assistant_targets[0].get("loss") is not True:
                    errors.append("recall must contain exactly one loss=true assistant completion")
                else:
                    completion = _message_text(assistant_targets[0])
                    if not completion.strip():
                        errors.append("recall completion is empty")
                    if row.get("payload_sha256") and row["payload_sha256"] != _sha256_text(completion):
                        errors.append("payload_sha256 does not match recall completion")
                    if any(completion in _message_text(message) for message in messages[:-1] if isinstance(message, Mapping)):
                        errors.append("recall completion is duplicated in the prompt")
        report["by_sample_type"].setdefault(sample_type, {"rows": 0, "passed": 0, "failed": 0})["rows"] += 1
        if errors:
            report["failed"] += 1
            report["by_sample_type"][sample_type]["failed"] += 1
            report["errors"].append({"row": row_index, "sample_type": sample_type, "errors": errors})
        else:
            report["passed"] += 1
            report["by_sample_type"][sample_type]["passed"] += 1
    return report


def _disable_message_losses(row: Mapping[str, Any], *, only_role: Optional[str] = None) -> dict[str, Any]:
    result = copy.deepcopy(dict(row))
    messages = result.get("messages") or []
    for message in messages:
        if not isinstance(message, MutableMapping):
            continue
        role = _role(message)
        if only_role is None or role == only_role:
            message["loss"] = False
    return result


def _encode(template: Any, row: Mapping[str, Any]) -> Mapping[str, Any]:
    encoded = template.encode(dict(row), return_template_inputs=True)
    if isinstance(encoded, (list, tuple)):
        if len(encoded) != 1:
            raise TemplateAuditError("template truncation split a sample; use max_length >= sample length")
        encoded = encoded[0]
    return encoded


def _assert_token_mask(template: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    encoded = _encode(template, row)
    input_ids = encoded.get("input_ids")
    labels = encoded.get("labels")
    if not isinstance(input_ids, list) or not isinstance(labels, list) or len(input_ids) != len(labels):
        raise TemplateAuditError("template did not return equal-length input_ids and labels")
    target_count = sum(label != -100 for label in labels)
    if target_count <= 0:
        raise TemplateAuditError("template emitted no supervised target tokens")

    # Re-encode the same conversation with every response loss disabled.  The
    # rendered input must stay byte/token-identical while all target labels
    # disappear. This catches both misplaced `loss=true` fields and accidental
    # supervision of tool/user context without requiring brittle token offsets.
    baseline_row = _disable_message_losses(row)
    baseline = _encode(template, baseline_row)
    if baseline.get("input_ids") != input_ids:
        raise TemplateAuditError("loss flags changed qwen3_5 input rendering")
    baseline_labels = baseline.get("labels") or []
    baseline_targets = [index for index, label in enumerate(baseline_labels) if label != -100]
    # qwen3_5 appends a mandatory assistant-turn terminator.  Resolve its
    # actual token ids from the template rather than accepting an arbitrary
    # tail: otherwise a misplaced prompt/tool target could hide at the end.
    suffix_ids, _ = template._encode_context_list(template.template_meta.suffix)
    if not suffix_ids or len(suffix_ids) > len(input_ids):
        raise TemplateAuditError("qwen3_5 template suffix could not be resolved")
    suffix_start = len(input_ids) - len(suffix_ids)
    expected_suffix = list(range(suffix_start, len(input_ids)))
    if baseline_targets != expected_suffix:
        raise TemplateAuditError(
            "qwen3_5 emitted labels outside its exact template suffix with all message losses disabled")
    if input_ids[suffix_start:] != suffix_ids:
        raise TemplateAuditError("qwen3_5 encoded suffix tokens do not match template metadata")
    original_targets = [index for index, label in enumerate(labels) if label != -100]
    if not set(baseline_targets).issubset(original_targets):
        raise TemplateAuditError("qwen3_5 dropped template-owned suffix labels")

    # Explicitly mark only tool responses as targets. They remain query-side
    # context in qwen3_5 and must never produce labels.
    tool_only = _disable_message_losses(row)
    for message in tool_only.get("messages", []):
        if isinstance(message, MutableMapping) and _role(message) == "tool_response":
            message["loss"] = True
    tool_encoded = _encode(template, tool_only)
    tool_labels = tool_encoded.get("labels") or []
    tool_targets = [index for index, label in enumerate(tool_labels) if label != -100]
    if tool_targets != baseline_targets:
        raise TemplateAuditError("tool_response loss leaked into target labels")
    transformed = encoded.get("template_inputs")
    transformed_messages = getattr(transformed, "messages", []) if transformed is not None else []
    tool_call_rendered = []
    for message in transformed_messages:
        if isinstance(message, Mapping) and _role(message) == "assistant" and "<tool_call>" in _message_text(message):
            tool_call_rendered.append(_message_text(message))
    return {
        "tokens": len(input_ids),
        "target_tokens": target_count,
        "template_suffix_targets": len(baseline_targets),
        "transformed_messages": len(transformed_messages),
        "tool_call_blocks": len(tool_call_rendered),
    }


def audit_tokens(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    swift_root: Optional[os.PathLike[str] | str] = None,
    max_length: int = 4096,
) -> dict[str, Any]:
    """Run the optional native ms-swift token-level audit."""

    if swift_root:
        root = str(Path(swift_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        from swift import get_processor, get_template
    except Exception as exc:  # pragma: no cover - exercised on dependency-poor login nodes
        raise TemplateAuditError("native token audit requires ms-swift and its torch/transformers dependencies") from exc
    processor = get_processor(model, download_model=False)
    template = get_template(
        processor,
        template_type="qwen3_5",
        max_length=max_length,
        truncation_strategy="raise",
        remove_unused_columns=False,
        loss_scale="default",
        is_binary_loss_scale=True,
        enable_thinking=False,
    )
    template.set_mode("train")
    report = {"model": model, "template": "qwen3_5", "rows": 0, "passed": 0, "failed": 0, "errors": [], "samples": []}
    for row_index, row in enumerate(rows):
        report["rows"] += 1
        try:
            result = _assert_token_mask(template, row)
            report["passed"] += 1
            report["samples"].append({"row": row_index, **result})
        except Exception as exc:  # report all rows rather than aborting after the first bad example
            report["failed"] += 1
            report["errors"].append({"row": row_index, "error": str(exc)})
    try:
        report["ms_swift_version"] = importlib.metadata.version("ms-swift")
    except importlib.metadata.PackageNotFoundError:
        report["ms_swift_version"] = "source-checkout"
    return report


def _read_rows(paths: Sequence[os.PathLike[str] | str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in paths:
        path = Path(source).expanduser().resolve()
        with path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TemplateAuditError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise TemplateAuditError(f"{path}:{line_no}: expected a JSON object")
                rows.append(value)
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Mem2W JSONL against ms-swift qwen3_5")
    parser.add_argument("--dataset", action="append", required=True, type=Path)
    parser.add_argument("--source-episodes", action="append", default=[], type=Path)
    parser.add_argument("--model", help="optional local/HF model; enables token-level audit")
    parser.add_argument("--swift-root", type=Path, help="native ms-swift checkout to import")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = _read_rows(args.dataset)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        rows = rows[:args.limit]
    report: dict[str, Any] = {"structural": audit_structure(rows, source_episode_paths=args.source_episodes)}
    if args.model:
        report["tokens"] = audit_tokens(rows, model=args.model, swift_root=args.swift_root, max_length=args.max_length)
    report["ok"] = all(section["failed"] == 0 for section in report.values() if isinstance(section, Mapping) and "failed" in section)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        destination = args.report.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
