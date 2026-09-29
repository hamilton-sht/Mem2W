#!/usr/bin/env python3
"""Normalize AutomationBench OpenAI tool calls for ms-swift/qwen3_5.

Some archived rows store each assistant ``tool_calls`` entry as a JSON string.
This makes the call envelope an object while keeping ``function.arguments``
as the standard JSON string expected by OpenAI/Qwen templates. Keeping the
arguments serialized also prevents the HF Arrow loader from trying to infer a
single schema for heterogeneous tool parameter objects.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must decode to an object")
    return value


def normalize_row(row: dict[str, Any]) -> tuple[dict[str, Any], int]:
    row = copy.deepcopy(row)
    changed = 0
    for message in row.get("messages", []):
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        # The HF JSON loader infers nested Arrow fields from the first rows.
        # AutomationBench legitimately uses null for messages without a tool
        # call, but a later assistant/tool message can contain a list/string.
        # Use stable empty values so native ms-swift can load the full JSONL
        # without changing the actual tool-call payloads.
        if isinstance(message, dict):
            if message.get("tool_calls") is None:
                message["tool_calls"] = []
            if message.get("tool_call_id") is None:
                message["tool_call_id"] = ""
                changed += 1
            calls = message.get("tool_calls")
        if not calls:
            continue
        if not isinstance(calls, list):
            raise ValueError("assistant.tool_calls must be a list")
        normalized = []
        for call in calls:
            obj = _json_object(call, "tool_call")
            function = obj.get("function", obj)
            function = _json_object(function, "tool_call.function")
            if not str(function.get("name", "")).strip():
                raise ValueError("tool call is missing function.name")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool call arguments must be an object")
            obj["function"] = {
                "name": function["name"],
                "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            }
            obj.setdefault("type", "function")
            normalized.append(obj)
            if not isinstance(call, dict) or call.get("function") != obj.get("function"):
                changed += 1
        message["tool_calls"] = normalized
    return row, changed


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_file(src: Path, dst: Path) -> tuple[int, int]:
    rows = changed = 0
    with src.open(encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            row, count = normalize_row(json.loads(line))
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            rows += 1
            changed += count
    return rows, changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    args = parser.parse_args()
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in ("action_train.jsonl", "recall_train.jsonl"):
        src, dst = args.src_dir / name, args.dst_dir / name
        rows, changed = normalize_file(src, dst)
        summary[name] = {"rows": rows, "tool_calls_normalized": changed, "sha256": _sha256(dst)}
    for name in ("manifest.json", "qa.json", "recall_audit_en.json"):
        src = args.src_dir / name
        if src.exists():
            shutil.copy2(src, args.dst_dir / name)
    manifest_path = args.dst_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = manifest.setdefault("outputs", {})
        for name, info in summary.items():
            outputs.setdefault(name, {}).update({"count": info["rows"], "sha256": info["sha256"]})
        manifest["format"] = str(manifest.get("format", "")) + "+tool-call-objects"
        manifest["normalization"] = {
            "tool_calls_as_openai_objects": True,
            "arguments_as_json_strings": True,
            "nullable_tool_fields_stabilized": True,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.dst_dir / "normalization.json").write_text(
        json.dumps({"source_dir": str(args.src_dir), "files": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
