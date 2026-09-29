#!/usr/bin/env python3
"""Create a schema-stable, loss-preserving ms-swift SFT JSONL view.

The custom Mem2W loader consumes the rich rows directly.  Hugging Face
Datasets (used by stock ``swift sft``) infers an Arrow schema across every
metadata field and can reject nullable mixed-type fields.  This view keeps
only the native training contract: messages, per-message loss flags, and
template kwargs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _message(message: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "role": str(message.get("role", "")),
        "content": message.get("content") if message.get("content") is not None else "",
        "loss": bool(message.get("loss", False)),
    }
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    if message.get("tool_call_id"):
        out["tool_call_id"] = str(message["tool_call_id"])
    if message.get("name"):
        out["name"] = str(message["name"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with args.src.open(encoding="utf-8") as inp, args.dst.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            out.write(json.dumps({
                "messages": [_message(m) for m in row["messages"]],
                "chat_template_kwargs": row.get("chat_template_kwargs", {"enable_thinking": False}),
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            rows += 1
    print(json.dumps({"source": str(args.src), "output": str(args.dst), "rows": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
