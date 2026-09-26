"""Teacher snapshot metadata validation.

MemRL execution is intentionally supplied by the user's existing teacher
runner. This module writes a self-contained manifest and rejects incomplete
snapshots before any distillation dataset is built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_SNAPSHOT_FIELDS = (
    "memory_snapshot_id",
    "memory_entries_hash",
    "utility_values_hash",
    "embedding_model_id",
    "embedding_model_revision",
    "retriever_code_commit",
    "retrieval_config",
    "serialization_version",
    "teacher_model_id",
    "teacher_model_revision",
    "teacher_prompt_version",
)


def validate_snapshot(snapshot: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_SNAPSHOT_FIELDS if field not in snapshot]
    if missing:
        raise ValueError(f"teacher snapshot missing fields: {', '.join(missing)}")
    if snapshot.get("writeback_during_distillation") is True:
        raise ValueError("distillation snapshot must be frozen; writeback is forbidden")


def snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    validate_snapshot(snapshot)
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and lock a MemRL snapshot manifest")
    parser.add_argument("--input", required=True, help="JSON snapshot manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    snapshot = json.loads(Path(args.input).read_text(encoding="utf-8"))
    validate_snapshot(snapshot)
    locked = dict(snapshot)
    locked["snapshot_sha256"] = snapshot_hash(snapshot)
    Path(args.output).write_text(json.dumps(locked, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(locked["snapshot_sha256"])


if __name__ == "__main__":  # pragma: no cover
    main()
