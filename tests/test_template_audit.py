import json
from pathlib import Path

from mem2w.ms_swift_adapter import convert_episodes
from mem2w.template_audit import audit_structure


ROOT = Path(__file__).parents[1]


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_structural_audit_covers_action_recall_and_payload_boundary(tmp_path):
    convert_episodes(ROOT / "data/example_episode.jsonl", tmp_path)
    rows = _rows(tmp_path / "action_train.jsonl") + _rows(tmp_path / "recall_train.jsonl")
    report = audit_structure(rows, source_episode_paths=[ROOT / "data/example_episode.jsonl"])
    assert report["rows"] == 2
    assert report["passed"] == 2
    assert report["failed"] == 0


def test_structural_audit_rejects_tool_call_and_loss_mask_corruption(tmp_path):
    convert_episodes(ROOT / "data/example_episode.jsonl", tmp_path)
    rows = _rows(tmp_path / "action_train.jsonl")
    rows[0]["messages"][2]["loss"] = False
    report = audit_structure(rows)
    assert report["failed"] == 1
    assert any("target role assistant" in error for error in report["errors"][0]["errors"])

    rows = _rows(tmp_path / "action_train.jsonl")
    rows[0]["messages"].append({
        "role": "tool_call",
        "content": {"name": "broken", "arguments": []},
        "loss": True,
    })
    report = audit_structure(rows)
    assert report["failed"] == 1
    assert any("arguments" in error for error in report["errors"][0]["errors"])


def test_structural_audit_rejects_recall_payload_duplication_and_hash_mismatch(tmp_path):
    convert_episodes(ROOT / "data/example_episode.jsonl", tmp_path)
    rows = _rows(tmp_path / "recall_train.jsonl")
    payload = rows[0]["messages"][-1]["content"]
    rows[0]["messages"].insert(1, {"role": "user", "content": payload, "loss": False})
    rows[0]["payload_sha256"] = "invalid"
    report = audit_structure(rows)
    assert report["failed"] == 1
    assert any("payload_sha256" in error for error in report["errors"][0]["errors"])
    assert any("duplicated" in error for error in report["errors"][0]["errors"])
