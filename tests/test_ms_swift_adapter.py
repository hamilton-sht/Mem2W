import json
from pathlib import Path

import pytest

from mem2w.ms_swift_adapter import Mem2WDataError, build_swift_config, convert_episodes


ROOT = Path(__file__).parents[1]


def test_convert_and_select_independent_modes(tmp_path):
    manifest = convert_episodes(ROOT / "data/example_episode.jsonl", tmp_path)
    assert manifest["files"]["action_train.jsonl"]["count"] == 1
    assert manifest["files"]["recall_train.jsonl"]["count"] == 1
    action = json.loads((tmp_path / "action_train.jsonl").read_text())
    recall = json.loads((tmp_path / "recall_train.jsonl").read_text())
    assert action["channel"] == "action"
    assert recall["channel"] == "recall"
    assert all(message.get("loss") is not True for message in action["messages"] if message["role"] in {"user", "tool_response"})
    assert recall["messages"][-1]["loss"] is True

    action_config = build_swift_config(action_data=[tmp_path / "action_train.jsonl"], recall_data=[], mode="action")
    recall_config = build_swift_config(action_data=[], recall_data=[tmp_path / "recall_train.jsonl"], mode="recall")
    assert action_config["dataset"] == [str((tmp_path / "action_train.jsonl").resolve())]
    assert recall_config["dataset"] == [str((tmp_path / "recall_train.jsonl").resolve())]


def test_memrl_provenance_is_preserved_without_payload_duplication(tmp_path):
    episode = json.loads((ROOT / "data/example_episode.jsonl").read_text())
    episode.update(
        {
            "task_id": "task-r-config-001",
            "epoch": 3,
            "_memq_task_description": "验证项目 R 的配置修改。",
            "_memq_retrieved_ids": ["mem-017", "mem-019"],
            "partial_credit": 0.75,
            "partial_credit_diagnostic": 0.5,
            "diagnostics": {"tool_calls": 2},
            "q_value": 0.2,
            "q_visits": 4,
            "source_algorithm": "memrl",
        }
    )
    source = tmp_path / "episode.jsonl"
    source.write_text(json.dumps(episode, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = convert_episodes(source, tmp_path / "out")
    action = json.loads((tmp_path / "out/action_train.jsonl").read_text())
    recall = json.loads((tmp_path / "out/recall_train.jsonl").read_text())
    for row in (action, recall):
        assert row["task_id"] == "task-r-config-001"
        assert row["epoch"] == 3
        assert row["partial_credit"] == 0.75
        assert row["partial_credit_diagnostic"] == 0.5
        assert row["retrieved_memory_ids"] == ["mem-017", "mem-019"]
    assert manifest["files"]["recall_train.jsonl"]["count"] == 1
    # The retrieved payload is a target only; it must not be copied into the
    # action prompt as a hidden teacher message.
    assert all("memories" not in str(message.get("content", "")) for message in action["messages"])


def test_missing_split_is_rejected_instead_of_becoming_train(tmp_path):
    episode = json.loads((ROOT / "data/example_episode.jsonl").read_text())
    episode.pop("split")
    source = tmp_path / "episode.jsonl"
    source.write_text(json.dumps(episode, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(Mem2WDataError, match="split"):
        convert_episodes(source, tmp_path / "out")


def test_structured_trajectory_alias_and_task_description(tmp_path):
    episode = json.loads((ROOT / "data/example_episode.jsonl").read_text())
    episode.pop("messages")
    episode["trajectory"] = {
        "messages": [
            {"role": "assistant", "content": "执行工具"},
            {"role": "tool", "content": "完成"},
        ]
    }
    episode["task"] = {"tools_schema_id": "sandbox-tools-v1"}
    episode["_memq_task_description"] = "恢复任务描述"
    source = tmp_path / "episode.jsonl"
    source.write_text(json.dumps(episode, ensure_ascii=False) + "\n", encoding="utf-8")
    convert_episodes(source, tmp_path / "out")
    action = json.loads((tmp_path / "out/action_train.jsonl").read_text())
    assert action["messages"][0]["role"] in {"system", "user"}
    assert any(message["role"] == "tool_response" for message in action["messages"])


def test_action_survives_missing_raw_recall_payload(tmp_path):
    episode = json.loads((ROOT / "data/example_episode.jsonl").read_text())
    event = episode["retrieval_events"][0]
    event["injected_context_text"] = ""
    event["recall_missing_payload"] = True
    source = tmp_path / "episode.jsonl"
    source.write_text(json.dumps(episode, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = convert_episodes(source, tmp_path / "out")
    assert manifest["files"]["action_train.jsonl"]["count"] == 1
    assert "recall_train.jsonl" not in manifest["files"]
    assert manifest["qa"]["recall_missing_payload"] == 1
