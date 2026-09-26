import json
from pathlib import Path

from mem2w.ms_swift_adapter import build_swift_config, convert_episodes


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
