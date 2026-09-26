import json
from pathlib import Path

from mem2w.raw_automationbench import ImportOptions, import_automationbench


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_import_trajectory_join_and_missing_payload_policy(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    trajectory = "\n".join(
        json.dumps(row, ensure_ascii=False)
        for row in [
            {"role": "user", "content": "配置项目 R"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "type": "function"}],
                "reasoning_content": "private chain of thought",
            },
            {"role": "tool", "content": "完成", "tool_call_id": "call-1"},
        ]
    )
    _write_jsonl(
        trajectories,
        [
            {
                "epoch": 0,
                "task": "task-a",
                "task_completed_correctly": 1,
                "partial_credit": 1.0,
                "retrieved_memory_ids": ["mem-1"],
                "retrieval_records": [{"memory_id": "mem-1", "q_estimate": 0.7}],
                "trajectory": trajectory,
            },
            {
                "epoch": 1,
                "task": "task-b",
                "task_completed_correctly": 0,
                "partial_credit": 0.0,
                "retrieved_memory_ids": ["mem-2"],
                "retrieval_records": [{"memory_id": "mem-2", "q_estimate": 0.1}],
                "trajectory": trajectory,
            },
        ],
    )
    prompts = tmp_path / "actor_prompts"
    prompts.mkdir()
    _write_jsonl(
        prompts / "epoch_0.jsonl",
        [
            {
                "task_id": "task-a",
                "messages": [
                    {"role": "system", "content": "[Reference Memories]\nEXACT PAYLOAD"}
                ],
            }
        ],
    )
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"task-a": "train", "task-b": "validation"}), encoding="utf-8")
    out = tmp_path / "out"
    manifest = import_automationbench(
        ImportOptions(
            trajectories=trajectories,
            output_dir=out,
            actor_prompts=prompts,
            split_manifest=split,
            snapshot_root=tmp_path / "snapshots",
        )
    )
    assert manifest["episodes"] == 2
    assert manifest["recall_missing_payload"] == 1
    assert manifest["prompt_missing_payload"] == 1
    rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    first, second = rows
    assert first["memory_snapshot_id"] == "snapshot/0"
    assert first["split"] == "train"
    assert first["retrieval_events"][0]["injected_context_text"] == "[Reference Memories]\nEXACT PAYLOAD"
    assert first["retrieval_events"][0]["recall_missing_payload"] is False
    assert second["retrieval_events"][0]["recall_missing_payload"] is True
    assert second["retrieval_events"][0]["injected_context_text"] == ""
    assert first["messages"][1]["tool_calls"][0]["id"] == "call-1"
    assert "reasoning_content" not in first["messages"][1]
    qa = json.loads((out / "qa.json").read_text())
    assert qa["split_counts"] == {"train": 1, "validation": 1}
    assert qa["prompt_missing_payload"] == 1


def test_manifest_split_is_authoritative(tmp_path):
    trajectory = json.dumps({"role": "assistant", "content": "action"})
    source = tmp_path / "trajectories.jsonl"
    _write_jsonl(
        source,
        [
            {
                "epoch": 0,
                "task": "task-a",
                "split": "validation",
                "trajectory": trajectory,
            }
        ],
    )
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"task-a": "train"}), encoding="utf-8")
    out = tmp_path / "out"
    import_automationbench(ImportOptions(trajectories=source, output_dir=out, split_manifest=split))
    row = json.loads((out / "episodes.jsonl").read_text())
    assert row["split"] == "train"


def test_epoch_directory_input_and_compact_payload_are_audited(tmp_path):
    run = tmp_path / "run"
    epoch_dir = run / "epoch_4"
    epoch_dir.mkdir(parents=True)
    payload = "[Reference Memories]\n" + ("x" * 80)
    trajectory = "\n".join(
        json.dumps(row)
        for row in [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "action"},
        ]
    )
    _write_jsonl(
        epoch_dir / "trajectories.jsonl",
        [{"task": "task-a", "partial_credit": 1, "retrieved_memory_ids": ["m"], "trajectory": trajectory}],
    )
    prompts = run / "actor_prompts"
    prompts.mkdir()
    _write_jsonl(
        prompts / "epoch_4.jsonl",
        [{"task_id": "task-a", "prompt": payload}],
    )
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["task-a"]}), encoding="utf-8")
    out = tmp_path / "out"
    import_automationbench(
        ImportOptions(
            trajectories=run,
            output_dir=out,
            actor_prompts=prompts,
            split_manifest=split,
            compact_payload=True,
            payload_budget_chars=100,
        )
    )
    row = json.loads((out / "episodes.jsonl").read_text())
    event = row["retrieval_events"][0]
    assert event["payload_compacted"] is True
    assert event["payload_original_chars"] > event["payload_chars"]
    assert "compacted recall payload" in event["injected_context_text"]
