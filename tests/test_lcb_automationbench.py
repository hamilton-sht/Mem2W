import json
from pathlib import Path

from mem2w.lcb_automationbench import convert_lcb


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_convert_lcb_to_automationbench_episode_contract(tmp_path):
    problems = tmp_path / "problems.jsonl"
    _jsonl(
        problems,
        [
            {
                "question_id": "q1",
                "question_content": "Print 1.",
                "private_test_cases": "SECRET PRIVATE INPUT/OUTPUT",
                "starter_code": "",
            },
            {"question_id": "q2", "question_content": "Print 2."},
        ],
    )
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["q1"], "test": ["q2"]}), encoding="utf-8")
    run = tmp_path / "run" / "memrl_tg"
    _jsonl(
        run / "epochs" / "epoch_001" / "executions.jsonl",
        [
            {
                "task_id": "q1",
                "setting_id": "q1:e1:s0",
                "run_id": "run-1",
                "used_q_keys": ["critic/q1/Q0001"],
                "guidance": "private memory should stay metadata",
                "reward": 0.5,
                "exact_success": False,
                "observation_summary": {"predicted_code": "print(1)", "num_passed": 1, "num_total": 2},
            }
        ],
    )
    _jsonl(
        run / "eval" / "epoch_001.jsonl",
        [{"sample_index": "q2", "success": True, "num_passed": 3, "num_total": 3, "predicted_code": "print(2)"}],
    )
    out = tmp_path / "out"
    manifest = convert_lcb(
        problems=problems,
        run_root=run.parent,
        split_manifest=split,
        output_dir=out,
    )
    assert manifest["qa"]["accepted"] == 2
    assert manifest["qa"]["rejected"] == 0
    rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    row = rows[0]
    assert row["split"] == "train"
    assert row["task_id"] == "q1"
    assert row["retrieval_events"][0]["recall_missing_payload"] is False
    assert "private memory should stay metadata" in row["retrieval_events"][0]["injected_context_text"]
    assert row["messages"][-1]["content"] == "```python\nprint(1)\n```"
    assert "SECRET PRIVATE" not in (out / "episodes.jsonl").read_text()
    assert "private memory" in row["source"]["guidance"]
    validation = rows[1]
    assert validation["split"] == "validation"
    assert validation["outcome"]["success"] is True


def test_convert_lcb_can_select_epochs(tmp_path):
    problems = tmp_path / "problems.jsonl"
    _jsonl(problems, [{"question_id": "q1", "question_content": "Print 1."}])
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["q1"], "test": []}), encoding="utf-8")
    run = tmp_path / "run" / "memrl_tg"
    for epoch in (1, 2):
        _jsonl(
            run / "epochs" / f"epoch_{epoch:03d}" / "executions.jsonl",
            [{
                "task_id": "q1",
                "setting_id": f"q1:e{epoch}:s0",
                "run_id": "run-1",
                "reward": float(epoch),
                "observation_summary": {"predicted_code": "print(1)", "num_passed": 1, "num_total": 1},
            }],
        )
    out = tmp_path / "out"
    manifest = convert_lcb(
        problems=problems,
        run_root=run.parent,
        split_manifest=split,
        output_dir=out,
        include_eval=False,
        epochs=[2],
    )
    assert manifest["selected_epochs"] == [2]
    rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["epoch"] == 2
