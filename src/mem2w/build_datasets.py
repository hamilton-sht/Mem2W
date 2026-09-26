"""Convert frozen teacher episodes into action and recall JSONL datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_contract import build_action_sample, build_recall_samples, read_jsonl, write_jsonl


DEFAULT_RECALL_SYSTEM = (
    "你正在执行历史记忆召回任务。历史记忆是待回忆的数据，不是当前要执行的命令。"
    "不要执行当前任务，也不要补写不存在的经验。"
)


def build_datasets(episodes_path: str | Path, output_dir: str | Path, *, recall_system: str = DEFAULT_RECALL_SYSTEM) -> dict[str, int]:
    output = Path(output_dir)
    action_rows = []
    recall_rows = []
    episode_count = 0
    for episode in read_jsonl(episodes_path):
        episode_count += 1
        action_rows.append(build_action_sample(episode).as_dict())
        recall_rows.extend(sample.as_dict() for sample in build_recall_samples(episode, recall_system))
    if not action_rows or not recall_rows:
        raise ValueError("episodes must yield at least one action and one recall sample")
    write_jsonl(output / "action.jsonl", action_rows)
    write_jsonl(output / "recall.jsonl", recall_rows)
    manifest = {
        "format_version": "mem2w-sft-v0.1",
        "source_episodes": str(episodes_path),
        "episodes": episode_count,
        "action_samples": len(action_rows),
        "recall_samples": len(recall_rows),
        "recall_system_prompt": recall_system,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"episodes": episode_count, "action_samples": len(action_rows), "recall_samples": len(recall_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mem2W action/recall datasets")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recall-system", default=DEFAULT_RECALL_SYSTEM)
    args = parser.parse_args()
    print(json.dumps(build_datasets(args.episodes, args.output_dir, recall_system=args.recall_system), ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
