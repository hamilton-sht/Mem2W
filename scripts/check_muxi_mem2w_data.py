#!/usr/bin/env python3
"""Fail-closed audit for the canonical AutomationBench Mem2W JSONL streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_no}: expected an object')
            rows.append(row)
    return rows


def _audit(data_dir: Path, accumulation_steps: int, world_size: int) -> dict[str, Any]:
    errors: list[str] = []
    action_path = data_dir / 'action_train.jsonl'
    recall_path = data_dir / 'recall_train.jsonl'
    manifest_path = data_dir / 'manifest.json'
    for path in (action_path, recall_path, manifest_path):
        if not path.is_file():
            errors.append(f'missing required file: {path}')
    if errors:
        return {'ok': False, 'errors': errors, 'data_dir': str(data_dir)}

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    action = _read_rows(action_path)
    recall = _read_rows(recall_path)
    counts = {'action': len(action), 'recall': len(recall)}
    epoch_counts = {
        'action': dict(sorted(Counter(int(row.get('epoch', -1)) for row in action).items())),
        'recall': dict(sorted(Counter(int(row.get('epoch', -1)) for row in recall).items())),
    }

    for name, path, rows in (
        ('action_train.jsonl', action_path, action),
        ('recall_train.jsonl', recall_path, recall),
    ):
        expected = manifest.get('outputs', {}).get(name, {})
        actual_hash = _sha256(path)
        if expected.get('count') != len(rows):
            errors.append(f'{name}: manifest count does not match ({expected.get("count")} != {len(rows)})')
        if expected.get('sha256') != actual_hash:
            errors.append(f'{name}: manifest sha256 does not match')

    if {row.get('sample_type') for row in action} != {'action'}:
        errors.append('action_train.jsonl contains a non-action row')
    if {row.get('sample_type') for row in recall} != {'recall'}:
        errors.append('recall_train.jsonl contains a non-recall row')

    recall_bad = 0
    feedback_wrapper = 0
    for row in recall:
        messages = row.get('messages') or []
        if len(messages) != 3:
            errors.append(f"recall {row.get('sample_id')}: expected system/user/assistant messages")
            continue
        system = str(messages[0].get('content') or '')
        user = str(messages[1].get('content') or '')
        target = str(messages[2].get('content') or '')
        if not system.startswith('You are performing a historical memory recall task.'):
            recall_bad += 1
        if not user.startswith('Retrieval query:'):
            recall_bad += 1
        if target and target in f'{system}\n{user}':
            recall_bad += 1
        if hashlib.sha256(target.encode()).hexdigest() != row.get('payload_sha256'):
            recall_bad += 1
        if '\n\nHISTORICAL EVALUATOR FEEDBACK:\n{' in target:
            feedback_wrapper += 1
    if recall_bad:
        errors.append(f'recall prompt/payload/hash checks failed on {recall_bad} rows')
    if feedback_wrapper:
        errors.append(f'{feedback_wrapper} recall rows still contain evaluator wrapper fields')

    # ``group_size`` is per Trainer rank.  DDP consumes one logical dataset
    # item on every rank per optimizer step, so an update covers
    # accumulation_steps * world_size rows globally.
    effective_rows_per_update = accumulation_steps * world_size
    updates_per_action_epoch = math.ceil(len(action) / effective_rows_per_update)
    updates_per_recall_epoch = math.ceil(len(recall) / effective_rows_per_update)
    return {
        'ok': not errors,
        'errors': errors,
        'data_dir': str(data_dir),
        'counts': counts,
        'epoch_counts': epoch_counts,
        'updates_per_action_epoch': updates_per_action_epoch,
        'updates_per_recall_epoch': updates_per_recall_epoch,
        'updates_per_paired_epoch': max(updates_per_action_epoch, updates_per_recall_epoch),
        'accumulation_steps': accumulation_steps,
        'world_size': world_size,
        'effective_rows_per_update': effective_rows_per_update,
        'manifest_policy': manifest.get('policy', {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--accumulation-steps', type=int, default=8)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()
    if args.accumulation_steps <= 0 or args.world_size <= 0:
        raise SystemExit('--accumulation-steps and --world-size must be positive')
    report = _audit(args.data_dir.expanduser().resolve(), args.accumulation_steps, args.world_size)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
