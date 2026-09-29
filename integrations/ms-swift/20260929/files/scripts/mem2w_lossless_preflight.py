"""Audit EVERY lossless row using native SWIFT encoding, without model weights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from swift import get_processor, get_template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads((Path(args.model) / 'config.json').read_text())
    limit = config.get('text_config', config)['max_position_embeddings']
    processor = get_processor(args.model, download_model=False)
    template = get_template(processor, template_type='qwen3_5', max_length=None,
                            truncation_strategy='raise', remove_unused_columns=False,
                            loss_scale='default', is_binary_loss_scale=True, enable_thinking=False)
    template.set_mode('train')
    report = {'model': args.model, 'model_context_limit': limit, 'streams': {}}
    for kind in ('action', 'recall'):
        path = args.data_dir / f'{kind}_train.jsonl'
        records, errors = [], []
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for i, raw in enumerate(stream, 1):
                digest.update(raw)
                row = json.loads(raw)
                try:
                    original = copy.deepcopy(row)
                    encoded = template.encode(row)
                    if isinstance(encoded, (list, tuple)):
                        raise ValueError('unexpected split')
                    ids, labels = encoded['input_ids'], encoded['labels']
                    count = sum(label != -100 for label in labels[1:])
                    if not count:
                        raise ValueError('zero supervised tokens')
                    if kind == 'recall':
                        target = original['messages'][-1]['content']
                        if hashlib.sha256(target.encode()).hexdigest() != original['payload_sha256']:
                            raise ValueError('recall target hash mismatch')
                    # Prove the per-message loss flag reaches the native encoder.
                    if i <= 3:
                        disabled = copy.deepcopy(original)
                        for message in disabled['messages']:
                            message['loss'] = False
                        masked = template.encode(disabled)
                        if sum(x != -100 for x in masked['labels'][1:]) > 2:
                            raise ValueError('loss-mask leak when every message is disabled')
                    records.append({'row': i, 'sample_id': original['sample_id'],
                                    'tokens': len(ids), 'supervised_tokens': count})
                except Exception as exc:
                    errors.append({'row': i, 'error': repr(exc)})
                if i % 128 == 0:
                    print(json.dumps({'stream': kind, 'processed': i, 'errors': len(errors)}), flush=True)
        lengths = sorted(r['tokens'] for r in records)
        summary = {'rows': len(records), 'errors': errors, 'source_sha256': digest.hexdigest(),
                   'min': min(lengths), 'max': max(lengths),
                   'p50': lengths[len(lengths) // 2], 'p95': lengths[int(len(lengths) * .95)],
                   'overflow': [r for r in records if r['tokens'] > limit],
                   'total_tokens': sum(lengths), 'records': records}
        report['streams'][kind] = summary
        print(json.dumps({kind: {k: v for k, v in summary.items() if k != 'records'}}), flush=True)
    report['ok'] = all(not value['errors'] and not value['overflow'] for value in report['streams'].values())
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
