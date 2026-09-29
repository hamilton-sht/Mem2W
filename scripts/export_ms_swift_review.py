"""Export a read-only source snapshot as a reviewable upstream patch and manifest.

Inputs are local snapshots, never a live training checkout. Only differing source
files are copied; caches and timestamped backup files are excluded and recorded.
"""
import argparse
import ast
import difflib
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-commit', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; use a new directory to preserve review evidence')
    args.output.mkdir(parents=True)
    entries, excluded, patches = [], [], []
    paths = sorted({p.relative_to(root) for root in (args.upstream, args.snapshot)
                    for p in root.rglob('*') if p.is_file()})
    for rel in paths:
        if any(p in {'.git', '__pycache__', '.pytest_cache'} or p.endswith('.egg-info') for p in rel.parts):
            continue
        old, new = args.upstream / rel, args.snapshot / rel
        a, b = old.read_bytes() if old.exists() else b'', new.read_bytes() if new.exists() else b''
        if a == b:
            continue
        if '.bak_' in rel.name:
            excluded.append({'path': str(rel), 'reason': 'timestamped backup, not active source',
                             'sha256': hashlib.sha256(b).hexdigest()})
            continue
        text_a, text_b = a.decode('utf-8'), b.decode('utf-8')
        status = 'A' if not old.exists() else 'D' if not new.exists() else 'M'
        entries.append({'path': str(rel), 'status': status, 'bytes': len(b),
                        'sha256': hashlib.sha256(b).hexdigest() if new.exists() else None,
                        'upstream_sha256': hashlib.sha256(a).hexdigest() if old.exists() else None})
        if new.exists():
            if rel.suffix == '.py':
                ast.parse(text_b, filename=str(rel))
            target = args.output / 'files' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(new, target)
        patches.append('diff --git a/{0} b/{0}\n'.format(rel))
        if status == 'A':
            patches.append('new file mode 100644\n')
        elif status == 'D':
            patches.append('deleted file mode 100644\n')
        patches.extend(difflib.unified_diff(text_a.splitlines(keepends=True), text_b.splitlines(keepends=True),
                        fromfile=f'a/{rel}' if old.exists() else '/dev/null',
                        tofile=f'b/{rel}' if new.exists() else '/dev/null'))
    (args.output / 'changes.patch').write_text(''.join(patches))
    shutil.copy2(args.upstream / 'LICENSE', args.output / 'LICENSE.upstream')
    manifest = {'captured_at_utc': datetime.now(timezone.utc).isoformat(),
                'upstream': 'https://github.com/modelscope/ms-swift', 'base_commit': args.base_commit,
                'source': 'A800 /home/sht/haoting/ms-swift-mem2w (disk snapshot; not a Git repository)',
                'runtime_caveat': 'Disk snapshot is not proof of source already imported by running processes.',
                'files': entries, 'excluded': excluded}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    table = ['# ms-swift source inventory\n', f'Base: `{args.base_commit}`\n',
             '| Status | File | Lines |\n| --- | --- | --- |\n']
    for e in entries:
        p = args.output / 'files' / e['path']
        lines = len(p.read_text().splitlines()) if p.exists() else 0
        table.append(f"| {e['status']} | [{e['path']}](files/{e['path']}) | {lines} |\n")
    (args.output / 'INDEX.md').write_text('\n'.join(table))
    print(json.dumps({'changed_files': len(entries),
                      'added': sum(e['status'] == 'A' for e in entries),
                      'modified': sum(e['status'] == 'M' for e in entries),
                      'excluded_backups': len(excluded)}, indent=2))


if __name__ == '__main__':
    main()
