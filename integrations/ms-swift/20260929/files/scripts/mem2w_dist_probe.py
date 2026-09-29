"""Small MACA distributed capability probe used before launching DDP smoke tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    report = {
        'torch': torch.__version__,
        'cuda_device_count': torch.cuda.device_count(),
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'distributed': dist.is_available(),
        'gloo': dist.is_gloo_available(),
        'nccl': dist.is_nccl_available(),
        'mpi': dist.is_mpi_available() if hasattr(dist, 'is_mpi_available') else False,
        'rank': int(os.getenv('RANK', '-1')),
        'world_size': int(os.getenv('WORLD_SIZE', '-1')),
    }
    if int(os.getenv('WORLD_SIZE', '1')) > 1:
        backend = os.getenv('MEM2W_DIST_BACKEND', 'gloo')
        try:
            dist.init_process_group(backend=backend)
            rank = dist.get_rank()
            device = torch.device('cuda', int(os.getenv('LOCAL_RANK', '0')))
            value = torch.tensor([rank + 1], device=device, dtype=torch.float32)
            dist.all_reduce(value)
            report.update({
                'backend': backend,
                'rank': rank,
                'world_size': dist.get_world_size(),
                'local_rank': int(os.getenv('LOCAL_RANK', '0')),
                'all_reduce_value': float(value.item()),
                'all_reduce_expected': sum(range(1, dist.get_world_size() + 1)),
            })
            dist.destroy_process_group()
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            report.update({'backend': backend, 'error': f'{type(exc).__name__}: {exc}'})
    default_output = f"/tmp/mem2w_dist_probe_rank{report['rank']}.json"
    output = Path(os.getenv('MEM2W_DIST_PROBE_OUTPUT', default_output))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
