"""Probe native MACA FlashAttention forward/backward before long-sequence SFT."""
import json
import torch
import flash_attn
from flash_attn import flash_attn_func

torch.manual_seed(42)
q = torch.randn(1, 128, 16, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, 128, 4, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
v = torch.randn_like(k, requires_grad=True)
out = flash_attn_func(q, k, v, causal=True)
out.float().square().mean().backward()
torch.cuda.synchronize()
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v))
print(json.dumps({'torch': torch.__version__, 'flash_attn': flash_attn.__version__,
                  'device': torch.cuda.get_device_name(), 'flash_backward': 'passed',
                  'peak_bytes': torch.cuda.max_memory_allocated()}), flush=True)
