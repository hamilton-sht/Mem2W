"""GPU/CPU-independent check for the Mem2W W/C content-gradient rule."""

import torch

from swift.mem2w import Mem2WConfig
from swift.mem2w.memory import PersistentKVMemory


def main() -> None:
    torch.manual_seed(7)
    memory = PersistentKVMemory(Mem2WConfig(hidden_size=32, slots=8, key_dim=8, value_dim=8))
    with torch.no_grad():
        memory.W_O.normal_(std=0.1)
    hidden = torch.randn(2, 5, 32)

    warmup_loss = memory(hidden, stop_content_grad=False).float().square().mean()
    warmup_loss.backward()
    warmup = {name: parameter.grad is not None for name, parameter in memory.named_parameters()}
    if set(name for name, present in warmup.items() if present) != {'W_Q', 'K', 'V', 'W_O'}:
        raise AssertionError(f'W stage gradient set mismatch: {warmup}')

    memory.zero_grad(set_to_none=True)
    constrained_loss = memory(hidden, stop_content_grad=True).float().square().mean()
    constrained_loss.backward()
    constrained = {name: parameter.grad is not None for name, parameter in memory.named_parameters()}
    if constrained != {'W_Q': True, 'K': True, 'V': False, 'W_O': False}:
        raise AssertionError(f'C stage gradient set mismatch: {constrained}')
    print({'status': 'ok', 'warmup_gradients': warmup, 'constrained_gradients': constrained})


if __name__ == '__main__':
    main()
