import copy

import torch

from swift.mem2w.config import Mem2WConfig
from swift.mem2w.memory import PersistentKVMemory


def test_memory_tiling():
    for stop_content in (False, True):
        torch.manual_seed(18)
        full = PersistentKVMemory(Mem2WConfig(hidden_size=8, slots=5, key_dim=4,
                                              value_dim=6, compute_chunk_size=100)).double()
        # Nonzero output matrix is essential: zero initialization hides Q/K/V bugs.
        with torch.no_grad():
            full.W_O.normal_()
        tiled = copy.deepcopy(full)
        tiled.compute_chunk_size = 3
        x = torch.randn(2, 11, 8, dtype=torch.double, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        mask = torch.ones(2, 11)
        mask[0, -2:] = 0
        a = full(x, attention_mask=mask, stop_content_grad=stop_content)
        b = tiled(y, attention_mask=mask, stop_content_grad=stop_content)
        torch.testing.assert_close(a, b)
        a.square().mean().backward()
        b.square().mean().backward()
        torch.testing.assert_close(x.grad, y.grad)
        for (name, p), (_, q) in zip(full.named_parameters(), tiled.named_parameters()):
            if stop_content and name in {'V', 'W_O'}:
                assert p.grad is None and q.grad is None
            else:
                torch.testing.assert_close(p.grad, q.grad)
    print('memory tiling: values, input/four-parameter gradients, C detach PASSED', flush=True)


if __name__ == '__main__':
    test_memory_tiling()
