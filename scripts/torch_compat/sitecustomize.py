"""Runtime compatibility for the A800 torch 2.6 environment.

The installed Qwen3.5/Fla wheel decorates a few kernels with ``torch.compile``
at import time.  This environment ships the private ``torch._functorch``
implementation but not the legacy top-level package that TorchDynamo imports.
For the training smoke/run we disable compile-time decoration and execute the
same Python kernels eagerly; model numerics remain unchanged, with a possible
throughput cost.
"""

import os

if os.environ.get("MEM2W_DISABLE_TORCH_COMPILE", "1") == "1":
    import torch

    def _eager_compile(fn=None, *args, **kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    torch.compile = _eager_compile
