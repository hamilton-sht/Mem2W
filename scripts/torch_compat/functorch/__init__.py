"""Compatibility shim for environments where torch 2.6 omits top-level functorch.

PyTorch 2.6 moved the implementation under ``torch._functorch`` but a few
TorchDynamo imports (and the Qwen3.5/Fla stack) still import
``functorch.compile``.  The shim only re-exports the public compile helpers
needed by those imports; it does not replace the underlying implementation.
"""

from . import compile  # noqa: F401

__all__ = ["compile"]
