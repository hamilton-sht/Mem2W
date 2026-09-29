"""Small top-level functorch.compile compatibility layer for PyTorch 2.6."""

from torch._functorch.compilers import (  # noqa: F401
    min_cut_rematerialization_partition,
    nop,
    ts_compile,
)
from torch._functorch._aot_autograd.runtime_wrappers import (  # noqa: F401
    make_boxed_func,
)

# These names are imported lazily by some TorchDynamo debug paths.  Keep them
# as explicit placeholders with clear errors rather than masking a bad setup.
try:
    from torch._functorch.decompositions import default_decompositions  # type: ignore
except Exception:  # pragma: no cover - version-dependent compatibility path
    def default_decompositions(*args, **kwargs):
        raise RuntimeError("default_decompositions is unavailable in this torch build")


try:
    from torch._functorch.partitioners import min_cut_rematerialization_partition as _min_cut
except Exception:  # pragma: no cover
    _min_cut = min_cut_rematerialization_partition


__all__ = [
    "default_decompositions",
    "make_boxed_func",
    "min_cut_rematerialization_partition",
    "nop",
    "ts_compile",
]
