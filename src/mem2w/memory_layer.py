"""Persistent query/key/value memory used by Mem2W.

The module has no dependency on Transformers.  ``K`` and ``V`` are persistent
parameters shared across batches and tasks.  They are intentionally not projected
from the current hidden states.  The implementation also exposes separate read and
content parameter groups so the constrained recall stage can stop gradients only to
``V`` and ``W_O`` while keeping the read path differentiable.
"""

from __future__ import annotations

from typing import Any, Iterable

try:  # Keep import errors useful in environments without PyTorch.
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]

    class _MissingTorch:
        def __getattr__(self, name: str) -> Any:
            raise ImportError("PersistentKVMemory requires PyTorch") from exc

    nn = _MissingTorch()  # type: ignore[assignment]


if torch is not None:

    class RMSNorm0(nn.Module):
        """RMSNorm without trainable affine parameters."""

        def __init__(self, eps: float = 1.0e-6) -> None:
            super().__init__()
            if eps <= 0:
                raise ValueError("eps must be positive")
            self.eps = float(eps)

        def forward(self, hidden_states: Tensor) -> Tensor:
            # Compute the statistic in float32 for BF16/FP16 stability, then retain
            # the computation graph and cast back to the input dtype.
            work_dtype = (
                torch.float32
                if hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
                else hidden_states.dtype
            )
            values = hidden_states.to(work_dtype)
            rms = values.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return (values * rms).to(hidden_states.dtype)


    class PersistentKVMemory(nn.Module):
        """Read a persistent bank of ``J`` key/value memory slots.

        Parameters use the names in the plan: ``W_Q[d, d_k]``, ``K[J, d_k]``,
        ``V[J, d_v]`` and ``W_O[d_v, d]``.  ``b=0`` exits before normalization or
        any memory read, which is important for an exact backbone bypass.
        """

        def __init__(
            self,
            hidden_size: int,
            slots: int = 512,
            key_dim: int = 256,
            value_dim: int = 256,
            *,
            rms_eps: float = 1.0e-6,
            memory_dropout: float = 0.0,
        ) -> None:
            super().__init__()
            for name, value in (
                ("hidden_size", hidden_size),
                ("slots", slots),
                ("key_dim", key_dim),
                ("value_dim", value_dim),
            ):
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if memory_dropout < 0 or memory_dropout >= 1:
                raise ValueError("memory_dropout must be in [0, 1)")

            self.hidden_size = hidden_size
            self.slots = slots
            self.key_dim = key_dim
            self.value_dim = value_dim
            self.scale = key_dim**-0.5
            self.rms_norm = RMSNorm0(rms_eps)
            self.W_Q = nn.Parameter(torch.empty(hidden_size, key_dim))
            self.K = nn.Parameter(torch.empty(slots, key_dim))
            self.V = nn.Parameter(torch.empty(slots, value_dim))
            self.W_O = nn.Parameter(torch.empty(value_dim, hidden_size))
            # Checkpoint export uses this explicit marker to distinguish the
            # four deployable memory tensors from the frozen backbone.  A
            # Parameter is a Tensor subclass and safely carries this metadata
            # without changing its state-dict name or optimizer behavior.
            for parameter in (self.W_Q, self.K, self.V, self.W_O):
                parameter.is_memory_parameter = True
            self.dropout = nn.Dropout(memory_dropout) if memory_dropout else nn.Identity()
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.xavier_uniform_(self.W_Q)
            nn.init.normal_(self.K, mean=0.0, std=1.0)
            nn.init.normal_(self.V, mean=0.0, std=0.02)
            nn.init.zeros_(self.W_O)

        def read_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
            """Return the parameters updated by both action and recall reads."""

            return self.W_Q, self.K

        def content_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
            """Return value/output parameters whose recall gradients may be stopped."""

            return self.V, self.W_O

        def named_read_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
            return (("W_Q", self.W_Q), ("K", self.K))

        def named_content_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
            return (("V", self.V), ("W_O", self.W_O))

        @staticmethod
        def _valid_token_mask(attention_mask: Tensor | None, batch: int, length: int) -> Tensor | None:
            """Accept only a 2-D valid-token mask; leave causal masks untouched."""

            if attention_mask is None:
                return None
            if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch, length):
                return None
            return attention_mask.to(dtype=torch.bool)

        def forward(
            self,
            hidden_states: Tensor,
            *,
            b: int | bool = 1,
            attention_mask: Tensor | None = None,
            stop_content_grad: bool = False,
            return_attention: bool = False,
        ) -> Tensor | tuple[Tensor, Tensor]:
            """Apply ``H' = H + b * ((softmax(U W_Q Kᵀ) V) W_O)``.

            ``stop_content_grad`` detaches only the *parameters* ``V`` and ``W_O``;
            the memory output stays attached so gradients to ``W_Q`` and ``K`` pass
            through the attention/read path.  It is therefore safe to use this
            branch for the constrained recall stage.
            """

            # This must be the first operation.  In particular, do not compute a
            # zeroed branch: callers rely on an exact, allocation-free bypass.
            if isinstance(b, Tensor):
                if b.numel() != 1:
                    raise ValueError("b must be a scalar 0/1 value")
                b_scalar = b.detach().item()
                if b_scalar not in (0, 1, False, True):
                    raise ValueError("b must be 0 or 1")
                if not bool(b_scalar):
                    return (hidden_states, None) if return_attention else hidden_states
                b_value = 1
            else:
                if b not in (0, 1, False, True):
                    raise ValueError("b must be 0 or 1")
                if not bool(b):
                    return (hidden_states, None) if return_attention else hidden_states
                b_value = 1
            if hidden_states.ndim != 3:
                raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
            if hidden_states.shape[-1] != self.hidden_size:
                raise ValueError(
                    f"last dimension {hidden_states.shape[-1]} does not match hidden_size {self.hidden_size}"
                )

            batch, length, _ = hidden_states.shape
            normalized = self.rms_norm(hidden_states)
            # FP32 attention scores and memory content are stable for BF16/FP16
            # backbones.  Keep output in the backbone dtype before the residual add.
            work_dtype = (
                torch.float32
                if hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
                else hidden_states.dtype
            )
            query = normalized.to(work_dtype).matmul(self.W_Q.to(work_dtype))
            keys = self.K.to(work_dtype)
            values = self.V.detach().to(work_dtype) if stop_content_grad else self.V.to(work_dtype)
            output_projection = (
                self.W_O.detach().to(work_dtype) if stop_content_grad else self.W_O.to(work_dtype)
            )
            logits = torch.matmul(query, keys.transpose(0, 1)).mul(self.scale)
            attention = torch.softmax(logits, dim=-1)
            readout = torch.matmul(attention, values)
            residual = self.dropout(torch.matmul(readout, output_projection)).to(hidden_states.dtype)

            valid_mask = self._valid_token_mask(attention_mask, batch, length)
            if valid_mask is not None:
                residual = residual * valid_mask.unsqueeze(-1).to(residual.dtype)
            result = hidden_states + b_value * residual
            return (result, attention) if return_attention else result


else:  # pragma: no cover - import-time fallback for machines without torch

    class RMSNorm0:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("RMSNorm0 requires PyTorch")

    class PersistentKVMemory:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PersistentKVMemory requires PyTorch")


# Friendly aliases used by small downstream experiments.  The explicit class name
# remains the canonical one in checkpoints and documentation.
MemoryLayer = PersistentKVMemory
Mem2WMemoryLayer = PersistentKVMemory

__all__ = ["RMSNorm0", "PersistentKVMemory", "MemoryLayer", "Mem2WMemoryLayer"]
