"""Controlled insertion of a Mem2W layer into a Qwen decoder.

The integration deliberately wraps one existing decoder block instead of changing
``num_hidden_layers`` or registering an opaque forward hook.  This keeps the
original block identities and cache calling convention intact.  Transformers is an
optional dependency and is imported only by ``load_qwen_with_memory``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Mapping as TypingMapping

from .config import MemoryConfig
from .memory_layer import PersistentKVMemory


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("Qwen integration requires PyTorch") from exc
    return torch, nn


def _find_decoder_layers(model: Any) -> tuple[Any, str]:
    """Find the original decoder ``ModuleList`` without assuming one Qwen class."""

    candidates = (
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        (
            "model.language_model.layers",
            getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        ),
        (
            "language_model.layers",
            getattr(getattr(model, "language_model", None), "layers", None),
        ),
        ("layers", getattr(model, "layers", None)),
    )
    for path, layers in candidates:
        if layers is not None and hasattr(layers, "__getitem__") and hasattr(layers, "__len__"):
            return layers, path
    raise AttributeError(
        "Could not find decoder layers; expected model.model.layers, "
        "model.language_model.layers, or model.layers"
    )


def _replace_hidden(output: Any, hidden_states: Any) -> Any:
    """Replace the first decoder output while preserving its tuple/list contract."""

    if isinstance(output, tuple):
        if not output:
            raise TypeError("decoder block returned an empty tuple")
        return (hidden_states, *output[1:])
    if isinstance(output, list):
        if not output:
            raise TypeError("decoder block returned an empty list")
        return [hidden_states, *output[1:]]
    if isinstance(output, Mapping):
        # Decoder blocks normally return tuples; this branch supports small dummy
        # models used in tests without changing mapping keys.
        result = dict(output)
        first_key = "hidden_states" if "hidden_states" in result else next(iter(result), None)
        if first_key is None:
            raise TypeError("decoder block returned an empty mapping")
        result[first_key] = hidden_states
        return type(output)(result) if type(output) is not dict else result
    return hidden_states


def _first_hidden(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        if not output:
            raise TypeError("decoder block returned an empty sequence")
        return output[0]
    if isinstance(output, Mapping):
        return output.get("hidden_states", next(iter(output.values()), None))
    return output


def _valid_attention_mask(value: Any) -> Any:
    """Only pass a padding mask to memory; causal 4-D masks are not padding masks."""

    if value is None or getattr(value, "ndim", 0) != 2:
        return None
    # Positional IDs are also commonly shaped [B, L] and can arrive positionally
    # at a decoder block.  Only a binary 0/1 tensor is a valid padding mask here.
    try:
        if getattr(value, "dtype", None) is not None and str(value.dtype) == "torch.bool":
            return value
        if bool(value.numel()) and bool(value.detach().min().item() >= 0) and bool(
            value.detach().max().item() <= 1
        ):
            return value
    except (AttributeError, RuntimeError, ValueError):
        return None
    return None


try:  # Importing this module should still work when only config tooling is installed.
    import torch as _torch_import
    from torch import nn as _nn
except ImportError:  # pragma: no cover - depends on environment
    _torch_import = None
    _nn = None

_MemoryBlockBase = _nn.Module if _nn is not None else object


class MemoryAfterDecoderBlock(_MemoryBlockBase):
    """A decoder block plus a residual memory read.

    The wrapper defaults to the enabled path.  ``memory_enabled=False`` sets ``b=0``
    and the memory module exits before doing any work.  Reserved keyword arguments
    are accepted for direct testing and integration adapters; ordinary Qwen calls
    pass through unchanged to the wrapped block.
    """

    def __init__(self, block: Any, memory: PersistentKVMemory) -> None:
        if _nn is None:  # pragma: no cover - depends on environment
            raise ImportError("Qwen integration requires PyTorch")
        super().__init__()
        self.block = block
        self.memory = memory
        self.memory_enabled = True
        self.stop_content_grad = False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        memory_b = kwargs.pop("memory_b", None)
        memory_enabled = kwargs.pop("memory_enabled", None)
        stop_content_grad = kwargs.pop("stop_content_grad", None)
        output = self.block(*args, **kwargs)
        hidden_states = _first_hidden(output)
        if hidden_states is None:
            raise TypeError("decoder block did not return hidden states")
        if memory_b is None:
            memory_b = self.memory_enabled if memory_enabled is None else memory_enabled
        if stop_content_grad is None:
            stop_content_grad = self.stop_content_grad
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None and len(args) > 1:
            attention_mask = args[1]
        attention_mask = _valid_attention_mask(attention_mask)
        memory_output = self.memory(
            hidden_states,
            b=memory_b,
            attention_mask=attention_mask,
            stop_content_grad=bool(stop_content_grad),
        )
        return _replace_hidden(output, memory_output)


class Mem2WModelAdapter(_MemoryBlockBase):
    """Forward adapter that exposes the explicit per-branch stop-grad flag.

    Hugging Face model classes do not accept a project-specific
    ``stop_content_grad`` keyword.  This adapter consumes that keyword, sets it on
    the wrapped block for the duration of one forward, and restores the previous
    value afterwards.  All ordinary model attributes (``config``, ``generate``,
    ``model`` and so on) are delegated to the loaded checkpoint.
    """

    def __init__(self, base_model: Any) -> None:
        if _nn is None:  # pragma: no cover - depends on environment
            raise ImportError("Qwen integration requires PyTorch")
        super().__init__()
        self.base_model = base_model

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            base_model = super().__getattr__("base_model")
            return getattr(base_model, name)

    def forward(
        self,
        *args: Any,
        stop_content_grad: bool = False,
        memory_b: int | bool = 1,
        **kwargs: Any,
    ) -> Any:
        wrappers = _memory_wrappers(self.base_model)
        previous = [(wrapper.memory_enabled, wrapper.stop_content_grad) for wrapper in wrappers]
        try:
            for wrapper in wrappers:
                wrapper.memory_enabled = bool(memory_b)
                wrapper.stop_content_grad = bool(stop_content_grad)
            return self.base_model(*args, **kwargs)
        finally:
            for wrapper, (enabled, previous_stop) in zip(wrappers, previous):
                wrapper.memory_enabled = enabled
                wrapper.stop_content_grad = previous_stop

    def memory_parameters(self) -> list[Any]:
        return memory_parameters(self.base_model)


def attach_memory(
    model: Any,
    memory: PersistentKVMemory | None = None,
    config: MemoryConfig | None = None,
    *,
    insertion_index: int | None = None,
) -> Any:
    """Attach one memory layer after the configured decoder block.

    All parameters belonging to the original model are frozen before the wrapper is
    installed.  The new memory parameters retain ``requires_grad=True`` and are the
    only parameters returned by :func:`memory_parameters`.
    """

    torch, _ = _torch_modules()
    config = config or MemoryConfig()
    layers, path = _find_decoder_layers(model)
    index = config.insertion_index if insertion_index is None else int(insertion_index)
    if index < 0 or index >= len(layers):
        raise IndexError(f"insertion_index {index} is outside {len(layers)} decoder blocks")
    if isinstance(layers[index], MemoryAfterDecoderBlock):
        raise ValueError(f"memory is already attached at decoder block index {index}")

    model_config = getattr(model, "config", None)
    model_hidden_size = getattr(model_config, "hidden_size", None)
    if model_hidden_size is None:
        text_config = getattr(model_config, "text_config", None)
        model_hidden_size = getattr(text_config, "hidden_size", None)
    hidden_size = int(model_hidden_size or config.hidden_size)
    if hidden_size != config.hidden_size:
        raise ValueError(
            f"memory hidden_size={config.hidden_size} does not match base model hidden_size={hidden_size}"
        )
    if memory is None:
        memory = PersistentKVMemory(
            hidden_size=config.hidden_size,
            slots=config.slots,
            key_dim=config.key_dim,
            value_dim=config.value_dim,
            rms_eps=config.rms_eps,
            memory_dropout=config.memory_dropout,
        )
    elif memory.hidden_size != hidden_size:
        raise ValueError("provided memory hidden_size does not match base model")

    # Freeze only what existed before insertion.  Assigning the memory wrapper after
    # this loop leaves its newly created parameters trainable.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # A caller may pass a memory module that was previously attached to a frozen
    # module or loaded under a global ``requires_grad_(False)`` context.  The
    # memory branch is the sole trainable component in this integration, so make
    # that invariant explicit for both newly created and supplied modules.
    for parameter in memory.parameters():
        parameter.requires_grad_(True)
    layers[index] = MemoryAfterDecoderBlock(layers[index], memory)

    # Bypass nn.Module.__setattr__ so the same memory is not registered a second time
    # under the model root (the wrapper is its sole state_dict owner).
    object.__setattr__(model, "_mem2w_memory", memory)
    object.__setattr__(model, "_mem2w_insertion_index", index)
    object.__setattr__(model, "_mem2w_layers_path", path)
    object.__setattr__(model, "_mem2w_wrapper_version", "0.1")
    # Keep the base layer count untouched; this is an explicit invariant in v0.1.
    object.__setattr__(model, "_mem2w_original_num_hidden_layers", len(layers))
    if model_config is not None and hasattr(model_config, "num_hidden_layers"):
        if int(model_config.num_hidden_layers) != len(layers):
            raise ValueError("model config num_hidden_layers does not match decoder layer count")
    del torch  # make it explicit that no inference-mode context was used here
    return model


def get_memory_module(model: Any) -> PersistentKVMemory:
    """Return the attached memory module or raise a useful error."""

    memory = getattr(model, "_mem2w_memory", None)
    if memory is None:
        layers, _ = _find_decoder_layers(model)
        memory = next(
            (layer.memory for layer in layers if isinstance(layer, MemoryAfterDecoderBlock)),
            None,
        )
    if memory is None:
        raise ValueError("model has no attached Mem2W memory")
    return memory


def _memory_wrappers(model: Any) -> list[MemoryAfterDecoderBlock]:
    layers, _ = _find_decoder_layers(model)
    return [layer for layer in layers if isinstance(layer, MemoryAfterDecoderBlock)]


def set_memory_enabled(model: Any, enabled: bool) -> None:
    """Enable or bypass all attached memory wrappers (callers must clear caches)."""

    wrappers = _memory_wrappers(model)
    if not wrappers:
        raise ValueError("model has no attached Mem2W memory")
    for wrapper in wrappers:
        wrapper.memory_enabled = bool(enabled)


def set_stop_content_grad(model: Any, enabled: bool) -> None:
    """Set the default recall content-gradient policy on attached wrappers."""

    wrappers = _memory_wrappers(model)
    if not wrappers:
        raise ValueError("model has no attached Mem2W memory")
    for wrapper in wrappers:
        wrapper.stop_content_grad = bool(enabled)


def memory_parameters(model: Any) -> list[Any]:
    """Return the trainable memory parameters for an optimizer."""

    memory = get_memory_module(model)
    return [parameter for parameter in memory.parameters() if parameter.requires_grad]


def load_qwen_with_memory(
    model_id: str = "Qwen/Qwen3.5-9B",
    *,
    revision: str | None = None,
    memory_config: MemoryConfig | None = None,
    loader: Any | None = None,
    **from_pretrained_kwargs: Any,
) -> Any:
    """Load an official Qwen checkpoint and attach Mem2W.

    No model constructor or random fallback is used.  ``loader`` is an optional
    ``AutoModelForCausalLM``-compatible object for tests and pinned local loaders;
    production callers should leave it unset so Transformers performs
    ``from_pretrained`` resolution.
    """

    if loader is None:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "load_qwen_with_memory requires Transformers; install the pinned version "
                "from the project lock before loading Qwen3.5"
            ) from exc
        loader = AutoModelForCausalLM
    if not hasattr(loader, "from_pretrained"):
        raise TypeError("loader must expose from_pretrained")
    kwargs = dict(from_pretrained_kwargs)
    if revision is not None:
        kwargs["revision"] = revision
    model = loader.from_pretrained(model_id, **kwargs)
    return Mem2WModelAdapter(attach_memory(model, config=memory_config))


def load_mem2w_model(config: TypingMapping[str, Any]) -> Mem2WModelAdapter:
    """Load the model described by the complete project YAML config.

    This compatibility entry point is used by ``mem2w.train``.  It resolves the
    checkpoint through ``from_pretrained`` and never constructs a random fallback.
    """

    base_config = config.get("base_model", {})
    if not isinstance(base_config, TypingMapping):
        raise ValueError("base_model config must be a mapping")
    model_id = base_config.get("model_id", "Qwen/Qwen3.5-9B")
    revision = base_config.get("revision")
    memory_config = MemoryConfig.from_mapping(config)
    kwargs: dict[str, Any] = {}
    # The project default is BF16.  Resolve the string through torch so the loader
    # receives a dtype object and errors early for an unsupported declared value.
    dtype_name = base_config.get("backbone_dtype")
    if dtype_name is not None:
        try:
            import torch

            dtype = getattr(torch, str(dtype_name))
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"unsupported backbone_dtype: {dtype_name!r}") from exc
        kwargs["torch_dtype"] = dtype
    return load_qwen_with_memory(
        str(model_id), revision=revision, memory_config=memory_config, **kwargs
    )


__all__ = [
    "MemoryAfterDecoderBlock",
    "Mem2WModelAdapter",
    "attach_memory",
    "get_memory_module",
    "load_qwen_with_memory",
    "load_mem2w_model",
    "memory_parameters",
    "set_memory_enabled",
    "set_stop_content_grad",
]
