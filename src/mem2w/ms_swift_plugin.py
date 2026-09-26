"""ms-swift registration for the minimal Mem2W action/recall pilot.

The stock Qwen3.5 registration loads the official model but has no knowledge of
the extra memory module.  Passing this file through ``--external_plugins``
replaces only the loader for the existing ``qwen3_5`` model type.  The loader
freezes the loaded backbone, attaches one Mem2W layer, and optionally restores
``memory.safetensors``.  The project runner performs the final freeze after
ms-swift's ``tuner_type=full`` preparation; calling stock ``swift sft`` alone
is a data/template smoke path and may reopen the backbone.

This plugin deliberately does not implement the later paired W/C gradient
schedule.  Use one independent ``--mode action`` or ``--mode recall`` config
for the first end-to-end smoke run.

Environment variables:

``MEM2W_MEMORY_CONFIG``
    Optional JSON/YAML file with a standalone ``MemoryConfig`` mapping.
``MEM2W_MEMORY_CHECKPOINT``
    Optional directory containing ``memory.safetensors``.  The file is loaded
    after the module is attached and before the Trainer is created.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping


def _load_memory_config() -> Any:
    from mem2w.config import MemoryConfig

    path = os.environ.get("MEM2W_MEMORY_CONFIG")
    if not path:
        return MemoryConfig()
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"MEM2W_MEMORY_CONFIG does not exist: {source}")
    if source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError("PyYAML is required for a YAML MEM2W_MEMORY_CONFIG") from exc
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    else:
        value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"MEM2W_MEMORY_CONFIG must contain an object: {source}")
    return MemoryConfig.from_mapping(value)


def _restore_memory(model: Any) -> None:
    checkpoint = os.environ.get("MEM2W_MEMORY_CHECKPOINT")
    if not checkpoint:
        return
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError("safetensors is required for MEM2W_MEMORY_CHECKPOINT") from exc
    from mem2w.qwen_integration import get_memory_module

    path = Path(checkpoint).expanduser()
    if path.is_dir():
        path = path / "memory.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"MEM2W_MEMORY_CHECKPOINT does not exist: {path}")
    state = load_file(str(path), device="cpu")
    memory = get_memory_module(model)
    expected = memory.state_dict()
    # Accept both the standalone memory keys and keys copied from a wrapped
    # module (e.g. ``memory.W_Q``) to make checkpoint migration explicit.
    normalized = {}
    for key, value in state.items():
        if ".memory." in key:
            key = key.rsplit(".memory.", 1)[-1]
        normalized[key.removeprefix("memory.")] = value
    missing = sorted(set(expected) - set(normalized))
    unexpected = sorted(set(normalized) - set(expected))
    if missing or unexpected:
        raise RuntimeError(f"memory checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    memory.load_state_dict(normalized, strict=True)


def _model_memory_config(model: Any) -> Any:
    """Resolve the memory width after ms-swift has loaded the base model.

    The target 9B checkpoint is width 4096.  For a small Qwen3.5 smoke
    checkpoint, such as the local 4B fixture, use its text width when no
    explicit memory config was supplied.  An explicitly supplied config is
    never silently changed; a mismatch then fails in ``attach_memory``.
    """

    config = _load_memory_config()
    if os.environ.get("MEM2W_MEMORY_CONFIG"):
        return config
    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None) or getattr(model_config, "hidden_size", None)
    if hidden_size is not None:
        config.hidden_size = int(hidden_size)
    return config


def _register() -> None:
    # Imports are delayed until ms-swift imports this external plugin.  This
    # keeps dataset conversion and config generation usable without ms-swift.
    from swift.model import MODEL_MAPPING, ModelMeta, register_model
    from swift.model.models.qwen import Qwen3_5Loader
    from mem2w.qwen_integration import attach_memory

    base_meta = MODEL_MAPPING.get("qwen3_5")
    if base_meta is None:  # pragma: no cover - depends on ms-swift registry
        raise RuntimeError("ms-swift qwen3_5 model registration is unavailable")

    class Mem2WQwen3_5Loader(Qwen3_5Loader):
        def get_model(
            self,
            model_dir: str,
            config: Any,
            processor: Any,
            model_kwargs: dict[str, Any],
        ) -> Any:
            model = super().get_model(model_dir, config, processor, model_kwargs)
            attach_memory(model, config=_model_memory_config(model))
            _restore_memory(model)
            return model

    meta = copy.deepcopy(base_meta)
    assert isinstance(meta, ModelMeta)
    meta.loader = Mem2WQwen3_5Loader
    # In recent ms-swift releases the built-in registry has already resolved
    # ``model_arch`` from an enum/string into a ``MultiModelKeys`` instance.
    # Calling ``register_model`` on that copied metadata tries to resolve the
    # object a second time and raises ``TypeError: unhashable type``.  Keep the
    # resolved architecture and replace only the loader in that case; older
    # releases still need the public registration helper.
    if hasattr(meta.model_arch, "arch_name"):
        MODEL_MAPPING[meta.model_type] = meta
    else:
        register_model(meta, exist_ok=True)


_register()
