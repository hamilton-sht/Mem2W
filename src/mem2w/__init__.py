"""Mem2W v0.1 package."""

__version__ = "0.1.0"

from .config import (
    MemoryConfig,
    load_config,
    load_memory_config,
    save_memory_config,
    validate_config,
)
from .memory_layer import MemoryLayer, PersistentKVMemory
from .qwen_integration import (
    Mem2WModelAdapter,
    MemoryAfterDecoderBlock,
    attach_memory,
    get_memory_module,
    load_mem2w_model,
    load_qwen_with_memory,
    memory_parameters,
    set_memory_enabled,
    set_stop_content_grad,
)
from .checkpointing import (
    load_memory_checkpoint,
    model_parameter_fingerprint,
    save_memory_checkpoint,
)

__all__ = [
    "MemoryConfig",
    "MemoryLayer",
    "PersistentKVMemory",
    "MemoryAfterDecoderBlock",
    "Mem2WModelAdapter",
    "attach_memory",
    "get_memory_module",
    "load_mem2w_model",
    "load_qwen_with_memory",
    "memory_parameters",
    "set_memory_enabled",
    "set_stop_content_grad",
    "load_memory_checkpoint",
    "model_parameter_fingerprint",
    "save_memory_checkpoint",
    "load_config",
    "load_memory_config",
    "save_memory_config",
    "validate_config",
    "__version__",
]
