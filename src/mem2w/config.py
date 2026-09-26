"""Configuration loading and invariant checks for Mem2W."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a configuration violates a v0.1 invariant."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config without silently filling in model-specific defaults."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("PyYAML is required to load Mem2W configs") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigError(f"config must be a mapping: {path}")
    validate_config(value)
    return value


def _get(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the constraints that must hold before model construction."""

    if config.get("project") != "Mem2W":
        raise ConfigError("project must be Mem2W")
    if _get(config, "base_model", "freeze_all_original_parameters") is not True:
        raise ConfigError("the original backbone must be frozen")
    if _get(config, "memory", "insert_after_block_1based") != 16:
        raise ConfigError("v0.1 inserts memory after block 16")
    if _get(config, "memory", "insertion_index_0based") != 15:
        raise ConfigError("insertion_index_0based must be 15")
    if _get(config, "memory", "persistent_independent_kv") is not True:
        raise ConfigError("K/V must be independent persistent Parameters")
    if _get(config, "memory", "slots") != 512:
        raise ConfigError("v0.1 defaults to 512 persistent memory slots")
    if _get(config, "memory", "query_key_dim") != 256 or _get(config, "memory", "value_dim") != 256:
        raise ConfigError("v0.1 defaults to d_k=d_v=256")
    if _get(config, "memory", "query_projection_bias") is not False:
        raise ConfigError("W_Q must not use a bias")
    if _get(config, "memory", "output_projection_bias") is not False:
        raise ConfigError("W_O must not use a bias")
    if _get(config, "memory", "affine_norm") is not False:
        raise ConfigError("RMSNorm0 must not add affine parameters")
    if _get(config, "loss", "action") != "masked_autoregressive_cross_entropy":
        raise ConfigError("action loss must be masked autoregressive CE")
    if _get(config, "loss", "recall") != "payload_completion_cross_entropy":
        raise ConfigError("recall loss must be payload completion CE")
    warmup = float(_get(config, "schedule", "warmup_stage_fraction", default=-1))
    if not 0.0 < warmup < 1.0:
        raise ConfigError("warmup_stage_fraction must lie strictly between 0 and 1")
    if _get(config, "training", "packing") is not False:
        raise ConfigError("cross-episode packing is not validated in v0.1")
    if _get(config, "training", "use_cache") is not False:
        raise ConfigError("training must run with use_cache=false")


@dataclass(frozen=True)
class RuntimeLock:
    """Small serializable record embedded in every training/checkpoint manifest."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    chat_template_hash: str
    data_hash: str
    teacher_snapshot_hash: str
    code_revision: str

    def as_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_hash": self.chat_template_hash,
            "data_hash": self.data_hash,
            "teacher_snapshot_hash": self.teacher_snapshot_hash,
            "code_revision": self.code_revision,
        }


@dataclass
class MemoryConfig:
    """Standalone memory-layer configuration, independent of Transformers.

    The project-level YAML validator above remains the source of truth for complete
    experiment configs.  This compact config is used by ``memory_config.json`` in a
    deployable memory package and can be loaded without PyYAML.
    """

    hidden_size: int = 4096
    slots: int = 512
    key_dim: int = 256
    value_dim: int = 256
    insertion_index: int = 15
    rms_eps: float = 1.0e-6
    query_projection_bias: bool = False
    output_projection_bias: bool = False
    persistent_independent_kv: bool = True
    trainable_gate: bool = False
    memory_dropout: float = 0.0
    slot_rope: bool = False
    read_frequency: str = "every_token"
    output_init: str = "zeros"
    version: str = "0.1"

    def __post_init__(self) -> None:
        for name in ("hidden_size", "slots", "key_dim", "value_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ConfigError(f"{name} must be a positive integer, got {value!r}")
        if self.insertion_index < 0:
            raise ConfigError("insertion_index must be zero-based and non-negative")
        if self.rms_eps <= 0:
            raise ConfigError("rms_eps must be positive")
        if not 0 <= self.memory_dropout < 1:
            raise ConfigError("memory_dropout must be in [0, 1)")
        if self.query_projection_bias or self.output_projection_bias:
            raise ConfigError("v0.1 projections do not use bias")
        if not self.persistent_independent_kv:
            raise ConfigError("K and V must be independent persistent parameters")
        if self.trainable_gate:
            raise ConfigError("v0.1 has no trainable memory gate")
        if self.slot_rope:
            raise ConfigError("v0.1 does not apply RoPE to memory slots")
        if self.read_frequency != "every_token":
            raise ConfigError("v0.1 memory is read for every input token")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MemoryConfig":
        """Build from flat fields or the plan's nested ``memory`` mapping."""

        if "memory" in values and isinstance(values["memory"], Mapping):
            values = values["memory"]  # type: ignore[assignment]
        aliases = {
            "query_key_dim": "key_dim",
            "insertion_index_0based": "insertion_index",
        }
        normalized: dict[str, Any] = {}
        for key, value in values.items():
            if key == "insert_after_block_1based":
                normalized["insertion_index"] = int(value) - 1
            else:
                normalized[aliases.get(key, key)] = value
        fields = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in normalized.items() if key in fields})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_memory_config(path: str | Path) -> MemoryConfig:
    """Load a standalone JSON ``memory_config.json``."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return MemoryConfig.from_mapping(json.load(handle))


def save_memory_config(config: MemoryConfig, path: str | Path) -> None:
    """Write a deterministic, human-readable memory config."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
