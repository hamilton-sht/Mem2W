"""Small native configuration object for the Mem2W model extension."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass
class Mem2WConfig:
    hidden_size: int = 4096
    slots: int = 512
    key_dim: int = 256
    value_dim: int = 256
    insertion_index: int = 15
    rms_eps: float = 1e-6
    memory_dropout: float = 0.0
    # Bound the temporary [sequence, slots] memory projection. This changes
    # only execution tiling; it never changes the token sequence or target.
    compute_chunk_size: int = 16384
    # Detach the content/value path while retaining gradients for the query and
    # key path.  This is part of the checkpointed configuration so a resumed
    # run has exactly the same objective semantics as the original run.
    stop_content_grad: bool = False

    def __post_init__(self):
        for name in ('hidden_size', 'slots', 'key_dim', 'value_dim'):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.insertion_index < 0:
            raise ValueError('insertion_index must be non-negative')
        if self.rms_eps <= 0 or not 0 <= self.memory_dropout < 1:
            raise ValueError('invalid RMSNorm epsilon or memory dropout')
        if not isinstance(self.compute_chunk_size, int) or self.compute_chunk_size <= 0:
            raise ValueError('compute_chunk_size must be a positive integer')

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> 'Mem2WConfig':
        if value is None:
            return cls()
        if isinstance(value.get('memory'), Mapping):
            value = value['memory']
        aliases = {'query_key_dim': 'key_dim', 'insertion_index_0based': 'insertion_index'}
        fields = set(cls.__dataclass_fields__)
        normalized = {}
        for key, item in value.items():
            if key == 'insert_after_block_1based':
                normalized['insertion_index'] = int(item) - 1
            else:
                normalized[aliases.get(key, key)] = item
        return cls(**{key: item for key, item in normalized.items() if key in fields})

    @classmethod
    def from_json(cls, path: Optional[str]) -> 'Mem2WConfig':
        if not path:
            return cls()
        value = json.loads(Path(path).expanduser().read_text(encoding='utf-8'))
        if not isinstance(value, Mapping):
            raise ValueError('mem2w_config must contain an object')
        return cls.from_mapping(value)
