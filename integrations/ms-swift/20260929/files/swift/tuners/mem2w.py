"""Native ms-swift tuner for the Mem2W persistent memory parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from swift.tuner_plugin.base import Tuner
from swift.mem2w import Mem2WConfig, attach_memory, freeze_memory_only, get_memory_module


def _config_from_args(args, model) -> Mem2WConfig:
    model_config = getattr(model, 'config', None)
    text_config = getattr(model_config, 'text_config', None)
    hidden_size = getattr(model_config, 'hidden_size', None) or getattr(text_config, 'hidden_size', None)
    if hidden_size is None:
        hidden_size = 4096
    return Mem2WConfig(
        hidden_size=int(hidden_size),
        slots=args.mem2w_slots,
        key_dim=args.mem2w_key_dim,
        value_dim=args.mem2w_value_dim,
        insertion_index=args.mem2w_insertion_index,
        rms_eps=args.mem2w_rms_eps,
        memory_dropout=args.mem2w_dropout,
        stop_content_grad=args.mem2w_stop_content_grad,
        compute_chunk_size=args.mem2w_compute_chunk_size,
    )


def _normalize_state(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        marker = '._mem2w_memory.'
        if marker in key:
            normalized[key.rsplit(marker, 1)[-1]] = value
        elif key.startswith('_mem2w_memory.'):
            normalized[key[len('_mem2w_memory.'):]] = value
        elif key.startswith('memory.'):
            normalized[key[len('memory.'):]] = value
        elif key in {'W_Q', 'K', 'V', 'W_O'}:
            normalized[key] = value
    return normalized


def _materialize_memory_state(model, state_dict):
    """Extract and validate exactly the four Mem2W tensors.

    ``state_dict`` is supplied by Trainer/Accelerate and may be a sharded or
    incomplete view under ZeRO/FSDP.  Silently falling back to the local module
    in that case can save rank-local garbage or stale weights.  We therefore
    fail closed and require the caller to configure a full-state gather before
    invoking the tuner save path.
    """
    expected = set(get_memory_module(model).state_dict())
    normalized = _normalize_state(state_dict)
    missing = expected - set(normalized)
    extra = set(normalized) - expected
    if missing or extra:
        raise RuntimeError(
            'Mem2W checkpoint requires a complete gathered state dict; '
            f'missing={sorted(missing)}, unexpected={sorted(extra)}. '
            'Configure FSDP/DeepSpeed full-state gathering before saving.')
    memory_state = {}
    module_state = get_memory_module(model).state_dict()
    for key in expected:
        value = normalized[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'Mem2W state {key!r} is not a torch.Tensor: {type(value)!r}')
        if value.is_meta:
            raise RuntimeError(f'Mem2W state {key!r} is still on the meta device')
        if tuple(value.shape) != tuple(module_state[key].shape):
            raise RuntimeError(
                f'Mem2W state {key!r} has shape {tuple(value.shape)}, '
                f'expected {tuple(module_state[key].shape)}')
        memory_state[key] = value.detach().cpu().contiguous()
    return memory_state


class Mem2WTuner(Tuner):
    """Attach Mem2W before Trainer/Accelerate wrapping and save only its tensors."""

    @staticmethod
    def prepare_model(args, model):
        model = attach_memory(model, _config_from_args(args, model))
        return freeze_memory_only(model)

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        if state_dict is None:
            state_dict = model.state_dict()
        memory_state = _materialize_memory_state(model, state_dict)
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(memory_state, str(output / 'memory.safetensors'))
        else:
            torch.save(memory_state, output / 'memory.pt')
        config = getattr(model, '_mem2w_config', None)
        if config is None:
            config = Mem2WConfig(
                hidden_size=get_memory_module(model).hidden_size,
                slots=get_memory_module(model).slots,
                key_dim=get_memory_module(model).key_dim,
                value_dim=get_memory_module(model).value_dim,
                insertion_index=getattr(model, '_mem2w_insertion_index', 15),
            )
        (output / 'mem2w_config.json').write_text(
            json.dumps(config.__dict__, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs):
        checkpoint = Path(model_id)
        config_path = checkpoint / 'mem2w_config.json'
        if hasattr(model, '_mem2w_memory') or getattr(model, '_mem2w_memory_layer', None) is not None:
            memory = get_memory_module(model)
        else:
            config = Mem2WConfig.from_json(str(config_path) if config_path.is_file() else None)
            memory = get_memory_module(attach_memory(model, config))
        safetensors_path = checkpoint / 'memory.safetensors'
        if safetensors_path.is_file():
            from safetensors.torch import load_file

            state = load_file(str(safetensors_path), device='cpu')
        else:
            state = torch.load(checkpoint / 'memory.pt', map_location='cpu', weights_only=True)
        memory_state = _normalize_state(state)
        expected = set(memory.state_dict())
        if set(memory_state) != expected:
            raise RuntimeError(
                f'Invalid Mem2W checkpoint {checkpoint}: '
                f'missing={sorted(expected - set(memory_state))}, '
                f'unexpected={sorted(set(memory_state) - expected)}')
        memory.load_state_dict(memory_state, strict=True)
        if kwargs.get('is_trainable', True):
            return freeze_memory_only(model)
        model.requires_grad_(False)
        return model

    @staticmethod
    def load_checkpoint(model: torch.nn.Module, checkpoint: str, **kwargs) -> torch.nn.Module:
        """Load a native Mem2W checkpoint during Trainer resume.

        Transformers' generic Trainer loader only knows model/PEFT filenames;
        Mem2W checkpoints intentionally contain ``memory.safetensors``.  This
        entry point lets ms-swift's trainer mixin restore the adapter while
        leaving optimizer/scheduler state to the standard Trainer path.
        """
        return Mem2WTuner.from_pretrained(model, checkpoint, **kwargs)
