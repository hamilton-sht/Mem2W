"""Attach Mem2W to a loaded ms-swift/Transformers decoder without replacing layers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
import inspect
from typing import Optional

import torch

from .config import Mem2WConfig
from .memory import PersistentKVMemory


def _decoder_layers(model):
    candidates = (
        getattr(getattr(model, 'model', None), 'layers', None),
        getattr(getattr(getattr(model, 'model', None), 'language_model', None), 'layers', None),
        getattr(getattr(model, 'language_model', None), 'layers', None),
        getattr(model, 'layers', None),
    )
    for layers in candidates:
        if layers is not None and hasattr(layers, '__len__') and hasattr(layers, '__getitem__'):
            return layers
    raise AttributeError('could not find decoder layers on the loaded model')


def _first_hidden(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, Mapping):
        return output.get('hidden_states', next(iter(output.values())))
    return output


def _replace_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    if isinstance(output, Mapping):
        key = 'hidden_states' if 'hidden_states' in output else next(iter(output))
        # ModelOutput (used by Transformers decoder blocks) is a dict subclass
        # with attribute access.  Converting it to a plain dict silently drops
        # that contract and can break downstream layers.  Preserve the concrete
        # mapping type whenever possible, with a namedtuple-style fallback for
        # immutable mapping outputs.
        try:
            result = copy(output)
            result[key] = hidden
            return result
        except (TypeError, AttributeError, KeyError):
            if hasattr(output, '_replace'):
                return output._replace(**{key: hidden})
            try:
                return type(output)(**{**dict(output), key: hidden})
            except Exception as exc:
                raise TypeError(f'cannot preserve decoder output mapping type {type(output)!r}') from exc
    return hidden


def _is_binary_2d_mask(candidate, shape):
    """Return whether *candidate* is an actual 2-D padding/attention mask.

    Decoder hooks receive positional arguments whose second slot is normally
    ``attention_mask`` but can be ``position_ids`` for model variants.  Both can
    be rank-2 tensors, so rank/shape alone is not a safe discriminator.  A
    binary mask check prevents position ids from being interpreted as a mask.
    """
    if candidate is None or not isinstance(candidate, torch.Tensor):
        return False
    if candidate.ndim != 2 or tuple(candidate.shape[:2]) != tuple(shape[:2]):
        return False
    if candidate.dtype == torch.bool:
        return True
    if not candidate.is_floating_point() and candidate.dtype not in (torch.uint8, torch.int8, torch.int16,
                                                                        torch.int32, torch.int64):
        return False
    # Avoid materializing a large unique tensor.  Attention masks are finite and
    # bounded to {0, 1}; position ids generally contain values greater than 1.
    probe = candidate.detach()
    if probe.numel() == 0 or (probe.is_floating_point() and not torch.isfinite(probe).all()):
        return False
    return bool((probe.min() >= 0).item() and (probe.max() <= 1).item())


def _attention_mask_positional_index(layer):
    try:
        parameters = list(inspect.signature(layer.forward).parameters.values())
        for index, parameter in enumerate(parameters):
            if parameter.name == 'attention_mask':
                return index
        return None
    except (TypeError, ValueError):
        return None


def _memory_hook(layer, args, kwargs, output):
    hidden = _first_hidden(output)
    attention_mask = kwargs.get('attention_mask')
    if not _is_binary_2d_mask(attention_mask, hidden.shape):
        attention_mask = None
    if attention_mask is None:
        # Prefer the decoder signature over a value-only heuristic.  Qwen3.5
        # puts ``position_embeddings`` before ``attention_mask``; looking only
        # at args[1] would miss its mask (or confuse it with position ids).
        positional_index = getattr(layer, '_mem2w_attention_arg_index', None)
        if positional_index is not None and positional_index < len(args):
            candidate = args[positional_index]
            if _is_binary_2d_mask(candidate, hidden.shape):
                attention_mask = candidate
        # The fallback is retained for wrappers exposing only *args/**kwargs.
        elif len(args) > 1 and _is_binary_2d_mask(args[1], hidden.shape):
            attention_mask = args[1]
    hidden = layer._mem2w_memory(
        hidden,
        enabled=layer._mem2w_enabled,
        attention_mask=attention_mask,
        stop_content_grad=layer._mem2w_stop_content_grad,
    )
    return _replace_hidden(output, hidden)


def attach_memory(model, config: Optional[Mem2WConfig] = None):
    layers = _decoder_layers(model)
    model_config = getattr(model, 'config', None)
    text_config = getattr(model_config, 'text_config', None)
    hidden_size = getattr(model_config, 'hidden_size', None) or getattr(text_config, 'hidden_size', None)
    if config is None:
        config = Mem2WConfig(hidden_size=int(hidden_size)) if hidden_size is not None else Mem2WConfig()
    if config.insertion_index >= len(layers):
        raise IndexError(f'insertion_index {config.insertion_index} is outside {len(layers)} decoder blocks')
    layer = layers[config.insertion_index]
    if hasattr(layer, '_mem2w_memory'):
        raise ValueError('Mem2W is already attached to this decoder block')
    if hidden_size is not None and int(hidden_size) != config.hidden_size:
        raise ValueError(f'Mem2W hidden_size={config.hidden_size} does not match model hidden_size={hidden_size}')

    reference = next(layer.parameters(), None)
    memory = PersistentKVMemory(config)
    if reference is not None:
        # Keep the four persistent tensors in FP32 as the optimizer/master
        # representation.  ``PersistentKVMemory.forward`` already casts its
        # matmul work to the model dtype where appropriate; storing BF16 here
        # would make AdamW updates and checkpoint round-trips BF16 as well.
        memory.to(device=reference.device, dtype=torch.float32)
    layer.add_module('_mem2w_memory', memory)
    layer._mem2w_enabled = True
    layer._mem2w_stop_content_grad = bool(config.stop_content_grad)
    # Cache this once: inspecting a decoder signature inside every forward
    # would add avoidable overhead to long SFT runs.
    layer._mem2w_attention_arg_index = _attention_mask_positional_index(layer)
    layer._mem2w_hook_handle = layer.register_forward_hook(_memory_hook, with_kwargs=True)
    object.__setattr__(model, '_mem2w_memory_layer', layer)
    object.__setattr__(model, '_mem2w_insertion_index', config.insertion_index)
    object.__setattr__(model, '_mem2w_config', config)
    return model


def get_memory_module(model):
    return _memory_layer(model)._mem2w_memory


def _memory_layer(model):
    layer = getattr(model, '_mem2w_memory_layer', None)
    if layer is not None:
        return layer
    for candidate in _decoder_layers(model):
        if hasattr(candidate, '_mem2w_memory'):
            return candidate
    raise ValueError('model has no attached Mem2W memory')


def freeze_memory_only(model):
    model.requires_grad_(False)
    memory = get_memory_module(model)
    for parameter in memory.parameters():
        parameter.requires_grad_(True)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if len(trainable) != 4 or not all(getattr(parameter, 'is_mem2w_parameter', False) for _, parameter in trainable):
        raise RuntimeError(f'Mem2W memory-only freeze failed: {[name for name, _ in trainable]}')
    return model


def memory_parameters(model):
    return [parameter for parameter in get_memory_module(model).parameters() if parameter.requires_grad]


def set_memory_enabled(model, enabled: bool):
    _memory_layer(model)._mem2w_enabled = bool(enabled)


def set_stop_content_grad(model, enabled: bool):
    _memory_layer(model)._mem2w_stop_content_grad = bool(enabled)
