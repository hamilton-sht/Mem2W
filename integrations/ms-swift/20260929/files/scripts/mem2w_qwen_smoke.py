"""Run a small native Mem2W smoke test against a local Qwen3.5 checkpoint.

This intentionally exercises the same native tuner entry point used by
``swift sft --tuner_type mem2w`` while keeping the sequence and memory adapter
small enough for a single Muxi C500.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from swift.tuners.mem2w import Mem2WTuner


def _args(insertion_index: int, slots: int, key_dim: int, value_dim: int):
    return SimpleNamespace(
        mem2w_insertion_index=insertion_index,
        mem2w_slots=slots,
        mem2w_key_dim=key_dim,
        mem2w_value_dim=value_dim,
        mem2w_rms_eps=1e-6,
        mem2w_dropout=0.0,
        mem2w_stop_content_grad=False,
    )


def _trainable(model):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--insertion-index', type=int, default=15)
    parser.add_argument('--slots', type=int, default=16)
    parser.add_argument('--key-dim', type=int, default=32)
    parser.add_argument('--value-dim', type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('MUSA/CUDA device is not available')
    device = torch.device('cuda:0')
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    print(json.dumps({
        'model_type': getattr(model.config, 'model_type', None),
        'class': type(model).__name__,
        'device': str(next(model.parameters()).device),
        'dtype': str(next(model.parameters()).dtype),
        'layers': len(getattr(getattr(getattr(model, 'model', None), 'language_model', None), 'layers', [])),
    }, ensure_ascii=False))

    messages = [
        {'role': 'user', 'content': 'Remember that the deployment target is a Muxi C500 GPU.'},
        {'role': 'assistant', 'content': 'I will retain that deployment constraint.'},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_tensors='pt',
    )
    input_ids = rendered['input_ids'] if isinstance(rendered, Mapping) else rendered
    attention_mask = rendered.get('attention_mask') if isinstance(rendered, Mapping) else None
    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    model = Mem2WTuner.prepare_model(
        _args(args.insertion_index, args.slots, args.key_dim, args.value_dim), model)
    trainable = _trainable(model)
    names = [name for name, _ in trainable]
    if len(trainable) != 4 or not all(name.endswith(('W_Q', 'K', 'V', 'W_O')) for name in names):
        raise AssertionError(f'expected four Mem2W tensors, got {names}')

    labels = input_ids.clone()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    loss = outputs.loss
    loss.backward()
    grads = {name: param.grad for name, param in trainable}
    if any(grad is None or not torch.isfinite(grad).all().item() for grad in grads.values()):
        raise AssertionError('a Mem2W parameter has no finite gradient')

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    Mem2WTuner.save_pretrained(model, str(output))
    before = {name: param.detach().cpu().clone() for name, param in trainable}
    with torch.no_grad():
        trainable[0][1].add_(1.0)
    Mem2WTuner.from_pretrained(model, str(output), is_trainable=True)
    after = {name: param.detach().cpu() for name, param in trainable}
    if any(not torch.equal(before[name], after[name]) for name in before):
        raise AssertionError('checkpoint reload did not restore Mem2W tensors')

    print(json.dumps({
        'loss': float(loss.detach().cpu()),
        'input_tokens': int(input_ids.shape[-1]),
        'trainable': names,
        'checkpoint': str(output),
        'checkpoint_files': sorted(path.name for path in output.iterdir()),
        'status': 'ok',
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
