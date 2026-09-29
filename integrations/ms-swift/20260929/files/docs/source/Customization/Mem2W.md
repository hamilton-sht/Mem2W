# Native Mem2W SFT

This checkout contains an opt-in native Mem2W path. It is implemented inside
the ms-swift SFT pipeline rather than through an external plugin or a second
Trainer implementation.

## Enable the model extension

```bash
swift sft \
  --model Qwen/Qwen3.5-9B \
  --template qwen3_5 \
  --tuner_type mem2w \
  --dataset /path/to/action_train.jsonl \
  --output_dir output/mem2w_action
```

The native `mem2w` tuner has arguments for the insertion index, slot count,
key/value dimensions, RMSNorm epsilon and dropout. Its adapter/checkpoint path
is compatible with ms-swift's normal `--adapters` and resume flow.

The native SFT pipeline loads the model and template using the normal ms-swift
path, then the native tuner attaches Mem2W after decoder block index 15 and
freezes every parameter except `W_Q`, `K`, `V` and `W_O`. The original decoder
layer is kept intact and receives a forward hook, so gradient checkpointing,
FSDP/DeepSpeed wrapping, layer indices and state-dict ownership remain native.
The trainer's existing tuner save path writes a memory-only checkpoint.

The standard ms-swift causal-LM loss and message-level `loss` masks are used;
no upstream Trainer or chat-template fork is needed. This first integration
does not yet implement the later paired action/recall W/C schedule.
