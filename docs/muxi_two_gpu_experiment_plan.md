# Muxi 双卡实验计划

当前设计包含两个可独立启动的实验，默认 `DRY_RUN=1`，先只做数据检查和命令打印。

## Experiment A：Mem2W staged W/C/W

入口：`scripts/run_muxi_mem2w_staged_2gpu.sh`

```text
W: 1 action epoch -> checkpoint-W
C: 1 recall epoch -> checkpoint-C
W: 1 action epoch -> checkpoint-final
```

当前正式数据为 6804 条 action、864 条 recall，双卡、每卡 accumulation 为 8，因此一个
全局 update 消耗 16 条样本：action epoch 为 426 个 update，recall epoch 为 54 个
update，计划为：

```text
W:426,C:54,W:426
```

W 阶段只计算 action 分支，C 阶段只计算 recall 分支；recall 不会为了匹配 action
数量而循环 7.9 倍。每个 rank 仍以 `accumulation_steps` 条样本组成一个 logical
update；DDP 全局有效样本数为 `world_size * accumulation_steps`。阶段边界由 Trainer
callback 精确保存，因此 checkpoint 为 `checkpoint-426`、`checkpoint-480`、
`checkpoint-906`。

入口固定 `--mem2w-insertion-index 15`，即 zero-based decoder block 15 / 人类计数第 16
层。当前 Mem2W tuner 只训练该层挂载的四个参数 `W_Q/K/V/W_O`；Qwen 原始第 16 层的
attention/MLP 权重仍然冻结。“第 16 层完全调”在本实验中指四个 Mem2W 参数全部可训练。

脚本额外检查上述三个阶段边界是否存在 `memory.safetensors`。

## Experiment B：LoRA action baseline

入口：`scripts/run_muxi_lora_sft_2gpu.sh`

默认只使用 W/action 语料 `action_train.jsonl`，采用原生 `swift sft`、`qwen3_5`、
LoRA rank 8 / alpha 32，训练 3 个 epoch，按 epoch 保存 checkpoint。它不是 Mem2W
tuner，也不会加载或更新 Mem2W memory 参数。

## 并行方式

如果有 4 张空闲卡，可以将两个实验放到不重叠的卡组：

```bash
GPU_IDS=0,1 DRY_RUN=0 scripts/run_muxi_mem2w_staged_2gpu.sh
GPU_IDS=2,3 DRY_RUN=0 scripts/run_muxi_lora_sft_2gpu.sh
```

如果只有 2 张卡，两个实验不能物理并行，应先后运行。两个入口都会先执行
`check_muxi_mem2w_data.py`；数据不完整、manifest hash 不一致、C prompt 泄漏或 payload
hash 错误时会直接退出，不会开始训练。

正式启动前先分别使用 `DRY_RUN=1` 检查命令和 `experiment_plan.json`，确认模型路径、卡号、
最大长度和输出目录后再改为 `DRY_RUN=0`。
