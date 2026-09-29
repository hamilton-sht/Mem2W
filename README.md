# Mem2W

Mem2W 是一个插入冻结 Qwen3.5 主干的持久 KV memory layer。仓库提供数据转换、模板/loss-mask 审计和独立 action/recall SFT；配套 native ms-swift fork 还提供 action+recall 配对的 W/C 双目标入口，两个路径都只更新 memory 层的 `W_Q/K/V/W_O`，并将结果保存为 `memory.safetensors`。

## 安装

在已安装 PyTorch/CUDA 的训练环境中：

```bash
python -m pip install -e '.[sft]'
```

本项目针对 ms-swift `>=4.5` 和支持 Qwen3.5 的 Transformers 版本（`>=5.2.0`）。运行时必须记录实际版本和模型 revision。

## 最小训练通路

先把 MemRL episode JSONL 转成 ms-swift 数据：

```bash
mem2w-sft prepare \
  --input data/teacher_episodes.jsonl \
  --data-dir data/ms_swift \
  --config configs/action.json \
  --mode action

mem2w-sft prepare \
  --input data/teacher_episodes.jsonl \
  --data-dir data/ms_swift \
  --config configs/recall.json \
  --mode recall
```

然后分别启动两个 mode：

```bash
mem2w-sft-train --mode action \
  --dataset data/ms_swift/action_train.jsonl \
  --config configs/default.yaml \
  --output-dir artifacts/memory_action

mem2w-sft-train --mode recall \
  --dataset data/ms_swift/recall_train.jsonl \
  --config configs/default.yaml \
  --output-dir artifacts/memory_recall
```

LiveCodeBench 的单轮 execution 记录也可先转换为当前 AutomationBench/Mem2W episode
协议：

```bash
mem2w-convert-lcb-automationbench \
  --problems /path/to/release_v6_test6.jsonl \
  --run-root /path/to/lcb_run \
  --split-manifest /path/to/lcb_split.json \
  --output-dir data/lcb_automationbench \
  --epochs 10
```

`--epochs` 可限制转换范围；省略时转换全部 epoch，`--epochs 10` 表示只保留最后一个
训练 epoch 及其对应验证集。

两个命令都会通过 ms-swift 加载模型、processor、chat template 和 Trainer；Mem2W 在 Trainer 建立前插入第 16 个 block 后，冻结原模型并只把四个 memory 参数交给 optimizer。训练结束后输出：

```text
artifacts/memory_action/memory.safetensors
artifacts/memory_action/memory_config.json
```

`recall` mode 使用完全独立的 recall JSONL。需要真正的双目标训练时，在 native fork 中运行：

```bash
python swift/cli/main.py mem2w-sft \
  --model /mnt/public/model/Qwen3.5-4B \
  --action-dataset data/ms_swift/action_train.jsonl \
  --recall-dataset data/ms_swift/recall_train.jsonl \
  --output-dir artifacts/mem2w_dual \
  --max-steps 1000 \
  --warmup-fraction 0.20
```

该入口按显式的 W/C 阶段运行：W 阶段只计算 action，C 阶段只计算 recall；因此不会把 recall 数据循环配对到每个 action update，也不会在一个 update 中保留 16 个长上下文计算图。C 阶段按约束分支 detach `V/W_O`，保留 `W_Q/K` 的读路径梯度。阶段边界由 native Trainer callback 保存，checkpoint 可用 `--resume-from-checkpoint` 恢复。

## 关键约束

- 原 Qwen 参数全部冻结，但 forward 仍保留梯度路径，保证 memory 能从 LM loss 学习。
- memory 旁路时直接返回 block 激活，原模型层数和 cache 契约不变。
- `memory.safetensors` 只保存 memory 参数，不包含外部 MemRL memory index 或训练样本。
- Qwen3.5-9B 目标模型的 memory hidden size 是 4096；可以用官方 Qwen3.5-4B 做 smoke，但需使用其实际 hidden size。

完整算法边界和当前交付/待验收项见 [docs/PLAN.md](docs/PLAN.md)，ms-swift 入口说明见 [docs/ms_swift_sft.md](docs/ms_swift_sft.md)。
