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

该入口在一个逻辑更新内完成 action/recall 两次 forward/backward 和一次 optimizer step；W 阶段两个分支更新四个参数，C 阶段 recall 分支 detach `V/W_O`，保留 `W_Q/K` 梯度。每步 checkpoint 可用 `--resume-from-checkpoint` 恢复。

## 关键约束

- 原 Qwen 参数全部冻结，但 forward 仍保留梯度路径，保证 memory 能从 LM loss 学习。
- memory 旁路时直接返回 block 激活，原模型层数和 cache 契约不变。
- `memory.safetensors` 只保存 memory 参数，不包含外部 MemRL memory index 或训练样本。
- Qwen3.5-9B 目标模型的 memory hidden size 是 4096；可以用官方 Qwen3.5-4B 做 smoke，但需使用其实际 hidden size。

完整算法边界和当前交付/待验收项见 [docs/PLAN.md](docs/PLAN.md)，ms-swift 入口说明见 [docs/ms_swift_sft.md](docs/ms_swift_sft.md)。
