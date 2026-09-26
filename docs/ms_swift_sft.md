# Mem2W 的 ms-swift SFT 接入

`src/mem2w/ms_swift_adapter.py` 将 Mem2W 教师 episode 转换为 ms-swift 能直接读取的标准 JSONL，并生成 `swift sft` 的 JSON 配置。转换器不重新检索、不重写教师 payload，也不会把外部记忆正文放进行动 prompt。

当前核对的上游版本是 ms-swift `main`：

```text
c08110b30a1ccb60bcfb70adf87d2cd72f5b9f3c  (2026-09-26)
```

官方支持表将 `Qwen/Qwen3.5-9B` 映射到 `qwen3_5` 模板；Qwen3.5 需要 Transformers 5.2.0 或更高版本。训练环境仍须在 preflight 中记录实际安装的 ms-swift、Transformers 版本和模型 revision，不能把这里的上游 commit 当成已经安装的依赖。

## 1. 转换数据

输入是计划文档 9.1 的 episode JSONL。最少需要 `episode_id`、`task`、`messages`、`retrieval_events`；每个 retrieval event 必须包含 `query_text`、`k_requested`、`injected_context_text`。无命中不能留空字符串，要把真实教师返回的固定空结果（例如 `{"memories":[]}`）写入 `injected_context_text`。

```bash
python3 -m mem2w.ms_swift_adapter convert \
  --input data/teacher_episodes.jsonl \
  --output-dir data/ms_swift
```

也可以用一次 `prepare` 完成转换和配置：

```bash
python3 -m mem2w.ms_swift_adapter prepare \
  --input data/teacher_episodes.jsonl \
  --data-dir data/ms_swift \
  --config configs/ms_swift_sft.json \
  --mode both \
  --external-plugin src/mem2w/ms_swift_plugin.py
```

输出按来源 split 分成：

```text
data/ms_swift/action_train.jsonl
data/ms_swift/recall_train.jsonl
data/ms_swift/action_validation.jsonl
data/ms_swift/recall_validation.jsonl
data/ms_swift/manifest.json
```

action 样本保留完整交互历史。`system`、`user`、`tool_response` 设为 `loss=false`，`assistant` 和 `tool_call` 设为 `loss=true`，因此仅 action token 参与 SFT。recall 样本使用固定系统指令、查询和 `k_requested`，assistant completion 逐字等于该检索 event 的 `injected_context_text`，只有 completion 参与 loss。

转换器检查 action 的非 assistant 输入是否包含教师 payload；发现泄漏会直接报错。它也保留 `sample_type`、`source_episode_id`、`source_retrieval_event_id`、`memory_snapshot_id`、`success`、`reward`、`termination_reason` 等审计字段，并给每行写 `raw_content_hash`。这些字段不参与 prompt，不应被训练脚本当成标签。

## 2. 生成 ms-swift 配置

先分别生成两个最小 pilot mode。数据、模板和模型 loader 仍复用 ms-swift；真正的
memory-only 更新请使用项目的 `mem2w-sft-train` runner。原因是 ms-swift 的
`tuner_type=full` 会在 `prepare_model` 阶段重新打开全模型梯度，而项目 runner
在该阶段之后再次冻结主干并断言 optimizer 只看到四个 memory 参数。这一步不实现
W/C 阶段的梯度路由。

```bash
# action：只用教师轨迹中的 assistant/tool_call 输出
python3 -m mem2w.ms_swift_adapter config \
  --mode action \
  --action-data data/ms_swift/action_train.jsonl \
  --action-val-data data/ms_swift/action_validation.jsonl \
  --external-plugin src/mem2w/ms_swift_plugin.py \
  --output configs/ms_swift_action.json

mem2w-sft-train --mode action \
  --dataset data/ms_swift/action_train.jsonl \
  --config configs/default.yaml \
  --output-dir artifacts/memory_action

# recall：只用检索 payload completion
python3 -m mem2w.ms_swift_adapter config \
  --mode recall \
  --recall-data data/ms_swift/recall_train.jsonl \
  --recall-val-data data/ms_swift/recall_validation.jsonl \
  --external-plugin src/mem2w/ms_swift_plugin.py \
  --output configs/ms_swift_recall.json

mem2w-sft-train --mode recall \
  --dataset data/ms_swift/recall_train.jsonl \
  --config configs/default.yaml \
  --output-dir artifacts/memory_recall
```

`--mode both` 仍可用于把两个文件合成一个普通 ms-swift smoke test：

```bash
python3 -m mem2w.ms_swift_adapter config \
  --mode both \
  --action-data data/ms_swift/action_train.jsonl \
  --recall-data data/ms_swift/recall_train.jsonl \
  --output configs/ms_swift_both.json
```

生成的配置也可交给原生 ms-swift 做数据/template smoke test：

```bash
swift sft configs/ms_swift_sft.json
```

但原生命令的 full tuner 不保证 memory-only 参数过滤，也不会导出本项目约定的
`memory.safetensors`；正式最小训练以 `mem2w-sft-train` 为准。

已有配置只想查看启动命令时：

```bash
python3 -m mem2w.ms_swift_adapter command --config configs/ms_swift_sft.json
```

默认关键参数对应计划文档：`model=Qwen/Qwen3.5-9B`、`template=qwen3_5`、`tuner_type=full`、`torch_dtype=bfloat16`、`max_length=4096`、`packing=false`、`gradient_checkpointing=false`、`enable_thinking=false`、`remove_unused_columns=false`。ms-swift 的 SFT pipeline 会在训练前把 `model.config.use_cache` 设为 `false`；它不是 SftArguments 的 JSON 字段。项目 runner 在 tuner 准备后重新 attach/freeze memory，optimizer 只接收 Mem2W memory 参数。

## 3. 双目标训练（native ms-swift fork）

ms-swift 的标准 SFT Trainer 能正确完成 chat template、assistant `loss` mask 和 causal-LM shift。它的 `loss_scale` 还支持 token 权重，`enable_channel_loss=true` 可以按 `channel` 统计 action/recall 的诊断指标。

Mem2W 目标要求：

1. `L_act` 与 `L_recall` 分别按自己的有效 token 数归一化，再乘 `lambda_recall`；
2. warmup 阶段两个分支都更新 `W_Q/K/V/W_O`；约束阶段 recall 分支只更新 `W_Q/K`；
3. 两分支在一个逻辑 optimizer step 内完成，且主干梯度始终可穿过冻结 block 传到 memory。

当前普通 `swift sft --tuner_type mem2w` 仍是单流 SFT，用于兼容性 smoke test；它不会根据
`sample_type` 自动做双分支归一化或 recall-only 梯度路由。双目标训练已经迁移到
native ms-swift fork 的 `swift mem2w-sft` 入口：它复用 ms-swift 的模型加载、
`qwen3_5` template encoder 和 data collator，但由 native Mem2W trainer 在一个逻辑
optimizer step 内依次完成 action/recall 两次 forward/backward，再执行一次 step。

```bash
python swift/cli/main.py mem2w-sft \
  --model /mnt/public/model/Qwen3.5-4B \
  --action-dataset data/ms_swift/action_train.jsonl \
  --recall-dataset data/ms_swift/recall_train.jsonl \
  --template qwen3_5 \
  --output-dir artifacts/mem2w_dual \
  --max-steps 1000 \
  --warmup-fraction 0.20 \
  --lambda-recall 1.0 \
  --mem2w-insertion-index 15 \
  --mem2w-slots 512 \
  --mem2w-key-dim 256 \
  --mem2w-value-dim 256
```

`swift mem2w-sft` 的 W 阶段让两个分支更新 `W_Q/K/V/W_O`；C 阶段仅对 recall
分支 detach `V/W_O`，保留 `W_Q/K` 的梯度。每个分支先按自身有效 label token
总数归一化（默认每分支 8 个 microbatch），再乘 `lambda_recall`；每个逻辑 step
只执行一次 optimizer step，并按 cosine/LR warmup 更新 scheduler。每一步输出
`checkpoint-N/memory.safetensors`、`mem2w_config.json`、`optimizer.pt`、
`scheduler.pt`、`rng_state.pt` 和 `mem2w_dual_state.json`，可以用
`--resume-from-checkpoint` 恢复；`training_metrics.jsonl` 和
`training_summary.json` 保留逐步指标。普通 `swift sft`
和此入口共享同一个 native `mem2w` tuner 与四参数 checkpoint 格式。

## 4. token mask 验证

在正式训练前，必须对至少一条 action 和一条 recall 运行 ms-swift 的 template encoder，检查 `labels`/`loss_scale`：

```python
from swift import get_processor, get_template

processor = get_processor("Qwen/Qwen3.5-9B")
template = get_template(processor, loss_scale="default")
template.set_mode("train")
encoded = template.encode(row)
print(template.safe_decode(encoded["labels"]))
```

应看到 action 的 user、tool response 是 `-100`，每个 assistant action（包括合法终止标记）有标签；recall 只有 payload completion 有标签。不要只依据 `loss=true` 的布尔值判断 mask 正确性。Qwen3.5 首版关闭 packing；混合 linear/full attention 的 var-len 行为以及跨样本状态隔离没有在 Mem2W wrapper 中验证之前，不要打开 `packing`。

仓库提供了可复用的审计入口。它先在无训练依赖的情况下检查 JSONL 的结构化
tool-call、`loss` 标记、payload hash 和 teacher payload 边界；传入 native
ms-swift checkout 与本地模型后，还会真实编码 `qwen3_5`，验证 token-level
`labels`、全关闭 loss 的零标签基线，以及 `tool_response` 不会产生监督标签：

```bash
PYTHONPATH=src python -m mem2w.template_audit \
  --dataset data/ms_swift/action_train.jsonl \
  --dataset data/ms_swift/recall_train.jsonl \
  --source-episodes data/episodes.jsonl \
  --model /mnt/public/model/Qwen3.5-4B \
  --swift-root /path/to/ms-swift \
  --report artifacts/template_audit.json
```

省略 `--model` 可在无 torch/transformers 的登录节点上执行纯结构审计；有
`--model` 时不加载模型权重，只调用 processor/tokenizer，因此仍不会启动训练。
