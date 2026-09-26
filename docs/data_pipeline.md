# MemRL / AutomationBench 数据管线

研究材料中的 `0916_memrl_96+24_ds0731` 原始输出不是 SFT 对话数据，而是
`trajectories.jsonl`：每行包含 `task`、`epoch`、reward/成功标记、检索 ID、
`retrieval_records` 以及 JSON-lines 字符串形式的 `trajectory`。因此需要先导入
Mem2W 的 canonical episode，再交给 `ms_swift_adapter` 生成 action/recall JSONL。

## 原始导入

```bash
mem2w-import-automationbench \
  --trajectories /path/to/epochs \
  --actor-prompts /path/to/actor_prompts \
  --split-manifest /path/to/split_manifest.json \
  --snapshot-root /path/to/snapshots \
  --output-dir data/mem2w_episodes
```

`--actor-prompts` 可以是包含 `epoch_*.jsonl` 的目录，也可以是单个 JSONL。导入器
按 `(task_id, epoch)` 连接 prompt；split manifest 中的 task-id 映射是固定来源，
不会被 trajectory 行里的 split 覆盖。官方 AutomationBench manifest 的
`selected_train_task_ids`/`selected_eval_task_ids` 会分别映射为 `train`/`validation`；
没有 split 映射的行进入 `rejected.jsonl`。
`--trajectories` 也可以指向一个包含 `epoch_*/trajectories.jsonl` 的 run 目录。
默认保留完整 payload；只有显式加 `--compact-payload --payload-budget-chars N` 才会
写入带标记的头尾压缩目标，`qa.json` 会记录原文/压缩后的 hash 和长度。压缩目标
不再是 exact actor prompt，适合上下文长度受限的 pilot；正式结果应优先重新导出
compact serializer，而不是依赖头尾截断。

导入器输出：

```text
episodes.jsonl   # action-ready canonical episodes
rejected.jsonl   # 行号、task、epoch、拒绝原因
qa.json          # split、join、缺 payload 和拒绝统计
manifest.json    # 输入 hash、策略和输出位置
```

每个 episode 的 snapshot ID 固定为 `snapshot/{epoch}`。如果提供
`--snapshot-root`，只记录发现的 snapshot 路径，不读取或重建 MemoryStore 内容。
trajectory 中的 `tool_calls` 和 `tool_call_id` 原样保留；`reasoning_content` 和
`reasoning` 字段递归删除。

## Recall payload 的边界

`retrieval_records` 只有 ID、相似度和 Q 诊断，不能当作 recall 监督目标。只有
joined actor prompt 中出现 `[Reference Memories]` 或 `[MEMRL MEMORY` 标记时，
才把该消息的原文作为 `injected_context_text`。actor prompt 缺失或没有标记时，
canonical episode 仍保留，事件标记 `recall_missing_payload=true`、payload 为空；
这样 action 可以训练，但 recall 行会被明确跳过并计入 `qa.json` 的
`recall_missing_payload`/`prompt_missing_payload`，不会凭空构造目标。

没有检索发生的 episode 不创建空 retrieval event；有检索但缺 exact payload 的
episode 仍保留 action，recall event 标记 `recall_missing_payload=true`，且不会进入
recall JSONL。

## 转换为 ms-swift

```bash
mem2w-sft convert \
  --input data/mem2w_episodes/episodes.jsonl \
  --output-dir data/ms_swift
```

输出按原始 split 分成 `action_train.jsonl`、`recall_train.jsonl` 等。action 会按
角色/metadata 结构移除 teacher memory；recall 只监督 exact `injected_context_text`。
`partial_credit`、`partial_credit_diagnostic`、`q_value`、`q_visits`、retrieved IDs
和诊断信息保留为审计列，不进入 prompt，也不改变 loss。

对 0916 归档真实回放的结果是：960 个 action episode，864 个发生过检索，27 个
有完整 actor-prompt recall payload，837 个检索 episode 缺 payload，最终生成
`action_train.jsonl=960`、`recall_train.jsonl=27`。默认 exact 模式下 27 条 payload
中有超长样本（首条约 585k 字符），所以实际 SFT 前必须选择新的 compact 导出或显式
启用带审计 hash 的 compaction；不能让 `max_length` 静默截断。

## 质量门

训练前至少检查：

1. `qa.json` 中 rejected、split、join 和 `recall_missing_payload` 数量符合预期；
2. 每个 action 行有 assistant/tool action，且 prompt 中不含 teacher payload；
3. recall 行的 `payload_sha256` 与 actor prompt 原文一致；
4. 没有把 `retrieval_records` 或空字符串猜成 recall 目标；
5. `manifest.json` 的输入 hash、snapshot ID、模型/模板 manifest 被一并归档。
