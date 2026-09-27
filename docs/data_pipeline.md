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

如果只把公开结果目录和抽样 actor prompt 交给通用 importer，它只能得到 960 条
episode-level action、27 条可逐字验证的 recall；这不是 0916 的正式全量训练导出。
正式全量导出必须使用下面的归档专用转换器，从完整 trajectory 和 frozen snapshot
恢复 6804 条单步 action 与 864 条 retrieval-event recall。

## 0916 归档的正式 Mem2W 样本

`0916_memrl_96+24_ds0731` 的结果目录经过归档，完整文件位置由其
`ARCHIVED.json` 指向：

```text
result_root  = /mnt/public/haoting/mrl-textgrad/results/automationbench/0916_memrl_96+24_ds0731
archive_root = /mnt/public/code/haoting/mrl-textgrad-archive/0916_memrl_96+24_ds0731
```

这里使用专用转换器，而不是通用的 episode importer：

```bash
mem2w-convert-automationbench-0916 \
  --result-root /mnt/public/haoting/mrl-textgrad/results/automationbench/0916_memrl_96+24_ds0731 \
  --archive-root /mnt/public/code/haoting/mrl-textgrad-archive/0916_memrl_96+24_ds0731 \
  --output-dir /mnt/public/haoting/mem2w_data/automationbench_0916 \
  --payload-budget-chars 12000 \
  --action-context-budget-chars 16000
```

该转换器的两个输出流与 native paired trainer 对应：

```text
action_train.jsonl   # W_action_single_step：6804 条，每个 assistant action 一条
recall_train.jsonl   # C_memory_reconstruction：864 条，每个真实 retrieval event 一条
qa.json
manifest.json
```

W 样本使用完整 trajectory 的可见前缀；历史 assistant/tool response 只作为条件，
当前 assistant action 是唯一 `loss=true` 的消息。C 样本使用
`checkpoints/snapshot/(epoch-1)/payloads.json`，按 trajectory 中
`retrieval_records` 的顺序重建真实 `[Reference Memories]` / `[MEMRL MEMORY n]`
上下文。该映射在归档的 27 条 actor prompt 抽样上逐字 hash 校验为 27/27；ID、相似度
和 Q 值只保留为审计字段。

由于原始 W 前缀和 C payload 很长，转换器对历史上下文和 recall target 采用显式头尾
压缩，并保留 `context_original_chars`、`payload_original_chars`、`payload_sha256`、
`*_compacted` 等字段。这样 ms-swift 不会静默截断；如果要做 exact teacher-payload
复现实验，应把预算提高并单独处理超长样本，而不是直接使用默认 `max_length`。

## 质量门

训练前至少检查：

1. `qa.json` 中 rejected、split、join 和 `recall_missing_payload` 数量符合预期；
2. 每个 action 行有 assistant/tool action，且 prompt 中不含 teacher payload；
3. recall 行的 `payload_sha256` 与 actor prompt 原文一致；
4. 没有把 `retrieval_records` 或空字符串猜成 recall 目标；
5. `manifest.json` 的输入 hash、snapshot ID、模型/模板 manifest 被一并归档。
