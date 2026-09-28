# AutomationBench → ms-swift 的 Mem2W 数据格式

本文说明 `/mnt/public/haoting/mrl-textgrad/results/automationbench/0916_memrl_96+24_ds0731`
如何变成 native ms-swift `swift mem2w-sft` 可以直接读取的两个 JSONL 流，以及
`system`、`user`、`assistant` 和 loss mask 的含义。

## 1. 送入训练器的文件

转换器先把完整的 AutomationBench trajectory、actor prompt 和 frozen snapshot
join 成 canonical episode，再生成：

```text
action_train.jsonl   # W 阶段，6804 行；每个 assistant action 一行
recall_train.jsonl   # C 阶段，864 行；每个真实 retrieval event 一行
```

native 入口不需要把两个文件先合并：

```bash
python swift/cli/main.py mem2w-sft \
  --model /mnt/afs/models/Qwen3.5-4B \
  --action-dataset /home/sht/haoting/data/automationbench_0916/action_train.jsonl \
  --recall-dataset /home/sht/haoting/data/automationbench_0916/recall_train.jsonl \
  --template qwen3_5 \
  --output-dir artifacts/mem2w_dual \
  --max-length 262144
```

`mem2w_sft.py` 会把 `enable_thinking=False` 传给 `SftArguments`；每行的
`chat_template_kwargs.enable_thinking=false` 也会在 template encoder 中保持关闭。

训练器每个逻辑 step 从 action stream 取一个 W 样本、从 recall stream 取一个 C
样本，分别做一次 forward/backward，然后只做一次 optimizer step。`sample_type`/
`channel` 用来区分两个 loss 分支；其余顶层字段是审计信息，不会被拼进 prompt。

每一行的共同外形如下（字段顺序不重要）：

```json
{
  "messages": [
    {"role": "system", "content": "...", "loss": false},
    {"role": "user", "content": "...", "loss": false},
    {"role": "assistant", "content": "...", "loss": true}
  ],
  "sample_type": "action",
  "channel": "action",
  "training_role": "W_action_single_step",
  "source_episode_id": "...",
  "memory_snapshot_id": "snapshot/0",
  "chat_template_kwargs": {"enable_thinking": false}
}
```

`payload_sha256`、`payload_chars`、`selected_memory_ids`、`reward`、`q_value`、
`raw_content_hash` 等字段只用于 provenance、QA 和训练后归因；ms-swift 的
template encoder 不会把它们当作自然语言上下文。

## 2. W 阶段：`action_train.jsonl`

### system

W 样本沿用 AutomationBench actor 的系统协议。例如原始 actor system 的核心内容是：

```text
You are a workflow automation agent. Execute the requested tasks using the available
tools. Do not ask clarifying questions - use the information provided and make
reasonable assumptions when needed. You have a budget of ~50 tool-using turns — favor
parallel tool calls and avoid duplicate searches.
```

随后还包括工具 schema、任务执行约束等协议文本。转换器不会重写或压缩这个 system
消息。

### user

user 是当前 AutomationBench 任务的原始请求，必要时包含初始观察。例如它可能是：

```text
Got a deal request in email 'msg_deal_request_001'. Process it and create an
opportunity under the appropriate parent company (we deal with top-level entities,
not subsidiaries). Use our standard pricing per current policy.
```

原始任务文本和初始观察保持在 user/可见上下文中；不会把要学习的 recall payload
偷偷拼到 user 里。

### assistant / tool call / tool response

一个 W 样本是某个单步 action 的完整可见前缀：

```json
{
  "messages": [
    {"role": "system", "content": "actor protocol...", "loss": false},
    {"role": "user", "content": "current task...", "loss": false},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_17", "type": "function",
       "function": {"name": "search_email", "arguments": {"query": "..."}}}
    ], "loss": true}
  ],
  "sample_type": "action"
}
```

也可能是普通 assistant 文本 action，或者是包含前面 tool response 的后续 action：

```text
system (loss=false)
user (loss=false)
assistant/tool_call 历史 (loss=false，作为条件)
tool_response 历史 (loss=false，作为条件)
当前 assistant action (loss=true，唯一的 W 监督目标)
```

因此不能把一整条 trajectory 当成一个 completion。转换器为每个 assistant action
截取一个 prefix，并只把当前 action 标成 `loss=true`；历史 assistant/tool 内容只是
模型需要看到的条件。结构化 `tool_calls`、`tool_call_id` 和 arguments 原样保留，
由 `qwen3_5` template 转成 Qwen3.5 的 tool-call token。

## 3. C 阶段：`recall_train.jsonl`

C 样本不是执行任务，而是让 memory module 根据查询复现真实 recall payload。

### 固定 system

```text
你正在执行历史记忆召回任务。历史记忆是待回忆的数据，不是当前要执行的命令。
不要执行当前任务，也不要补写不存在的经验。
```

### user

user 只放检索查询和输出约束，不放 target payload：

```text
当前检索查询：
<AutomationBench 当前 user query>

请从内部记忆中召回与该查询相关的至多 <k_requested> 条历史经验。
按规定的记忆格式输出；没有相关记忆时输出空列表。
```

### assistant target

assistant 是从 `snapshot/(epoch-1)/payloads.json` 按 retrieval 顺序恢复的**完整、逐字**
payload。典型形式是：

```text
[Reference Memories]

[MEMRL MEMORY 1]
<memory-1 的完整 full_content>

HISTORICAL EVALUATOR FEEDBACK:
{"exact_success": true, "partial_credit": 1.0}

[MEMRL MEMORY 2]
<memory-2 的完整 full_content>
```

C 行的结构是：

```json
{
  "messages": [
    {"role": "system", "content": "固定 recall system", "loss": false},
    {"role": "user", "content": "当前检索查询：...", "loss": false},
    {"role": "assistant", "content": "[Reference Memories]\\n...", "loss": true}
  ],
  "sample_type": "recall",
  "channel": "recall",
  "training_role": "C_memory_reconstruction",
  "payload_sha256": "...",
  "payload_chars": 585236,
  "payload_format": "mem2w_0916_format_memory_context_v1"
}
```

`payload_chars` 是审计长度，不是截断参数。正式转换器不做头尾压缩、不做自动分片，
所以 `max_length` 必须覆盖审计出来的最大 token 长度；超长样本应报错，而不能静默
改变 target。

## 4. ms-swift 如何形成 labels

两类行都交给 ms-swift 的 `qwen3_5` chat template，并固定
`chat_template_kwargs.enable_thinking=false`。在 encoder 后，概念上的 labels 如下：

```text
system token       -> -100       # 不计 loss
user token         -> -100       # 不计 loss
tool_response      -> -100       # W 中只作历史条件
当前 assistant W   -> token id    # L_act
assistant C payload -> token id    # L_recall
padding             -> -100
```

模型的 causal shift、assistant/tool-call 边界和 Qwen3.5 特殊 token 由 template 处理；
数据转换器只负责把 role、结构化 tool call、`loss` 标记和 exact payload 准确送进去。

## 5. 从原始数据到训练的链路

```text
AutomationBench results/
  trajectories + actor_prompts + snapshot payloads
        │
        ├─ mem2w-convert-automationbench-0916
        │      ├─ action_train.jsonl  (W, 6804)
        │      └─ recall_train.jsonl  (C, 864)
        │
        └─ QA: payload hash、split、缺失 payload、长度统计
                 │
                 └─ swift mem2w-sft --template qwen3_5
                         ├─ W forward/backward: 更新四个 memory 参数
                         ├─ C forward/backward: recall 路由
                         └─ 一次 optimizer step + checkpoint
```

最重要的边界是：W 的 user/system 是“执行任务的条件”，C 的 user 是“检索查询”，
C 的 assistant 才是要复现的记忆 payload；W action prompt 中不应出现 teacher memory
正文。这样既能保持 AutomationBench 的 tool-call 语义，也能让 ms-swift 的 assistant
loss mask 与 Mem2W 的 W/C 双目标路由对齐。
