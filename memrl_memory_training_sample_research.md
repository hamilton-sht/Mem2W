# `0916_memrl_96+24_ds0731`：记忆如何构造为训练样本

## 结论

目标结果是经典 MemRL baseline：`algorithm=memrl`，`memrl_tg.enabled=false`，`textgrad.enabled=false`。因此它没有 TextGrad Critic、Composer 或 Q-text 生成；Actor 也没有通过梯度更新。所谓“训练样本”是每个 AutomationBench episode 的执行轨迹，写入外部 MemoryStore，随后作为下一轮 Actor 的上下文 replay。

## 一条完整链路

```text
task description
  -> Actor rollout
  -> serialize_trajectory(output)
  -> {trajectory, success, partial_credit, diagnostics, q metadata}
  -> MemoryService.add_memories()
  -> TrajectoryBuilder 原样保存 full_content
  -> task description 做 query embedding 并建立 memory-id bucket
  -> 下一轮 retrieve_query(top-k=3)
  -> similarity + numeric Q 排序
  -> compacted trajectory + historical feedback 注入 Actor prompt
```

代码位置（目标 commit `daa3fda33f392186be3d052a8b454f2cd68f68b6`）：

- `mrl_tg/run/automationbench_runner.py::_learn_batch`
- `mrl_tg/automationbench_eval/adapter.py::serialize_trajectory`、`_ensure_retrieval`、`format_memory_context`
- `mrl_tg/service/updater.py::VanillaUpdater.prepare_update_op`
- `mrl_tg/service/builders.py::TrajectoryBuilder.build`
- `mrl_tg/service/memory_service.py::add_memories`、`retrieve_query`、`update_values`

## 一个 episode 被拆成两个学习信号

### 1. 新建一条可 replay 的 memory

对每个未中止的训练输出，runner 取：

```python
task_description = output["_memq_task_description"]
trajectory = serialize_trajectory(output)
success = exact_success(output)
partial_credit = output["reward"]
```

并写入 metadata：`task_id`、`epoch`、`success`、`partial_credit_diagnostic`、`reflection_diagnostics`、`source_algorithm="memrl"`、`q_value`、`q_visits` 等。

本实验配置为 `build_strategy=trajectory`、`update_strategy=vanilla`，所以 `TrajectoryBuilder` 不调用 LLM，直接原样返回完整 trajectory。`VanillaUpdater` 最终保存：

```json
{
  "id": "memory id",
  "memory": "task description",
  "metadata": {
    "full_content": "Task: ...\\n\\n<serialized trajectory>",
    "success": true,
    "partial_credit_diagnostic": 1.0,
    "source_algorithm": "memrl",
    "q_value": 0.0,
    "q_visits": 0
  }
}
```

成功和失败都会保存；该 run 没有失败反思生成，因为使用的是 `vanilla`，而不是 `adjustment`/TextGrad Q-text 路径。

### 2. 用本轮 reward 更新已检索的旧 memory

本轮 Actor 开始前检索到的 memory IDs 会放在 `_memq_retrieved_ids`。batch 学习时，runner 对这些旧 memory 做数值 Q 更新：

```text
reward = +1  (exact success)
       = -1  (exact failure)
q_new = (1 - alpha) * q_old + alpha * reward + gamma * next_max_q
```

本 run 的配置是 `alpha=0.1`、`gamma=0`、`success_reward=1`、`failure_reward=-1`，所以实际为 `q_new = 0.9*q_old + 0.1*(+1/-1)`。`memrl_partial_credit_q=false`，所以 partial credit 不进入 Q target；它只作为诊断写入 memory metadata，并在后续 prompt 中展示为 historical evaluator feedback。

## MemoryStore 的索引与取样

- 向量索引的 key 是 **task description**，不是 trajectory。
- 因为 `retrieved_memory_queries=[[]]`，新 memory 默认按自己的完整 task description 建 key。
- `latest_memory_per_query=false`，同一个 task 的历史 memory 会累积在同一个 bucket 中。
- 检索先按 task-description embedding 找 query keys，再展开 bucket 内的 memory IDs；本 run `k_retrieve/topk=3`、`epsilon=0`、`weight_sim=0.5`、`weight_q=0.5`。
- Actor 看见的是按 `[MEMRL MEMORY i]` 包装的压缩 action/tool trace，以及 `exact_success` 和 `partial_credit`；不会看到 Q key、owner 或内部 Q 更新证据。

## 结果快照证明了什么

目标结果目录：

`/mnt/public/haoting/mrl-textgrad/results/automationbench/0916_memrl_96+24_ds0731`

原始 memory snapshot 归档在：

`/mnt/public/code/haoting/mrl-textgrad-archive/0916_memrl_96+24_ds0731`

- 每个 epoch 96 个训练任务，因此每轮新增约 96 条 memory；epoch 0 快照有 96 条，epoch 1 快照有 192 条，10 轮结束约 960 条。
- epoch 0：23 条 exact success、73 条 failure；memory 的 `full_content` 中位长度约 155,824 字符。
- epoch 1 新增 96 条，其中 44 success、52 failure；每个任务的 bucket 已包含 epoch 0 和 epoch 1 两条历史 memory。
- epoch 1 有 `288` 条 provenance edges，符合 96 个任务每个检索 3 条 memory；`q_updates=198` 是实际被更新的去重 memory 数量。
- epoch 1 的 q 均值约 `-0.024`，范围约 `[-0.760, 0.554]`；`td_count=0`，说明不是 TD 链式学习。
- 该 run 的 Critic/TextGrad 请求为 0；训练提升主要来自同任务历史轨迹 replay。训练 SR 从 epoch 0 的 `0.2396` 升到 epoch 9 的 `0.5938`，最终 held-out eval 为 `5/24=0.2083`。

因此，0916 的“样本”更准确的叫法是：

> **episode-level trajectory memory + retrieval-time prompt replay + numeric Q side update**。

它不是监督微调样本，也不是发给优化器的 `(input, label)` 对；模型参数没有被训练，变化的是外部 memory/Q 状态和下一轮 Actor 的上下文。

## 与当前 MemRL-TG 的区别

当前 `mrl_tg/docs/mrl-tg0915.md` 定义的 MemRL-TG 走另一条路径：

```text
task embedding -> Critic
criteria + Q-text -> Composer settings
settings + selected Q-text -> Actor rollout
rollout + reward/evidence -> TextGrad patch
patch -> task-owned Q-text / criteria / world facts
```

TG 的训练样本是按 `task:epoch:setting` 形成的 execution record，Q 是带 provenance 的文本经验（压缩轨迹、条件、反思、成功/失败标签和 reward evidence），不是 `q_value` 标量。0916 结果不能用来证明当前 TG 的 Q-text 构造；它只能作为“原始 episode replay + numeric Q baseline”。

## 复现时最应检查的四个字段

1. `serialize_trajectory(output)`：实际写入的轨迹内容。
2. `metadata.full_content`：memory 被 Actor 读取的主体。
3. `_memq_retrieved_ids` / `retrieval_records`：本轮哪些旧 memory 参与了 prompt 和 Q 更新。
4. `q_value`、`q_visits`、`partial_credit_diagnostic`：数值 Q 与部分得分的边界。
