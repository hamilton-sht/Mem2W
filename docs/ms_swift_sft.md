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

action 样本保留完整交互历史。`system`、`user`、`tool_response` 设为 `loss=false`，`assistant` 和 `tool_call` 设为 `loss=true`，因此仅 action token 参与 SFT。recall 样本使用英文固定系统指令、英文 query wrapper 和 `k_requested`，assistant completion 逐字等于该检索 event 的 `injected_context_text`，只有 completion 参与 loss；payload 不会进入 recall prompt。

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
3. W/C 采用交替阶段：W 只计算 action，C 只计算 recall；主干梯度始终可穿过冻结
   block 传到 memory，避免每个 update 无谓地执行两个长上下文分支。

`swift sft --tuner_type mem2w` 现在就是正式的双目标入口。native fork 在
`swift/pipelines/train/sft.py` 中加载 action/recall 两条数据流，在
`swift/trainers/mem2w_trainer.py` 中复用 ms-swift Trainer 的 DDP、DataLoader、
autocast、optimizer、scheduler、checkpoint 和 resume 生命周期，只替换 staged
W/C loss。每个 Trainer example 包含 `mem2w_accumulation_steps` 个当前阶段
microbatch；当前阶段分支按有效 token 数归一化，再执行一次 optimizer step。长上下文
必须设置 `mem2w_loss_chunk_size>0`，这样只分块计算 lm_head+CE，不物化完整 vocabulary
logits；训练仍然保留全部输入和监督 token，不做压缩、截断或分片。

```bash
python -m torch.distributed.run --nproc_per_node=2 swift/cli/sft.py \
  --model /mnt/public/model/Qwen3.5-4B \
  --dataset data/ms_swift/action_train.jsonl \
  --mem2w_recall_dataset data/ms_swift/recall_train.jsonl \
  --template qwen3_5 \
  --tuner_type mem2w \
  --output-dir artifacts/mem2w_dual \
  --max_steps 1000 \
  --mem2w_stage_plan W:333,C:334,W:333 \
  --mem2w_lambda_recall 1.0 \
  --mem2w_accumulation_steps 8 \
  --mem2w_loss_chunk_size 256 \
  --mem2w_insertion_index 15 \
  --mem2w_slots 512 \
  --mem2w_key_dim 256 \
  --mem2w_value_dim 256 \
  --truncation_strategy delete
```

native `mem2w` tuner 会冻结主干并断言只有四个第 16 层 Mem2W 参数可训练，Trainer
checkpoint 直接导出 `memory.safetensors`、`mem2w_config.json`、optimizer/scheduler
和标准 `trainer_state.json`。`truncation_strategy=delete` 在当前 fork 的 template
参数映射为严格的 `raise`，超长样本会报错而不会静默截断。

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

## 5. 远端 A800 的 Conda 环境（按 ms-swift 标准配置）

截至 2026-09-28，A800 训练节点上的环境已经在独立 Conda prefix 中配置并通过
单卡、双卡 DDP 的 Mem2W W/C smoke。远端连接和路径如下：

```text
ssh -p 8022 sht@106.75.235.231
project: /home/sht/haoting/Mem2W
ms-swift fork: /home/sht/haoting/ms-swift-mem2w
conda prefix: /home/sht/haoting/ms-swift-conda
model: /mnt/afs/models/Qwen3.5-4B
full data: /home/sht/haoting/data/automationbench_0916
```

每次启动前使用项目环境，而不是旧的 virtualenv：

```bash
source /mnt/afs/anaconda3/etc/profile.d/conda.sh
conda activate /home/sht/haoting/ms-swift-conda
export PYTHONPATH=/home/sht/haoting/ms-swift-mem2w:$PYTHONPATH
```

该环境的关键版本已经写入远端清单
`/home/sht/haoting/ENVIRONMENT_MANIFEST_CONDA.json`、
`/home/sht/haoting/ms-swift-conda-requirements.txt` 和
`/home/sht/haoting/ms-swift-conda-environment.yml`：

```text
Python 3.11.16                  PyTorch 2.6.0+cu124
Transformers 5.5.0             datasets 4.8.4
TRL 0.24.0                     PEFT 0.20.0
Accelerate 1.10.1               safetensors 0.6.2
huggingface_hub 1.14.0         qwen-vl-utils 0.0.14
flash-linear-attention 0.5.2   fla-core 0.5.2
ms-swift                       editable: /home/sht/haoting/ms-swift-mem2w
Mem2W                          editable: /home/sht/haoting/Mem2W
```

PyTorch、torchvision、torchaudio 和 CUDA 相关大包通过 prefix 内的符号链接复用
`/mnt/afs/anaconda3`，以避免在共享盘重复占用约 5 GB；Python 依赖本身仍由该
Conda prefix 隔离。`pip check` 唯一的提示是 `decord 0.6.0 is not supported on
this platform`（上游 wheel 的平台标签过旧），但 `import decord` 已验证成功，
不影响当前训练链路。`causal-conv1d` 因与 AFS PyTorch 的 C++ ABI 不兼容已移除，
所以 A800 稳定 profile 固定使用 `--attn-impl sdpa`；不要把这个环境描述成
`causal-conv1d` 已安装。

### A800 单卡 smoke

以下参数是当前可复现的稳定最小配置；`--gradient-checkpointing`、
`--loss-chunk-size 256` 和 `--mem2w-compute-chunk-size 2048` 对显存很关键：

```bash
CUDA_VISIBLE_DEVICES=4 SWIFT_SINGLE_DEVICE_MODE=1 \
python /home/sht/haoting/Mem2W/swift/cli/mem2w_sft.py \
  --model /mnt/afs/models/Qwen3.5-4B \
  --action-dataset /home/sht/haoting/data/smoke/action_train.jsonl \
  --recall-dataset /home/sht/haoting/data/smoke/recall_train.jsonl \
  --output-dir /home/sht/haoting/logs/smoke_conda_1gpu_0928_v2 \
  --template qwen3_5 --max-length 32768 --max-steps 2 \
  --mem2w-slots 512 --mem2w-key-dim 256 --mem2w-value-dim 256 \
  --warmup-fraction 0.5 --accumulation-steps 1 --attn-impl sdpa \
  --gradient-checkpointing --loss-chunk-size 256 \
  --mem2w-compute-chunk-size 2048 --lazy-encode \
  --distributed-backend nccl
```

该 run 已完成 W/C 两个逻辑 step、checkpoint-1/2 和四个 Mem2W 参数梯度检查；
峰值约 24.5 GiB。结果目录为
`/home/sht/haoting/logs/smoke_conda_1gpu_0928_v2`。

### A800 双卡 DDP smoke

```bash
CUDA_VISIBLE_DEVICES=4,5 SWIFT_SINGLE_DEVICE_MODE=1 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  /home/sht/haoting/Mem2W/swift/cli/mem2w_sft.py \
  --model /mnt/afs/models/Qwen3.5-4B \
  --action-dataset /home/sht/haoting/data/smoke/action_train.jsonl \
  --recall-dataset /home/sht/haoting/data/smoke/recall_train.jsonl \
  --output-dir /home/sht/haoting/logs/smoke_conda_ddp2_0928 \
  --template qwen3_5 --max-length 32768 --max-steps 2 \
  --mem2w-slots 512 --mem2w-key-dim 256 --mem2w-value-dim 256 \
  --warmup-fraction 0.5 --accumulation-steps 1 --attn-impl sdpa \
  --gradient-checkpointing --loss-chunk-size 256 \
  --mem2w-compute-chunk-size 2048 --lazy-encode \
  --distributed-backend nccl
```

双卡 run 已验证 `world_size=2`、NCCL、W/C 两阶段、checkpoint-1/2 以及四个
memory 参数的梯度；每卡峰值仍约 24.5 GiB。DDP 会提升样本吞吐，但不会把单条
超长 recall 样本自动切到多卡；正式全量训练前仍需按数据预检报告确认最长样本的
显存 profile。两个验证结果和完整依赖清单均保留在上述远端目录，便于复现和审计。
