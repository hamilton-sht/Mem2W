# Muxi C500 上的 Mem2W SFT runbook

这份 runbook 固定了从 0916 AutomationBench 归档到 native ms-swift `qwen3_5`
paired trainer 的可复现实验链路。数据是逐字无损导出；训练入口使用
`truncation_strategy=raise`，不会因为长度超限而静默截断、压缩或分片监督目标。

## 环境核验

当前核验的远端是 `memrl-40949`（Muxi MetaX C500）。容器内 `mx-smi` 报告 4 张
卡，每张 64 GiB，测试时均为空闲。native Qwen3.5-4B 可以加载；FlashAttention
forward/backward probe 已通过。

```text
host: memrl-40949
model: /mnt/public/model/Qwen3.5-4B
data: /mnt/public/haoting/mem2w_data/automationbench_0916
swift fork: /mnt/public/haoting/ms-swift-mem2w
venv: /mnt/public/haoting/venvs/mem2w-smoke
```

全量 native template 预检结果保存在远端：
`/mnt/public/haoting/mem2w_data/preflight_full_0927.json`。6804 条 action 和
864 条 recall 均能用 `qwen3_5` 编码，监督 token 非零，payload hash 全部匹配，且
没有超过 Qwen3.5 的 262144-token context limit。recall 的最大样本是 212062
tokens；这代表模型上下文合法，不代表在单张 64-GiB 卡上已经完成过这个长度的
反向传播。

## 已完成的 Muxi smoke

## 直接脚本启动（推荐）

启动器位于 [scripts/run_muxi_mem2w_sft.sh](/Users/haoting/Documents/ChatGPT/Mem2W/scripts/run_muxi_mem2w_sft.sh)。它会在启动前校验模型、两个 JSONL、无损预检报告和 source SHA256，并创建输出目录锁；运行中写入 `launcher.log`、`launcher_status.json`、checkpoint 和训练摘要。预检或设备放置不匹配会直接失败，不会截断、压缩或静默跳过样本。

远端同步后可直接运行：

```bash
source /mnt/public/haoting/venvs/mem2w-smoke/bin/activate
cd /mnt/public/haoting/ms-swift-mem2w
MODE=model_parallel MAX_STEPS=2 \
  /mnt/public/haoting/ms-swift-mem2w/scripts/run_muxi_mem2w_sft.sh
```

`model_parallel` 是当前已验证的稳定路径：一个进程使用 `device_map=auto` 跨 4 张 C500，保留完整上下文；首次运行可能触发 MACA kernel 编译。若需四进程数据并行，可设置 `MODE=data_parallel`，但单卡长上下文受限。`SEQUENCE_PARALLEL_SIZE=4` 已接入实验入口，但仍需单独完成 Muxi kernel 编译后的完整验证，暂不作为默认 profile。

2026-09-28 直接脚本 smoke 已完成：全量 canonical 数据中的一条真实 recall 行（164096 tokens）完成 1 个 paired step，loss_action=0.6074、loss_recall=1.1843、峰值显存 43.8 GB，4 个 Mem2W 参数均被保存；首步耗时 221.6 秒（包含首次 kernel 编译）。

当前已知边界：canonical 数据最长 recall 行为 212062 tokens。普通 4 卡 model-parallel 在该行的反向峰值仍会尝试额外分配 3.64 GiB 而 OOM；因此不能宣称“6804+864 全量、最长行全部训练完成”。这不是静默截断：启动器会保留失败现场并标记 `status=failed`。要覆盖该行，需要完成 sequence-parallel Muxi profile 或更低层的 Qwen3.5 长序列 kernel 优化。

使用真实的无损数据行（不是压缩副本）完成了 2 个 paired logical steps：

```text
W: step 1, action + recall backward, all four memory parameters receive gradients
C: step 2, recall content path detached, W_Q/K still receive gradients
```

配置是 `flash_attn`、non-reentrant gradient checkpointing、chunked hidden CE
（只分块输出投影，不修改 token 或 target）和 lazy encoding。远端结果：

```text
output: /mnt/public/haoting/mem2w_data/lossless_fa_gc_0927
peak memory: about 17.5 GiB on the 20k-token recall smoke row
checkpoints: checkpoint-1 and checkpoint-2, each with memory.safetensors,
             optimizer.pt, scheduler.pt, rng_state.pt and dual state
```

## 复现命令

先在容器内设置 native fork 的 import path：

```bash
export PYTHONPATH=/mnt/public/haoting/venvs/mem2w-smoke/lib/python3.10/site-packages:/mnt/public/haoting/ms-swift-mem2w
```

短 smoke（已验证）：

```bash
python /mnt/public/haoting/ms-swift-mem2w/swift/cli/mem2w_sft.py \
  --model /mnt/public/model/Qwen3.5-4B \
  --action-dataset /mnt/public/haoting/mem2w_data/automationbench_0916_smoke/action_train.jsonl \
  --recall-dataset /mnt/public/haoting/mem2w_data/automationbench_0916_smoke/recall_train.jsonl \
  --output-dir /mnt/public/haoting/mem2w_data/lossless_fa_gc_0927 \
  --template qwen3_5 --max-length 32768 --max-steps 2 \
  --mem2w-slots 512 --mem2w-key-dim 256 --mem2w-value-dim 256 \
  --warmup-fraction 0.5 --accumulation-steps 1 \
  --attn-impl flash_attn --gradient-checkpointing \
  --loss-chunk-size 256 --lazy-encode
```

## 多卡并行（已验证 2 卡和 4 卡）

Muxi 的 NCCL backend 在 2 张和 4 张 C500 上均完成 CUDA tensor all-reduce。native
paired trainer 对四个 Mem2W 参数显式做梯度平均，并按 rank 分片 action/recall
行；只有 rank 0 写 checkpoint。由于 ms-swift 在多卡可见时会默认尝试 MP+DDP，当前
Qwen3.5 环境还需要关闭这个自动 MP 分支：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SWIFT_SINGLE_DEVICE_MODE=1
export PYTHONPATH=/mnt/public/haoting/venvs/mem2w-smoke/lib/python3.10/site-packages:/mnt/public/haoting/ms-swift-mem2w

/mnt/public/haoting/venvs/mem2w-smoke/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  /mnt/public/haoting/ms-swift-mem2w/swift/cli/mem2w_sft.py \
  --model /mnt/public/model/Qwen3.5-4B \
  --action-dataset /mnt/public/haoting/mem2w_data/automationbench_0916_smoke/action_train.jsonl \
  --recall-dataset /mnt/public/haoting/mem2w_data/automationbench_0916_smoke/recall_train.jsonl \
  --output-dir /mnt/public/haoting/mem2w_data/lossless_ddp4_0927 \
  --template qwen3_5 --max-length 32768 --max-steps 2 \
  --warmup-fraction 0.5 --accumulation-steps 1 \
  --attn-impl flash_attn --gradient-checkpointing \
  --loss-chunk-size 256 --lazy-encode --distributed-backend nccl
```

该命令已在 4 卡上完成 W/C 两步，并生成两个 rank-0 checkpoint。多卡数据并行会
提高吞吐并让不同样本分布到不同卡，但不会把一个 212k-token 样本自动切到多张卡；
最长无损样本仍需要 sequence parallel 或模型/激活切分才能解决单样本 OOM。

2 卡 DDP 的短无损 paired smoke 也已通过（W/C 各两步、四个 Mem2W 参数均有梯度、
checkpoint 可写出），结果目录为 `/mnt/public/haoting/mem2w_data/lossless_ddp2_0927`。
但是对 canonical 长样本，2 卡 `model_parallel` 在首个约 164k-token 样本的反向重算阶段
于 GPU1 OOM（已分配约 47.94 GiB，reserved 约 14.37 GiB，还需 2.51 GiB）。因此
2 卡可用于短/中等样本 DDP smoke，不可把当前 2 卡 profile 视为长样本稳定训练方案。

本轮还探测了 `SEQUENCE_PARALLEL_SIZE=2`：212062-token 样本完成了双 rank 初始化和
前向/反向，但 Mem2W 梯度为 non-finite；约 164k-token canonical 样本则停在首次
Inductor/MACA kernel 编译，尚未得到 `step_completed`。sequence-parallel 仍是实验入口，
暂不作为稳定训练 profile。

## 四卡模型并行长样本探针

单进程、`device_map=auto` 的四卡模型并行也已验证。完整数据前缀的 164096-token
recall 可以完成实际 W/C 两步训练：

```bash
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/mnt/public/haoting/venvs/mem2w-smoke/lib/python3.10/site-packages:/mnt/public/haoting/ms-swift-mem2w

/mnt/public/haoting/venvs/mem2w-smoke/bin/python \
  /mnt/public/haoting/ms-swift-mem2w/swift/cli/mem2w_sft.py \
  --model /mnt/public/model/Qwen3.5-4B \
  --action-dataset /mnt/public/haoting/mem2w_data/automationbench_0916/action_train.jsonl \
  --recall-dataset /mnt/public/haoting/mem2w_data/automationbench_0916/recall_train.jsonl \
  --output-dir /mnt/public/haoting/mem2w_data/lossless_mp_w_c_0927 \
  --template qwen3_5 --max-length 262144 --max-steps 2 \
  --warmup-fraction 0.5 --accumulation-steps 1 \
  --attn-impl flash_attn --gradient-checkpointing \
  --loss-chunk-size 256 --lazy-encode
```

```text
/mnt/public/haoting/mem2w_data/lossless_mp_w_c_0927
step 1: W, 164096 recall tokens, completed
step 2: C, 134335 recall tokens, completed
checkpoint-1 and checkpoint-2: present
```

为支持模型并行，chunked CE 现在会在 hidden、lm_head 和 labels 分属不同 MACA
设备时显式搬运 target，并在正确的输出设备上合并分块 loss。最长的 212062-token
recall 仍在 GPU3 约 54.36 GiB 已分配时需要额外 3.64 GiB，因 OOM 失败；这是一条
明确的显存边界，不是数据或模板错误。

正式全量运行前必须保留完整 context，并先读取 `preflight_full_0927.json`：

```bash
python /mnt/public/haoting/ms-swift-mem2w/swift/cli/mem2w_sft.py \
  --model /mnt/public/model/Qwen3.5-4B \
  --action-dataset /mnt/public/haoting/mem2w_data/automationbench_0916/action_train.jsonl \
  --recall-dataset /mnt/public/haoting/mem2w_data/automationbench_0916/recall_train.jsonl \
  --output-dir /mnt/public/haoting/mem2w_data/lossless_full_run \
  --template qwen3_5 --max-length 262144 \
  --attn-impl flash_attn --gradient-checkpointing \
  --loss-chunk-size 256 --lazy-encode \
  --max-steps <N> --warmup-fraction 0.2
```

这条命令不会截断样本；如果最长 recall 在单卡上触发 OOM，应采用 sequence
parallel/长序列内存优化或按样本调度到更大显存，而不是修改 payload。当前 native
paired trainer 是一进程一设备的 memory-only trainer，4 卡可以做 data parallel
式的后续扩展，但还没有在本次 smoke 中宣称 212k-token 样本的单卡全量反向已完成。

边界探测的实际记录是：最长 `hr.visa_expiration_monitoring:epoch:1:train:recall`
在 212062 tokens 时于 C500 64-GiB 单卡 OOM（PyTorch 已分配约 52.88 GiB，下一次
分配需要 1.62 GiB）。因此当前“全链路打通”的验收口径是数据无损、模板/mask/hash
预检通过、短无损 paired W/C smoke 和 checkpoint 通过；不是宣称最长样本已经完成
单卡训练。
