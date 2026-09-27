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
