# 两组实验状态与实际代码链路

检查时间：2026-09-30 13:36–13:40（Asia/Shanghai）。本次只读检查远程训练，没有停止、重启或修改任务。

## 1. 结论

两组任务均在运行，扫描当前训练日志未见 OOM、NaN 或异常栈。但 **Mem2W C 阶段存在 optimizer 参数漂移，不能判为正确性通过**。

| 项目 | LoRA | Mem2W |
| --- | --- | --- |
| 真实阶段顺序 | C → W1 → W2 | W1 → C → W2 |
| GPU | A800 2、3；sequence parallel=2 | A800 4、5；sequence parallel=2 |
| 已完成 | C 108/108，checkpoint-108 存在 | W1 851/851，checkpoint-851 存在 |
| 当前采样进度 | W1 765/851（89.9%） | C 4/108；global_step=855/1810 |
| 最后阶段 | W2 尚未开始，watcher PID 3689574 等待 W1 | W2 尚未开始 |
| 阶段用时 | C 5h57m11s；W1 已 3h14m55s | W1 累计 step 计算 2h22m03s；C 前4步 22m34s |
| 实测速度 | W1 累计 15.29 s/step；最近10步 13.6 s/step | W1 平均10.01 s/step、末10步7.19 s/step；C平均338.53 s/step |
| 最近 loss / gradient norm | loss=1.4641，grad_norm=7.9039 | recall loss=0.7114，memory_grad_norm=0.2236 |
| GPU瞬时利用率 | 98% / 98% | 100% / 99% |
| GPU瞬时已用显存 | 34285 / 33033 MiB | 80105 / 78765 MiB（已接近80GiB） |

估算（不是保证）：LoRA W1 日志预计还需约22分钟，若 W2 速度相似，全链还需约4小时。Mem2W 按前4个 C step 外推，剩余 C 约9.8小时，再加 W2 约2.4小时；C 样本长度不同且存在下面的正确性问题，这个时间只表示当前运行配置的计算量。

W1 按 epoch 保存，所以 LoRA W1 目前没有中间 checkpoint；不是保存失败。Mem2W 每个 step 保存，检查时已有855个 checkpoint。LoRA C→W1 只加载 adapter 权重，重新建立 optimizer/scheduler，不是连续恢复完整优化器状态。

注意：此前用户要求过 C→W→W；当前 Mem2W 被改成 W→C→W，属于实验设置改变，不应称为与 LoRA 同顺序的对照。本次未再调整顺序。

## 2. Mem2W 新发现的问题：detach 不等于 optimizer 冻结

C 阶段的代码用 `V.detach()`、`W_O.detach()` 切断 recall 对内容参数的梯度。
但 `_synchronize_gradients()` 又把缺失梯度 `None` 补成了零张量，然后统一 `optimizer.step()`。
checkpoint-851 的 AdamW state 显示已有非零动量，weight_decay 实际为 **0.01**（构造器未显式设置，沿用默认值），不是早期计划写的0。

实测证据：

- step855：V/W_O 梯度范数均为0，但本步参数差值范数分别为0.0022163 / 0.0112749。
- 直接读取 checkpoint-851 与 checkpoint-855 的 memory.safetensors：V 累计差值范数0.0112845，W_O 累计差值范数0.0531543。
- 因此这不只是日志显示问题，而是实参在 C 阶段被 Adam 动量/weight decay 更新。

对应位置：

- [memory.py](../integrations/ms-swift/20260929/files/swift/mem2w/memory.py)：`forward()` 中的内容参数 detach。
- [dual_trainer.py](../integrations/ms-swift/20260929/files/swift/mem2w/dual_trainer.py)：`_synchronize_gradients()`（328行）和 `train_step()` 中的 optimizer.step。
- [mem2w_sft.py](../integrations/ms-swift/20260929/files/swift/cli/mem2w_sft.py)：217行构建 AdamW。

建议后续修复方向（本次未实施）：C step 在 optimizer 层排除 V/W_O 的更新，包括动量和衰减；增加“先 W 建立 optimizer 状态，再 C 断言内容参数逐值不变”的回归测试。恢复点可选未受 C 漂移影响的 checkpoint-851，具体重跑方案需要确认。

其他未完成的正确性审查：Mem2W SP 在切分前重复计数完整 labels，使用完整样本 token 数给 local mean loss 加权，尚未证明与单卡目标归一化等价；不能把其 loss 与 LoRA loss 直接横向比较，也不能把 tokens_recall 当独立全局 token 总数。阶段游标已重置，但851×8=6808，会用 modulo 重复4条 action，不能称严格无重复遍历。通用 chunked loss 的 loss_scale/累积归一化还需独立等价测试。

## 3. LoRA 实际执行链路

```text
原始 AutomationBench action / recall JSONL
  → scripts/normalize_ms_swift_tool_calls.py
      规范化工具调用结构 / arguments 表示
  → scripts/prepare_ms_swift_sft_jsonl.py
      生成仅包含 messages、消息 loss 标记、template kwargs 的稳定训练视图
  → W1_fixed2 实际 torchrun 命令
      swift/cli/sft.py --tuner_type lora --adapters C/checkpoint-108
  → swift/pipelines/train/sft.py : SwiftSft
      HF Datasets → 原生 qwen3_5 template → PEFT LoRA → Trainer
  → swift/trainers/seq2seq_trainer.py
      _prepare_inputs → sequence_parallel.prepare_inputs
      compute_loss → _chunked_sft_loss
  → swift/mem2w/dual_trainer.py : chunked_hidden_cross_entropy
      LM head / CE 分块256；不截断上下文；处理本rank零监督标签
  → Seq2SeqTrainer.training_step → _sync_chunked_sft_grads
      原生 Trainer/Accelerate 的 optimizer、scheduler、epoch保存
  → W1_fixed2/checkpoint-851（尚未生成）
  → scripts/continue_muxi_lora_w2_fixed.sh
      等W1进程退出并检查adapter文件 → W2_fixed2
```

LoRA 不注入 PersistentKVMemory。虽然复用了 `swift/mem2w/dual_trainer.py` 中的 CE helper，但没有运行 Mem2WDualObjectiveTrainer。

### LoRA 改过哪些代码

| 文件 | 修改内容 |
| --- | --- |
| [normalize_ms_swift_tool_calls.py](../scripts/normalize_ms_swift_tool_calls.py) | 工具调用结构规范化，不压缩正文 |
| [prepare_ms_swift_sft_jsonl.py](../scripts/prepare_ms_swift_sft_jsonl.py) | 去除非训练metadata，避免HF Arrow的string→null类型冲突 |
| [run_muxi_lora_cww_2gpu.sh](../scripts/run_muxi_lora_cww_2gpu.sh) | 为后续启动固化规范化视图及C/W/W串联流程；当前修复版W1由单独torchrun启动 |
| [continue_muxi_lora_w2_fixed.sh](../scripts/continue_muxi_lora_w2_fixed.sh) | 当前正在运行的W2接续watcher |
| [seq2seq_trainer.py](../integrations/ms-swift/20260929/files/swift/trainers/seq2seq_trainer.py) | SP输入准备、chunked CE、绕过DDP wrapper后的手动梯度同步 |
| [dual_trainer.py](../integrations/ms-swift/20260929/files/swift/mem2w/dual_trainer.py) | 公共CE helper：已shift labels、零本地监督token时返回连图零损失 |
| [sft_args.py](../integrations/ms-swift/20260929/files/swift/arguments/sft_args.py) 与 [pipelines/train/sft.py](../integrations/ms-swift/20260929/files/swift/pipelines/train/sft.py) | 注册/传递sft_loss_chunk_size；数据schema相关接入 |
| [model/models/qwen.py](../integrations/ms-swift/20260929/files/swift/model/models/qwen.py) | 两组共用的FLA chunk-size控制及fallback开关 |

历史两个失败：W1.log 的 `Couldn't cast array of type string to null`；W1_fixed.log 的 `branch has zero supervised tokens`。后者是我们CE helper对SP边界的处理错误，不是LoRA初始化问题。当前W1_fixed2已越过两处错误并持续训练。

**不能称为完全未修改的上游 LoRA**：其训练生命周期是原生 ms-swift，但启用 `sft_loss_chunk_size` 后走的是我们新增的 loss/梯度同步路径。

## 4. Mem2W 实际执行链路

```text
scripts/run_muxi_mem2w_cww_2gpu.sh（文件名仍为cww，实际plan为WCW）
  → check_muxi_mem2w_data.py（完整数据审计）
  → torchrun swift/cli/mem2w_sft.py --stage-plan W:851,C:108,W:851
      read_jsonl → SftArguments.get_model_processor
      → sequence_parallel.prepare → qwen3_5 template
  → swift/tuners/mem2w.py : Mem2WTuner.prepare_model
  → swift/mem2w/integration.py : attach_memory → freeze_memory_only
      block15（第16层）输出forward hook；冻结全部Qwen原参数
  → swift/mem2w/memory.py : PersistentKVMemory
      RMSNorm(H) → W_Q → 对512个K/V槽做单头attention → W_O → 残差相加
  → swift/mem2w/dual_trainer.py : Mem2WDualObjectiveTrainer
      _stage_and_local_update → W只取action / C只取recall
      template.encode / data_collator → SP.prepare_inputs
      _forward_loss → chunked_hidden_cross_entropy → backward
      _synchronize_gradients → clip_grad_norm → AdamW.step → scheduler.step
  → CLI每步 Mem2WTuner.save_pretrained
      memory.safetensors + mem2w_config.json + optimizer/scheduler/RNG/step
```

### Mem2W 改过哪些代码

- 新增模型部分：[config.py](../integrations/ms-swift/20260929/files/swift/mem2w/config.py)、[memory.py](../integrations/ms-swift/20260929/files/swift/mem2w/memory.py)、[integration.py](../integrations/ms-swift/20260929/files/swift/mem2w/integration.py)、[tuners/mem2w.py](../integrations/ms-swift/20260929/files/swift/tuners/mem2w.py)。
- 新增独立训练入口：[cli/mem2w_sft.py](../integrations/ms-swift/20260929/files/swift/cli/mem2w_sft.py)、[dual_trainer.py](../integrations/ms-swift/20260929/files/swift/mem2w/dual_trainer.py)。
- 本轮已修：W/C只走当前数据分支、不再每步同时计算两路；阶段内游标重置；零本地labels；跨rank loss日志；C零梯度保护；启动顺序改为WCW。
- 原生Trainer集成也存在：[trainers/mem2w_trainer.py](../integrations/ms-swift/20260929/files/swift/trainers/mem2w_trainer.py)、TrainerFactory、SwiftSft数据接入、tuner mapping、参数注册、mixin checkpoint恢复。**当前Mem2W进程没有走这个Trainer子类**，走的是上面的独立循环。
- 未修改上游 `qwen3_5` 模板文件或 `swift/sequence_parallel/`，是在我们的入口调用上游实现。

完整ms-swift差异为29文件（17新增、12修改，含测试与文档），不是29处训练算法修改。查看 [完整文件索引](../integrations/ms-swift/20260929/INDEX.md)、[完整补丁](../integrations/ms-swift/20260929/changes.patch)。

## 5. 位置、证据与推荐阅读顺序

远程实际源码：`a800-mem2w:/home/sht/haoting/ms-swift-mem2w`。
本地审查快照：`integrations/ms-swift/20260929/files/`；是差异文件集合，不是完整可运行checkout。
本次核对5个关键文件（dual_trainer、CLI、seq2seq_trainer、memory、integration）SHA-256，本地快照与远程磁盘一致；磁盘一致不单独证明进程加载版本，运行日志提供上述阶段/参数行为的补充证据。

运行目录：

```text
Mem2W: /home/sht/haoting/runs/mem2w_wcw_fixed_2gpu_4b_sp2_20260930_1120
LoRA:  /home/sht/haoting/runs/lora_cww_sp2_2gpu_4b_20260929_2045
```

优先阅读：本报告 → dual_trainer.py（328/346行）→ cli/mem2w_sft.py（217行）→ memory.py → seq2seq_trainer.py（133行）→ 两组启动脚本 → 全量patch。

本次没有重跑GPU kernel导入测试，不把启动参数/历史skill中的版本记录当成当前kernel实测证明；当前GPU速度、loss与checkpoint均来自本次读取。历史结构图只用于说明模块概念，准确串联应以本报告为准：block1–16 → Mem2W → block17–32 → LM head。
