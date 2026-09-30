# ms-swift 内的 Mem2W 改动清单与审查入口

审查日期：2026-09-29。本文件描述服务器**磁盘源码快照**，不把历史口头说明当作实现证据。

## 1. 对比范围与交付物

- 来源：A800 `/home/sht/haoting/ms-swift-mem2w`，该目录没有 `.git`。
- 基线：本地 `/Users/haoting/Documents/ChatGPT/ms-swift-mem2w` 保存的上游提交 `c08110b30a1ccb60bcfb70adf87d2cd72f5b9f3c`。
- 全量文件内容逐字节对比：**29 个源码/测试/文档文件，17 个新增、12 个修改，无删除**。
- [完整逐文件索引](../integrations/ms-swift/20260929/INDEX.md)、[完整 unified diff](../integrations/ms-swift/20260929/changes.patch)、[SHA-256 清单](../integrations/ms-swift/20260929/manifest.json)。
- `files/` 只收录相对基线有变化的文件，不是一个完整可运行的 ms-swift checkout；补丁应应用到对应上游 commit。
- 排除缓存、egg-info，以及 `swift/model/models/qwen.py.bak_20260929_chunk`、`qwen.py.bak_20260929_fla` 两个旧备份（名称及哈希记录在 manifest）。保留上游 Apache 2.0 LICENSE。
- 不覆盖本地另一个 ms-swift 工作目录的未提交修改，不修改远程代码、不停止/重启训练。

## 2. 优先检查：两条训练入口并不等价

| 入口 | 实际调用 | W/C 语义 | 生命周期 |
| --- | --- | --- | --- |
| `swift/cli/mem2w_sft.py`（CWW 任务命令） | `Mem2WDualObjectiveTrainer` | 显式 stage plan 下 W 只计算 action、C 只计算 recall；每个阶段重置数据游标 | 自建 optimizer、scheduler、循环、手动 all-reduce、每步保存 |
| `swift sft --tuner_type mem2w` | `Mem2WDualTrainer(Seq2SeqTrainer)` | W 只计算 action；C 只计算 recall 并 detach V/W_O | ms-swift/Transformers Trainer，覆盖 paired training_step |
| `swift sft --tuner_type lora --sft_loss_chunk_size 256` | 修改过的 `Seq2SeqTrainer` | 每次调用只用一份 dataset；外层脚本串联 C/W/W | 原生 Trainer 生命周期，但 loss 和梯度同步有自定义修改 |

**运行注意**：修复后的源码已同时更新到 `ms-swift-mem2w` 和本目录快照；已经启动的旧进程不会热加载新代码。旧进程若在修复前启动，仍应按历史混合 action+recall 结果处理，不能当作严格的 C-only → W-only → W-only 实验。普通 LoRA 也不是完全未修改的上游 loss 路径。

对应代码：

- [CLI run / 循环与 checkpoint](../integrations/ms-swift/20260929/files/swift/cli/mem2w_sft.py)（run 约 142 行；构造 trainer 约 249 行）。
- [自定义 train_step](../integrations/ms-swift/20260929/files/swift/mem2w/dual_trainer.py)（316 行）：先构造两路各 8 个 batch，然后两路各 backward，最后一次 optimizer.step。
- [原生 Trainer 分阶段选择](../integrations/ms-swift/20260929/files/swift/trainers/mem2w_trainer.py)（188 行 `_iter_loss_terms`；273 行 `training_step`）。

## 3. 全部 29 个文件及用途

所有链接均指向本次从服务器提取的本地快照。

### 模型层、参数与 checkpoint：6 个新增

| 文件 | 用途 |
| --- | --- |
| [swift/mem2w/config.py](../integrations/ms-swift/20260929/files/swift/mem2w/config.py) | slots、key/value dim、插入层、RMS epsilon、dropout、compute chunk 配置和校验 |
| [swift/mem2w/memory.py](../integrations/ms-swift/20260929/files/swift/mem2w/memory.py) | 四参数 W_Q/K/V/W_O；RMSNorm→query→KV attention→残差；W_O 零初始化；C 的 content detach；CUDA SDPA 与无损计算分块 |
| [swift/mem2w/integration.py](../integrations/ms-swift/20260929/files/swift/mem2w/integration.py) | 查找 decoder；第 16 个 block 后 forward hook；保持输出结构；冻结主干；只开放 4 个 memory 张量；运行期开关 |
| [swift/mem2w/__init__.py](../integrations/ms-swift/20260929/files/swift/mem2w/__init__.py) | 导出 config、memory、attach/freeze 等 API |
| [swift/tuners/mem2w.py](../integrations/ms-swift/20260929/files/swift/tuners/mem2w.py) | Mem2WTuner prepare/save/load；保存 memory.safetensors 与 config；恢复形状和命名 |
| [swift/mem2w/dual_trainer.py](../integrations/ms-swift/20260929/files/swift/mem2w/dual_trainer.py) | 自定义 paired 循环；chunked hidden CE；shifted labels；SP prepare_inputs 时机；梯度同步及指标 |

### 原生 pipeline、配置与注册：11 个修改、2 个新增

| 状态 | 文件 | 用途 |
| --- | --- | --- |
| 修改 | [swift/arguments/tuner_args.py](../integrations/ms-swift/20260929/files/swift/arguments/tuner_args.py) | 新增 memory 超参数 |
| 修改 | [swift/arguments/base_args/base_args.py](../integrations/ms-swift/20260929/files/swift/arguments/base_args/base_args.py) | 允许 checkpoint 参数恢复 Mem2W 配置 |
| 修改 | [swift/arguments/sft_args.py](../integrations/ms-swift/20260929/files/swift/arguments/sft_args.py) | recall dataset、stage plan、loss/accumulation/offload 参数；Mem2W 校验；通用 sft_loss_chunk_size |
| 修改 | [swift/tuner_plugin/__init__.py](../integrations/ms-swift/20260929/files/swift/tuner_plugin/__init__.py) | tuner 导出 |
| 修改 | [swift/tuner_plugin/mapping.py](../integrations/ms-swift/20260929/files/swift/tuner_plugin/mapping.py) | 注册 tuner_type=mem2w |
| 修改 | [swift/cli/main.py](../integrations/ms-swift/20260929/files/swift/cli/main.py) | 注册独立 mem2w CLI 子命令 |
| 新增 | [swift/cli/mem2w_sft.py](../integrations/ms-swift/20260929/files/swift/cli/mem2w_sft.py) | 独立入口；模型与模板加载；SP 初始化；4 参数断言；自定义 optimizer/scheduler/resume/保存 |
| 修改 | [swift/pipelines/train/sft.py](../integrations/ms-swift/20260929/files/swift/pipelines/train/sft.py) | 显式 Arrow tool-call schema；加载 action/recall；配对数据；向 training_args 传递新增参数 |
| 新增 | [swift/trainers/mem2w_trainer.py](../integrations/ms-swift/20260929/files/swift/trainers/mem2w_trainer.py) | 配对 dataset/collator；原生 Trainer 子类；按阶段分支 loss；逐 microbatch backward；边界保存 |
| 修改 | [swift/trainers/__init__.py](../integrations/ms-swift/20260929/files/swift/trainers/__init__.py) | 导出 Mem2WDualTrainer |
| 修改 | [swift/trainers/trainer_factory.py](../integrations/ms-swift/20260929/files/swift/trainers/trainer_factory.py) | tuner_type=mem2w 路由到新增 Trainer |
| 修改 | [swift/trainers/mixin.py](../integrations/ms-swift/20260929/files/swift/trainers/mixin.py) | Trainer resume 支持 memory.safetensors/memory.pt |
| 修改 | [swift/trainers/seq2seq_trainer.py](../integrations/ms-swift/20260929/files/swift/trainers/seq2seq_trainer.py) | 通用 LoRA/full chunked CE；绕开整段 vocab logits；处理 SP shifted labels；手动同步可训练梯度 |

### Qwen 内核：1 个修改

| 文件 | 用途 |
| --- | --- |
| [swift/model/models/qwen.py](../integrations/ms-swift/20260929/files/swift/model/models/qwen.py) | `SWIFT_QWEN35_DISABLE_FLA` fallback 开关；`SWIFT_QWEN35_FLA_CHUNK_SIZE` 16/32/64；为长序列控制 FLA 内核分块 |

未修改上游 `qwen3_5` 模板文件，也未修改 `swift/sequence_parallel/` 文件；复用的是上游实现。兼容修复主要在调用入口和 loss 处理位置。

### 测试、探针与说明：9 个新增

| 文件 | 用途 |
| --- | --- |
| [scripts/mem2w_dist_probe.py](../integrations/ms-swift/20260929/files/scripts/mem2w_dist_probe.py) | 分布式通信 smoke |
| [scripts/mem2w_gradient_routing_smoke.py](../integrations/ms-swift/20260929/files/scripts/mem2w_gradient_routing_smoke.py) | W/C 梯度路由 smoke |
| [scripts/mem2w_lossless_preflight.py](../integrations/ms-swift/20260929/files/scripts/mem2w_lossless_preflight.py) | 无损模板编码、长度与监督检查 |
| [scripts/mem2w_maca_probe.py](../integrations/ms-swift/20260929/files/scripts/mem2w_maca_probe.py) | MACA 设备探针 |
| [scripts/mem2w_qwen_smoke.py](../integrations/ms-swift/20260929/files/scripts/mem2w_qwen_smoke.py) | 小型 Qwen forward/backward/checkpoint 验证 |
| [tests/test_mem2w_chunked_loss.py](../integrations/ms-swift/20260929/files/tests/test_mem2w_chunked_loss.py) | 分块 CE 与普通 CE 的数值/梯度比较 |
| [tests/test_mem2w_memory_tiling.py](../integrations/ms-swift/20260929/files/tests/test_mem2w_memory_tiling.py) | memory 分块与非分块比较 |
| [tests/test_mem2w_native.py](../integrations/ms-swift/20260929/files/tests/test_mem2w_native.py) | 插入、冻结、梯度、保存恢复基本测试 |
| [docs/source/Customization/Mem2W.md](../integrations/ms-swift/20260929/files/docs/source/Customization/Mem2W.md) | fork 内使用说明；较早文档，不应替代源码核对 |

## 4. 实现审查时需要重点关注的地方（尚未修复）

1. **“第16层全调”的含义**：`freeze_memory_only()` 冻结整个 Qwen 主干，包括原第16层；只训练插入其后的 4 个 memory 参数，绝不是原 decoder block 第16层所有权重全调。
2. **两种入口仍需区分**：修复后的自定义循环和原生 Trainer 都按显式 stage plan 只走一路；但是已运行的旧进程使用的是修复前代码，不能与新 smoke 或新实验混合统计。
3. **C-only 从零初始化开始的风险**：`W_O` 初始化为 0，而 C recall 将 V/W_O detach；如果没有先学到非零 W_O 或加载非零 checkpoint，recall 对 W_Q/K 的梯度可能为零。这需要专门测试；此前“首个 C step 只有 W_O 更新符合 C 设计”的说明不成立，那次更新可能来自 action 分支。
4. **SP token 归一化/日志**：自定义循环在 prepare_inputs 之前统计完整样本 labels，然后对各 rank 做 SUM；SP rank 共享完整样本，这会重复计数。随后 local mean loss 和梯度同步的权重也需要核对，日志 token 数不能直接当作独立监督 token 数或用于计算绝对吞吐。
5. **原生 Mem2W Trainer 的 SP 覆盖**：其自定义 `_prepare_inputs` / `_forward_branch` 未像独立循环显式调用 SP prepare_inputs，且 `_valid_count` 与 chunked CE 默认 shift 路径不同于 LoRA SP 路径；不能把 LoRA SP 成功直接等同于该 Trainer 的 SP 验证通过。
6. **通用 chunked SFT 路径**：`_chunked_sft_loss` 丢弃 `loss_scale`，通过 base.model 绕过 DDP wrapper 并手动同步梯度。需要审核 token 权重语义、每个 microbatch 的同步开销、zero-local-labels 情况；并非完全原封不动的原生 loss。
7. **dataset epoch 边界**：独立循环两路共享 `(global_step-1)*accumulation_steps`，并通过 modulo 循环；W 阶段不会自动从 action 第0行重置。`ceil(6804/8)=851` 也会重复最后不足一组的行，不能宣称严格无重复遍历。
8. **checkpoint 策略**：独立 CLI 每个 logical step 保存而非只在 epoch 末；原生 Trainer 则有阶段边界 callback。两种入口的 I/O 与恢复语义需要分别检查。

以上为静态源码证据及待验证风险，不擅自修改训练或声称性能/数值问题已修好。磁盘文件可能在进程启动后被其他任务更新，审查快照不是运行进程已加载代码的完整证明。

## 5. 验证和 GitHub 同步状态

- 已对服务器快照与上游文件全集逐字节比较，导出全部差异文件和哈希。
- Python 文件已做 AST 语法检查。
- `git apply --check changes.patch` 在对应上游导出树上通过；这不是 GPU 数值正确性测试。
- GitHub 上传范围应只包含经过检查的代码、补丁和文档；不包含训练数据、模型/checkpoint、日志中的完整样本、认证图片或临时修补目录。
- 本地 Mem2W 未配置 Git remote；旁边 ms-swift 的 origin 是 `modelscope/ms-swift` 上游，不应把我们的修改直接推给上游。当前等待用户提供个人/组织 GitHub 目标仓库 URL（新建仓库还需可见性），尚未 push。
