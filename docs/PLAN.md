# Mem2W 新仓库、模型改造与 ms-swift SFT 实施计划

日期：2026-09-27 · 版本：v0.1 · 状态：模型层、数据管线、native W/C paired SFT 和 Muxi 4B smoke 已完成；正式 9B 长跑与 benchmark 仍待单独安排。

## 1. 本次交付与实施范围

本次文档描述独立的 `Mem2W` 仓库、Qwen3.5-9B 模型层改造及基于 ms-swift 的双目标 SFT。附件 [Mem2W 算法与需求 v0.1](/Users/haoting/Downloads/Mem2W_Qwen3.5_9B_Algorithm_and_Requirements_v0.1.md) 作为算法设计与验收依据。当前已经落地的是可插拔 memory 层、冻结主干、action/recall 数据管线、`qwen3_5` 模板和 token mask 审计、native W/C paired trainer、`memory.safetensors` 导出及 Muxi Qwen3.5-4B 两步 GPU smoke。

实现范围：持久 KV 记忆模块、受控模型接入、主干冻结、action/recall 数据接入、模板与 mask 验证、双分支 Trainer、检查点与恢复、最小推理验证。沿用已有 MemRL 系统导出的固定快照和日志，不在本阶段重写教师检索器或教师采集平台。

暂不纳入：在线记忆写入、Q 更新、RL/DPO、视觉任务、vLLM/SGLang 插件、完整 benchmark、多种子论文实验。正式训练需要真实数据与环境预检通过；本计划不预设训练时长、GPU 数量或成功率。

## 2. 仓库与运行位置

本地工作目录为 `/Users/haoting/Documents/ChatGPT/Mem2W`，代码和最小测试已落在该 Git 仓库。当前未配置远程仓库；如需推送，后续再由用户指定 GitHub 组织、可见性和 remote。

结合前文的 Muxi 环境，远端候选工作目录为 `memrl-40949:/mnt/public/haoting/Mem2W`。这是拟采用的独立目录，尚未创建或验证存储条件；既有 `/mnt/public/haoting/mrl-textgrad` 可作为教师导出来源。`memrl-tg` 当前不是本机可解析的 SSH 别名，不能直接把它写成可用连接命令。

ms-swift 以锁定 commit 的依赖接入，Mem2W 的模型扩展、数据处理和 Trainer 留在本项目中。优先使用项目级子类和注册接口；如确需修改上游，用单独的可审查补丁记录。GitHub 组织、可见性与 remote 尚未指定，本计划不假定已建立远程仓库。

当前目录（最小实现）：

```text
Mem2W/
  pyproject.toml
  README.md
  configs/default.yaml
  src/mem2w/{memory_layer,qwen_integration,checkpointing}.py
  src/mem2w/{ms_swift_adapter,ms_swift_plugin,ms_swift_train}.py
  tests/{test_memory_checkpoint,test_ms_swift_adapter}.py
  docs/{PLAN,ms_swift_sft}.md
```

正式双目标入口位于配套的 native ms-swift fork `/Users/haoting/Documents/ChatGPT/ms-swift-mem2w`，命令为 `swift mem2w-sft`；本仓库的 `mem2w-sft-train` 保留为独立 action/recall 单流兼容入口。

模型权重、真实教师正文、凭证、训练日志和结果不提交 Git。小型合成 fixture 可以提交，并明确标记其用途。

## 3. 已核对的上游接口与版本策略

本轮通过 Git 检查到 ms-swift `main` 为 `c08110b30a1ccb60bcfb70adf87d2cd72f5b9f3c`，它作为接口调查的基准，不代表已经安装或训练验证。

| 上游位置 | 本计划的接入用途 |
|---|---|
| `swift/model/models/qwen.py` | 复用 Qwen3.5 loader；该版本加载 `Qwen3_5ForConditionalGeneration`，注册依赖要求包含 `transformers>=5.2.0` |
| `swift/template/templates/qwen.py` | 使用 `qwen3_5` 模板，核验非思考模式与工具消息序列化 |
| `swift/pipelines/train/sft.py` | 基于 `SwiftSft` 建立项目训练入口，在模型准备后接入 memory |
| `swift/trainers/` 与 `TrainerFactory` | 接入项目 Trainer，控制配对 batch、两次 backward、单次更新与保存 |
| `swift/loss_scale/` | 使用消息监督标记辅助模板编码，并检查最终 token labels |

上述代码可在 [锁定的 ms-swift 源码](https://github.com/modelscope/ms-swift/tree/c08110b30a1ccb60bcfb70adf87d2cd72f5b9f3c/swift) 查看。具体子类方法和注册方式需在 P0 用锁定版本做契约测试；不能把 `external_plugins` 导入普通 Python 文件等同于完成模型/Trainer 注册。

P0 将 ms-swift commit、Transformers 实际版本、PyTorch/Accelerate、Python、设备运行时、模型与 tokenizer revision 写入 runtime lock。当前远端 `clin-swift` 环境已验证 Python 3.12、PyTorch 2.14.0+cu130、Transformers 5.16.1、ms-swift 4.5.3；实际 4B/9B GPU 训练仍需独立资源预检。`transformers>=5.2.0` 只是该上游版本的声明下界，最终采用通过 checkpoint 加载、模板和 backward 测试的固定版本。

## 4. 模型层改造

### 4.1 加载与插入位置

默认加载官方完整 `Qwen/Qwen3.5-9B` checkpoint，保留视觉模块但仅输入文本。官方配置文本侧为 `hidden_size=4096`、32 个 decoder block、词表 248320，完整架构为 `Qwen3_5ForConditionalGeneration`。模型配置依据：[官方 config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)。

在第 16 个 block 输出之后插入残差模块，零起始索引为 15。原层数、layer index、混合注意力缓存、位置参数和原 state dict 的映射必须可审计。接入时从实际加载对象定位文本 decoder，检查路径、层数与类型，不凭其他 Qwen 版本猜测成员路径。

采用受控 wrapper/子类。完整模型与文本模型的加载接口分别验证；文本专用类不是默认替代路径。相关官方接口参见 [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)。

### 4.2 持久记忆模块

```text
U = RMSNorm0(H)
A = softmax((U W_Q) K^T / sqrt(d_k))
H_out = H + (A V) W_O
```

| 参数 | 默认形状 | 初始化 | 分组 |
|---|---|---|---|
| W_Q | 4096 × 256 | Xavier uniform | read |
| K | 512 × 256 | Normal(0, 1) | read |
| V | 512 × 256 | Normal(0, 0.02) | content |
| W_O | 256 × 4096 | 全零 | content |

总计 2,359,296 个新增参数。K/V 为跨任务持久 Parameter；槽数 512 与教师返回记忆条数相互独立。没有可训练 norm、bias、gate、新 token 或独立 decoder。

`memory_enabled=False` 直接返回输入 H。RMS 统计与 softmax 使用稳定精度，新增参数默认 FP32；即使外层开启 autocast，也必须验证实际计算 dtype。逐 token 读取，padding 使用真实有效位置 mask，不拿 loss mask 代替 attention mask；缓存 decode 时只处理当前新 token。

### 4.3 冻结、梯度与缓存

注入前记录原参数身份及校验信息，冻结全部原参数；注入后验证可训练名单精确等于四组 memory 参数，optimizer 只持有这些参数。后半段冻结 block 和 LM head 仍需对输入反传，不能把整体 forward 放入 `no_grad()`。

每次 forward 显式携带不可变的分支参数 `stop_content_grad`。受限 recall 使用 `V.detach()` 和 `W_O.detach()` 的局部视图，保留 Q/K 到词表 CE 的完整梯度链。不要依赖 forward 后恢复的全局开关：checkpoint 重算发生在 backward，可能读到错误状态。首轮关闭 gradient checkpointing；后续仅在 non-reentrant 重算与梯度等价测试通过后开启。

原混合 cache 不接收新增持久 K/V。推理会话需绑定 memory revision；memory 更新、替换、禁用后令原会话缓存失效并重新 prefill，不能只切开关继续旧缓存。

## 5. 数据、模板与 mask

输入为固定教师快照对应的 episode JSONL，至少包含任务来源/分组、split、snapshot 标识与 hash、真实检索事件、教师实际注入 payload、完整 action/tool 轨迹以及 outcome。快照在蒸馏阶段禁止写回，保留成功和失败样本，reward 不参与过滤或加权。

派生两条独立数据流：

- **Action：** 保留任务、工具协议、初始观察及完整交互历史；按来源字段移除仅教师可见的 memory 消息。每个 assistant action 及需要生成的结束标记计一次 loss，system/user/tool/padding 为 `-100`。
- **Recall：** 每个真实检索事件生成一条样本；prompt 只含查询、召回指令与预先设定的 `k_req`，completion 精确等于当时注入 payload。空结果使用真实固定空结果文本；payload 内历史 tool 文本同样属于监督目标。

转换器保留结构化 `tool_calls`、参数及 `tool_call_id`，不能降格为自然语言或丢弃工具 schema。去除 memory 按结构执行；不能全局字符串替换正常任务内容。split 缺失报错，不能自动归到 train；近重复与同任务重试按来源组隔离。

交由 ms-swift 模板编码，显式设置非思考模式。逐 token 检查最终 `input_ids/labels`，输出人工可读的 token 审计样例，覆盖多轮工具调用、失败、空召回、结束标记、历史 action 只计一次，以及 tool 内容不会被监督的情况。标准因果位移只做一次；shift 后有效目标数为零须拒绝。

先统计两类长度分布。默认最大长度 4096，关闭 packing/padding-free 跨样本拼接；超长样本整条拒绝并记录原因、覆盖率与成功/失败比例。工具 JSON 和 recall payload 不允许静默截断。

## 6. 基于 ms-swift 的双目标 SFT

### 6.1 接入架构

`ms_swift_train.py` 提供独立 action/recall 单流兼容 runner，复用 ms-swift 模型/processor、模板编码、训练参数和 Trainer；native fork 额外提供 `swift mem2w-sft`，复用同一套模型加载和 `qwen3_5` template/data collator，在 native Mem2W tuner 上实现配对 dataset、双分支 token 分母、W/C 梯度路由、单次逻辑 optimizer step 和原生 checkpoint/resume。

配对 dataset/collator 每次提供 action、recall 两组张量及各自 shift 后目标 token 数，metadata 留在审计侧，不传给模型。不能直接混合两份 JSONL 后依赖 stock SFT 的一个总平均 loss；那既改变权重，也不实现 recall 分支的梯度限制。

因此 native 入口不把两份 JSONL 混入 stock SFT，而是在同一模型和 template 上显式编码两条流，再执行 paired logical update。普通 `swift sft --tuner_type mem2w` 仍保留为单流兼容 smoke path。

### 6.2 损失与阶段

`L = S_act/N_act + lambda_recall × S_recall/N_recall`，其中 S 为各分支目标 token NLL 总和，N 为该分支完整有效 batch 的目标 token 数。

| 阶段/分支 | W_Q/K | V/W_O | 主干 |
|---|---|---|---|
| W：action | 更新 | 更新 | 冻结 |
| W：recall | 更新 | 更新 | 冻结 |
| C：action | 更新 | 更新 | 冻结 |
| C：recall | 更新 | 无该分支梯度 | 冻结 |

共同预热为前 `floor(0.2 × U)` 次逻辑更新；其余为 C。W_O 零初始化使初次梯度主要到达 W_O，需要至少一轮更新后再检查 Q/K/V 链路。学习率 warmup 3% 与共同预热 20% 是独立设置。

每个逻辑更新：清梯度 → 汇总该更新的两类 token 分母 → action microbatches 顺序 forward/backward → recall microbatches 顺序 forward/backward → 检查/clip → 一次 optimizer step → 一次 scheduler step。默认每分支 microbatch 1、累积 8，配对后仍只算一次逻辑更新。

不能累加 microbatch 的平均 loss 再平均，长度不同会改变目标。单卡每个 microbatch 使用 `NLL_sum/N_branch_total`；DDP 按全局分母并考虑梯度平均系数。例如标准 world-size 平均语义下，每卡 numerator 乘 `world_size/N_branch_global`。Accelerate 的自动累积缩放只能应用一次，必须用不等长、多 microbatch、双进程参考测试证明等价。DeepSpeed/FSDP 延后至单卡与 DDP 验收通过。

### 6.3 默认超参数与消融

AdamW：lr `1e-4`、betas `(0.9,0.999)`、weight decay `0`、grad clip `1`；cosine 调度，LR warmup `0.03`；`lambda_recall=1`；pilot 暂定 1000 逻辑步、每 100 步验证。

配置同时保留 `action_only`、`full_memory_both_losses`、`strict_no_warmup` 三个变体。默认值是工程起点，槽数/维度/插入位置允许显式消融，不应被配置校验误当成永不可改的硬约束。双目标数学与无泄漏等约束仍须保持。

## 7. 保存、恢复和最小推理

部署包计划包含 `memory.safetensors`、`memory_config.json`、`base_model_lock.json`、模板 manifest 和训练 manifest。只保存新增参数，校验名称/形状/插入位置/基础模型 revision；禁止未知字段静默忽略。实际序列化后报告文件大小。

恢复 checkpoint 另外保存 optimizer/scheduler、逻辑 update、阶段、随机数、两类 sampler 位置、累积边界及分布式状态。首版在逻辑更新边界保存；恢复后按全程总更新预算继续阶段调度。通过连续训练与中断恢复对照检查 loss 和参数轨迹。

推理直接输入任务历史并输出行动，使用原 LM head。通过禁用外部记忆文件访问验证部署路径；自由 recall 是独立诊断接口，不自动插入 action 前。

## 8. 执行顺序与验收门

| 阶段 | 主要工作 | 完成条件 |
|---|---|---|
| P0：版本与仓库 | 建立代码结构、锁 ms-swift/模型/依赖、Muxi 环境预检 | 已验证 native fork、Transformers 5.16.1、MACA 容器和 4×C500；9B 长跑仍未启动 |
| P1：记忆层 | 数学模块、冻结、受控插入、保存/加载 | 已完成零分支/冻结/四参数检查、真实 Qwen3.5 结构 attach smoke 和 safetensors round-trip |
| P2：SFT 数据 | 固定 episode 接入、Swift 模板、mask 和长度审计 | 已完成 action/recall 转换、tool-call/payload 泄漏检查和 qwen3_5 token-level 审计 |
| P3：双目标 Trainer | 阶段、token 分母、累积、梯度、scheduler、resume | native paired W/C 两步 Muxi smoke 通过；梯度路由断言、W/C 日志、四参数、checkpoint/resume 等价性已验证 |
| P4：真实模型集成 | 4B/9B 短序列、缓存/padding、BF16/FP32、资源 profiling | 4B BF16 两步 GPU smoke 通过；9B 长跑、cached/full-prefix 误差仍待单独验收 |
| P5：小样本 SFT | 少量合成未知规则与 action/recall 数据 | 可复现训练与自由生成报告，memory 开关干预和损失切片可解释 |

P1 数学测试与 P2 数据处理可在 P0 接口明确后并行；P3 依赖二者。P4 通过前不启动长 pilot。

关键测试补充：

1. 第 16 个完整 block 后注入，32 层及缓存索引未改变。
2. 整个主干逐参数核验，包括视觉塔、embedding、norm 和 LM head。
3. C 阶段 recall-only 测试在 optimizer 已有动量/weight decay 状态时执行，确认内容参数不变。
4. 不等长 microbatches 和多卡得到与完整有效 batch 相同的每分支归一化梯度。
5. 改变未来环境文本不会改变较早 action logits；padding 不改变有效 token 输出。
6. memory 更新/禁用/替换后旧 cache 被拒绝或清空；prefill/decode 与完整前缀对齐。
7. checkpoint round-trip 和 resume 等价；最小部署无需外部 memory/index。

P5 的报告区分 teacher-forced NLL、自由 recall 格式/事实准确性与实际工具执行。小数据拟合证明工程链路可学习，不代表正式 benchmark 结论。

## 9. 资源与待确认事项

资源评估从 microbatch 1、短序列开始，再逐步增加到 4096。报告权重驻留、峰值训练显存、有效目标 tokens/s、memory checkpoint 大小与推理增量。完整词表 logits 和后半段反传仍有显著成本，不能按 236 万可训练参数推断训练显存很低。

启动真实实验前需确定：Muxi 具体 GPU/运行时与可用磁盘；教师 snapshot/数据位置及来源划分；模型权重的固定 revision；工具模板；验证集与 checkpoint 选择规则。首轮使用固定人工工具环境验证学习，再安排真实 benchmark。

本轮交付状态：需求文档已读完，上游关键接口已核对；可插拔 memory 层、两种独立 ms-swift SFT mode、冻结校验、native tuner 注册、数据转换、qwen3_5/tool-call/loss-mask 审计、native W/C paired Trainer、checkpoint/resume 和 `memory.safetensors` 导出均已实现。Muxi 上 Qwen3.5-4B 已完成两步 BF16 paired GPU smoke；正式 9B 长跑与 benchmark 不属于 smoke 验收，后续可直接复用同一入口。
