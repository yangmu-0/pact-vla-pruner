# Table 1 全条件、小样本复测协议

日期：2026-08-31；工作区：`/home/ubuntu/PACT-VLA`。

## 范围与实验身份

- 模型：OpenVLA、OpenVLA-OFT。
- 方法：Vanilla、FastV、SparseVLM、DivPrune、VLA-Cache、VLA-Pruner。
- 剪枝率：25%、50%、75%、87.5%，分别保留 75%、50%、25%、12.5%。VLA-Cache 的比率是 `local_fixed_reuse_v2` 目标最终复用率，不是真正删除 token；首次调用/缓存重置仍全量计算。
- 套件：LIBERO Spatial、Object、Goal、Long（`libero_10`），每套件全部任务 0–9。
- 每任务 3 episodes，使用既有 evaluator 的初态索引 0、1、2，seed=7。每条件 30 episodes。
- 主表：2 × (1 + 5 × 4) × 4 = 168 条件。额外 DivPrune 和 VLA-Cache 的匹配 Vanilla：2 × 2 × 4 = 16 条件。合计 184 条件、5520 episodes。
- 论文是每任务 50 次，本次按用户要求缩减到 3 次。这是覆盖全部方法/套件/比例的缩小样本量复测，不具备原表相同的统计精度。每套件成功率最小步长为 3.33 个百分点；不针对这三个初态调参。

## 配置与边界

沿用新入口已验证的实现和环境，不因为成功率高低修改模型/任务。OpenVLA native 和 DivPrune 保持 `use_cache=False`；VLA-Cache 必须使用其专用缓存路径，不能把全矩阵强制为 `--decode-cache off`。`--decode-cache auto` 会做上述分流。OFT 为双相机图像、proprio、L1 head、8-action chunk，不启用 FiLM/diffusion。

`use_prefil_attention=False`；FastV/SparseVLM/VLA-Pruner 的 CLI prune layer=3。VLA-Pruner 的时序 warmup、当前 attention 汇总实现保持现状，由实际源码哈希和配置记录标识，不宣称与论文未公开协议完全等价。DivPrune 保留现有 pre-LLM 适配和官方 max-min 算法对照。VLA-Cache 固定复用适配是本地实现，不冒充作者确认的原表实现。

每个条件使用独立进程，按顺序共享 GPU0；24 个基线条件先执行，再按方法和比例执行。已有暂停实验继续保持暂停。短测和失败尝试留在各自目录，不与正式结果混合。

统一 worker/package 冻结到运行的 `code_snapshot/`。后端源码和环境仍是共享安装，因此记录每条件的实际源码 SHA256、环境、attention 类和配置；运行中不更改这些共享代码。缓存加载器使用每 attempt 私有 checkpoint 元数据副本，权重只引用，不改历史 checkpoint。

## 指标口径

### 成功率与 Acc.(%)

原始成功率为成功 episode 数/完成 episode 数。完成条件必须达到 30 episodes，异常不能充当普通任务失败。四套件均完成后才写整体汇总；整体成功率为四套件等权均值。

本次明确采用 `Acc.(%) = 100 × mean_s(SR_method,s / SR_native_vanilla,s)`。`Acc_matched(%)` 使用同后端 Vanilla；`Acc_ratio_of_means(%)` 为 `100 × sum_s(SR_method,s) / sum_s(SR_native_vanilla,s)`。基线为零或未完成时不生成无意义的相对值。

论文称 Acc 为 relative accuracy，但未明确汇总公式。一些印刷值不能从其展示的四套件成功率复算：例如 OpenVLA FastV 保留 50% 的四套件均值之比/比值宏平均并不能精确得到所列 97.43%，后者算得约 97.50%。本次保留可核算的明确定义，不为匹配印刷值回填数据。

### FLOPs 三种口径并列，不能相互替换

所有公式都基于 forward hook 观测到的真实每层 batch、序列长度 n、hidden d 和 intermediate m，而不是仅用目标保留率乘基线。

1. 主列 **FLOPs(T)**：每次策略调用的首次 LLM pass，逐层求和 `B × (4nd² + 2n²d + 3ndm) / 1e12`。这是当前缓存仓库使用的 gated-FFN 公式，MAC=1，排除视觉编码器/projector/action head 和后续 decode。
2. **FLOPs_paper_Eq9(T)**：同样范围，但按论文附录 A.2 Eq.9 将 FFN 系数改为 2。论文公式与当前仓库代码不一致，故单列，不能声称已恢复作者未公开 profiler。
3. **FLOPs_profiled_call(T)**：每 episode 调用 1、5 和每 50 次调用，用 `torch.profiler` 采样真实 `get_action` 的算子 FLOPs，MAC=2。融合 SDPA 若 profiler 未登记 FLOPs，按记录的 Q/K/V 形状补计一次。不重放动作、不改变 inference/no-grad 模式。包含已登记的视觉/投影/LLM/后续 decode/head/选择矩阵运算，缺少未登记的 elementwise/softmax/sort 和 TensorFlow 预处理；不是完整硬件指令计数。另列冷启动、warm 样本。

短测中 OpenVLA Vanilla 的主公式约 1.907 T，与原表 1.906 T 量级一致；字面 Eq.9 为约 1.487 T。完整 no-cache 策略调用包含 7 次 LLM forward，算子口径自然大得多。这仅说明主公式与原表尺度接近，不足以证明作者实际采用的全部统计边界。正式结果由各任务真实 token 长度重新统计。

两种首次 pass 公式都平均所有成功策略调用，含 cold/warmup。完整调用 FLOPs 是抽样均值，冷/暖组成与全调用不同，单独标注。逐次公式和样本算子分项均留档。

### Latency(ms) 与加速比

`Latency(ms)` 在 `get_action` 边界 CUDA 同步测量，包含策略预处理和动作构造，排除模型加载、仿真和视频保存。OpenVLA 单位是每 action；OFT 单位是每 8-action chunk。另列 `Model_latency(ms)`，边界为 `predict_action`。

全条件前 10 次调用和全部 FLOPs profiler 调用不参与延迟均值，防止 profiler 开销污染速度。没有足够有效调用则留空。在线轨迹平均时间并非固定输入/相同轨迹微基准；成功/失败轨迹长度会影响采样组成。主加速比用 native Vanilla，匹配加速比使用同模型/套件/后端 Vanilla。四套件整体延迟等权平均各套件均值，不混合未完成套件。

## 输出和重启

- `summary.csv`：184 条件各自的状态、episodes、成功率、延迟、FLOPs 和匹配加速比。
- `table1.csv`：42 行主表，列出四套件原始成功率、Acc、FLOPs、Latency 和补充口径。
- `table1_with_backend_baselines.csv`：包含四种额外后端 Vanilla 行，共 46 行。
- `metric_definitions.json`：机器可读口径。
- 每 attempt 的 `policy_timing_calls.csv`、`flops_profile_samples.jsonl`、`token_audit.jsonl`、`result.json`、`environment.json` 和原始日志：审计依据。

条件完成时原子更新 CSV；正在跑的条件可从 `result.json` 看实时计数。异常 fail-fast 停队列，保留失败尝试；修复后新 attempt 重跑该条件，不把半个条件续入正式分母。暂停为条件边界暂停。

```bash
cd /home/ubuntu/PACT-VLA
python evaluate.py --model all --strategy all --suite all \
  --ratio 25 50 75 87.5 --trials 3 --with-baseline \
  --condition-order method --collect-flops --prefill-attention false \
  --name table1_all_prune25_50_75_875_trials3_20260831 --tmux
python evaluate.py --status runs/table1_all_prune25_50_75_875_trials3_20260831
tail -f runs/table1_all_prune25_50_75_875_trials3_20260831/queue.log
```
