# PACT-VLA：统一实验工作区

服务器位置：`/home/ubuntu/PACT-VLA`。本仓库整合已测试的 OpenVLA / OpenVLA-OFT 实现、模型、历史实验和日志，提供新的 LIBERO 评测入口。**整合不等于重新复现论文数值**；旧结果、失败尝试和不同协议均保留，不混成一张正式成绩表。

## 快速开始

无需手动激活 Conda 环境；入口会根据模型和策略选择原先验证过的 Python 环境。

```bash
cd /home/ubuntu/PACT-VLA
python evaluate.py --list
python evaluate.py --list-tasks --suite all
python evaluate.py --doctor

# OFT + SparseVLM，四套件，各任务 5 次，三档剪枝率；在 tmux 后台运行
python evaluate.py --model oft --strategy sparsevlm --suite all \
  --ratio 50 75 87.5 --trials 5 --with-baseline --tmux --name oft_sparse_5trials

# OpenVLA + VLA-Pruner，Spatial 任务 0 和 3，每任务 1 次
python evaluate.py --model openvla --strategy vla-pruner --suite spatial \
  --ratio 87.5 --task-ids 0 3 --trials 1 \
  --prefill-attention false --decode-cache off --with-baseline

# OpenVLA-OFT + PACT-VLA：从候选保留率中逐步动态选择预算
python evaluate.py --model oft --strategy pact-vla --suite all \
  --pact-budget-rates 0.125,0.25,0.5,0.75,1.0 --trials 3 --with-baseline

# 只验证推理路径，不把短测当成功率：跨过 OFT 前 3 个 chunk 的 warmup
python evaluate.py --model oft --strategy divprune --suite long \
  --ratio 75 --task-ids 3 --mode verify --verify-calls 5

# 查看完整矩阵，检查资源但不使用 GPU、不启动实验
python evaluate.py --model all --strategy all --suite all \
  --ratio 50 75 87.5 --trials 1 --with-baseline --dry-run
```

策略可以一次选择多个，例如 `--strategy fastv sparsevlm divprune`。模型支持 `openvla`、`oft`、`all`；套件支持 `spatial`、`object`、`goal`、`long`、`all`，也接受原始 `libero_*` 名称。每套件任务编号为 **0–9**；`--task-ids` 不填即全部任务。`--trials` 是每个所选任务的 episode 数（1–50），默认 1。

## 方法与参数含义

| `--strategy` | 实现来源 | 比率意义 | 匹配基线 |
|---|---|---|---|
| `vanilla` | 指定后端关闭加速策略 | 不剪枝 | 可用 `--baseline-backend` 选择 |
| `fastv` | VLA-Pruner 中的 FastV 分支 | 删除视觉 token 比例 | native |
| `sparsevlm` | 原 OpenVLA 分支 / 已补充的 OFT 适配 | 删除视觉 token 比例 | native |
| `divprune` | 已验证的 pre-LLM 适配，使用官方 max-min 选择算法对照 | 删除视觉 token 比例 | divprune |
| `vla-cache` | VLA-Cache 后端 + `local_fixed_reuse_v2` | **目标最终缓存复用比例，不是删除比例** | vla-cache |
| `vla-pruner` | 当前工作树中的 VLA-Pruner 分支 | 删除视觉 token 比例 | native |
| `pact-vla` | 当前工作树中的 PACT-VLA 自适应预算分支（仅 OpenVLA-OFT） | 运行时从候选**保留率**中动态选择 | native |

`--ratio 50 75 87.5` 等同 `--ratio .5 .75 .875`，对应保留率 50%、25%、12.5%。Vanilla 自动忽略该参数且只运行一次。PACT-VLA 同样只生成一条 adaptive 条件，忽略固定 `--ratio`，候选保留率由 `--pact-budget-rates` 设置；当前没有 OpenVLA（非 OFT）实现。VLA-Cache 首帧/缓存重置后仍计算全部 token；固定复用适配不是作者公开确认的 Table 1 完整协议。

`--with-baseline` 自动为每种实现添加相应 Vanilla，避免跨后端比较。所有方法、两模型、四套件、三档固定比例（PACT-VLA 仅 OFT 各套件一条自适应条件）：**132 个条件**；补齐 native / DivPrune / VLA-Cache 各自基线后共 **148 个条件**。本次整合只运行有界集成检查，没有自动启动这个完整矩阵。

默认配置保留既有实验约定：seed=7、center crop、环境等待 10 步；OpenVLA native / DivPrune 的 `use_cache=False`；OFT 为双图像、proprio、L1 head、每次 8 个 action。`use_prefil_attention=False` 默认关闭。VLA-Cache 使用其专用缓存后端，不允许强制 `--decode-cache off`。保留各后端原有物理环境、终止逻辑和动作变换。

VLA-Pruner 当前源代码（包括 OFT attention 汇总与 warmup）不一定等于历史某次 500-episode 实验的代码。每次新评测记录实际源码哈希、模型类、attention 类和参数；历史日志不被重写。DivPrune 使用匹配的 pre-LLM/no-cache OpenVLA、显式双向 OFT 适配，所以它的 Vanilla 不能冒充 native Vanilla。

高级参数：`--prune-layer 3`；`--backend-option FIELD VALUE` 传入后端已有 dataclass 参数。不允许用它覆盖模型、策略、套件、次数等评测身份字段；不同后端不支持的参数会在加载 GPU 前报错。覆盖模型路径用 `--checkpoint /absolute/path`，此时须只选一个模型和套件。配置详见 `python evaluate.py --help`。

## 进度、暂停、继续

```bash
cd /home/ubuntu/PACT-VLA
tmux ls
tmux attach -t pact_oft_sparse_5trials
# 离开 tmux 而不停止：Ctrl-b，然后 d

python evaluate.py --status runs/oft_sparse_5trials
tail -f runs/oft_sparse_5trials/queue.log

# 当前条件完成后暂停（不是当前 episode 完成后）
python evaluate.py --pause runs/oft_sparse_5trials

# 在 tmux 中继续：跳过已完成的条件
tmux new -s pact_resume
python evaluate.py --resume runs/oft_sparse_5trials

# 失败/中断条件从头重试，创建 attempt_02，不覆盖 attempt_01
python evaluate.py --resume runs/oft_sparse_5trials --retry-failed
```

前台 `Ctrl-C` 会结束该入口启动的子进程，并标记 INTERRUPTED。按条件恢复，**不从半个 episode/半个条件续写**。旧实验目录中的暂停标记和队列完全不被本入口控制，也不会自动恢复。GPU 已被其他任务使用时入口会拒绝启动，不擅自终止其他进程；同一 GPU 的新入口还受文件锁保护。

## 输出与指标

### Table 1 全条件复测（每任务 3 次，新增 25% 剪枝）

```bash
python evaluate.py --model all --strategy all --suite all \
  --ratio 25 50 75 87.5 --trials 3 --with-baseline \
  --condition-order method --collect-flops --prefill-attention false \
  --name table1_all_prune25_50_75_875_trials3_20260831 --tmux
```

共 **172 个主表条件 + 16 个额外后端匹配基线 = 188 条件、5640 episodes**。新增 25% 指删除 25%、保留 75%，不是保留 25%。固定预算方法保持四档；PACT-VLA 在 OFT 上增加一条自适应预算结果，主表共 43 行。所有基线优先执行；不复用历史暂停队列的结果。

开启 `--collect-flops` 后，条件完成时自动更新 `table1.csv`、`table1_with_backend_baselines.csv` 和逐套件 `summary.csv`。主表提供四套件成功率、`Acc.(%)`、`FLOPs(T)`、`Latency(ms)` 和加速比；尚未完成四套件的行不填整体均值。

`Acc.(%)` 明确定义为四套件“本方法成功率 / native Vanilla 成功率”的宏平均 ×100，另列匹配后端版本和均值之比。论文未明确给出汇总公式，部分印刷数值不能由其展示的四个成功率复算，因此不能声称这一列已完全还原作者未公开的聚合方式。

主列 `FLOPs(T)` 使用已安装仓库的 Llama gated-FFN 公式（FFN 系数 3，MAC=1），仅统计每个策略调用的首次 LLM pass；附加列保留论文 Eq.9 的系数 2 版本，以及抽样的完整策略调用算子 FLOPs（MAC=2，包含后续 decode）。三者范围不同，不可互换。默认每个 episode 的第 1、5 次调用及每 50 次调用采样；**采样调用全部排除在延迟均值外**。详见 [评测协议](docs/TABLE1_20260831_PROTOCOL.md) 与每次运行的 `metric_definitions.json`。

每个运行冻结统一 worker/package 代码到 `code_snapshot/`，后端仍来自已有环境及 vendor 源码，并记录实际哈希。中途不要修改共享后端源码/环境。3 次/任务是用户指定的缩小样本量，论文为 50 次/任务；不能把本次称为相同统计精度的原表复现。

VLA-Cache 的原加载器会同步 checkpoint 里的配置/Python 代码。统一入口为它创建每次运行独立的 `checkpoint_shadow/`：大权重仅引用，配置与代码复制后再同步，避免改写历史 checkpoint。来源映射见 `checkpoint_shadow.json`。集成检查曾触发的 3 个旧元数据/代码文件已从 SHA256 匹配的自动备份恢复，加载器生成的版本另外保留在 `provenance/loader_side_effect_before_isolation/`。

每次运行创建独立 `runs/<name>/`，重名拒绝覆盖：

```text
runs/<name>/
  manifest.json / summary.csv / queue.log（tmux 时）
  <模型_套件_策略_后端_比例>/attempt_01/
    job.json / invocation.json / tasks.json
    environment.json                 # 实际环境、类、配置、源码 SHA256
    stdout.log / internal_logs/      # 原始日志
    policy_timing_calls.csv          # 逐次同步测时
    token_audit.jsonl                # 实际 LLM 层序列长度、token/复用记录
    result.json                     # 分母、成功次数、状态、错误、延迟
```

成功率为已完成 episode 的成功数/总数；正常 eval 必须完成预期分母，推理异常不会被伪装成普通失败。`mode=verify` 的状态为 VERIFIED，成功率不作为实验结果。

`policy_ms_mean` 在 `get_action` 边界测时，包括策略侧预处理与动作构造；`model_ms_mean` 在 `predict_action` 边界测时。均 CUDA 同步，排除模型加载、仿真和视频保存，默认排除前 10 次调用。OFT 单位是 **8-action chunk**，OpenVLA 是单次 action。样本不够时延迟留空，不用含 warmup 的均值替代。

`summary.csv` 的 `policy_speedup` / `model_speedup` 用同一次运行、同模型、同套件、同后端已完成 Vanilla 的均值计算。**这是在线轨迹平均延迟，不是相同输入固定回放的微基准**；不同方法轨迹/调用数量不同，不能不加说明地与论文某个 kernel 时间比较。历史固定输入延迟脚本仍完整保留在归档中。

## 文件组织与迁移

```text
/home/ubuntu/PACT-VLA/
  evaluate.py / pact_eval/ / configs/ / tests/ / tools/
  vendor/vla-pruner/                 # 原仓库、修改、checkpoint、所有历史实验
  vendor/vla-cache/                  # 原缓存仓库、适配、历史实验
  vendor/divprune-official/          # 官方算法参考仓库
  checkpoints/openvla -> ../vendor/vla-pruner/src/openvla/checkpoints
  checkpoints/oft     -> ../vendor/vla-pruner/src/openvla-oft/checkpoints
  dependencies/LIBERO/
  references/                       # 官方 OFT/Transformers、PDF、原始参考树
  references/runtime_sources/       # 缓存环境自定义 Transformers 的完整源码快照
  experiments/archive/              # 按原仓库整理的历史实验快捷入口
  experiments/legacy/               # 其他已做实验平台的旧文件，非统一 LIBERO 后端
  provenance/                       # 迁移日志、校验清单、环境导出、验证证据
  runs/                             # 新统一评测结果
```

主体目录通过同文件系统原子重命名迁入；旧路径保留兼容软链接。因此权重、日志和源码不是两套重复副本；既有绝对路径和 editable install 仍可访问。VLA-Cache 的共享权重链接已改成新工作区内相对链接。不要删除旧兼容链接，也不要直接搬动 Conda 环境目录（环境前缀敏感）。

Conda 仍在 `/home/ubuntu/miniconda3/envs/`，四套环境为 `vla-pruner`、`vla-oft-official`、`vla-cache-openvla`、`vla-cache-oft`。版本导出见 `provenance/environments_20260831/`；可用 `PACT_CONDA_ROOT` 覆盖 Miniconda 根目录。VLA-Cache 从其已验证环境导入自定义 Transformers，不能随意换成 pip 默认包；对应源码已另外快照保存，并有逐文件 SHA256 校验。

原上游 Git 历史和用户未提交修改均保留。新根 Git 仓库管理统一入口、配置、说明、测试；`vendor/` 内保持独立原仓库，模型/历史大文件不重复加入根 Git 索引。**只 clone 根 Git 并不会自动带走 120+ GB 实验数据**；搬到新服务器须另外传输这些目录及恢复环境。`provenance/migration_20260831/` 保留原路径到新路径的映射、文件/inode 清单及小文件备份。

本入口仅支持上述四个 LIBERO 套件。历史 SimplerEnv / RoboTwin 实验保留原入口用于追溯，并未声明由该 CLI 支持或通过了本次重测；独立 VLABench 项目不属于本次迁移的 LIBERO 后端。

## 检查

```bash
python -m unittest discover -s tests -v
python evaluate.py --doctor
python evaluate.py --list-tasks --suite all
```

迁移与归档脚本是一次性维护工具，不是日常评测入口。`tools/run_integration_checks.py` 会实际运行 12 个单 episode 检查，需要空闲 GPU；不要把它当只读 doctor 使用。
