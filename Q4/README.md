# 问题四复现代码

本目录公开问题四的核心计算链，包括 C8 重建、实体匹配、能力与算力前沿、Loss--Benchmark 桥接、支持边界下的贡献分解、情景预测和回测。竞赛数据、问题三的中间数据、运行结果、图片和绘图脚本不随本目录上传。

## 数据目录

**把题目附件解压到仓库根目录的 `data/` 下**，使 `real_attachments/` 正好落在
`data/real_attachments/`；问题一、问题二、问题四读的是**同一份** `data/`。请在运行前准备以下结构。

```text
<仓库根目录>/
├─ Q4/
│  ├─ Q4_1/
│  └─ Q4_2/
└─ data/                                  # 题目附件解压到这里（不随仓库分发）
   └─ real_attachments/
      └─ C_efficiency_evolution/
         ├─ leaderboard_cleaned.csv
         ├─ leaderboard_enhanced.csv
         ├─ leaderboard_extended_timeseries.csv
         ├─ epoch_all_ai_models.csv
         ├─ loss_benchmark_bridge.csv
         ├─ loss_benchmark_bridge_expanded.csv
         └─ detailed_results/             # 1,863 个模型目录，用于重建 C8 逐任务快照
```

`detailed_results/` 是重建 C8 逐任务快照的数据来源，**不能只保留上面那六个 CSV**。
`Q4/Q4_1/Q_4_1_1_rebuild_C8.py` 两种形式都支持：解压出 `detailed_results/` 后直接读它；
若手里有原始 `F题.zip`，放到 `data/real_attachments/` 下也会被优先读取（压缩包内
4 个截断记录只有它带有，读取时会在 `C8_rebuild_summary.json` 的 `source` 字段里注明用了哪一种）。

问题四的第二阶段还需要问题三的成本分配结果（提供三档预算下的最优损失锚点，是统一状态方程的机制输入）。
该文件的具体位置写在 `Q4/Q4_2/q4_utils.py` 顶部的 `Q3_RESULT` 常量里，请按那里的路径放置；
问题三的产物形态请见 [Q3/README.md](../Q3/README.md)。

不要把附件放入 `Q4/`，也不要提交 `Q4_1/results` 或 `Q4_2/results`。若改变目录层级，
需要同步修改 `Q4/Q4_1` 三个脚本和 `Q4/Q4_2/q4_utils.py` 顶部的路径常量。

## 环境与运行

```powershell
conda env create -f environment.yml
conda activate huawei-modeling
.\run_all.ps1
```

也可以指定 Python 可执行文件。

```powershell
.\run_all.ps1 -Python "C:\path\to\python.exe"
```

脚本必须按入口给出的顺序执行。`Q4_2` 依赖 `Q4_1/results`，模型结果写入 `Q4_2/results`。

## 文件说明

### 数据准备

| 文件 | 用法 |
| --- | --- |
| `Q4_1/Q_4_1_1_rebuild_C8.py` | 从原始压缩包严格解析 C8 JSON，并保留每个模型最新的可解析快照。 |
| `Q4_1/Q_4_1_2_match_C1_C4.py` | 对 C1 排行榜模型与 C4 元数据进行分级实体匹配。 |
| `Q4_1/Q_4_1_3_c6_bridge_plan.py` | 审计 C5、C6 的 Loss 可比性层级和桥接支持范围。 |

### 统一状态模型

| 文件 | 用法 |
| --- | --- |
| `Q4_2/q4_utils.py` | 定义公共路径、滚动分位前沿、Q3 损失插值和桥接工具。 |
| `Q4_2/Q_4_2_1_sample_and_frontier.py` | 构造能力样本，验证综合指标并计算能力前沿。 |
| `Q4_2/Q_4_2_2_compute_frontier.py` | 筛选开放权重语言模型并计算训练算力前沿。 |
| `Q4_2/Q_4_2_3_bridge_and_mechanism.py` | 比较桥接模型并完成历史贡献的部分识别。 |
| `Q4_2/Q_4_2_4_forecast.py` | 在统一状态方程下运行条件模拟和情景预测。 |
| `Q4_2/Q_4_2_5_validation.py` | 完成敏感性、滚动回测、基线比较和匹配样本检验。 |
| `Q4_2/Q_4_2_7_verify_outputs.py` | 检查结果范围、数据合同、文件完整性和输出哈希。 |
