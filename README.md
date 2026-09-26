# 算力约束下提升大语言模型能力的资源配置建模

“华为杯”第二十三届中国研究生数学建模竞赛项目代码。作者：彭铸、郑徐瀚宇、阳挺。

四个问题各自独立成目录，脚本命名统一为 `Q_<题号>_<阶段>_<序号>_<英文名>.py`：

| 位置                | 内容                                         | 入口                               |
| ------------------- | -------------------------------------------- | ---------------------------------- |
| [Q1/](Q1/README.md) | 问题一：语料质量表征、质量冲突消解与领域配比 | `Q1/run_all.ps1`                   |
| [Q2/](Q2/README.md) | 问题二：经典与广义标度律、边际效用与替代分析 | `Q2/run_all.ps1`                   |
| [Q3/](Q3/README.md) | 问题三：算力预算下的资源配置与质量投入       | `Q3/run.py`                        |
| [Q4/](Q4/README.md) | 问题四：能力前沿、Loss–Benchmark 桥接与预测  | `Q4/run_all.ps1`（依赖问题三结果） |

题目附件与生成结果不随仓库分发：`data/` 与 `results/` 都被忽略，
两者都位于仓库根目录下，克隆后需自行准备附件。

## 一、数据存放位置（全仓库统一）

**请把题目附件解压到仓库根目录的 `data/` 下**，使解压出来的 `real_attachments/`
正好落在 `data/real_attachments/`；解压后的层级与压缩包内保持一致，不需要再手工调整目录名。
`Q1`/`Q2`/`Q4` 都从这一份数据读取，不存在第二套路径：

```text
仓库根目录/
├─ data/real_attachments/            # 题目附件解压到这里（不随仓库分发）
│  ├─ A_data_value/                 # 配比实验、质量实验、质量信号样本（Q1、Q2、Q3）
│  │  ├─ regmix_tables/             #   3 个观测尺度 × 配比 × 13 评测域 Loss
│  │  ├─ slimpajama_quality_extended/       # 扩展质量信号（Q1；.jsonl 与 .jsonl.xz 均可）
│  │  └─ slimpajama_quality_signal_sample.jsonl
│  ├─ B_scaling_laws/               # 训练日志与质量实验（Q2、Q3）
│  │  ├─ pythia_training_log_existing.csv
│  │  ├─ cerebras_training_log.csv
│  │  ├─ supplementary_NQ_experiment*.csv
│  │  └─ training_trajectories/
│  └─ C_efficiency_evolution/       # 排行榜、模型元数据与逐模型评测明细（Q2、Q4）
│     ├─ leaderboard_*.csv、epoch_all_ai_models.csv
│     ├─ loss_benchmark_bridge*.csv
│     ├─ model_architecture_metadata.csv
│     └─ detailed_results/          #   1,863 个模型目录，重建 C8 快照的数据来源
├─ results/                          # 全仓库共用的产物目录（不随仓库分发）
│  ├─ Q_1_to_2/                     # 问题一交给问题二的接口
│  ├─ Q2_to_Q3/                     # 问题二交给问题三的接口
│  └─ …                             # 各阶段的结果、报告与图件
└─ Q1 / Q2 / Q3 / Q4                # 四个问题的代码
   └─ Q<n>_*/results/               # 各阶段自己的中间产物，同样被忽略
```

**关于两个 `results/`**：根目录的 `results/` 是四个问题共用的产物树，跨题目交接
（`results/Q_1_to_2/`、`results/Q2_to_Q3/`）就放在这里；`Q1/Q1_1/results/`、`Q3/results/`
这类是各阶段自己的中间结果，位置由脚本自身决定。两者都被忽略，都不会提交。

部分附件同时提供 `.jsonl` 与 `.jsonl.xz` 两种形式（质量信号），脚本两种都能读，
磁盘紧张时保留 `.xz` 即可。若想把附件放在别处，问题二的路径常量集中在
`Q2/_q2_paths.py`，可用 `HUAWEI_DATA_DIR`、`HUAWEI_RESULTS_DIR` 两个环境变量重定向。

问题三自带交接输入副本（`Q3/Q3_1/00_交接输入/`），因此可以独立运行；
问题四需要问题三的成本分配结果（`Q4/Q4_2/q4_utils.py` 顶部的 `Q3_RESULT`）。
附件清单、各文件用途与缺失项见各子目录 README。

## 二、环境

```powershell
python -m pip install -r requirements.txt
```

**要求 Python >= 3.12**。

## 三、运行

### 统一入口

```powershell
python run_all.py --list                  # 列出每个题目的脚本、前置条件与缺失标记
python run_all.py q2                      # 只跑问题二
python run_all.py q1 q2 --stop-on-error   # 依次跑问题一、问题二，遇错即停
python run_all.py --all                   # 跑全部四个题目
```

`run_all.py` 按依赖顺序在被调脚本自己的目录中执行，并自动为 matplotlib
指定可写的缓存目录。问题三通过它自己的入口 `Q3/run.py --grid` 调用，
参数可用 `--q3-args` 覆盖。

### 日志

每次运行的完整输出都保存在仓库根目录的 `log/` 下，按运行时间与题目分目录，
文件名与脚本同名：

```text
log/
└─ 20260926-215716/            # 一次运行的开始时间
   ├─ q1/Q_1_1_1_audit_A1.log  # 每个脚本一份，含 stdout/stderr 与退出码
   └─ q2/Q_2_1_1_classic_law.log
```

`python run_all.py` 会自动写入这些日志（加 `--no-logs` 可关闭）；
各题目的 `run_all.ps1` 也会在 `log/<时间戳><题目>/` 下留一份。
`log/` 已被 `.gitignore` 忽略，不会被提交。

### 各题目单独运行

问题一（准备好 `data/real_attachments/` 后）：

```powershell
.\Q1\run_all.ps1
```

入口按 `Q1_1 → Q1_2 → Q1_3` 执行，产物位于各阶段的 `results/`。问题二需要把交接文件
放到 `results/Q_1_to_2/`：

```powershell
$src = '.\Q1\Q1_3\results'
$dst = '.\results\Q_1_to_2'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
foreach ($name in @('Q17_mapping.csv', 'train_domain_names.csv', 'loss_domain_names.csv',
                    'loss_standardization.json', 'Q1_to_Q2_inputs.json')) {
    Copy-Item -LiteralPath (Join-Path $src $name) -Destination $dst
}
```

问题二（确认 `data/real_attachments/` 与上述交接文件已就位）：

```powershell
.\Q2\run_all.ps1
```

阶段顺序为 `Q2.1 经典标度律 → Q2.2 广义标度律 → Q2.3 交付第三问 → Q2.4 弹性与替代`；
`Q2.4` 依赖 `Q2.3` 生成的求值器。`Q2.1` 写出的规模指数是 `Q2.2` 的输入，
顺序不能交换。详细脚本清单与产物路径见 [Q2/README.md](Q2/README.md)。

问题三（**需要 Python ≥ 3.10**；自带交接输入副本，因此也能独立运行）：

```powershell
python .\Q3\Q3_1\sync_q3_handoff.py   # 把本次 Q1/Q2 的产物同步进交接副本（从零跑整链时必需）
python .\Q3\run.py                    # 单个算力档（默认 1e22 FLOPs、上下文 4096）
python .\Q3\run.py --grid             # 全部算力档 × 上下文 × 成本函数 × 决策模式
python .\Q3\Q3_3\write_q4_anchors.py --input Q3\results\full_grid.jsonl   # 导出问题四所需锚点
```

结果写入 `Q3/results/`（已被忽略）。再次运行请换输出文件名，避免覆盖已有记录
（`run.py` 不会覆盖已存在的输出，这是有意的保护）。

本机若已有 WSL 环境，可用其中的 Python 3.12 直接运行，例如：

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/d/<路径>/huawei_code/Q3' && python run.py --grid --output results/full_grid.jsonl"
```

问题四（须先备齐 C 类附件；锚点表由上一步导出）：

```powershell
.\Q4\run_all.ps1
```

结果写入 `Q4/Q4_1/results/` 与 `Q4/Q4_2/results/`。
`Q4/Q4_1/Q_4_1_1_rebuild_C8.py` 既支持从 `data/real_attachments/F题.zip` 读取，
也支持直接读取已解压的 `detailed_results/`，两者产出同一张表。

## 四、出图

出中文图需系统中有可用的中文字体；

## 五、已知前置条件与边界

- **附件不随仓库分发**：`data/`、`results/`、各题目的 `results/` 都被忽略，
  克隆后必须自行准备附件才能从零复现。
- **问题四依赖问题三的结果**：三档预算下的最优损失锚点由问题三给出；
  该结果未就位前，问题四的第二阶段无法从零运行。
- **问题三需要 Python ≥ 3.10**：其注解写法在 3.9 上导入即失败，建议整体使用 3.12。
- 各题目自身的数据边界与可识别性限制（哪些参数不可单独辨识、哪些数据不可用）
  在其 README 的末节列出，引用结论时请一并带上。

## 许可证

仓库内的原创代码与文档采用 [MIT License](LICENSE)。题目附件、第三方数据及其授权不因本许可证而改变。
