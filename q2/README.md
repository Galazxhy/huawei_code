# 问题二：算力约束下的标度律与资源配置

本目录公开第二问的完整计算链：**经典标度律**、**含数据质量与训练配比的广义标度律**、
**面向第三问的交付接口**，以及**质量与配比的边际效益和替代关系**。

所有脚本只读题目附件，不写入任何附件目录；全部产物写入仓库根目录的 `results/`。
克隆仓库后 `data/` 与 `results/` 都是空的，需要按 §2 准备数据，再按 §5 的顺序重跑。

---

## 一、任务与两条计算线

题目给定两种实验数据：一组训练日志记录「模型规模 `N` × 训练 token 数 `D` → 验证损失
`L`」，另一组配比实验记录「在固定 `(N, D)` 下，各训练域的采样配比 `p` 与数据质量 `Q`
→ 逐评测域损失」。本目录据此回答：

1. **经典标度律** `L(N, D) = E + A·N^(−α) + B·D^(−β)`
   只用参数量 `N` 与训练 token 数 `D`，在训练日志上辨识五个参数，
   再用另外几组独立日志做跨来源验证、用百亿参数以上的模型做外推检验。
2. **广义标度律**：在经典项之上追加**数据质量 `Q` 与训练配比 `p`** 的贡献项。
   配比实验只有 3 个 `(N, D)` 观测点，`(α, β)` 无法从配比实验独立辨识，
   因此必须借用第一条线辨识出的指数基底——两条线有先后依赖，不能交换顺序。

两条线的损失标定不同，**它们的导数不可相加、也不可相除**来推替代率；
需要跨线比较时，本目录用独立的质量实验单独标定质量效应（见 §5.4）。

---

## 二、数据目录

**把题目附件解压到仓库根目录的 `data/` 下**，使 `real_attachments/` 正好落在
`data/real_attachments/`；解压后的层级与压缩包内保持一致，不需要再手工调整目录名。
本目录与问题一、问题四读的是**同一份** `data/`，不存在第二套路径。

```text
<仓库根目录>/
├─ Q2/                                    # 本目录
├─ data/                                  # 题目附件（解压到这里；不随仓库分发）
│  └─ real_attachments/
│     ├─ A_data_value/
│     │  ├─ regmix_tables/                 # 配比实验：3 个尺度 × 配比 × 13 评测域
│     │  │  ├─ train_mixture_1m.csv        train_pile_loss_1m.csv
│     │  │  ├─ test_mixture_1m.csv         test_pile_loss_1m.csv
│     │  │  ├─ test_mixture_60m.csv        test_pile_loss_60m.csv
│     │  │  ├─ test_mixture_1B.csv         test_pile_loss_1B.csv
│     │  │  ├─ est_mixture_10b.csv         est_pile_loss_10b.csv
│     │  │  └─ est_mixture_70b.csv         est_pile_loss_70b.csv
│     │  └─ slimpajama_quality_signal_sample.jsonl      # 质量信号样本（问题一使用）
│     └─ B_scaling_laws/
│        ├─ pythia_training_log_existing.csv            # 主训练日志（8 个规模 × 147 步）
│        ├─ cerebras_training_log.csv                   # 独立来源的训练日志
│        ├─ supplementary_NQ_experiment.csv             # 质量实验（10 个质量等级）
│        ├─ supplementary_NQ_experiment_expanded.csv    # 质量实验（450 组）
│        └─ training_trajectories/*.csv                 # 逐规模轨迹
└─ results/                          # 全部产物（不随仓库分发）
   ├─ Q_1_to_2/                            # 问题一交给问题二的接口
   │  ├─ Q17_mapping.csv                   # 17 个训练域的质量分
   │  ├─ train_domain_names.csv            # 17 个训练域的名字与顺序
   │  ├─ loss_domain_names.csv             # 13 个评测域的名字与顺序
   │  └─ loss_standardization.json         # 跨来源尺度归一化口径
   └─ Q2_to_Q3/                            # 问题二交给问题三的接口（由脚本生成）
```

`results/Q_1_to_2/` 是问题一的交付物，**不在本目录内**。若只下载 Q2，
需要从问题一的交付包（或问题一代码的 `Q1/Q1_3/results/`）补齐这四项。

附件较大，其中 `slimpajama_quality_signal_sample.jsonl` 约 413 MB（同目录另附
`.xz` 压缩版，脚本两种都能读），放置时请确认磁盘空间充足。

### 路径约定

路径只在 `Q2/_q2_paths.py` 里定义一次，其余脚本一律从那里导入：

* **不调用 `resolve()`**：路径常量以 `Path(__file__).parent` 为准，保持成
  调用者给出的形式，不额外做符号链接与大小写改写；Python 3.9 起 `__file__`
  本来就是绝对路径，因此从这里推导出的常量也已经是绝对路径。
* **Windows 上统一小写**：文件系统报告的是磁盘上的真实大小写（`Q2`），
  而调用者可能写 `q2`；两者混用会让 `Path.relative_to` 抛错，所以常量在这里
  一次性归一化。这一处理只影响大小写，不改变路径指向。
* **写进产物的路径一律是仓库相对路径**：JSON 载荷与报告里用
  `_q2_paths.repo_relative(...)`，输出形如 `data/real_attachments/...`，
  不会记录仓库在某一台机器上的位置；仓库之外的路径（例如 `--input` 指向别处）
  原样保留，因为没有更短的写法。
* 目录本身可以整体搬移：脚本靠自身位置定位仓库根，不依赖工作目录。
  想把附件放到别的磁盘，用下面两个环境变量：

```powershell
$env:HUAWEI_DATA_DIR     = "<附件所在目录>"
$env:HUAWEI_RESULTS_DIR = "<产物输出目录>"
```

不要手工把附件复制进 `Q2/`，也不要把 `results/` 提交到版本库。

---

## 三、数据附件的对外称谓

脚本内部沿用题目附件的原始标号（`A_data_value`、`B1`…`B10`、`C7`、`Q17`），
它们是与数据表一一对应的交叉引用，改成别的名字会让产物无法回溯到原始表格。
面向读者的对应关系如下，阅读代码时按此换算即可：

| 代码里的标号 | 含义 |
| --- | --- |
| 数据集 A | 训练配比与数据质量实验（`A_data_value/`） |
| `regmix_tables/` | 数据集 A 的配比实验表：3 个观测尺度 × 13 个评测域的损失 |
| `Q17` | 17 个训练域的固定质量分，由问题一给出（`Q17_mapping.csv`） |
| 数据集 B | 公开模型训练日志与已发表的标度律数据（`B_scaling_laws/`） |
| `B1` | 主训练日志（Pythia 8 个规模 × 147 个检查点），经典律的唯一参数拟合数据 |
| `B2`–`B5` | 4 组独立来源日志，用于跨来源验证与结构早停 |
| `B6`、`B7` | 数据质量实验（10 个质量等级；B7 为其扩展版 450 组） |
| `B8` | 已废弃：138 个格点 100% 方向反转，1704 行中 362 行压在硬性地板上 |
| `B9`、`B10` | 百亿参数以上的大模型日志，用于外推检验 |
| 数据集 C | 公开排行榜与模型元数据（本目录只用其中的上下文长度元数据 `C7`） |
| `C7` | 上下文长度可行性输入，随第三问交付 |

---

## 四、环境与运行

```powershell
conda env create -f .\Q2\environment.yml
conda activate huawei-modeling
python --version          # 已在 Python 3.12 与 3.9 上验证
```

依赖：`numpy`、`scipy`、`scikit-learn`、`matplotlib`（`fonttools` 只被中文子集字体脚本用到；
仓库根目录的 `requirements.txt` 给出同样的清单）。

出图前若 matplotlib 的缓存目录不可写，先指定一个可写目录（`run_all.ps1` 会自动处理）：

```powershell
$env:MPLCONFIGDIR = (Join-Path $PWD ".mplcfg")
```

**每个脚本都用 `python Q2/<阶段>/<脚本>.py` 调用**，不要用 `python -m`：脚本靠
「自身所在目录自动进入 `sys.path`」来互相导入，`Q2/` 不是包（没有 `__init__.py`）。

一条命令跑完整条链：

```powershell
.\Q2\run_all.ps1                                   # 用默认 python
.\Q2\run_all.ps1 -Python "C:\path\to\python.exe"   # 指定解释器
```

也可以在仓库根目录用统一入口按题目选择（含问题一、三、四）：

```powershell
python run_all.py --list           # 打印每个题目的脚本清单与状态
python run_all.py q2               # 只跑问题二
python run_all.py q1 q2            # 跑问题一和问题二
```

两种入口都会把每个脚本的完整输出写到仓库根目录的 `log/` 下
（`log/<时间戳>/q2/<脚本名>.log`，含 stdout/stderr 与退出码）；
`log/` 已被忽略，不会提交。想只看屏幕输出可给统一入口加 `--no-logs`。

---

## 五、复现顺序

依赖是有向的，按下面顺序执行即可从零重建全部产物。`run_all.ps1` 与
`run_all.py` 已按该顺序编排。

### 5.1 经典标度律（阶段 `Q_2_1`）

| 顺序 | 脚本 | 作用 | 读 | 写 |
| --- | --- | --- | --- | --- |
| 1 | `Q_2_1_1_classic_law.py` | 独立口径的五参数拟合（可分离非线性最小二乘：解析解 `E,A,B` + 网格/模式搜索 `α,β`），含 CV 与 bootstrap | 主训练日志 | `results/traditional_scaling_law/` |
| 2 | `Q_2_1_2_classic_law_early_stopping.py` | **自适应加权结构早停**：用归一化后的 B2–B5 约束模型结构，防止 B1 过拟合 | B1–B5 | `results/scaling_law_early_stopping/` |
| 3 | `Q_2_1_3_classic_law_analysis.py` | **第二问主流程**：B1 主拟合（默认取早停选中模型）→ B2–B5 交叉尺度归一化验证 → B9/B10 大模型外推 | B1–B5、B9、B10 | `results/scaling_law_full/` |
| 4 | `Q_2_1_4_classic_law_figures_zh.py` | 论文用图（中文标注；单图 6×4 in，左右合成 12×4 in） | 上两步的产物 | `results/traditional_scaling_law/*.png` |

`Q_2_1_3` 只做经典律，不含任何质量/配比项；它写出的 `B1_main_fit` 是下游广义律的指数来源。
`B6`–`B8` **不参与任何参数估计**，只在需要时作事后对照。

**指数搜索阶梯**：指数不是一次放开的，先在共享指数下解最小二乘，再逐级释放自由度，
每一级都用 B2–B5 的验证分数决定是否值得继续（阶段 0 共享指数基准 → 阶段 2N/2D 单独释放
`α` 或 `β` → 阶段 3 两指数自由 → 阶段 4 局部精化）。搜索范围 `[0.01, 1.50]`，
初始 31 点网格，随后按八邻域模式搜索并逐次减半步长至 `1e−7`。

### 5.2 广义标度律（阶段 `Q_2_2`）

| 顺序 | 脚本 | 作用 | 读 | 写 |
| --- | --- | --- | --- | --- |
| 1 | `Q_2_2_1_generalized_law.py` | **广义律主体**：三折划分、配比响应的岭回归 + `clr(p)` 中心化、逐域规模幅度、`ε` 敏感性、残差校正消融，并自动生成 Markdown 报告 | `regmix_tables/`、`Q_1_to_2/`、`scaling_law_full/` 的 `B1_main_fit` | `results/Q_1_to_2/generalized_scaling_compact.json`、`results/generalized_scaling_law/` 报告 |
| 2 | `Q_2_2_2_generalized_law_figures.py` | 广义律图集（默认 6 张读者向图，`--figures all` 出全部 21 张）；**每次运行都自检**：重算的留出集指标必须复现 `generalized_scaling_compact.json`，不一致直接报错退出 | 同上（并把 `Q_2_2_1` 作为模块导入） | `results/generalized_scaling_law_drawings/` |

模型（`N_b`、`D_b` 以十亿为单位）：

```text
L_k(N, D, p, Q) = E_k + A_k·N_b^(−α) + B_k·D_b^(−β) + S_k(N, D)·Φ_k(p, Q)
S_k(N, D)       = (N_b / N_ref)^(−η_k) · (D_b / D_ref)^(−ζ_k)
Φ_k(p, Q)       = slope_ref_k · [ Ridge_k(Standardize([p_i·Q_i, clr_ε(p)_i])) − center_k ]
```

`α = β` 取自 B1，**不重新辨识**；`N_ref = 1e6` 参数、`D_ref = 1e9` token。

### 5.3 面向第三问的交付（阶段 `Q_2_3`）

| 顺序 | 脚本 | 作用 | 写 |
| --- | --- | --- | --- |
| 1 | `Q_2_3_1_handoff_build.py` | 生成第三问的接口层：广义律形式与参数、配比响应可复算参数、上下文长度可行性、质量成本函数、弹性与替代关系，并落地一个可直接 `import` 的求值器 | `results/Q2_to_Q3/`（含 `generalized_law_evaluator.py`） |
| 2 | `Q_2_3_2_handoff_paper_form.py` | 把广义律整理成可直接写进论文的形式（含 13×17 系数表），并验证该写法与原模型的最大偏差 | 同上 |
| 3 | `Q_2_3_3_handoff_identification.py` | **配对跨规模辨识**：只用同一配比在两个规模上的配对损失把形状项消掉，从而免形状假设、免 `ε` 地辨识 `(η_k, ζ_k)`，并用留出折检验预测力 | 同上 |

`Q_2_4` 的 4 个分析脚本依赖 `Q_2_3_1` 生成的 `generalized_law_evaluator.py`，
因此**必须先跑完 `Q_2_3`**。

### 5.4 边际效益、弹性与替代（阶段 `Q_2_4`）

| 顺序 | 脚本 | 作用 |
| --- | --- | --- |
| 1 | `Q_2_4_1_elasticity_analysis.py` | 质量的边际效益与弹性（独立的 `B·(D·Q)^(−β)` 形式）；配比在单纯形可行方向上的边际效益与弹性（组成广义律） |
| 2 | `Q_2_4_2_elasticity_figures.py` | 上一步的图集（中文标注） |
| 3 | `Q_2_4_3_factor_elasticity_analysis.py` | **各因素边际效用与弹性**：`N`/`D` 报「相对 +1%」的精确降损 `L(x)−L(1.01x)` 与解析弹性（中心差分复核）；`p` 报份额 +1 个百分点的可行方向降损（仅内点、扰动前后都在支撑域内）；`Q` 报 `Q+0.01` 的精确降损。含 10B/250B 情景外推行与「汇总表可由明细重算」的验收自检 |
| 4 | `Q_2_4_4_factor_elasticity_figures.py` | 上一步的图件（中文标注，PNG） |
| 5 | `Q_2_4_5_quality_substitution_analysis.py` | **等损失替代**：`N` 与配比不变时，提高 token 质量能少用多少 token。先定义两种量（**可节省 token** `D₀(1−Q₀/Q₁)` 与**等效新增 token** `D₀(Q₁/Q₀−1)`），再用「留一质量等级」交叉验证检验 `D·Q` 假设，并对照自由指数模型 `B·D^(−β_D)·Q^(−β_Q)` 报告 `r = β_Q/β_D` 的 bootstrap 区间 |
| 6 | `Q_2_4_6_quality_substitution_figures.py` | 上一步的图件（一图：维持参考损失所需 token 随质量的变化） |
| 7 | `Q_2_4_7_joint_substitution_analysis.py` | **广义律指导下的联合等损失替代**：锚定式 `L_k† = E_k + A_k·N^(−α) + (Q/Q_0)^(−γ)·[B_k·D^(−β) + S_k·φ_k(Q⁽⁰⁾, p)]`，`S_k`、`φ_k` 及全部原有参数沿用已验证的广义律（`Q = Q_0` 时严格退化，实测偏差 4e−16），`γ` 由独立质量实验标定；固定 `N` 后沿单纯形可行方向移动 `p`，逐目标求解维持同一损失所需的 `D`；同时报「完全耦合」与「质量只作用于 `B_k·D^(−β)`」两种耦合的替代区间 |
| 8 | `Q_2_4_8_joint_substitution_figures.py` | 上一步的图件（等损失曲线、三维柱图、配比影响对比图） |

**口径提醒**：边际效益 `g_x = −∂L/∂x`（正值 = 降损）、弹性 `ε_x = ∂lnL/∂lnx`（负值 = 增大该因素降损）。
注意 `−∂L/∂lnx` **不等于**「提高 1% 的降损」，后者按 `L(x) − L(1.01x)` 精确计算，两者在明细里都有但表头分开标注。
跨目标的四分位区间是**分布范围**，不是置信区间；**各因素工作点不同**（`N`/`D`/`p` 在配比实验的观测锚点上，
`Q` 在质量实验设计域内），因此它们的弹性**不可横向相减**。

---

## 六、目录结构

脚本名统一为 `Q_<题号>_<阶段>_<序号>_<英文名>.py`，阶段目录与脚本序号一一对应：

```text
Q2/
├─ _q2_paths.py            # 唯一路径定义处（附件、产物、交付目录、模块搜索路径）
├─ environment.yml         # conda 环境规格
├─ run_all.ps1             # 按依赖顺序运行全部 Q2 脚本
├─ README.md               # 本文件
├─ Q_2_1/                  # 经典标度律
├─ Q_2_2/                  # 广义标度律（质量 + 配比）
├─ Q_2_3/                  # 面向第三问的交付接口与辨识
└─ Q_2_4/                  # 边际效益、弹性与替代
```

每个脚本开头都有同一段引导代码（由 `_q2_paths.py` 提供）：

```python
_PKG_DIR = _Path(__file__).resolve().parents[1]      # Q2/
_sys.path.insert(0, str(_PKG_DIR))
import _q2_paths
_q2_paths.install_source_dirs(__file__)              # 让同阶段与跨阶段导入都能解析
PROJECT_ROOT = _q2_paths.PROJECT_ROOT
```

跨阶段导入按**内容名**书写（如 `import Q_2_2_1_generalized_law as compact`），
`install_source_dirs()` 会自行找到该模块所在的阶段目录，因此脚本里没有硬编码的兄弟目录路径。

### 与旧版扁平目录的对应

原先 17 个脚本平铺在 `Q2/` 下，现已按阶段归档并加序号前缀；旧的模块名一律保留为
脚本名的后缀，便于对照：

| 旧脚本名 | 新位置 |
| --- | --- |
| `classic_law.py` | `Q_2_1/Q_2_1_1_classic_law.py` |
| `classic_law_early_stopping.py` | `Q_2_1/Q_2_1_2_classic_law_early_stopping.py` |
| `classic_law_analysis.py` | `Q_2_1/Q_2_1_3_classic_law_analysis.py` |
| `classic_law_figures_zh.py` | `Q_2_1/Q_2_1_4_classic_law_figures_zh.py` |
| `generalized_law.py` | `Q_2_2/Q_2_2_1_generalized_law.py` |
| `generalized_law_figures.py` | `Q_2_2/Q_2_2_2_generalized_law_figures.py` |
| `handoff_build.py` | `Q_2_3/Q_2_3_1_handoff_build.py` |
| `handoff_paper_form.py` | `Q_2_3/Q_2_3_2_handoff_paper_form.py` |
| `handoff_identification.py` | `Q_2_3/Q_2_3_3_handoff_identification.py` |
| `elasticity_analysis.py` | `Q_2_4/Q_2_4_1_elasticity_analysis.py` |
| `elasticity_figures.py` | `Q_2_4/Q_2_4_2_elasticity_figures.py` |
| `factor_elasticity_analysis.py` | `Q_2_4/Q_2_4_3_factor_elasticity_analysis.py` |
| `factor_elasticity_figures.py` | `Q_2_4/Q_2_4_4_factor_elasticity_figures.py` |
| `quality_substitution_analysis.py` | `Q_2_4/Q_2_4_5_quality_substitution_analysis.py` |
| `quality_substitution_figures.py` | `Q_2_4/Q_2_4_6_quality_substitution_figures.py` |
| `joint_substitution_analysis.py` | `Q_2_4/Q_2_4_7_joint_substitution_analysis.py` |
| `joint_substitution_figures.py` | `Q_2_4/Q_2_4_8_joint_substitution_figures.py` |

产物目录名（`results/` 下）**没有改动**，因此已有的中间结果与文档引用仍然有效。

---

## 七、图像约定

论文级图统一遵守以下约定，改图时请沿用：

* **不画标题**，单位写在轴标题里（损失用 `val_loss` 原始单位）；
* 单图 `6×4 in`，左右合成图 `12×4 in`；`axes.labelsize=14`、`xtick/ytick/legend=12`、
  `savefig.bbox="tight"`、白底；
* 坐标轴标题用**普通 Unicode 字符**而不是 `$...$`：mathtext 渲染不了汉字，
  混排会直接抛 `Font 'default' does not have a glyph`；
* 调色板：`#4C78A8`（主）、`#7BA7CC`（次）、`#BFBFBF`（弱化）。

出图脚本支持 `--formats png,svg,pdf`、`--dpi`、`--output-dir`（`--formats` 默认只出 png）；
`Q_2_2_2_generalized_law_figures.py` 还支持 `--figures <逗号分隔的 panel 名>` 与
`--skip-verify`（不建议使用，自检是这套图的主要保障）。

**中文字体**：仓库不再随附字体文件。中文标注脚本按
`DSH_CJK_FONT` 环境变量 → `.fontwork/NotoSansCJKsc-Regular.otf` → 系统 Noto → 系统已装 CJK 字族的顺序查找，
找不到时 matplotlib 只发 `UserWarning` 并把汉字画成方框，不会中断出图。

---

## 八、已知边界与可识别性（引用结论时请一并带上）

这些不是实现缺陷，而是数据本身给出的上限，写进论文时应如实标注：

* **`(η_k, ζ_k)` 是插值而非独立验证**：3 个观测尺度 × 逐域 2 个指数，残差维数为 0。
  `Q_2_3_3_handoff_identification.py` 用配对回归给出了不依赖该形状假设的独立辨识
  （配对 R² 最小 0.9761 / 中位 0.9868，留出折 60M 损失中位 relRMSE 1.16%），
  13 个域中有 9 个域的 `η ≠ ζ` 在 95% 水平上成立；
* **`Q_i` 不可单独辨识**：`Φ_k` 只通过乘积 `p_i·Q_i` 使用质量，`(p, Q)` 的缩放不可分；
  因此本目录不报告「质量对损失的独立边际效应」，该效应改由 §5.4 用独立的质量实验单独估计
  （10 点质量网格 `Q_score ∈ {0.1, 0.2, …, 1.0}` 配 `N/D` 组合，共 450 行）；
* **`w^pQ` / `w^clr` 不可逐域解释**：**43.67%** 的系数为负（标准化后符号分布相同），
  且系数只与 `Q_i` 识别到共同尺度（换 `Q` 的刻度不改变预测）。它们是拟合出来的响应方向，
  不是「域的客观价值」；
* **1B 锚点只有 64 个配比**，不足以支撑独立的实际损失估计量；
* **`B6` 不是 `B7` 的独立重复**（`B6 ⊂ B7`，360/360 键完全相同、`val_loss` 逐行一致），
  两者同时使用不构成交叉验证；**`B8` 不可用**（138 个格点 100% 方向反转，
  1704 行中有 362 行压在硬性 `L = 0.5` 地板上）；
* **数值列在 CSV 里同时存在两种字符串写法**：质量实验的 `N_params_B` / `D_tokens_B`
  同一数值会出现 `0.7` 与 `0.70`、`1.0` 与 `1.00` 两种形式（14 种字符串对应 7 个数值）。
  按**字符串**分组会凭空多出分组，读入时一律先 `float()` 再比较；
* **秩相关指标对运行环境敏感**：`scaling_law_full/` 里已存的 `scaling_law_full_results.json`
  由更早的环境（旧版 numpy）产出，在本文档环境重跑时，除 `B1_main_fit` 的五参数逐位一致外，
  各 `spearman_rho` 会有小差别（B1 约 7e−7、B4 约 1.5e−3、B5 约 3e−4），
  自适应权重随自举分数有 1e−5 量级的漂移。这是环境漂移而非代码差异。引用历史数字时注意环境口径；
* **文件来源标注**：`results/*/` 下的产物都是脚本生成的中间结果而非实验观测，
  只有 `data/real_attachments/` 下的是题目给定的原始数据。

---

## 九、不在本目录中的前置脚本

`results/generalized_scaling_law/` 里有四份前置分析
（`A_标准标度律验证.md`、`AB_标度律归一化可行性.md`、`A_平移到B基底.md`、`mixture_structure.json`），
它们由当时存在、但**从未提交进版本库**的脚本产生，只能作为已有交付物引用，**无法在本目录内重算**；
`results/scaling_law_full/scaling_law_academic_flowchart.{png,pdf,svg}` 同理。

当前 `Q2/` 下的脚本已不依赖它们中的任何一个。
