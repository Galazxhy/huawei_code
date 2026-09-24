# 第二问代码：算力约束下的大语言模型标度律与资源配置

本目录是第二问的全部可执行代码。任务分两条互不混合的线：

1. **经典标度律** `L(N, D) = E + A·N^(−α) + B·D^(−β)`：只用参数量 `N` 与训练
   token 数 `D`，在 B1 上辨识五参数，再用 B2–B5 做跨来源验证、用 B9/B10 做大模型外推。
2. **广义标度律**：在同一经典项之上，追加**数据质量 `Q` 与训练配比 `p`** 的贡献项。
   由于附件 A 的配比实验只有 3 个 `(N, D)` 观测点，`(α, β)` 无法从 A 独立辨识，
   必须借用第一条线在 B1 上得到的指数基底，两条线因此有先后依赖。

---

## 〇、文件命名约定

脚本名统一为 `{主题}_{角色}.py`，同一主题的文件在目录里自然相邻：

| 主题 | 含义 | 文件 |
| --- | --- | --- |
| `classic_law*` | 经典标度律（`N`、`D` 两自变量） | `classic_law.py`、`_early_stopping.py`、`_analysis.py`、`_figures_zh.py` |
| `generalized_law*` | 广义标度律（加 `Q`、`p`） | `generalized_law.py`、`_figures.py` |
| `handoff_*` | 交付给第三问 | `handoff_build.py`、`_paper_form.py`、`_identification.py` |
| `elasticity_*` | 质量/配比的边际效益与弹性 | `elasticity_analysis.py`、`_figures.py` |

`*_analysis.py` 是主题主流程，`*_figures.py` / `*_figures_zh.py` 只出图（`_zh` 为中文标注）。

> **调用方式**：所有脚本的默认路径都相对仓库根目录解析（`Path(__file__).parents[1]`），
> 所以在任何工作目录下都可以直接 `python q2/<script>.py`。同目录互相 import 的脚本
> （如 `from classic_law import Observation`）依赖“脚本自身所在目录自动进入 `sys.path`”
> 这一行为，请用 `python q2/x.py` 调用，**不要用 `python -m q2.x`**。
> `q2/` 下没有 `__init__.py`，它是脚本目录而不是包。

---

## 一、脚本一览

### 1.1 经典标度律

| 脚本 | 作用 | 读 | 写 |
| --- | --- | --- | --- |
| `classic_law.py` | 在主训练日志上做五参数经典拟合（含指数约束的 5 级搜索阶梯、CV、bootstrap；见 §1.5） | `data/real_attachments/B_scaling_laws/pythia_training_log_existing.csv` | `data_analysis/traditional_scaling_law/{traditional_scaling_fit.json, traditional_scaling_predictions.csv}` |
| `classic_law_early_stopping.py` | **自适应加权结构早停**：用归一化的 B2–B5 约束模型结构，防止 B1 过拟合 | `data/real_attachments/B_scaling_laws/*.csv` | `data_analysis/scaling_law_early_stopping/{early_stopping_results.json, early_stopping_report.md, adaptive_weights.csv, chosen_model_predictions.csv, early_stopping_history.csv}` |
| `classic_law_analysis.py` | **第二问主流程**：B1 主拟合（默认取早停选中模型）→ B2–B5 交叉标度归一化验证 → B9/B10 大模型外推 | `data/real_attachments/B_scaling_laws/*.csv` | `data_analysis/scaling_law_full/{scaling_law_full_results.json, scaling_law_report.md, classic_validation_metrics.csv, large_model_extrapolation.csv, adaptive_weights.csv, early_stopping_history.csv, early_stopping_predictions.csv}` |
| `classic_law_figures_zh.py` | 论文用单图（**中文标注**，单图 6×4、左右合成图 12×4） | `data_analysis/traditional_scaling_law/*`、`data_analysis/scaling_law_full/*` | `data_analysis/traditional_scaling_law/traditional_scaling_law_*.png` |

`classic_law_analysis.py` **只做经典律**，不含任何质量/配比项。它写出的
`B1_main_fit` 是下游广义标度律的指数基底来源。

### 1.2 广义标度律（质量 + 配比）

| 脚本 | 作用 | 读 | 写 |
| --- | --- | --- | --- |
| `generalized_law.py` | **当前交付的广义标度律**：三折划分、配比响应的岭回归 + `clr(p)` 中心化、逐域规模幅度 `S_k`、`ε` 敏感性、残差校正消融、自动生成 Markdown 报告 | 附件 A `regmix_tables/`、`data_analysis/Q_1_to_2/`、`data_analysis/scaling_law_full/scaling_law_full_results.json`（取 `B1_main_fit`） | `data_analysis/Q_1_to_2/generalized_scaling_compact.json`、`data_analysis/generalized_scaling_law/generalized_scaling_law_compact_report.md` |
| `generalized_law_figures.py` | 广义律图集，默认 6 张读者向图，`--figures all` 出全部 21 张；**每次运行都强制自检**：重算的留出集指标必须复现 `generalized_scaling_compact.json`，不一致直接报错退出 | 同上，另把 `generalized_law` 作为模块 import | `data_analysis/generalized_scaling_law_drawings/generalized_law_<panel>.png` |

模型（`N_b`、`D_b` 以十亿为单位）：

```text
L_k(N, D, p, Q) = E_k + A_k·N_b^(−α) + B_k·D_b^(−β) + S_k(N, D)·Φ_k(p, Q)
S_k(N, D)       = (N_b / N_ref)^(−η_k) · (D_b / D_ref)^(−ζ_k)
Φ_k(p, Q)       = slope_ref_k · [ Ridge_k(Standardize([p_i·Q_i, clr_ε(p)_i])) − center_k ]
```

`α = β` 取自 B1，**不重新辨识**；`N_ref = 1e6` 参数、`D_ref = 1e9` token。

### 1.3 第二问 → 第三问交付

| 脚本 | 作用 | 写 |
| --- | --- | --- |
| `handoff_build.py` | 生成第三问的接口层：广义律形式与参数、配比响应可复算参数、C7 上下文长度可行性、附录 B 质量成本函数、弹性与替代关系，并落地一个可直接 `import` 的求值器 | `data_analysis/Q2_to_Q3/{Q2_to_Q3_inputs.json, generalized_scaling_law.json, generalized_scaling_law_parameters.csv, mixture_response.json, quality_mixture_inputs.csv, context_length_feasibility.csv, quality_cost_functions.json, elasticity_and_substitution.json, generalized_law_evaluator.py}` |
| `handoff_paper_form.py` | 把广义律整理成**可直接写进论文的形式**（含 13×17 系数表），并验证该写法与原模型的最大偏差 | `data_analysis/Q2_to_Q3/{广义标度律_论文写法.md, generalized_scaling_law_paper_coefficients.csv}` |
| `handoff_identification.py` | **配对跨规模辨识（PCXI）**：只用同一配比在两个规模上的配对损失，把形状项消掉，从而免形状假设、免 `ε` 地辨识 `(η_k, ζ_k)`，并用留出折检验其预测力 | `data_analysis/Q2_to_Q3/{paired_cross_scale_identification.json, 配对跨规模回归辨识.md}` |

`data_analysis/Q2_to_Q3/交付文档.md` 与 `generalized_law_evaluator.py` 是**手写**的交付物，
不由脚本生成；改脚本名时要一并手改其中的复现命令。

### 1.4 质量与配比的边际效益 / 弹性

| 脚本 | 作用 | 读 | 写 |
| --- | --- | --- | --- |
| `elasticity_analysis.py` | 质量的边际效益与弹性（用独立的 `B·(D·Q)^(−β)` 形式）；配比在单纯形上的可行方向边际效益与弹性（用组成广义律） | `Q2_to_Q3/generalized_law_evaluator.py`、B7 质量实验、Q2 交付 | `data_analysis/quality_mixture_elasticity/{quality_elasticity.json, allocation_elasticity.json, quality_factorial_grid.csv, mixture_share_quantile_grid.csv, mixture_domain_detail.csv, mixture_source_summary.csv, analysis_report.md}` |
| `elasticity_figures.py` | 上述分析的图集（中文标注） | `data_analysis/quality_mixture_elasticity/*.csv`、`*.json` | `figures/{quality_curves, quality_heatmap, representative_mixtures, representative_mixtures_loss_cylinders, mixture_gain_intervals, mixture_quantile_slices}.png` |

> 两条律的损失标定不同，**它们的导数不可相加、也不可相除**来推替代率；
> 脚本里对这一点有显式注释与产物说明。

### 1.5 `classic_law.py` 的搜索阶梯

指数不是一次放开的：先在共享指数下解最小二乘，再逐级释放自由度，每一级都用
B2–B5 的验证分数决定是否值得继续。`classic_law_analysis.py` 的早停就沿这条路径走。

| 阶段 | 指数约束 | 最小二乘拟合参数 | 网格搜索参数 | 搜索维数 | 作用 |
| --- | --- | --- | --- | --- | --- |
| 0 | `α = β = s₀`（`s₀` 给定） | `E, A, B` | 无 | 0 | 初始化和数值基准 |
| 1 | `α = β = s` | `E, A, B` | `s` | 1 | 共享指数基准 |
| 2N | `β = s*`，`α` 自由 | `E, A, B` | `α` | 1 | 检验是否需要单独释放参数规模指数 |
| 2D | `α = s*`，`β` 自由 | `E, A, B` | `β` | 1 | 检验是否需要单独释放数据规模指数 |
| 3 | `α, β` 均自由 | `E, A, B` | `(α, β)` | 2 | 完整五参数标准标度律 |
| 4 | 同阶段 3 | `E, A, B` | `(α, β)` 局部邻域 | 2（局部） | 数值精化，不属于新模型结构 |

| 搜索阶段 | 搜索范围 | 初始网格 | 后续操作 |
| --- | --- | --- | --- |
| 共享指数 | `s ∈ [0.01, 1.50]` | 31 点 | 在最优点附近缩小区间 |
| 单指数释放 | 释放的指数 `∈ [0.01, 1.50]` | 31 点 | 以最佳值为中心局部细化 |
| 完整二维搜索 | `(α, β) ∈ [0.01, 1.50]²` | 31×31 | 八邻域模式搜索 |
| 局部精化 | 最佳点周围一个网格步长 | 逐次减半 | 直到步长小于 `1e−7` |

---

## 二、运行环境

已在 `python 3.12.14` 上全量验证：

| 包 | 版本 | 被谁需要 |
| --- | --- | --- |
| `numpy` | 2.5.3 | 全部 |
| `scipy` | 1.18.1 | `classic_law*` 的 `spearmanr`；`generalized_law.py` 的 `minimize`/`nnls` |
| `scikit-learn` | 1.9.1 | `generalized_law.py`（`Ridge`、`StandardScaler`）及其下游 |
| `matplotlib` | 3.11.2 | 全部出图脚本 |

两个环境上的坑：

```bash
# 1. matplotlib 默认缓存目录不可写，出图脚本前必须指定
export MPLCONFIGDIR=$PWD/.mplcfg

# 2. 无 GUI 后端即可，脚本内部已 matplotlib.use("Agg")
```

推荐调用形式：

```bash
MPLCONFIGDIR=$PWD/.mplcfg python q2/generalized_law_figures.py
```

---

## 三、依赖：**除原始 `data/` 之外还需要什么**

`.gitignore` 忽略了 `/data`、`/data_analysis/`、`/.*/`。也就是说**克隆仓库后既没有原始
数据、也没有任何中间产物**。要让本目录的脚本跑起来，除了自行获取 `data/`
（1.9 GB 附件，见仓库说明）以外，还必须有下面两项：

### 3.1 `data_analysis/Q_1_to_2/` —— 第一问的交付（**必需，且不在本仓库内**）

由第一问产出（也以 `data_analysis/Q_1_to_2.zip` 的形式随交付包分发）。第二问至少要用到：

| 文件 | 用途 |
| --- | --- |
| `Q17_mapping.csv` | 17 个训练域的质量 `Q17`（`Q_i` 在广义律中被固定，不参与拟合） |
| `train_domain_names.csv`、`loss_domain_names.csv` | 17 训练域 / 13 评测域的名字与顺序 |
| `scale_normalization.json`、`loss_standardization.json` | 跨来源尺度归一化口径 |

`generalized_law.py` 还会把 `generalized_scaling_compact.json` **写回**
这个目录，后续出图、交付、弹性分析都从这里读。

### 3.2 `data_analysis/scaling_law_full/scaling_law_full_results.json` —— 自己跑出来的（**必需**）

广义标度律的 `(α, β)` 直接取自这个文件的 `B1_main_fit.parameters`
（`parameter_exponent` / `data_exponent`），因此**必须先跑完 §四 里的 `classic_law_analysis.py`**，
否则 `generalized_law.py` 会因为找不到指数基底而失败。

### 3.3 中文字体（**不在本仓库内**）

两个中文出图脚本（`classic_law_figures_zh.py`、`elasticity_figures.py`）在运行时需要
环境里有一份 CJK 字体。仓库此前自带过一份裁剪子集
（`q2/assets/NotoSansSC-Regular-subset.otf`）与其构建脚本 `q2/figure_font_subset.py`，
**现均已移除**，本仓库不再随附任何字体文件。

本机已装字体列表里没有任何 CJK 字族，因此中文标注需要由外部提供字体。注意 matplotlib
在缺字时只发 `UserWarning` 并把汉字画成方框，不会中断出图。

### 3.4 原始数据 `data/`（另附，不在 git 中）

| 路径 | 用途 |
| --- | --- |
| `data/real_attachments/B_scaling_laws/*.csv` | B1–B5、B9、B10 的全部经典标度律观测 |
| `data/real_attachments/A_data_value/regmix_tables/` | 附件 A 的 3 个观测尺度 × 配比 × 13 域 Loss |
| `data/real_attachments/C_efficiency_evolution/model_architecture_metadata.csv` | 交付给第三问的上下文长度可行性（C7） |

---

## 四、复现顺序

依赖是有向的，按下面顺序执行即可从零重建全部产物。

```bash
export MPLCONFIGDIR=$PWD/.mplcfg

# ── 经典标度律：两件事，产物不同，都要跑 ──────────────────────────────
python q2/classic_law.py                # 独立的无约束五参数拟合（传统口径）
python q2/classic_law_early_stopping.py # 自适应加权结构早停（提供被选中的约束模型）
# 前者写 traditional_scaling_fit.json + predictions.csv，是下面出图命令的输入；
# 后者写 scaling_law_early_stopping/，主流程默认使用它选中的模型。

# ── 主流程：B1 主拟合 + B2–B5 验证 + B9/B10 外推（必需） ───────────────
python q2/classic_law_analysis.py
python q2/classic_law_figures_zh.py     # 图集（中文，论文用单图）

# ── 广义标度律（依赖上一步写出的 B1_main_fit） ────────────────────────
python q2/generalized_law.py
python q2/generalized_law_figures.py                 # 6 张读者向图 + 自检
python q2/generalized_law_figures.py --figures all    # 21 张全量

# ── 第二问 → 第三问交付 ─────────────────────────────────────────────
python q2/handoff_build.py
python q2/handoff_paper_form.py
python q2/handoff_identification.py

# ── 质量 / 配比的边际效益与弹性 ──────────────────────────────────────
python q2/elasticity_analysis.py
python q2/elasticity_figures.py
```

出图脚本都支持 `--formats png,svg,pdf`、`--dpi`、`--output-dir`（`--formats` 默认只出 png）；
`generalized_law_figures.py` 还支持 `--figures <逗号分隔的 panel 名>` 与
`--skip-verify`（不建议使用，自检是这套图的主要保障）。

---

## 五、图像约定

论文级图统一遵守以下约定，改图时请沿用：

* **不画标题**，单位写在轴标题里（Loss 用 `val_loss` 原始单位，不用 `nats`）；
* 单图 `6×4 in`，左右合成图 `12×4 in`；`axes.labelsize=14`、`xtick/ytick/legend=12`、
  `savefig.bbox="tight"`、白底；
* 坐标轴标题用**普通 Unicode 字符**而不是 `$...$`：mathtext 渲染不了汉字，
  混排会直接抛 `Font 'default' does not have a glyph`；
* 调色板：`#4C78A8`（主）、`#7BA7CC`（次）、`#BFBFBF`（弱化）。

---

## 六、已知边界与可识别性（引用结论时请一并带上）

这些不是实现缺陷，而是数据本身给出的上限，写进论文时应如实标注：

* **`(η_k, ζ_k)` 是插值而非独立验证**：3 个观测尺度 × 逐域 2 个指数，残差维数为 0。
  `handoff_identification.py` 用配对回归给出了不依赖该形状假设的独立辨识
  （配对 R² 最小 0.9761 / 中位 0.9868，留出折 60M 损失中位 relRMSE 1.16%），
  13 个域中有 9 个域的 `η ≠ ζ` 在 95% 水平上成立；
* **`Q_i` 不可单独辨识**：`Φ_k` 只通过乘积 `p_i·Q_i` 使用质量，
  `(p, Q)` 的缩放不可分；因此本目录不报告“质量对 Loss 的独立边际效应”，
  该效应改由 §1.4 用独立的质量实验单独估计：B7 的 10 点质量网格
  （`Q_score ∈ {0.1, 0.2, …, 1.0}`，配 `N/D` 组合，共 450 行）；
* **`w^pQ` / `w^clr` 不可逐域解释**：69% 的系数为负，且在熵控制下逐域排序会塌掉
  （Spearman 由正转 −0.098）。它们是拟合出来的响应方向，不是“域的客观价值”；
* **1B 锚点只有 64 个配比**，不足以支撑独立的实际损失估计量；
* **B6 不是 B7 的独立重复**（B6 ⊂ B7，360/360 键完全相同、`val_loss` 逐行一致），
  两者同时使用不构成交叉验证；**B8 不可用**（138 个格点 100% 方向反转，
  1704 行中有 362 行压在硬性 `L = 0.5` 地板上）；
* **数值列在 CSV 里同时存在两种字符串写法**：B7 的 `N_params_B` / `D_tokens_B`
  同一数值会出现 `0.7` 与 `0.70`、`1.0` 与 `1.00` 两种形式（14 种字符串对应
  7 个数值）。按**字符串**分组会凭空多出分组，读入时一律先 `float()` 再比较；
* **秩相关指标对运行环境敏感**：`scaling_law_full/` 里已存的 `scaling_law_full_results.json`
  由更早的 `GNN` 环境（旧版 numpy）产出，在本文档环境重跑时，除 `B1_main_fit` 的五参数
  逐位一致外，各 `spearman_rho` 会有小差别（B1 约 7e−7、B4 约 1.5e−3、B5 约 3e−4），
  自适应权重随自举分数有 1e−5 量级的漂移。这是环境漂移而非代码差异：早先的
  `rank_correlation.py` 垫片在有 scipy 时就转发 `scipy.stats.spearmanr`，与现在直接
  import 等价，换成直接 import 不改变任何数值。引用历史数字时注意环境口径；
* **文件来源标注**：`data_analysis/*/` 下的产物都是脚本生成的中间结果而非实验观测，
  只有 `data/real_attachments/` 下的是题目给定的原始数据。

---

## 七、不在本仓库中的前置脚本

`data_analysis/generalized_scaling_law/` 里的四份前置分析
（`A_标准标度律验证.md`、`AB_标度律归一化可行性.md`、`A_平移到B基底.md`、`mixture_structure.json`）
由当时存在、但**从未提交进本仓库**的脚本产生：
`verify_a_standard_scaling_law.py`、`normalize_a_and_b_scaling_laws.py`、
`translate_a_law_to_b_basis.py`、`mixture_structure_analysis.py`（以及旧版
`generalized_scaling_law.py`）。它们只能作为已有交付物引用，**无法在本仓库内重算**；
`data_analysis/scaling_law_full/scaling_law_academic_flowchart.{png,pdf,svg}` 同理。

当前仓库内的 `q2/` 脚本已不依赖它们中的任何一个。
