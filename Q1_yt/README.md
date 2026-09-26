# 问题一复现代码

本目录公开问题一的完整计算链，包括质量表征、质量冲突消解和领域配比建模。原始竞赛数据、运行结果、缓存、写作材料和临时图片不随仓库上传。所有结果会在各阶段的 `results/` 目录中重新生成。

## 数据目录

代码按下列相对位置读取题目附件。`Q1_yt` 与 `F 题` 必须位于同一个仓库根目录下。

```text
huawei_code/
├─ Q1_yt/
│  ├─ Q1_1/
│  ├─ Q1_2/
│  └─ Q1_3/
└─ F 题/
   └─ F题/
      └─ real_attachments/
         └─ A_data_value/
            ├─ slimpajama_quality_signal_sample.jsonl/
            │  └─ slimpajama_quality_signal_sample.jsonl
            ├─ slimpajama_quality_extended/
            │  ├─ arxiv_part-6777d8857c6e-000486.jsonl
            │  └─ github_part-6777d8857c6e-000275.jsonl
            └─ regmix_tables/
               ├─ train_mixture_1m.csv
               ├─ train_pile_loss_1m.csv
               ├─ test_mixture_1m.csv
               ├─ test_pile_loss_1m.csv
               ├─ test_mixture_60m.csv
               ├─ test_pile_loss_60m.csv
               ├─ test_mixture_1B.csv
               ├─ test_pile_loss_1B.csv
               ├─ est_mixture_10b.csv
               ├─ est_pile_loss_10b.csv
               ├─ est_mixture_70b.csv
               └─ est_pile_loss_70b.csv
```

不要把数据复制进 `Q1_yt`，也不要把生成的 `results/` 提交到 GitHub。若改变上述目录层级，需要同步修改 `Q_1_1_1_audit_A1.py`、`Q_1_1_6_validate_extensions.py`、`Q_1_2_1_prepare_signals.py`、`Q_1_2_9_text_cases.py` 和 `Q_1_3_1_load_and_standardize.py` 顶部的数据路径。

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

完整运行顺序为 `Q1_1`、`Q1_2`、`Q1_3`。后两阶段会读取前一阶段生成的 `results/`，不能交换顺序。盲审抽样脚本只生成待复核文本和密封键。仓库同时保留原实验中的 AI 辅助复核脚本，以保证代码披露完整，该步骤不能替代独立人工复核。

## 文件说明

### Q1_1 质量表征

| 文件 | 用法 |
| --- | --- |
| `Q_1_1_1_audit_A1.py` | 定位 A1，审计字段、缺失、类型和领域样本量。 |
| `Q_1_1_2_scalarize_A1.py` | 将列表型和 logits 型质量信号转为标量。 |
| `Q_1_1_3_freeze_scale.py` | 在 A1 上冻结方向和域均衡经验分布标度。 |
| `Q_1_1_3_audit_standardized_A1.py` | 审计标准化后的 22 项指标。 |
| `Q_1_1_4_structure.py` | 计算平衡相关矩阵并选择指标簇结构。 |
| `Q_1_1_5_score_A1.py` | 合成样本质量分和领域质量分。 |
| `Q_1_1_6_validate_extensions.py` | 将冻结模型应用到 A2、A3 并检验代表性。 |
| `Q_1_1_7_blind_text_review.py` | 按领域和分位层抽取 210 条盲审样本。 |
| `Q_1_1_7b_ai_review_complete.py` | 复现原实验的 AI 辅助复核结果，仅作披露和对照。 |
| `Q_1_1_8_baselines.py` | 比较等权、CRITIC 和 PCA 基线。 |
| `Q_1_1_9_ai_text_audit.py` | 输出原实验中的少量 AI 辅助文本案例审计。 |
| `Q_1_1_10_verify_outputs.py` | 检查样本量、取值范围和关键产物。 |
| `Q_1_1_11_sensitivity_A1.py` | 检验方向、分档和聚类数敏感性。 |
| `Q_1_1_12_direction_evidence.py` | 汇总质量信号方向设定的经验证据。 |

### Q1_2 冲突消解

| 文件 | 用法 |
| --- | --- |
| `Q_1_2_1_prepare_signals.py` | 汇总 A1 至 A3 的方向化信号。 |
| `Q_1_2_2_relation_graph.py` | 构造质量指标依赖图。 |
| `Q_1_2_3_crossfit_residuals.py` | 计算域内交叉拟合条件残差。 |
| `Q_1_2_4_conflict_scores.py` | 由异常残差和符号冲突构造冲突分数。 |
| `Q_1_2_5_cause_analysis.py` | 分解冲突来源并绘制诊断图。 |
| `Q_1_2_6_reliability_and_Qstar.py` | 估计可靠性并生成校正质量分。 |
| `Q_1_2_7_baselines.py` | 与替代冲突规则和未校正分数比较。 |
| `Q_1_2_8_sensitivity.py` | 检查冲突阈值和校正强度敏感性。 |
| `Q_1_2_9_text_cases.py` | 导出高冲突文本案例。 |
| `Q_1_2_11_tail_evidence.py` | 统计高冲突尾部的修正量贡献。 |
| `Q_1_2_10_verify_outputs.py` | 验证冲突消解阶段的关键产物。 |

### Q1_3 领域配比

| 文件 | 用法 |
| --- | --- |
| `Q_1_3_1_load_and_standardize.py` | 读取配方与损失表并完成配对和标准化。 |
| `Q_1_3_2_scheffe_features.py` | 构造 Scheffe 一阶、交互和 ILR 特征。 |
| `Q_1_3_3_fit_scheffe_models.py` | 拟合逐任务及多任务稀疏配比模型。 |
| `Q_1_3_4_quality_prior.py` | 将领域质量作为配比响应的先验信息。 |
| `Q_1_3_5_substitution_complement.py` | 计算来源替代与互补关系。 |
| `Q_1_3_6_validation.py` | 进行同尺度留出和跨尺度验证。 |
| `Q_1_3_7_bootstrap_uncertainty.py` | 对配比效应进行 Bootstrap 不确定性分析。 |
| `Q_1_3_10_evidence_checks.py` | 检查效应方向和跨规模证据。 |
| `Q_1_3_11_final_checks.py` | 比较模型族并执行最终稳健性检验。 |
| `Q_1_3_13_experiment_manifest.py` | 生成实验数据和证据等级清单。 |
| `Q_1_3_12_handoff.py` | 导出供问题二读取的标准接口。 |
| `Q_1_3_9_verify_outputs.py` | 完整性检查并确认全部输出存在。 |
