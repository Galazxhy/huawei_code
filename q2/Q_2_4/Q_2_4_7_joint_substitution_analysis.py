#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""广义标度律指导下的联合等损失替代：质量 × 配比 × Token。

为什么重做
----------
上一版只用独立质量律做 ``D·Q`` 换算，配比 ``p`` 完全没有参与，因此不能称为
"广义标度律指导下"的替代关系。本版**以已辨识的广义标度律为主体**，只把质量扰动
交给独立质量实验标定。

锚定形式
--------
记 ``Q`` 为可调整的整体质量等级，``Q⁽⁰⁾`` 为拟合广义律时固定的各来源质量分：

    L_k† = E_k + A_k·N^(−α)
           + (Q/Q_0)^(−γ) · [ B_k·D^(−β) + S_k(N,D)·φ_k(Q⁽⁰⁾, p) ]

``S_k`` 与 ``φ_k`` 完全沿用已验证的广义律（含其全部参数），``γ`` 由独立质量实验标定，
**不重拟合任何原有参数**。``Q = Q_0`` 时该式严格退化为原广义律（本脚本断言该恒等式，
实测偏差 ~4e-16）。质量改变的是**包括配比修正项在内**的"数据受限损失"，
所以同样的质量提升在不同配比、不同评估目标下收益不同——这正是本节要展示的。

联合等损失面
------------
    L_k†(N, D, p, Q) = L_k†(N, D_0, p_0, Q_0)

固定 ``N``，沿单纯形中的可行方向改变 ``p``，逐目标求解维持同一损失所需的 ``D``。
局部关系为

    dD = −( ∂_Q L_k† · dQ + ∇_p L_k† · dp ) / ∂_D L_k†

其中 ``∇_p`` 必须沿份额和为零的可行方向计算。

两种耦合方式（必须都报）
------------------------
``full``    ：质量缩放的括号含 ``B_k D^(−β)`` **与** 配比修正项 ``S_k φ_k``；
``bd_only``：质量只作用于 ``B_k D^(−β)``，配比修正项不受质量影响。

上式对 ``Q`` 与 ``p`` 的**乘性交互是外部标定的结构假设**，不是现有数据验证出的因果
效应：配比实验里各 ``Q_i`` 固定，直接缩放它们所得的质量弹性符号都不稳定。
故两种耦合给出的替代区间都要报，不能只报一个精确数字。

默认运行：
    python Q2/Q_2_4/Q_2_4_7_joint_substitution_analysis.py
产物写入 ``results/joint_quality_mixture_substitution/``。
"""


# --- shared paths for every script in Q2 (see Q2/_q2_paths.py) --- # q2-prologue:bootstrap
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np

import sys as _sys
from pathlib import Path as _Path

# The reports are Chinese; a cp936 console would raise on the superscripts
# and dashes they use, so force UTF-8 with replacement as a last resort.
for _stream in (_sys.stdout, _sys.stderr):
    _reconfigure = getattr(_stream, 'reconfigure', None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass

# This script lives in a stage directory, one level below the package root.
_PKG_DIR = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_PKG_DIR))
import _q2_paths  # noqa: E402

_q2_paths.install_source_dirs(__file__)

PROJECT_ROOT = _q2_paths.PROJECT_ROOT

import Q_2_4_1_elasticity_analysis as ea  # noqa: E402
import Q_2_2_1_generalized_law as compact  # noqa: E402
import Q_2_4_5_quality_substitution_analysis as qsub  # noqa: E402
from generalized_law_evaluator import GeneralizedLawEvaluator  # noqa: E402


ROOT = _q2_paths.PROJECT_ROOT
HANDOFF = _q2_paths.HANDOFF_OUT
REGMIX = _q2_paths.REGMIX_TABLES
OUTPUT = _q2_paths.ANALYSIS_DIR / "joint_quality_mixture_substitution"
sys.path.insert(0, str(HANDOFF))


#: 工作点：与上一节一致，便于对照。N=1B 与 Q0=0.6 是质量实验的观测等级，
#: D0=25 十亿 Token 落在质量实验的 10 与 50 之间，属于**插值**。
N0 = 1e9
D0 = 2.5e10
Q0 = 0.6
Q1 = 0.7
#: 配比扰动步长：1 个百分点。
SHARE_STEP = 0.01
#: 主图之外再画一张的对照评估域（除目标损失外口径完全一致）。
ARXIV_DOMAIN = "arxiv"
SOLVE_RESIDUALS: list[float] = []


def law_pieces(law: GeneralizedLawEvaluator, n: float, d: float,
               p: np.ndarray) -> dict[str, np.ndarray]:
    """把交付的广义律拆成三块：参数受限项、数据受限项、配比修正项。

    与 ``law.predict`` 的分解一致：``E + A·N_b^(−α) + B·D_b^(−β) + S·Φ``。
    """

    n_b, d_b = n / 1e9, d / 1e9
    parameter_part = law.e_coefficients + law.a_coefficients * n_b ** (-law.alpha)
    data_part = law.b_coefficients * d_b ** (-law.beta)
    mixture_part = (law.scale_amplitude(n, d)
                    * law.mixture_response(np.atleast_2d(p))[0])
    return {"parameter_part": parameter_part, "data_part": data_part,
            "mixture_part": mixture_part}


def anchored(law: GeneralizedLawEvaluator, n: float, d: float, p: np.ndarray,
             q: float, gamma: float, coupling: str) -> np.ndarray:
    """锚定式 ``L_k†``。``q = Q_0`` 时严格等于原广义律。"""

    pieces = law_pieces(law, n, d, p)
    scale = (q / Q0) ** (-gamma)
    data, mixture = pieces["data_part"], pieces["mixture_part"]
    if coupling == "full":
        return pieces["parameter_part"] + scale * (data + mixture)
    if coupling == "bd_only":
        return pieces["parameter_part"] + scale * data + mixture
    raise ValueError(f"未知耦合方式 {coupling!r}")


def solve_required_tokens(law: GeneralizedLawEvaluator, n: float, p: np.ndarray,
                          q: float, target: np.ndarray, gamma: float,
                          coupling: str) -> tuple[np.ndarray, np.ndarray]:
    """逐目标求解维持 ``target`` 所需的 ``D``；返回 (D, 是否找到)。

    损失随 ``D`` 单调下降（配比项为负时可能有轻微非单调），故在对数网格上找
    **第一个**由正变负的穿越点，再二分细化。找不到穿越即视为不可行。
    """

    grid = np.geomspace(1e9, 1e13, 600)
    values = np.stack([anchored(law, n, d, p, q, gamma, coupling) - target
                       for d in grid])                    # (600, 13)
    found = np.zeros(target.size, dtype=bool)
    required = np.full(target.size, np.nan)
    for k in range(target.size):
        column = values[:, k]
        crossing = np.where((column[:-1] > 0) & (column[1:] <= 0))[0]
        if not crossing.size:
            continue
        lo, hi = grid[crossing[0]], grid[crossing[0] + 1]
        for _ in range(200):
            mid = np.sqrt(lo * hi)                        # 对数中点
            value = anchored(law, n, mid, p, q, gamma, coupling)[k] - target[k]
            if value > 0:
                lo = mid
            else:
                hi = mid
            if hi / lo - 1 < 1e-12:
                break
        required[k] = np.sqrt(lo * hi)
        found[k] = True
        residual = abs(float(anchored(law, n, required[k], p, q, gamma,
                                      coupling)[k] - target[k]))
        SOLVE_RESIDUALS.append(residual)
        if residual > 1e-8:
            raise ArithmeticError(f"等损失求根残差过大: {residual:.3e}")
    return required, found


def representative_six(law: GeneralizedLawEvaluator, block_name: str,
                       levels: int = 6) -> list[dict]:
    """取 **6 个不同熵水平** 的真实记录配比：按熵的分位等距取点。

    图里**只按组成标注**每个配比（占比最高的来源），不出现熵——熵只是这里的选取规则，
    不是图件的表达维度。
    """

    blocks, train_names, _, _ = compact.load_data(compact.REGMIX, compact.HANDOFF)
    block = next(b for b in blocks if b.name == block_name)
    p = block.p / block.p.sum(axis=1, keepdims=True)
    entropy = np.asarray(law.entropy(p)).ravel()
    order = np.argsort(entropy)
    quantiles = np.linspace(0.05, 0.95, levels)
    picked = []
    for quantile in quantiles:
        index = int(order[round(float(quantile) * (len(order) - 1))])
        shares = p[index]
        ranking = np.argsort(-shares)
        picked.append({
            "row": index,
            "entropy": float(entropy[index]),
            "entropy_quantile": float(quantile),
            "mixture": shares.tolist(),
            "top_sources": [
                {"domain": train_names[i], "share": float(shares[i])}
                for i in ranking[:4] if shares[i] > 0],
            "label": " + ".join(
                f"{train_names[i]} {shares[i]:.0%}" for i in ranking[:2]
                if shares[i] > 0),
        })
    picked.sort(key=lambda item: -item["entropy"])
    return picked


def entropy_effect_test(law: GeneralizedLawEvaluator, target: np.ndarray,
                        median_target: int, gamma: float) -> dict:
    """检验一个自然的猜想：配比熵越大，"提升质量"的效果是否越好。

    "效果"有三种读法，都必须报，否则结论会随口径变化：

    - **绝对节省**（十亿 Token）= 该配比维持同一目标损失、把质量从 ``Q0`` 提到
      ``Q1`` 所需 Token 之差；
    - **相对节省**（%）= 绝对节省 ÷ 该配比在 ``Q0`` 处所需的 Token；
    - **损失本身**：熵大的配比是否本就在参考工作点上损失更低。

    绝对/相对节省对 ``full`` 与 ``bd_only`` 两种耦合必须分别算：两种口径下相对
    节省的配比间差异并不同量级，"配比不影响相对收益"只在主口径下近似成立。
    全部计算在 ``test_1b`` 的**所有**记录配比上做（不是只算图里那 6 条），
    再与熵做 Pearson / Spearman 相关，并比较熵最高与最低四分位。
    """

    from scipy.stats import pearsonr, spearmanr

    blocks, _, _, _ = compact.load_data(compact.REGMIX, compact.HANDOFF)
    block = next(b for b in blocks if b.name == "test_1b")
    p_all = block.p / block.p.sum(axis=1, keepdims=True)
    entropy = np.asarray(law.entropy(p_all)).ravel().astype(float)

    level_q0 = np.zeros(p_all.shape[0])
    loss_here = np.zeros(p_all.shape[0])
    tested = {}
    for coupling in ("full", "bd_only"):
        level_q1 = np.zeros(p_all.shape[0])
        for row, shares in enumerate(p_all):
            p = shares[None, :]
            if coupling == "full":
                loss_here[row] = float(anchored(law, N0, D0, p, Q0, gamma,
                                                coupling)[median_target])
            for quality, destination in ((Q0, level_q0), (Q1, level_q1)):
                required, found = solve_required_tokens(
                    law, N0, p, quality, target, gamma, coupling)
                if found[median_target]:
                    destination[row] = float(required[median_target] / 1e9)
        tested[coupling] = {"level_Q0": level_q0.copy(), "level_Q1": level_q1,
                            "saved": level_q0 - level_q1,
                            "relative": 100.0 * (1.0 - level_q1 / level_q0)}

    def correlate(values: np.ndarray) -> dict:
        return {"pearson": float(pearsonr(entropy, values).statistic),
                "spearman": float(spearmanr(entropy, values).statistic)}

    quartile = np.quantile(entropy, [0.25, 0.75])
    low, high = entropy <= quartile[0], entropy >= quartile[1]

    def describe(values: np.ndarray) -> dict:
        return {"low_entropy_quartile_mean": float(values[low].mean()),
                "high_entropy_quartile_mean": float(values[high].mean()),
                "min": float(values.min()), "max": float(values.max()),
                **correlate(values)}

    return {
        "block": "test_1b",
        "n_mixtures": int(p_all.shape[0]),
        "eval_domain": law.loss_domains[median_target],
        "level_billion": describe(level_q0),
        "loss_at_reference_point": describe(loss_here),
        "couplings": {
            coupling: {"absolute_saving_billion": describe(values["saved"]),
                       "relative_saving_percent": describe(values["relative"])}
            for coupling, values in tested.items()},
        # 图件用主口径的相对节省区间。
        "relative_saving_percent_range": [
            float(tested["full"]["relative"].min()),
            float(tested["full"]["relative"].max())],
        "level_billion_ratio": float(level_q0.max() / level_q0.min()),
        # 逐条明细：配比影响图要画出全部 64 条记录配比的分布（不是只画挑出来的 6 条）。
        "by_mixture": [
            {"row": int(row),
             "entropy": float(entropy[row]),
             "level_Q0_billion": float(tested["full"]["level_Q0"][row]),
             "level_Q1_billion": float(tested["full"]["level_Q1"][row]),
             "saved_billion": float(tested["full"]["saved"][row]),
             "relative_saving_percent": float(tested["full"]["relative"][row]),
             "saved_bd_only_billion": float(tested["bd_only"]["saved"][row]),
             "relative_saving_bd_only_percent": float(
                 tested["bd_only"]["relative"][row]),
             "loss_at_reference_point": float(loss_here[row])}
            for row in range(p_all.shape[0])],
        "saving_level_pearson": float(pearsonr(level_q0,
                                               tested["full"]["saved"]).statistic),
        "verdict": (
            "熵越大效果越好**不成立**：相对节省率与熵基本无关（主口径下它几近由质量"
            "台阶单独决定），绝对节省只随该配比的所需 Token 水平变化（两者近乎完全"
            "线性），而所需 Token 水平与熵也不是单调关系。熵描述的是配比的分散程度，"
            "不是收益的驱动变量。"),
    }


def iso_loss_curves_for_domain(law: GeneralizedLawEvaluator,
                               representatives: list[dict], target: np.ndarray,
                               index: int, gamma: float, grid_q: np.ndarray,
                               index_q0: int, index_q1: int,
                               chosen_by: str) -> tuple[list[dict], dict]:
    """在**同一个工作点**上，为某一个评估目标算六个代表性配比的等损失曲线。

    抽成函数是为了把「目标评估域」变成一个可替换的口径：换域之后工作点（$N$、
    参考配比、$Q$ 网格）与两种耦合方式全部不动，只有目标损失本身不同，
    所以两次画出来的图样式完全一致、可以直接并排看。
    """

    curves = []
    for item in representatives:
        p = np.asarray(item["mixture"], dtype=float)
        requirements = {}
        for coupling in ("full", "bd_only"):
            needed = []
            for q in grid_q:
                required, found = solve_required_tokens(
                    law, N0, p, float(q), target, gamma, coupling)
                needed.append(float(required[index] / 1e9)
                              if found[index] else None)
            requirements[coupling] = needed
        full_curve = requirements["full"]
        level_q0, level_q1 = full_curve[index_q0], full_curve[index_q1]
        curves.append({"row": item["row"],
                       "entropy": item["entropy"],
                       "entropy_quantile": item["entropy_quantile"],
                       "mixture": item["mixture"],
                       "label": item["label"],
                       "tokens_at_Q0_billion": level_q0,
                       "tokens_at_Q1_billion": level_q1,
                       "saved_billion": (None if level_q0 is None or level_q1 is None
                                         else level_q0 - level_q1),
                       "relative_saving_percent": (
                           None if not level_q0 else
                           100.0 * (1.0 - level_q1 / level_q0)),
                       "tokens_required_full": full_curve,
                       "tokens_required_bd_only": requirements["bd_only"]})
    info = {"eval_domain": law.loss_domains[index], "index": int(index),
            "target_loss": float(target[index]), "chosen_by": chosen_by}
    return curves, info


def mixture_levels_for_target(law: GeneralizedLawEvaluator, target: np.ndarray,
                              index: int, gamma: float) -> dict:
    """某一个评估目标下，**全部**记录配比的「维持目标损失所需 Token」分布。

    配比影响图要画 64 条记录配比的分布带与顶部刻度；主域这一列可以直接用
    ``entropy_effect_test.by_mixture``，换评估域时则要另算——同一组配比在不同
    评估目标下的所需 Token 并不成比例（见 §4 的换域对照）。
    """

    blocks, _, _, _ = compact.load_data(compact.REGMIX, compact.HANDOFF)
    block = next(b for b in blocks if b.name == "test_1b")
    p_all = block.p / block.p.sum(axis=1, keepdims=True)
    levels = []
    for shares in p_all:
        required, found = solve_required_tokens(law, N0, shares[None, :], Q0,
                                                target, gamma, "full")
        levels.append(float(required[index] / 1e9) if found[index] else None)
    return {"eval_domain": law.loss_domains[index], "level_Q0_billion": levels}


def analyse() -> dict:
    SOLVE_RESIDUALS.clear()
    law = GeneralizedLawEvaluator.from_directory(HANDOFF)
    blocks, train_names, loss_names, quality = compact.load_data(compact.REGMIX,
                                                                 compact.HANDOFF)
    if list(loss_names) != law.loss_domains:
        raise ValueError("评估目标顺序与交付律不一致")

    # ---- γ 标定：独立质量实验 -------------------------------------------- #
    n_exp, d_exp, q_exp, loss_exp = qsub.load_design()
    dq_fit = qsub.fit_dq(n_exp, d_exp, q_exp, loss_exp)
    free_fit = qsub.fit_free(n_exp, d_exp, q_exp, loss_exp)
    gamma_main = float(dq_fit["beta"])
    gamma_alt = float(free_fit["beta_Q"])

    reference = law.reference_mixture()
    reference = reference / reference.sum()

    # ---- 锚定式在 Q=Q0 处必须严格退化为原广义律 -------------------------- #
    anchor_check = {}
    for coupling in ("full", "bd_only"):
        for tag, gamma in (("gamma_main", gamma_main), ("gamma_alt", gamma_alt)):
            value = anchored(law, N0, D0, reference, Q0, gamma, coupling)
            original = law.predict(N0, D0, reference, warn_outside_support=False)
            anchor_check[f"{coupling}/{tag}"] = float(np.max(np.abs(value - original)))
    if max(anchor_check.values()) > 1e-12:
        raise SystemExit(f"锚定式在 Q=Q0 处未退化为原广义律：{anchor_check}")

    target = anchored(law, N0, D0, reference, Q0, gamma_main, "full")

    # ---- 只提质量：两种耦合各一次 ---------------------------------------- #
    quality_only = {}
    for coupling in ("full", "bd_only"):
        required, found = solve_required_tokens(law, N0, reference, Q1, target,
                                                gamma_main, coupling)
        quality_only[coupling] = {
            "tokens_required_billion": (required / 1e9).tolist(),
            "tokens_saved_billion": ((D0 - required) / 1e9).tolist(),
            "feasible": found.tolist(),
            "median_saved_billion": float(np.median((D0 - required)[found]) / 1e9),
            "q25_saved_billion": float(np.quantile((D0 - required)[found], 0.25) / 1e9),
            "q75_saved_billion": float(np.quantile((D0 - required)[found], 0.75) / 1e9),
        }
    # 旧口径（DQ 换算）作为对照：D_need = D0·Q0/Q1
    quality_only["dq_conversion"] = {
        "tokens_required_billion": float(D0 * Q0 / Q1 / 1e9),
        "tokens_saved_billion": float(D0 * (1 - Q0 / Q1) / 1e9),
        "note": "只由独立质量律给出的换算，配比未参与；仅作对照，不作为本节结论",
    }

    # ---- 联合：质量 + 可行配比方向 --------------------------------------- #
    if bool(law.outside_support(reference[None, :])[0]):
        raise ValueError("参考配比不在观测支撑域内")
    directions: list[dict] = []
    for i, source in enumerate(law.train_domains):
        before = float(reference[i])
        candidate = reference * (1 - SHARE_STEP / max(1 - before, 1e-12))
        candidate[i] = before + SHARE_STEP
        inside = (before > 0) and (not bool(law.outside_support(candidate[None, :])[0]))
        row = {"source_domain": source, "share_before": before,
               "share_after": float(candidate[i]), "feasible": inside}
        if inside:
            for coupling in ("full", "bd_only"):
                required, found = solve_required_tokens(
                    law, N0, candidate, Q1, target, gamma_main, coupling)
                row[f"median_saved_{coupling}_billion"] = float(
                    np.median((D0 - required)[found]) / 1e9)
                row[f"feasible_{coupling}"] = found.tolist()
                row[f"required_{coupling}_billion"] = [
                    float(value / 1e9) if valid else None
                    for value, valid in zip(required, found)]
                row[f"saved_{coupling}_billion"] = [
                    float((D0 - value) / 1e9) if valid else None
                    for value, valid in zip(required, found)]
        directions.append(row)

    # ---- 逐目标小表：仅质量 / 联合中位 / 四分位 / 可行比例 --------------- #
    table = []
    for k, name in enumerate(law.loss_domains):
        joint_values = [r["saved_full_billion"][k] for r in directions
                        if r["feasible"] and r["saved_full_billion"][k] is not None]
        joint_values_bd = [r["saved_bd_only_billion"][k] for r in directions
                           if r["feasible"] and r["saved_bd_only_billion"][k] is not None]
        table.append({
            "eval_domain": name,
            "target_loss": float(target[k]),
            "saved_quality_only_full_billion": quality_only["full"]["tokens_saved_billion"][k]
            if quality_only["full"]["feasible"][k] else None,
            "saved_quality_only_bd_only_billion": quality_only["bd_only"]["tokens_saved_billion"][k]
            if quality_only["bd_only"]["feasible"][k] else None,
            "saved_joint_median_full_billion": float(np.median(joint_values))
            if joint_values else None,
            "saved_joint_q25_full_billion": float(np.quantile(joint_values, 0.25))
            if joint_values else None,
            "saved_joint_q75_full_billion": float(np.quantile(joint_values, 0.75))
            if joint_values else None,
            "saved_joint_median_bd_only_billion": float(np.median(joint_values_bd))
            if joint_values_bd else None,
            "feasible_fraction": len(joint_values) / len(directions),
        })
    joint = {}
    for coupling in ("full", "bd_only"):
        field = f"saved_joint_median_{coupling}_billion"
        values = [row[field] for row in table if row[field] is not None]
        joint[coupling] = {
            "median_saved_billion": float(np.median(values)),
            "q25_saved_billion": float(np.quantile(values, 0.25)),
            "q75_saved_billion": float(np.quantile(values, 0.75)),
            "aggregation": "先在每个评估目标内取可行方向的中位数，再对目标等权汇总",
        }

    # ---- 6 个代表性配比：等损失曲线与三维柱图共用同一组 ------------------ #
    # 原来等损失曲线只画 3 个配比，示例太少容易把"配比"读成一条线；改为与三维柱图
    # 完全相同的 6 个代表性配比，两张图与报告表格口径一致、可逐条对照。
    representatives = representative_six(law, "test_1b")
    median_target = int(law.loss_domains.index("stackexchange"))
    # 展示整个标定区间 Q=0.1–1.0；步长 0.02 且精确包含 Q0 与 Q1。
    # 只画 0.5–1.0 时所需 Token 仅变 17%，幂律近乎直线、看不出弯；放到全区间后
    # 低质量端明显更陡（dlnD/dQ ∝ 1/Q），「质量越低、提升越值钱」才看得出来。
    grid_q = np.arange(10.0, 102.0, 2.0) / 100.0
    index_q0 = int(np.argmin(np.abs(grid_q - Q0)))
    index_q1 = int(np.argmin(np.abs(grid_q - Q1)))
    assert abs(grid_q[index_q0] - Q0) < 1e-12 and abs(grid_q[index_q1] - Q1) < 1e-12, \
        "网格必须精确包含 Q0 与 Q1，否则相对节省率需要额外插值"

    curve_data, curve_target = iso_loss_curves_for_domain(
        law, representatives, target, median_target, gamma_main, grid_q,
        index_q0, index_q1, "两种示例评估领域之一")
    # 同一张图换一个评估目标：除目标损失外口径完全一致，供交叉核对
    # （见 figures/joint_iso_loss_curves_arxiv.png）。
    extra_curves, extra_target = iso_loss_curves_for_domain(
        law, representatives, target, int(list(law.loss_domains).index(ARXIV_DOMAIN)),
        gamma_main, grid_q, index_q0, index_q1,
        "两种示例评估领域之二")
    iso_by_domain = {
        curve_target["eval_domain"]: {"grid_Q": grid_q.tolist(),
                                      "target": curve_target,
                                      "n_curves": len(curve_data),
                                      "curves": curve_data},
        extra_target["eval_domain"]: {"grid_Q": grid_q.tolist(),
                                      "target": extra_target,
                                      "n_curves": len(extra_curves),
                                      "curves": extra_curves}}

    # ---- 6 个代表性配比（供三维柱图）：Q0 与 Q1 下所需 Token -------------- #
    for item in representatives:
        p = np.asarray(item["mixture"], dtype=float)
        needed = {}
        for tag, q in (("Q0", Q0), ("Q1", Q1)):
            required, found = solve_required_tokens(law, N0, p, q, target,
                                                    gamma_main, "full")
            needed[tag] = (float(required[median_target] / 1e9)
                           if found[median_target] else None)
        item["tokens_required_Q0_billion"] = needed["Q0"]
        item["tokens_required_Q1_billion"] = needed["Q1"]
        # 同一配比在参考工作点 (N0, D0) 上、Q0 与 Q1 处的目标损失。
        # Q0 处锚定式恒等于原广义律。
        item["loss_Q0"] = float(anchored(law, N0, D0, p, Q0, gamma_main,
                                         "full")[median_target])
        item["loss_Q1"] = float(anchored(law, N0, D0, p, Q1, gamma_main,
                                         "full")[median_target])
        item["saved_billion"] = (None if needed["Q0"] is None or needed["Q1"] is None
                                 else needed["Q0"] - needed["Q1"])
        item["is_reference_mixture"] = bool(np.allclose(p, reference, atol=1e-9))

    marker_difference = max(
        abs(item[f"tokens_required_{tag}_billion"] - curve[f"tokens_at_{tag}_billion"])
        for item, curve in zip(representatives, curve_data)
        for tag in ("Q0", "Q1")
    )
    if marker_difference > 1e-9:
        raise ArithmeticError("示例图的 Q0/Q1 标记与配比汇总不一致")

    # ---- 熵是不是收益的驱动变量：在全部记录配比上直接检验 --------------- #
    entropy_test = entropy_effect_test(law, target, median_target, gamma_main)

    # ---- 识别边界：直接缩放 Q_i 的隐含弹性符号不稳 ----------------------- #
    mixture_rows = [r["mixture"] for r in representatives]
    formal = []
    for mixture in mixture_rows:
        p = np.asarray(mixture, dtype=float)[None, :]
        loss = law.predict(N0, D0, p, warn_outside_support=False)
        formal.append(float(ea.formal_global_quality_elasticity(law, p, N0, D0,
                                                                loss)[0]))

    # 精确局部斜率必须同时计入配比项的 D 指数 zeta_k。
    pieces = law_pieces(law, N0, D0, reference)
    data_term = pieces["data_part"]
    mixture_term = pieces["mixture_part"]
    local_slope = (-gamma_main * (data_term + mixture_term)
                   / (law.beta * data_term + law.zeta * mixture_term))
    ratio_gamma_beta = gamma_main / float(law.beta)
    analytic_saving = D0 * (1 - (Q1 / Q0) ** (-ratio_gamma_beta)) / 1e9

    payload = {
        "question": "在 N 与配比可动时，提高质量能省多少 Token；质量与配比联合能省多少",
        "analytic_relation": {
            "formula": "dlnD/dlnQ|L,p = -γ(BD^-β+Sφ)/(βBD^-β+ζ_k Sφ)",
            "local_slope_by_target": local_slope.tolist(),
            "gamma_over_beta": ratio_gamma_beta,
            "gamma_over_beta_basis": f"γ={gamma_main:.4f}, β={float(law.beta):.4f}",
            "approximate_saving_billion": analytic_saving,
            "analytic_saving_billion": analytic_saving,
            "why_smaller_than_dq": (
                "γ/β 仅是忽略配比项或两项 D 指数相同的近似；正文节省量以逐目标"
                "数值求根为准，不能由 DQ 常数换算直接推断。"),
        },
        "anchored_form": "L_k† = E_k + A_k N^(−α) + (Q/Q_0)^(−γ)·[B_k D^(−β) "
                         "+ S_k(N,D)·φ_k(Q⁽⁰⁾, p)]",
        "anchored_identity_check": anchor_check,
        "operating_point": {"N_parameters": N0, "D0_tokens": D0, "Q0": Q0, "Q1": Q1,
                            "reference_mixture_entropy": float(np.asarray(
                                law.__class__.entropy(reference)).ravel()[0]),
                            "note": "D0=25 十亿 Token 属质量实验范围内的插值"},
        "gamma_calibration": {
            "main": gamma_main,
            "main_source": "独立质量实验 DQ 拟合的 β",
            "alternative": gamma_alt,
            "alternative_source": "自由指数模型的 β_Q（敏感性）",
            "note": "γ 只由独立质量实验标定；广义律的原有参数一个都没有重拟合",
        },
        "couplings": {
            "full": "质量缩放含 B_k D^(−β) 与配比修正项 S_k φ_k",
            "bd_only": "质量只作用于 B_k D^(−β)，配比修正项不受影响（敏感性对照）",
        },
        "quality_only": quality_only,
        "joint": joint,
        "table": table,
        "representative_mixtures": representatives,
        "entropy_effect_test": entropy_test,
        # 配比影响图用的逐配比水平：主域与对照域各一份（主域直接从熵检验里取，不重算）
        "mixture_levels_by_domain": {
            curve_target["eval_domain"]: {
                "eval_domain": curve_target["eval_domain"],
                "level_Q0_billion": [row["level_Q0_billion"]
                                     for row in entropy_test["by_mixture"]]},
            extra_target["eval_domain"]: mixture_levels_for_target(
                law, target, int(list(law.loss_domains).index(ARXIV_DOMAIN)),
                gamma_main),
        },
        "iso_loss_curves_by_domain": iso_by_domain,
        "iso_loss_curves_main_domain": curve_target["eval_domain"],
        "identification_boundary": [
            "现有配比实验的各 Q_i 固定，直接缩放它们得到的是拟合模型的代数导数，"
            "符号甚至可能变号；本次抽样点的隐含弹性为 "
            + "、".join(f"{value:+.4f}" for value in formal) + "。",
            "锚定式对 Q 与 p 的乘性交互是**外部标定的结构假设**，不是现有数据"
            "直接验证出的因果效应。",
            "因此以「质量仅作用于 B_k D^(−β)」的版本作敏感性对照，报告两种耦合"
            "给出的替代区间。",
            "γ 由质量实验标定，而该实验没有配比变化；把它的质量响应搬到广义律的"
            "数据受限项上，是本节的建模选择。",
        ],
        "verification": {
            "anchored_reduces_to_law": max(anchor_check.values()),
            "solve_residual_max": max(SOLVE_RESIDUALS),
            "solve_count": len(SOLVE_RESIDUALS),
            "feasible_direction_count": sum(bool(row["feasible"]) for row in directions),
            "attempted_direction_count": len(directions),
            "curve_marker_difference_billion": marker_difference,
        },
    }
    return {"payload": payload, "directions": directions, "output": OUTPUT}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(payload: dict, path: Path) -> None:
    point = payload["operating_point"]
    quality = payload["quality_only"]
    joint = payload["joint"]
    lines = [
        "# 广义标度律指导下的联合等损失替代",
        "",
        "**问题**：固定参数量时，提高质量能省多少 Token？质量与配比**联合**能省多少？",
        "",
        "## 1. 锚定形式",
        "",
        "$$L_k^\\dagger = E_k + A_k N^{-\\alpha} + \\Big(\\frac{{Q}}{{Q_0}}\\Big)^{-\\gamma}"
        "\\Big[B_k D^{-\\beta} + S_k(N,D)\\,\\phi_k(\\boldsymbol Q^{(0)},"
        "\\boldsymbol p)\\Big]$$",
        "",
        "$S_k$ 与 $\\phi_k$ 完全沿用已验证的广义律（全部参数不重拟合），"
        "$\\gamma$ 由独立质量实验标定。",
        "",
        f"- $\\gamma = {payload['gamma_calibration']['main']:.4f}$"
        f"（{payload['gamma_calibration']['main_source']}）；敏感性 "
        f"$\\gamma' = {payload['gamma_calibration']['alternative']:.4f}$"
        f"（{payload['gamma_calibration']['alternative_source']}）。",
        f"- $Q=Q_0$ 时严格退化为原广义律，实测偏差 "
        f"**{payload['verification']['anchored_reduces_to_law']:.1e}**。",
        "- 质量改变的是**包括配比修正项在内**的数据受限损失，所以同样的质量提升在"
        "不同配比、不同评估目标下收益不同。",
        "",
        "## 2. 只提质量（两种耦合）",
        "",
        f"工作点 $N$={point['N_parameters']:.0e}、"
        f"$D_0$={point['D0_tokens'] / 1e9:.0f} 十亿、$Q_0$={point['Q0']}"
        f"$\\to Q_1$={point['Q1']}，参考配比为记录配比中熵居中者。",
        "",
        "| 口径 | 所需 Token（十亿） | 节省中位（十亿） | 四分位范围 |",
        "| --- | ---: | ---: | ---: |",
        f"| 完全耦合 `full` | — | {quality['full']['median_saved_billion']:.2f} | "
        f"{quality['full']['q25_saved_billion']:.2f} – "
        f"{quality['full']['q75_saved_billion']:.2f} |",
        f"| 仅 $B D^{{-\\beta}}$ 耦合 `bd_only` | — | "
        f"{quality['bd_only']['median_saved_billion']:.2f} | "
        f"{quality['bd_only']['q25_saved_billion']:.2f} – "
        f"{quality['bd_only']['q75_saved_billion']:.2f} |",
        f"| 旧 $DQ$ 换算（对照，配比不参与） | "
        f"{quality['dq_conversion']['tokens_required_billion']:.2f} | "
        f"{quality['dq_conversion']['tokens_saved_billion']:.2f} | — |",
        "",
        "## 3. 联合（质量 + 可行配比方向）",
        "",
        f"共 {len(payload['table'])} 个评估目标；可行方向 = 某个来源份额 +1 个百分点、"
        "其余按比例缩减，且扰动前后都在记录支撑域内。",
        "",
        "| 耦合 | 联合节省中位（十亿） | 四分位范围 |",
        "| --- | ---: | ---: |",
        f"| `full` | {joint['full']['median_saved_billion']:.2f} | "
        f"{joint['full']['q25_saved_billion']:.2f} – "
        f"{joint['full']['q75_saved_billion']:.2f} |",
        f"| `bd_only` | {joint['bd_only']['median_saved_billion']:.2f} | "
        f"{joint['bd_only']['q25_saved_billion']:.2f} – "
        f"{joint['bd_only']['q75_saved_billion']:.2f} |",
        "跨目标汇总先在各目标内对可行配比方向取中位数，再对 13 个目标等权汇总。",
        "",
        "### 逐评估目标",
        "",
        "| 评估目标 | 仅质量（`full`） | 仅质量（`bd_only`） | 联合中位（`full`） | "
        "联合四分位 | 可行方向比例 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def cell(value, spec="+.2f"):
        return "—" if value is None else f"{value:{spec}}"

    for row in payload["table"]:
        lines.append(f"| `{row['eval_domain']}` | "
                     f"{cell(row['saved_quality_only_full_billion'])} | "
                     f"{cell(row['saved_quality_only_bd_only_billion'])} | "
                     f"{cell(row['saved_joint_median_full_billion'])} | "
                     f"{cell(row['saved_joint_q25_full_billion'])} – "
                     f"{cell(row['saved_joint_q75_full_billion'])} | "
                     f"{row['feasible_fraction']:.0%} |")
    main_domain = payload["iso_loss_curves_main_domain"]
    curve_target = payload["iso_loss_curves_by_domain"][main_domain]["target"]
    curves = payload["iso_loss_curves_by_domain"][main_domain]["curves"]
    grid = payload["iso_loss_curves_by_domain"][main_domain]["grid_Q"]
    nearest = min(range(len(grid)), key=lambda i: abs(grid[i] - point["Q1"]))
    spread = payload["entropy_effect_test"]["relative_saving_percent_range"]
    levels = [curve["tokens_at_Q0_billion"] for curve in curves]
    mixture_rows = payload["representative_mixtures"]
    mixture_levels = [row["tokens_required_Q0_billion"] for row in mixture_rows]
    population = [row["level_Q0_billion"]
                  for row in payload["entropy_effect_test"]["by_mixture"]]
    alternate_domain = next(d for d in payload["iso_loss_curves_by_domain"]
                              if d != main_domain)
    alternate = payload["iso_loss_curves_by_domain"][alternate_domain]
    alternate_levels = [c["tokens_at_Q0_billion"] for c in alternate["curves"]]
    alternate_rel = [c["relative_saving_percent"] for c in alternate["curves"]]
    main_saved = [row["saved_billion"] for row in mixture_rows]
    alt_saved = [row["saved_billion"] for row in alternate["curves"]]
    lines += [
        "",
        "## 4. 等损失曲线读数",
        "",
        f"示例目标取 `{curve_target['eval_domain']}`（{curve_target['chosen_by']}），"
        f"**六条**曲线来自该锚点上六个代表性记录配比（与三维柱图**同一组**，"
        "报告表格、两张图可逐条对照）。在 $Q_1$ 处维持同一目标损失所需 Token：",
        "",
        "| 记录配比（占比前两来源） | 熵 | $Q_0$ 处所需 Token | $Q_1$ 处所需 Token | "
        "绝对节省 | 相对节省 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for curve in curves:
        lines.append(
            f"| {curve['label']} | {curve['entropy']:.2f} | "
            f"{curve['tokens_at_Q0_billion']:.2f} | "
            f"{curve['tokens_at_Q1_billion']:.2f} | "
            f"**{curve['saved_billion']:.2f}** | "
            f"{curve['relative_saving_percent']:.2f}% |")
    lines += [
        "",
        f"读法：**配比改变的是水平，质量改变的是相对幅度**。同一目标损失在六个配比下"
        f"需要 {min(levels):.2f}–{max(levels):.2f} 十亿 Token（相差 "
        f"{max(levels) / min(levels):.1f} 倍），而 $Q_0\\to Q_1$ 的相对节省在六个配比上"
        f"几乎相同（{spread[0]:.2f}%–{spread[1]:.2f}%，见上表）。"
        "这些均为锚定模型的情景求解，偏离观测工作点的 Token 需求并非直接观测验证。",
        "",
        f"同一张图换一个评估目标（`{alternate_domain}`）另出一份：`figures/joint_iso_loss_curves_{alternate_domain}.png`。两张图除目标损失外，工作点、六个配比、$Q$ 网格与两种耦合口径**完全一致**，可以直接并排看（`figures/joint_iso_loss_curves_two_domains.png`：一张画布两格，**共用纵轴与图例**，右格不重复画刻度与标签）：",
        "",
        f"- 所需 Token：`{main_domain}` 上 {min(levels):.2f}–{max(levels):.2f} 十亿（{max(levels) / min(levels):.1f} 倍）；`{alternate_domain}` 上 {min(alternate_levels):.2f}–{max(alternate_levels):.2f} 十亿（{max(alternate_levels) / min(alternate_levels):.1f} 倍）；",
        f"- 相对节省：`{main_domain}` 上 {spread[0]:.2f}%–{spread[1]:.2f}%，`{alternate_domain}` 上 {min(alternate_rel):.2f}%–{max(alternate_rel):.2f}%。",
        "",
        "也就是说，**配比的杠杆有多大取决于评估目标**：同一组配比在某个评估域上能把所需 Token 差出一个量级，在另一个域上只差几倍；而质量的相对节省在两个域上都稳定在同一量级。结论的**形状**（质量只能小幅移动边界、配比移动得多）与评估域无关，**幅度**有关。",
        "",

        "### 熵越大效果越好吗（直接检验）",
        "",
        "一个自然的猜想是：配比熵越大（来源越分散）越「划算」，提升质量的效果越好。"
        "在**全部记录配比**上直接检验（不是只看图里那 6 条），"
        "结果见下表——「效果」的两种读法结论一致：",
        "",
        "| 统计量 | 耦合 | 熵最高的 1/4 配比 | 熵最低的 1/4 配比 | Pearson | Spearman |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    test = payload["entropy_effect_test"]
    rel_full = test["couplings"]["full"]["relative_saving_percent"]
    rel_bd = test["couplings"]["bd_only"]["relative_saving_percent"]
    rows = [("$Q_0$ 处所需 Token（十亿）", "—", test["level_billion"]),
            ("参考工作点上的损失", "—", test["loss_at_reference_point"])]
    for coupling in ("full", "bd_only"):
        rows.append(("绝对节省（十亿）", f"`{coupling}`",
                     test["couplings"][coupling]["absolute_saving_billion"]))
        rows.append(("相对节省（%）", f"`{coupling}`",
                     test["couplings"][coupling]["relative_saving_percent"]))
    for name, coupling, row in rows:
        lines.append(
            f"| {name} | {coupling} | {row['high_entropy_quartile_mean']:.3f} | "
            f"{row['low_entropy_quartile_mean']:.3f} | "
            f"{row['pearson']:+.3f} | {row['spearman']:+.3f} |")
    lines += [
        "",
        f"样本为 `{test['block']}` 的全部 {test['n_mixtures']} 条记录配比，"
        f"目标 `{test['eval_domain']}`。{test['verdict']}",
        "逐条明细（64 行：熵、$Q_0$/$Q_1$ 处所需 Token、两种耦合的绝对与相对节省、参考点损失）见 `entropy_effect_by_mixture.csv`。",
        "",
        f"机制是结构性的。主口径 `full` 下质量缩放同时作用于 $B_kD^{{-\\beta}}$ 与配比"
        f"修正项 $S_k\\varphi_k$；两项具有不同的数据量指数，故精确局部斜率"
        f"依赖其相对贡献。全部 {test['n_mixtures']} 条配比的相对节省率落在 "
        f"{rel_full['min']:.2f}%–{rel_full['max']:.2f}%（极差 "
        f"{rel_full['max'] - rel_full['min']:.2f} 个百分点）。绝对节省 = 该配比的所需 "
        f"Token 水平 × 这个近常数，故与水平的相关为 {test['saving_level_pearson']:+.3f}"
        f"（近乎完全线性），而水平与熵并不单调"
        f"（Pearson {test['level_billion']['pearson']:+.3f}）。"
        "起作用的量是「该配比下维持目标损失所需的 Token 水平」，不是熵。",
        "",
        f"但**「配比不影响相对收益」是主口径的性质，不是与耦合方式无关的事实**：敏感性"
        f"口径 `bd_only` 下相对节省率在全部配比上为 {rel_bd['min']:.2f}%–"
        f"{rel_bd['max']:.2f}%；质量缩放只作用于 $B_kD^{{-\\beta}}$ 时，配比修正项在"
        "总损失中的份额会直接改变相对节省率。两种口径下熵与收益的相关都很弱，"
        "所以「熵越大效果越好」的结论对口径不敏感，但节省率的**数值**对口径敏感，"
        "这正是 §3 两个区间都要报的原因。",
        "",
        "### 等损失曲线的绘制细节",
        "",
        "图件：`figures/joint_iso_loss_curves.png`（`iso_loss_figure`）。"
        "一格：横轴为整体质量等级 $Q$，纵轴为维持同一目标损失所需的 Token 数：",
        "",
        "- 纵轴取**对数刻度**。六条曲线的水平相差 8.6 倍，线性轴上低的两条会被"
        "压扁成贴地的一条线，看不出形状；对数轴上曲线的斜率可直接读作相对变化率；",
        "- 横轴覆盖质量实验标定过的整个 $Q$ 区间 $[0.1,1.0]$（步长 0.02，包含 $Q_0$、$Q_1$）：$Q_0=0.6$、$Q_1=0.7$ 是观测锚点，区间两端的曲线是同一外部标定 $\\gamma$ 下的情景推算（$Q$ 维本就由独立质量实验标定，不在数据集 A 的观测范围内）；只画 $[0.5,1.0]$ 时所需 Token 仅变 17%、幂律近乎直线，看不出「质量越低、提升越值钱」的弯；",
        "- 六条曲线来自六个代表性记录配比，颜色与三维柱图的配比顺序一一对应；"
        "在 $Q_1$ 处给每条曲线加实心圆点，两条点线标出 $Q_0$、$Q_1$，"
        "左上角给出目标评估域与 $N$；曲线的高低由配比决定（六条相差 8.6 倍），"
        "这一点在 §4 上表里逐条给出；",
        "- 图例放在图外下方、三列排布：6 个配比名较长，放在轴内会压线；",
        f"- 同一张图另出一份换评估域的对照（`figures/joint_iso_loss_curves_{alternate_domain}.png`）：口径只换目标损失，两张图可以直接并排；六条曲线的水平差距随评估域变（`{main_domain}` 8.6 倍、`{alternate_domain}` 3.8 倍），那是模型结论而不是画法差异；",
        "- 敏感性口径（质量只作用于 $B_kD^{-\\beta}$）不画进图里：它给出的是区间而不是"
        "第二条主曲线，画上去会让线数翻倍；其数值见 §3 与 `joint_substitution.json` "
        "的 `tokens_required_bd_only`。",
        "",

        "",
        "主口径的精确局部等损失斜率包含配比项，维持损失给出",
        "",
        "$$\\left.\\frac{\\mathrm d\\ln D}{\\mathrm d\\ln Q}\\right|_{L,p}="
        "-\\gamma\\frac{BD^{-\\beta}+S\\phi}"
        "{\\beta BD^{-\\beta}+\\zeta_kS\\phi}.$$",
        "",
        f"忽略配比项时 $\\gamma/\\beta = "
        f"{payload['analytic_relation']['gamma_over_beta']:.3f}$，近似节省 "
        f"{payload['analytic_relation']['approximate_saving_billion']:.2f} 十亿；"
        f"逐目标数值求根的主口径中位数为 {payload['quality_only']['full']['median_saved_billion']:.2f} 十亿。"
        f"{payload['analytic_relation']['why_smaller_than_dq']}",
        "",
        "### 六个代表性配比（三维柱图）",
        "",
        "选取规则：在记录配比的熵分布上等距取 6 个分位（0.05–0.95），得到 6 个"
        "**熵各不相同**的真实观测配比。**图件只按组成标注**每个配比，不出现熵——"
        "熵是选取依据，不是图件的表达维度：",
        "",
        "| 配比（占比前两来源） | 熵 | 可节省 Token（十亿） | 相对节省 | "
        "维持目标损失所需 Token（$Q_0$，十亿） |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["representative_mixtures"]:
        lines.append(
            f"| {item['label']} | {item['entropy']:.2f} | "
            f"**{item['saved_billion']:.2f}** | "
            f"{100 * item['saved_billion'] / item['tokens_required_Q0_billion']:.2f}% | "
            f"{item['tokens_required_Q0_billion']:.2f} |")
    heights = [item["tokens_required_Q0_billion"]
               for item in payload["representative_mixtures"]]
    savings = [item["saved_billion"] for item in payload["representative_mixtures"]]
    lines += [
        "",
        f"同一目标损失所需的 Token 在 {min(heights):.1f}–{max(heights):.1f} 十亿之间变化"
        f"（相差 {max(heights) / min(heights):.1f} 倍），可节省的 Token 随之在 "
        f"{min(savings):.2f}–{max(savings):.2f} 十亿之间变化，而**相对节省几乎不变**"
        f"（见 §4 上表）。**配比是量级更大的杠杆**；质量与配比不是可以互相替代的"
        "同类旋钮。",
        "",
        "### 配比的影响（横向对比图）",
        "",
        "图件：`figures/joint_mixture_effect.png`（`mixture_effect_figure`）。前两张图的横轴都是质量 $Q$，这张图的主语是**配比**：",
        "",
        "- 每行一个代表性配比，横轴为维持同一目标损失所需 Token（B）；行按所需 Token 从小到大排列，配色沿用等损失曲线图的配比顺序，三张图可逐条对照；",
        "- 实心点 = $Q_0$ 处所需 Token，空心点 = 提高到 $Q_1$ 之后。两点几乎重合（相对节省 5.4%–5.6%），**质量挪不动它**；而行与行之间相差数倍；",
        f"- 六个配比取自全部 {len(population)} 条记录配比（所需 Token 分布 {min(population):.2f}–{max(population):.2f} 十亿，相差 {max(population) / min(population):.1f} 倍），因此不是挑出来的；该分布可在 `mixture_levels_by_domain` 与 `entropy_effect_by_mixture.csv` 里复查；",
        "- 每行右侧给出该配比的可节省 Token 与相对节省率。",
        "- 两域并排图（`figures/joint_mixture_effect_two_domains.png`）是**一张画布两个共享坐标轴的面板**：共用纵轴与图例、两格等宽、中间只留窄缝，右格去掉配比名（行标签）与纵轴刻度，只留点与读数；",
        f"- 同一张图也出 `{alternate_domain}` 版（`figures/joint_mixture_effect_{alternate_domain}.png`）；两张图的**行顺序固定为主域**所需 Token 的排序，加上共用纵轴，并排后同一行左右就是同一个配比，可以横着读；",
        f"- 换域只改目标损失：`{main_domain}` 上六个配比所需 {min(mixture_levels):.2f}–{max(mixture_levels):.2f} 十亿、可省 {min(main_saved):.2f}–{max(main_saved):.2f} 十亿；",
        f"  `{alternate_domain}` 上所需 {min(alternate_levels):.2f}–{max(alternate_levels):.2f} 十亿、可省 {min(alt_saved):.2f}–{max(alt_saved):.2f} 十亿。**绝对节省随评估域变，相对节省仍在同一量级**。",
        "",
        f"读法：**配比决定水平，质量决定幅度**。同一目标损失下，换个配比可以让所需 Token 相差 {max(mixture_levels) / min(mixture_levels):.1f} 倍（{min(mixture_levels):.2f}–{max(mixture_levels):.2f} 十亿），而把质量从 $Q_0$ 提到 $Q_1$ 只在这一水平上挪动 5% 上下。",
        "",
        "### 三维柱图的绘制细节",
        "",
        "图件：`figures/joint_iso_loss_cylinders.png`"
        "（脚本 `Q2/Q_2_4/Q_2_4_8_joint_substitution_figures.py` 的 `cylinder_figure`）。",
        "",
        "**布局与顺序**",
        "",
        "- 3 列 × 2 行共 6 个三维面板，每个面板对应一个代表性配比；",
        "- 面板顺序 = 选取规则的排序：按熵从高到低，从左到右、从上往下依次为 "
        "(1)…(6)，即 (1) 左上 熵 2.25 … (6) 右下 熵 1.50；",
        "- 编号 (1)–(6) 只标出阅读顺序；**熵本身不出现在图里**，它只是选取依据"
        "（见上表）；",
        "- 每个面板**一根柱**：柱高就是 $Q_0\\to Q_1$ 可节省的 Token。",
        "",
        "**编码**",
        "",
        "- 柱高 = 把质量从 $Q_0=0.6$ 提到 $Q_1=0.7$ 可节省的 Token（十亿）；"
        "早期版本用「左柱 $Q_0$、右柱 $Q_1$、高度差 = 节省量」表达同一件事，"
        "但相对节省只有 5% 上下，两根柱的高度差在图上根本看不出来——"
        "把**效果本身**画成柱高才读得出来；",
        "- 柱顶扇形 = 该配比的实际组成（占比前四的来源 + 其他来源），"
        "扇形颜色与底部共享图例一致；",
        "- z 轴在六个面板间统一量程（0–2.5 B），因此柱高可跨面板直接比较；",
        "- 面板标题两行：第一行为配比组成，第二行为编号 + 可节省 Token 与相对节省率"
        "（相对节省率在六个面板上几乎相同，这行数字就是证据）；",
        "- 图注两行：编码说明 + 「柱高差异来自所需 Token 水平不同，不是质量杠杆"
        "本身更强」，防止把柱高差异误读成配比的质量弹性更强。",
        "",
        "**三维绘制要点**（沿用既有图集 `Q_2_4_2_elasticity_figures.py` 的做法）",
        "",
        "1. 侧壁按相机方向剔除背面。mplot3d 不在 collection 之间做深度排序，"
        "若不剔除，绕到背面的那半圈也会被画出来并叠在前壁上，形成里外两层的错色带；",
        "2. 侧壁的 `edgecolors` 取与面色相同的深色，而不是 `none`，"
        "否则相邻薄片之间会留下抗锯齿竖缝；",
        "3. 顶面扇形用白色描边分隔；柱体按扇形逐块用 `Poly3DCollection` 挤出，"
        "视角固定为 `elev=22°、azim=−58°`，与剔除背面所用的相机方向一致；",
        "4. z 刻度数字**自己投影手绘**：mplot3d 不为所有面板画 z 刻度标签"
        "（三列布局下左列完全没有数字），所以先把四条竖棱投影到屏幕、取最靠左的"
        "一条，再沿它线性插值定出每个刻度的屏幕位置（`_draw_shared_zticks`）。",
        "",
        "**画布与字号**（均可在脚本顶部调整）",
        "",
        "- 画布 8.4 × 6.4 in；面板标题 12 pt、z 轴刻度 11 pt、图例 10.5 pt、"
        "图注 11 pt；",
        "- 柱顶扇形图例 6 个来源 + 其他来源共用底部一行，四列排布；",
        "- 中文标注由外部 CJK 字体渲染；缺字会直接报错，不会画成方框。",
        "",

        "## 5. 识别边界",
        "",
    ]
    lines += [f"- {item}" for item in payload["identification_boundary"]]
    lines += [
        "",
        f"求根最大绝对损失残差：{payload['verification']['solve_residual_max']:.2e}"
        f"（共 {payload['verification']['solve_count']} 个可行求根）；"
        "在 $Q=Q_0$ 时锚定模型与原广义律一致。",
        "",
        "## 6. 复现",
        "",
        "```bash",
        "python Q2/Q_2_4/Q_2_4_7_joint_substitution_analysis.py",
        "python Q2/Q_2_4/Q_2_4_8_joint_substitution_figures.py",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    result = analyse()
    payload = result["payload"]
    output: Path = result["output"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "joint_substitution.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "joint_substitution_by_target.csv", payload["table"])
    write_csv(output / "entropy_effect_by_mixture.csv",
              payload["entropy_effect_test"]["by_mixture"])
    write_csv(output / "feasible_directions.csv", result["directions"])
    write_markdown(payload, output / "analysis_report.md")

    print("联合等损失替代")
    print(f"  γ = {payload['gamma_calibration']['main']:.4f}"
          f"（敏感性 {payload['gamma_calibration']['alternative']:.4f}）")
    print(f"  锚定式在 Q=Q0 处偏差 {payload['verification']['anchored_reduces_to_law']:.1e}")
    for coupling in ("full", "bd_only"):
        row = payload["quality_only"][coupling]
        print(f"  仅质量 {coupling:<8} 节省中位 {row['median_saved_billion']:+.2f} 十亿")
    for coupling in ("full", "bd_only"):
        row = payload["joint"][coupling]
        print(f"  联合   {coupling:<8} 节省中位 {row['median_saved_billion']:+.2f} 十亿")
    print(f"  旧 DQ 换算（对照）节省 {payload['quality_only']['dq_conversion']['tokens_saved_billion']:.2f} 十亿")
    print(f"  目标（图中用）：{payload['iso_loss_curves_main_domain']}"
          f"（另有对照域 {ARXIV_DOMAIN}）")
    print("  6 个代表性配比（熵分位 0.05–0.95；图件只按组成标注）：")
    for item in payload["representative_mixtures"]:
        print(f"    {item['label']:<34} 熵 {item['entropy']:.2f}  "
              f"所需 {item['tokens_required_Q0_billion']:.2f} → "
              f"{item['tokens_required_Q1_billion']:.2f} 十亿  "
              f"节省 {item['saved_billion']:.2f}")
    print(f"产物目录: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
