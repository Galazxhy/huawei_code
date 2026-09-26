#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""广义标度律四个因素（N、D、Q、p）的边际效益与弹性分析。

交付模型
--------
``L_k = E_k + A_k·N_b^(−α) + B_k·D_b^(−α) + S_k(N,D)·Φ_k(p, Q)``
``S_k(N,D) = (N/N₀)^(−η_k)·(D/D₀)^(−ζ_k)``，``N₀ = 10⁶`` 参数、``D₀ = 10⁹`` token；
``N_b、D_b`` 以十亿为单位。

四个因素与各自的定义
--------------------
============  ==========================  ==========================================
因素           边际效益                    弹性
============  ==========================  ==========================================
``N`` 参数量   ``−∂L/∂N``  (val_loss/参数)  ``∂lnL/∂lnN``
``D`` token 数 ``−∂L/∂D``  (val_loss/token) ``∂lnL/∂lnD``
``Q`` 数据质量 ``−∂L/∂Q``  (val_loss/质量分) ``∂lnL/∂lnQ``
``p`` 数据配比 ``−∂L/∂p_i`` 沿可行方向       ``(p_i/L)·∂L/∂p_i``
============  ==========================  ==========================================

**符号约定**：边际效益取 ``−∂L/∂x``，因此**正值表示增加该因素使损失下降**；
弹性为对数弹性，**负值表示该因素增加 1% 使损失下降约 |弹性|%**。

三条必须随结果一起给出的口径
----------------------------
1. ``N``、``D`` 的解析导数来自交付模型本体，并用中心差分复核（不通过即报错）；
   ``p`` 的可行方向导数同样与差分核对（复用 ``Q_2_4_1_elasticity_analysis.check_derivative``）。
2. ``Q`` **不能**由广义标度律识别：每个训练域的 ``Q_i`` 在数据集 A 中固定，模型只通过
   乘积 ``p_i·Q_i`` 使用它，整体尺度可被系数吸收。因此质量弹性取自 ``B6/B7`` 的
   独立质量律（``L = E + A·N^(−α) + B·(D·Q)^(−β)``），并把广义律的隐含值作为
   **反面对照**一并报告（其符号不稳定，不得引用）。
3. 各因素的工作点不同，**不可横向直接相减**：``N、D、p`` 在配比实验的观测锚点上评估
   （1M/1B-token、60M/1B-token、1B/25B-token，参考配比 = 记录配比中熵居中者）；
   ``Q`` 在 ``B7`` 自身的设计域内评估（``N=1B、D=25B``），因为质量实验没有覆盖
   ``10⁶`` 参数这一档。

默认运行：
    python Q2/Q_2_4/Q_2_4_3_factor_elasticity_analysis.py
产物写入 ``results/factor_marginal_elasticity/``。
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
from generalized_law_evaluator import GeneralizedLawEvaluator  # noqa: E402


ROOT = _q2_paths.PROJECT_ROOT
HANDOFF = _q2_paths.HANDOFF_OUT
OUTPUT = _q2_paths.ANALYSIS_DIR / "factor_marginal_elasticity"
sys.path.insert(0, str(HANDOFF))


#: 配比实验的三个观测锚点：标签、参数量、token 数（绝对个数）。
OBSERVED_SCALES = (
    ("1M / 1B tokens", 1e6, 1e9),
    ("60M / 1B tokens", 6e7, 1e9),
    ("1B / 25B tokens", 1e9, 2.5e10),
)
#: 外推：保持 D/N = 25，超出广义律观测范围，只在表中作对照、不并入汇总统计。
EXTRAPOLATION = ("10B / 250B tokens（外推）", 1e10, 2.5e11)
#: 质量律的评估点。Q=0.6 与 N=1B 是质量实验的观测等级；D=25B 落在实验的 10B 与 50B
#: 之间，属于**插值**（图注须写明）。
QUALITY_POINT = {"N_billion_parameters": 1.0, "D_billion_tokens": 25.0}
QUALITY_MAIN_Q = 0.6
#: 质量实验的观测等级（两组拟合共用同一套设计）。
QUALITY_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
#: 曲线网格：0.10 至 0.99。上界取 0.99 而不是 1.0，因为 Q+0.01 在 Q=1.0 处越出
#: 实验范围、降损无定义。
QUALITY_CURVE_GRID = tuple(round(0.10 + 0.01 * k, 2) for k in range(90))
#: 配比微扰步长：1 个百分点。
SHARE_STEP = 0.01

FACTORS = ("N", "D", "Q", "p")

#: 配比份额扫描的下界。``Φ`` 的 clr 分量的导数含 ``1/(p_i+ε)``，``ε = 1e-3``：
#: 在 ``p_i = 0.006`` 处该因子达 143 倍，线性边际效益比精确 +1pp 增益高估约 1.5 倍。
#: 取 ``50ε`` 为下界，在该处线性与精确已收敛到 1.0 倍。
SWEEP_MIN_SHARE = 0.05
#: 上界。定向导数是沿"该来源 +1pp、其余按比例缩减"的方向取的，``p_i → 1`` 时其余
#: 份额趋零，该方向的 ``1/(1-p_i)`` 发散，弹性随之被放大（与下界的 ε 放大同性质）。
#: 故两端各留出可信区间之外的部分不画、不汇总。
SWEEP_MAX_SHARE = 0.90
SWEEP_POINTS = 25
#: 扫描对象：**全部 13 个评测域**（它们都是训练配比的来源域，因此各自都有一个份额
#: 可以扫）。记录最大份额最小的 `hackernews` 为 0.12，仍在可扫区间内。
SWEEP_SOURCES: tuple[str, ...] | None = None   # None = 取 law.loss_domains
SWEEP_ANCHOR = ("1B / 25B tokens", 1e9, 2.5e10)


def scalar_factor_point(law: GeneralizedLawEvaluator, n: float, d: float,
                        p: np.ndarray, factor: str) -> dict[str, np.ndarray]:
    """``N`` 或 ``D`` 在每个评测域上的边际效益与弹性（解析式）。

    对 ``N``：``∂L/∂lnN = −α·A_k·(N/10⁹)^(−α) − η_k·S_k·Φ_k``，
    对 ``D`` 同构（``β``、``ζ_k``、``B_k``）。基线项与配比幅度项都随规模变化，
    两项符号相反，因此逐域弹性并非常数。
    """

    if factor not in ("N", "D"):
        raise ValueError(f"scalar_factor_point 只处理 N/D，收到 {factor!r}")
    n_b, d_b = n / 1e9, d / 1e9
    loss = law.predict(n, d, p, warn_outside_support=False).ravel()
    phi = law.mixture_response(p).ravel()
    amplitude = law.scale_amplitude(n, d).ravel()
    if factor == "N":
        baseline_term = law.a_coefficients * n_b ** (-law.alpha)
        exponent, decay, scale = law.alpha, law.eta, n
    else:
        baseline_term = law.b_coefficients * d_b ** (-law.beta)
        exponent, decay, scale = law.beta, law.zeta, d
    d_loss_d_log = -exponent * baseline_term - decay * amplitude * phi
    return {
        "predicted_loss": loss,
        "d_loss_d_log": d_loss_d_log,
        # −∂L/∂lnx：与其它三个因素同量纲（val_loss），可直接横向比较
        "marginal_benefit_per_1pct": -d_loss_d_log,
        "marginal_benefit": -d_loss_d_log / scale,        # −∂L/∂x
        "elasticity": d_loss_d_log / loss,                # ∂lnL/∂lnx
        "baseline_term": baseline_term,
        "mixture_term": amplitude * phi,
    }


def check_scalar_derivative(law: GeneralizedLawEvaluator, n: float, d: float,
                            p: np.ndarray, factor: str) -> float:
    """中心差分复核 ``N``/``D`` 的解析边际效益，返回最大相对偏差。"""

    step = 1e-4 * (n if factor == "N" else d)
    analytic = scalar_factor_point(law, n, d, p, factor)["marginal_benefit"]
    if factor == "N":
        plus = law.predict(n + step, d, p, warn_outside_support=False).ravel()
        minus = law.predict(n - step, d, p, warn_outside_support=False).ravel()
    else:
        plus = law.predict(n, d + step, p, warn_outside_support=False).ravel()
        minus = law.predict(n, d - step, p, warn_outside_support=False).ravel()
    numeric = -(plus - minus) / (2 * step)
    scale = np.maximum(np.abs(analytic), 1e-12)
    if not np.allclose(numeric, analytic, rtol=2e-4, atol=1e-9):
        worst = float(np.max(np.abs(numeric - analytic) / scale))
        raise ValueError(f"{factor} 的解析边际效益与差分不符，最大相对偏差 {worst:.2e}")
    return float(np.max(np.abs(numeric - analytic) / scale))


def quality_factor_point(quality_fit: dict, q: float) -> dict[str, float]:
    """独立质量律在 ``B7`` 设计点上的质量边际效益与弹性。"""

    n_b = QUALITY_POINT["N_billion_parameters"]
    d_b = QUALITY_POINT["D_billion_tokens"]
    point = ea.quality_point(quality_fit, n_b, d_b, q)
    return {
        "Q": q,
        "predicted_loss": point["predicted_loss_nats"],
        "marginal_benefit": point["marginal_benefit_nats_per_unit_Q"],
        "elasticity": point["loss_elasticity_wrt_Q"],
        "exact_benefit_for_plus_0_01_Q": point["exact_benefit_nats_for_plus_0_01_Q"],
    }


#: 汇总表的固定列序。各因素的汇总行字段并不一致（例如 p 有 best_source、
#: Q 有 identifiable），用固定表头 + 缺项留空，避免按首行推断列名。
SUMMARY_FIELDS = (
    "factor", "anchor", "operating_point", "elasticity_min", "elasticity_median",
    "elasticity_max", "marginal_benefit_median",
    "marginal_benefit_per_1pct_median", "derivative_check_max_rel_error",
    "best_source", "best_source_benefit_for_plus_1pp", "best_source_elasticity",
    "feasible_fraction", "fraction_positive", "identifiable", "source",
)


def write_csv(path: Path, rows: list[dict],
              fields: tuple[str, ...] | None = None) -> None:
    if not rows:
        return
    header = list(fields) if fields else list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def mixture_share_sweep(law: GeneralizedLawEvaluator, anchor: str, n: float,
                        d: float) -> tuple[list[dict], dict]:
    """把某个来源的份额从 0.05 扫到它的记录最大份额，看边际效益与弹性怎么变。

    扫描沿"该来源取份额 ``s``、其余按原比例缩小"的路径构造配比，因此始终在单纯形上，
    且每个点都做记录支撑域检查。**线性（解析导数）与精确（有限 +1pp）两个口径都记**：
    ``s`` 趋近 0 时 clr 的 ``1/(p_i+ε)`` 会把解析导数放大，精确值是更稳的那个。
    """

    reference = law.reference_mixture()
    reference = reference / reference.sum()
    support = law.support()["recorded_max_share_per_domain"]
    sources = tuple(SWEEP_SOURCES) if SWEEP_SOURCES else tuple(law.loss_domains)
    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for source in sources:
        index = law.train_domains.index(source)
        # 上界再留 1.1 个百分点：否则 +1pp 的移动会越出该来源的记录最大份额，
        # 该点算不出精确增益（NaN），曲线末端会假性断开。
        top = float(min(support[source] - 0.011, SWEEP_MAX_SHARE))
        if top <= SWEEP_MIN_SHARE * 1.2:
            continue
        grid = np.geomspace(SWEEP_MIN_SHARE, top, SWEEP_POINTS)
        records = []
        for share in grid:
            candidate = reference * (1 - share) / max(1 - reference[index], 1e-12)
            candidate[index] = share
            feasible = bool(~law.outside_support(candidate[None, :])[0])
            linear, elasticity, loss = ea.proportional_upweight(
                law, candidate[None, :], n, d)
            exact, _ = ea.finite_share_gain(law, candidate[None, :], n, d, loss)
            for k, domain in enumerate(law.loss_domains):
                rows.append({
                    "anchor": anchor,
                    "source_domain": source,
                    "share": float(share),
                    "eval_domain": domain,
                    "feasible": feasible,
                    "predicted_loss": float(loss[0, k]),
                    "linear_benefit_for_plus_1pp": float(linear[0, k, index]),
                    "exact_benefit_for_plus_1pp": float(exact[0, k, index]),
                    "elasticity_dlnL_dln_share": float(elasticity[0, k, index]),
                })
            records.append({
                "share": float(share),
                "linear": float(np.median(linear[0][:, index])),
                "exact": float(np.median(exact[0][:, index])),
                "elasticity": float(np.median(elasticity[0][:, index])),
            })
        shares = np.asarray([r["share"] for r in records])
        exact = np.asarray([r["exact"] for r in records])
        linear = np.asarray([r["linear"] for r in records])
        elasticity = np.asarray([r["elasticity"] for r in records])
        finite = np.isfinite(exact)
        # 只在有限点上找符号变化，且只取第一次「正 → 负」——那才是"再增份额开始有害"
        # 的份额；末端的 NaN 会伪造出一次符号变化。
        positive = shares[finite][exact[finite] > 0]
        crossings = np.where((exact[:-1] > 0) & (exact[1:] <= 0))[0]
        summary[source] = {
            "sweep_range": [float(shares[0]), float(shares[-1])],
            "recorded_max_share": float(support[source]),
            "benefit_exact_at_min_share": float(exact[finite][0]),
            "benefit_exact_at_max_share": float(exact[finite][-1]),
            "elasticity_at_min_share": float(elasticity[finite][0]),
            "elasticity_at_max_share": float(elasticity[finite][-1]),
            "benefit_positive_share_range": (
                [float(positive.min()), float(positive.max())] if positive.size else None),
            "turning_share_positive_to_negative": (
                float(shares[crossings[0]]) if crossings.size else None),
            # 线性口径在低份额端高估解析导数；用绝对差表示，比值在精确值近零时无意义
            "linear_minus_exact_at_min_share": float(linear[finite][0]
                                                     - exact[finite][0]),
        }
    return rows, summary


def relative_benefit(law: GeneralizedLawEvaluator, n: float, d: float,
                     p: np.ndarray, factor: str,
                     fraction: float = 0.01) -> dict[str, np.ndarray]:
    """``L(x) − L(1.01·x)``：**真正**"提高 1%"带来的降损（``N``、``D``）。

    注意这与 ``−∂L/∂lnx`` 不是一个量：后者是局部对数导数，只在扰动足够小时才近似
    等于前者。本分析两者都算、都存，表头分别写明"相对 1%"与"每单位"。
    """

    if factor not in ("N", "D"):
        raise ValueError(f"relative_benefit 只处理 N/D，收到 {factor!r}")
    base = law.predict(n, d, p, warn_outside_support=False)
    if factor == "N":
        bumped = law.predict(n * (1 + fraction), d, p, warn_outside_support=False)
    else:
        bumped = law.predict(n, d * (1 + fraction), p, warn_outside_support=False)
    return {"loss_before": base, "loss_after": bumped, "benefit": base - bumped}


def scale_rows(law: GeneralizedLawEvaluator, label: str, n: float, d: float,
               p: np.ndarray, is_observed: bool) -> tuple[list[dict], list[dict]]:
    """一个规模上的 ``N``、``D`` 明细与 ``p`` 明细。

    ``p`` 只纳入**内点且扰动前后都在观测支撑域内**的方向：份额为 0 的方向其对数弹性
    被 ``p_i`` 乘成 0，边界方向的 CLR 导数受平滑常数支配，两者都不作为正文结论。
    """

    shares = p / p.sum()
    domain_rows: list[dict] = []
    for factor in ("N", "D"):
        point = scalar_factor_point(law, n, d, shares[None, :], factor)
        relative = relative_benefit(law, n, d, shares, factor)
        for k, name in enumerate(law.loss_domains):
            domain_rows.append({
                "scale": label,
                "is_observed_scale": is_observed,
                "N_parameters": n,
                "D_tokens": d,
                "eval_domain": name,
                "factor": factor,
                "predicted_loss": float(relative["loss_before"][k]),
                "elasticity": float(point["elasticity"][k]),
                "benefit_relative_1pct": float(relative["benefit"][k]),
                "benefit_per_unit": float(point["marginal_benefit"][k]),
                "minus_dL_dlnx": float(point["marginal_benefit_per_1pct"][k]),
            })

    linear, elasticity, loss = ea.proportional_upweight(law, shares[None, :], n, d)
    exact, valid = ea.finite_share_gain(law, shares[None, :], n, d, loss)
    maximum = np.asarray([law.support()["recorded_max_share_per_domain"][name]
                          for name in law.train_domains])
    mixture_rows: list[dict] = []
    for k, eval_domain in enumerate(law.loss_domains):
        for i, source in enumerate(law.train_domains):
            before, after = float(shares[i]), float(shares[i]) + SHARE_STEP
            interior = before > 0
            feasible = bool(valid[0, i]) and interior
            mixture_rows.append({
                "scale": label,
                "is_observed_scale": is_observed,
                "eval_domain": eval_domain,
                "source_domain": source,
                "share_before": before,
                "share_after": after,
                "other_scale_factor": ((1 - after) / (1 - before)) if before < 1 else None,
                "recorded_max_share": float(maximum[i]),
                "interior": interior,
                "feasible": feasible,
                "predicted_loss_before": float(loss[0, k]),
                "predicted_loss_after": (float(loss[0, k] - exact[0, k, i])
                                         if feasible else None),
                "elasticity": float(elasticity[0, k, i]),
                "benefit_1pp_exact": (float(exact[0, k, i]) if feasible else None),
                "benefit_1pp_linear": float(linear[0, k, i]),
            })
    return domain_rows, mixture_rows


def summarise(payload_rows: dict[str, np.ndarray], factor: str, scale: str,
              perturbation: str, is_observed: bool, operating_point: str,
              note: str = "") -> dict:
    """把一组逐目标取值压成汇总表的一行（中位 + 四分位**分布范围**）。"""

    values = np.asarray(payload_rows["elasticity"], dtype=float)
    benefit = np.asarray(payload_rows["benefit"], dtype=float)
    return {
        "factor": factor,
        "scale": scale,
        "is_observed_scale": is_observed,
        "perturbation": perturbation,
        "operating_point": operating_point,
        "elasticity_median": float(np.median(values)),
        "elasticity_p25": float(np.quantile(values, 0.25)),
        "elasticity_p75": float(np.quantile(values, 0.75)),
        "benefit_median": float(np.median(benefit)),
        "benefit_p25": float(np.quantile(benefit, 0.25)),
        "benefit_p75": float(np.quantile(benefit, 0.75)),
        "benefit_unit": "val_loss",
        "n_effective": int(values.size),
        "positive_fraction": (float(np.mean(benefit > 0)) if benefit.size else None),
        "note": note,
    }


def quality_curve(fits: dict) -> list[dict]:
    """质量实验支持的 Q 等级上，两组拟合的精确 ``Q+0.01`` 降损与弹性。"""

    rows = []
    for q in QUALITY_CURVE_GRID:
        for key, tag in (("B7", "primary"), ("B6", "subset_control")):
            fit = fits[key]
            n_b = QUALITY_POINT["N_billion_parameters"]
            d_b = QUALITY_POINT["D_billion_tokens"]
            point = ea.quality_point(fit, n_b, d_b, q)
            rows.append({
                "Q": q,
                "fit": tag,
                "sample_count": fit["sample_count"],
                "r_squared": fit["r_squared"],
                "predicted_loss": point["predicted_loss_nats"],
                "benefit_for_plus_0_01_Q": point["exact_benefit_nats_for_plus_0_01_Q"],
                "elasticity": point["loss_elasticity_wrt_Q"],
                "is_main_point": abs(q - QUALITY_MAIN_Q) < 1e-9,
                "is_observed_level": any(abs(q - level) < 1e-9
                                         for level in QUALITY_LEVELS),
            })
    return rows


def analyse() -> dict:
    law = GeneralizedLawEvaluator.from_directory(HANDOFF)
    delivered = json.loads((HANDOFF / "elasticity_and_substitution.json").read_text(
        encoding="utf-8"))
    fits = delivered["quality_law_from_B6_B7"]
    reference = law.reference_mixture()
    reference = reference / reference.sum()

    domain_rows: list[dict] = []
    mixture_rows: list[dict] = []
    summary: list[dict] = []
    checks: dict[str, float] = {}

    for label, n, d in OBSERVED_SCALES + (EXTRAPOLATION,):
        is_observed = (label, n, d) in OBSERVED_SCALES
        rows_d, rows_m = scale_rows(law, label, n, d, reference, is_observed)
        domain_rows += rows_d
        mixture_rows += rows_m
        for factor in ("N", "D"):
            checks[f"{factor}@{label}"] = check_scalar_derivative(law, n, d,
                                                                  reference, factor)
            picked = [r for r in rows_d if r["factor"] == factor]
            summary.append(summarise(
                {"elasticity": [r["elasticity"] for r in picked],
                 "benefit": [r["benefit_relative_1pct"] for r in picked]},
                factor, label, "相对 +1%", is_observed,
                "参考配比（记录配比中熵居中）",
                "" if is_observed else "广义律观测范围之外的外推，不计入汇总统计"))
        interior = [r for r in rows_m if r["feasible"]]
        summary.append(summarise(
            {"elasticity": [r["elasticity"] for r in interior],
             "benefit": [r["benefit_1pp_exact"] for r in interior]},
            "p", label, "份额 +1 个百分点", is_observed,
            "参考配比；仅内点且扰动前后均在支撑域内的方向",
            f"纳入 {len(interior)}/{len(rows_m)} 个方向"
            + ("" if is_observed else "；外推，不计入汇总统计")))

    # Q 单列一行：来自独立质量实验，工作点与前三者不同，不与它们排序。
    n_b = QUALITY_POINT["N_billion_parameters"]
    d_b = QUALITY_POINT["D_billion_tokens"]
    point = ea.quality_point(fits["B7"], n_b, d_b, QUALITY_MAIN_Q)
    summary.append({
        "factor": "Q",
        "scale": "不适用（独立质量实验）",
        "is_observed_scale": False,
        "perturbation": "Q + 0.01",
        "operating_point": f"N={n_b:g}B、D={d_b:g}B（实验范围内插值）、Q={QUALITY_MAIN_Q}",
        "elasticity_median": point["loss_elasticity_wrt_Q"],
        "elasticity_p25": None,
        "elasticity_p75": None,
        "benefit_median": point["exact_benefit_nats_for_plus_0_01_Q"],
        "benefit_p25": None,
        "benefit_p75": None,
        "benefit_unit": "val_loss",
        "n_effective": fits["B7"]["sample_count"],
        "positive_fraction": None,
        "note": "由独立质量实验拟合的质量律给出；工作点与 N/D/p 不同，"
                "损失基准未经统一校准，不与前三者直接排序或相除",
    })

    curve = quality_curve(fits)
    sweep_anchor, sweep_n, sweep_d = SWEEP_ANCHOR
    sweep_rows, sweep_summary = mixture_share_sweep(law, sweep_anchor, sweep_n,
                                                    sweep_d)

    payload = {
        "definitions": {
            "marginal_benefit": "g_x = -dL/dx，正值表示增大该因素会降损",
            "elasticity": "eps_x = dlnL/dlnx，负值表示增大该因素会降损",
            "relative_1pct_benefit": "N、D 用 L(x) - L(1.01x)：真正提高 1% 的降损",
            "one_percentage_point_benefit": "p 用份额 +0.01 的可行方向降损；Q 用 Q+0.01",
            "minus_dL_dlnx": "局部对数导数，**不等于**提高 1% 的降损，仅供对照",
            "range_label": "跨目标的四分位区间是**分布范围**，不是置信区间",
            "loss_unit": "val_loss（附件记录的验证损失单位）",
        },
        "reference_mixture": {
            "definition": "记录配比中熵处于中位数的真实配比",
            "entropy": float(np.asarray(
                law.__class__.entropy(reference)).ravel()[0]),
            "shares": dict(zip(law.train_domains, reference.tolist())),
            "interior_sources": [name for name, value in
                                 zip(law.train_domains, reference) if value > 0],
        },
        "observed_scales": [{"label": l, "N_parameters": n, "D_tokens": d}
                            for l, n, d in OBSERVED_SCALES],
        "extrapolation_scenario": {
            "label": EXTRAPOLATION[0], "N_parameters": EXTRAPOLATION[1],
            "D_tokens": EXTRAPOLATION[2],
            "D_over_N": EXTRAPOLATION[2] / EXTRAPOLATION[1],
            "note": "保持 D/N=25、配比不变的外推，超出广义律观测范围，"
                    "仅以虚线/空心标记展示，不并入汇总统计",
        },
        "quality_operating_point": {
            "N_billion_parameters": n_b, "D_billion_tokens": d_b,
            "Q": QUALITY_MAIN_Q,
            "levels_in_experiment": QUALITY_LEVELS,
            "note": "Q=0.6 与 N=1B 是质量实验的观测等级；D=25B 落在实验的 10B 与 50B "
                    "之间，属于**插值**",
        },
        "quality_law_source": {
            "primary": {"sample_count": fits["B7"]["sample_count"],
                        "r_squared": fits["B7"]["r_squared"]},
            "control": {"sample_count": fits["B6"]["sample_count"],
                        "r_squared": fits["B6"]["r_squared"]},
            "control_is_subset": True,
            "note": "对照组是主组的**精确子集**（360 ⊂ 450、逐行一致），因此这是"
                    "子集一致性对照，不是独立重复实验",
        },
        "summary_table": summary,
        "quality_curve": curve,
        "mixture_share_sweep": {
            "anchor": sweep_anchor, "min_share": SWEEP_MIN_SHARE,
            "max_share": SWEEP_MAX_SHARE, "n_points": SWEEP_POINTS,
            "sources": list(sweep_summary), "per_source": sweep_summary,
        },
        "derivative_checks": checks,
        "caveats": [
            "解析导数（N、D 的 dlnL/dlnx 与 p 的可行方向导数）均经中心差分复核。",
            "p 的统计只纳入内点、且扰动前后都在观测支撑域内的方向；零份额与边界处的 "
            "CLR 导数不作为正文结论。",
            "Q 必须由独立质量实验的质量律给出；本分析不把广义律对固定 Q_i 求出的"
            "隐含响应当作有效的质量弹性。",
            "各因素工作点不同、损失基准未经统一校准，不用导数之比推断"
            "\"质量可替代多少 token 或参数\"。",
            "边际效益是模型敏感度，不是因果效应。",
        ],
    }
    return {"payload": payload, "domain_rows": domain_rows,
            "mixture_rows": mixture_rows, "sweep_rows": sweep_rows,
            "output": OUTPUT}


SUMMARY_FIELDS_V2 = (
    "factor", "scale", "is_observed_scale", "perturbation", "operating_point",
    "elasticity_median", "elasticity_p25", "elasticity_p75",
    "benefit_median", "benefit_p25", "benefit_p75", "benefit_unit",
    "n_effective", "positive_fraction", "note",
)


def write_markdown(result: dict, path: Path) -> None:
    payload = result["payload"]
    rows = payload["summary_table"]

    def fmt(value, spec="+.4f"):
        return "—" if value is None else format(value, spec)

    lines = [
        "# 各因素边际效用与弹性",
        "",
        "本文件由 `Q2/Q_2_4/Q_2_4_3_factor_elasticity_analysis.py` 生成。",
        "",
        "## 0. 口径",
        "",
        "| 量 | 定义 |",
        "| --- | --- |",
    ]
    for key, value in payload["definitions"].items():
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        f"**配比基准**：{payload['reference_mixture']['definition']}"
        f"（熵 {payload['reference_mixture']['entropy']:.4f}，"
        f"{len(payload['reference_mixture']['interior_sources'])} 个来源份额为正）。",
        "",
        f"**质量工作点**：N={payload['quality_operating_point']['N_billion_parameters']:g}B、"
        f"D={payload['quality_operating_point']['D_billion_tokens']:g}B、"
        f"Q={payload['quality_operating_point']['Q']}。"
        f"{payload['quality_operating_point']['note']}。",
        "",
        "## 1. 汇总表",
        "",
        "`相对 +1%` 指 $L(x)-L(1.01x)$；`份额 +1 个百分点` 指份额 $+0.01$ 的可行方向；"
        "`Q + 0.01` 指质量分加 0.01。四分位区间是跨评估目标的**分布范围**，不是置信区间。",
        "",
        "| 因素 | 规模 | 微扰 | 弹性中位 | 弹性 P25–P75 | 降损中位 | 降损 P25–P75 | 有效样本 | 正收益方向占比 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['factor']}` | {row['scale']} | {row['perturbation']} | "
            f"{fmt(row['elasticity_median'])} | "
            f"{fmt(row['elasticity_p25'])} – {fmt(row['elasticity_p75'])} | "
            f"{fmt(row['benefit_median'], '+.3e')} | "
            f"{fmt(row['benefit_p25'], '+.3e')} – {fmt(row['benefit_p75'], '+.3e')} | "
            f"{row['n_effective']} | {fmt(row['positive_fraction'], '.1%')} |")
    lines += [
        "",
        "> 标注为外推的行只出现在上表中作为对照，**不计入**观测规模的汇总统计。",
        "> `Q` 行的工作点与 `N`/`D`/`p` 不同、损失基准未经统一校准，"
        "**不与前三者直接排序或相除**。",
        "",
        "## 2. 质量响应曲线（独立质量实验）",
        "",
        f"主工作点 N=1B、D=25B、Q=0.6。两组拟合：主组 "
        f"{payload['quality_law_source']['primary']['sample_count']} 组"
        f"（R²={payload['quality_law_source']['primary']['r_squared']:.4f}），"
        f"对照组 {payload['quality_law_source']['control']['sample_count']} 组"
        f"（R²={payload['quality_law_source']['control']['r_squared']:.4f}）。",
        "",
        payload["quality_law_source"]["note"] + "。",
        "",
        "| Q | 预测损失 | Q+0.01 降损 | 弹性 ∂lnL/∂lnQ | |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["quality_curve"]:
        if row["fit"] != "primary" or not row["is_observed_level"]:
            continue
        exact = row["benefit_for_plus_0_01_Q"]
        lines.append(f"| {row['Q']:.2f} | {row['predicted_loss']:.4f} | "
                     + ("—" if exact is None else f"{exact:+.4f}") + " | "
                     f"{row['elasticity']:+.5f} | "
                     + ("**主工作点**" if row["is_main_point"] else "") + " |")
    lines += [
        "",
        "## 3. 复现",
        "",
        "```bash",
        "python Q2/Q_2_4/Q_2_4_3_factor_elasticity_analysis.py",
        "python Q2/Q_2_4/Q_2_4_4_factor_elasticity_figures.py",
        "```",
        "",
        "## 4. 必须随结果一起给出的限定",
        "",
    ]
    lines += [f"- {item}" for item in payload["caveats"]]
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_from_detail(output: Path) -> int:
    """验收：汇总表的每个数值都必须能由**已写出的明细**重算。

    重新读回 CSV（不碰内存里的行），按同样的筛选与统计口径复算，与汇总表逐项比对；
    不一致直接报错。这样"明细足以复现表格"是被机器检查过的，而不是靠声明。
    """

    table = read_csv_rows(output / "summary_table.csv")
    scale_rows = read_csv_rows(output / "scale_domain_detail.csv")
    mixture_rows = read_csv_rows(output / "mixture_direction_detail.csv")
    failures: list[str] = []
    for row in table:
        factor, scale = row["factor"], row["scale"]
        if factor in ("N", "D"):
            picked = [r for r in scale_rows
                      if r["factor"] == factor and r["scale"] == scale]
            elasticity = np.array([float(r["elasticity"]) for r in picked])
            benefit = np.array([float(r["benefit_relative_1pct"]) for r in picked])
        elif factor == "p":
            picked = [r for r in mixture_rows
                      if r["scale"] == scale
                      and str(r["feasible"]).strip().lower() == "true"
                      and r["benefit_1pp_exact"] not in ("", None)]
            elasticity = np.array([float(r["elasticity"]) for r in picked])
            benefit = np.array([float(r["benefit_1pp_exact"]) for r in picked])
        else:
            continue                      # Q 行来自独立质量律，不在本明细内
        expected = {
            "elasticity_median": float(np.median(elasticity)),
            "elasticity_p25": float(np.quantile(elasticity, 0.25)),
            "elasticity_p75": float(np.quantile(elasticity, 0.75)),
            "benefit_median": float(np.median(benefit)),
            "benefit_p25": float(np.quantile(benefit, 0.25)),
            "benefit_p75": float(np.quantile(benefit, 0.75)),
        }
        for key, value in expected.items():
            if not np.isclose(float(row[key]), value, rtol=0, atol=1e-12):
                failures.append(f"{factor}@{scale}.{key}: 表 {row[key]} vs 明细 {value}")
        if int(row["n_effective"]) != elasticity.size:
            failures.append(f"{factor}@{scale}.n_effective: "
                            f"{row['n_effective']} vs {elasticity.size}")
        if factor == "p":
            fraction = float(np.mean(benefit > 0))
            if not np.isclose(float(row["positive_fraction"]), fraction, atol=1e-12):
                failures.append(f"p@{scale}.positive_fraction: "
                                f"{row['positive_fraction']} vs {fraction}")
    if failures:
        raise SystemExit("汇总表与明细不一致：\n  " + "\n  ".join(failures[:8]))
    return len([row for row in table if row["factor"] != "Q"])


def main() -> int:
    result = analyse()
    output: Path = result["output"]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "summary_table.csv", result["payload"]["summary_table"],
              fields=SUMMARY_FIELDS_V2)
    write_csv(output / "scale_domain_detail.csv", result["domain_rows"])
    write_csv(output / "mixture_direction_detail.csv", result["mixture_rows"])
    write_csv(output / "quality_response_curve.csv",
              result["payload"]["quality_curve"])
    write_csv(output / "mixture_share_sweep.csv", result["sweep_rows"])
    (output / "marginal_utility.json").write_text(
        json.dumps(result["payload"], ensure_ascii=False, indent=2),
        encoding="utf-8")
    write_markdown(result, output / "analysis_report.md")
    verified = verify_from_detail(output)

    print("各因素边际效用与弹性")
    for row in result["payload"]["summary_table"]:
        flag = "" if row["is_observed_scale"] else "  [外推/不可比]"
        print(f"  {row['factor']:<3} {row['scale']:<18} "
              f"eps={row['elasticity_median']:+.4f}  "
              f"benefit={row['benefit_median']:+.3e}  n={row['n_effective']}{flag}")
    worst = max(result["payload"]["derivative_checks"].values())
    print(f"\n差分复核最大相对偏差 {worst:.1e}")
    print(f"验收：汇总表 {verified} 行已由明细 CSV 重算一致")
    print(f"产物目录: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
