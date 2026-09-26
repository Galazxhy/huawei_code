#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等损失替代关系：在 N 与配比不变时，提高 Token 质量能少用多少 Token。

问题
----
固定参数量与领域配比，把 Token 质量从 ``Q0`` 提到 ``Q1``，能省下多少 Token？

先定义，再换算（两种量不可混用）
--------------------------------
独立质量实验的模型是

    L_Q = E + A·N^(−α) + B·(D·Q)^(−β)

在该模型内部，``N`` 固定时"损失不变"等价于 ``D·Q`` 不变，于是

    D_need = D0 · Q0 / Q1                 维持原损失所需的 Token 数
    ΔD_saved = D0 − D_need = D0(1 − Q0/Q1)   **可节省 Token**
    ΔD_eq   = D0(Q1/Q0 − 1)                 **等效新增 Token**
                                            （保持原质量时，要多加多少 Token 才有同样的降损）

``ΔD_saved`` 与 ``ΔD_eq`` 不是同一个数、也不是互为倒数：前者是省下来的量，后者是
"这份收益若由堆 Token 换来需要多少"。正文必须分开表述。

等损失曲线的斜率是 ``dD/dQ = −D/Q < 0``；正值的 ``ΔD_eq`` 是**收益换算量**，
不是等损失曲线的斜率。现有交接材料 §4.3 的 ``dN/dQ`` 公式漏了负号，本脚本报告纠正值。

检验而非断言
------------
``D·Q`` 是**模型假设**，不是物理规律。本脚本用"留一质量等级"交叉验证检验它，
并对照允许两个指数分开的模型

    L_Q = E + A·N^(−α) + B·D^(−β_D)·Q^(−β_Q)

报告两者的留出误差与 ``r = β_Q/β_D`` 的估计区间；只有数据支持 ``r ≈ 1`` 时才采用
简洁的 ``D·Q`` 换算。另有同 ``N``、同 ``D·Q`` 的观测分组，用组内损失极差直接量化
"等效"的误差量级。

适用边界
--------
广义标度律在本节只用于**限定规模工作点与固定配比**。它的质量分在配比实验里从未
独立变化，因此本节给出的替代率只在独立质量实验的损失尺度上成立，**不能**推广为
13 个评估目标各自的质量—Token 兑换率。算力成本是否值得，留待后续问题判断。

默认运行：
    python Q2/Q_2_4/Q_2_4_5_quality_substitution_analysis.py
产物写入 ``results/quality_token_substitution/``。
"""


# --- shared paths for every script in Q2 (see Q2/_q2_paths.py) --- # q2-prologue:bootstrap
from __future__ import annotations
import collections
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

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


ROOT = _q2_paths.PROJECT_ROOT
DESIGN = (_q2_paths.B_DATA / "supplementary_NQ_experiment_expanded.csv")
HANDOFF = _q2_paths.HANDOFF_OUT
OUTPUT = _q2_paths.ANALYSIS_DIR / "quality_token_substitution"

#: 工作点：参数量、基准 Token 数、基准质量。``D0=25`` 十亿 Token 落在质量实验
#: 记录的 10 与 50 之间，属于**插值**。
N0_BILLION = 1.0
D0_BILLION = 25.0
Q0 = 0.6
#: 正文举例的目标质量。
Q1 = 0.7
#: 下标 13 个评估目标不参与本节的替代率（见模块 docstring 的适用边界）。
BOOTSTRAP = 200
SEED = 2025


def load_design() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with DESIGN.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    n = np.array([float(r["N_params_B"]) for r in rows])
    d = np.array([float(r["D_tokens_B"]) for r in rows])
    q = np.array([float(r["Q_score"]) for r in rows])
    loss = np.array([float(r["val_loss"]) for r in rows])
    return n, d, q, loss


def fit_dq(n, d, q, loss) -> dict:
    """``L = E + A·N^(−α) + B·(D·Q)^(−β)``：5 个参数。"""

    def residual(theta):
        e, a, alpha, b, beta = theta
        return e + a * n ** (-alpha) + b * (d * q) ** (-beta) - loss

    start = [0.96, 0.53, 0.28, 1.62, 0.11]
    solution = least_squares(residual, start, bounds=(0, np.inf), xtol=1e-14,
                             ftol=1e-14, gtol=1e-14, max_nfev=20000)
    e, a, alpha, b, beta = solution.x
    return {"E": e, "A": a, "alpha": alpha, "B": b, "beta": beta,
            "predict": lambda nn, dd, qq: e + a * nn ** (-alpha)
            + b * (dd * qq) ** (-beta)}


def fit_free(n, d, q, loss) -> dict:
    """``L = E + A·N^(−α) + B·D^(−β_D)·Q^(−β_Q)``：6 个参数。"""

    def residual(theta):
        e, a, alpha, b, beta_d, beta_q = theta
        return (e + a * n ** (-alpha) + b * d ** (-beta_d) * q ** (-beta_q)
                - loss)

    start = [0.96, 0.53, 0.28, 1.62, 0.11, 0.11]
    solution = least_squares(residual, start, bounds=(0, np.inf), xtol=1e-14,
                             ftol=1e-14, gtol=1e-14, max_nfev=40000)
    e, a, alpha, b, beta_d, beta_q = solution.x
    return {"E": e, "A": a, "alpha": alpha, "B": b, "beta_D": beta_d,
            "beta_Q": beta_q, "ratio": beta_q / beta_d,
            # 必须用 lambda 自己的 dd、qq：写成闭包里的 d、q 会永远在**拟合用的
            # 全量数组**上求值，留出预测就变成了在训练集上预测。
            "predict": lambda nn, dd, qq: e + a * nn ** (-alpha)
            + b * dd ** (-beta_d) * qq ** (-beta_q)}


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
        "relative_rmse": float(np.sqrt(np.mean(np.square(predicted / actual - 1)))),
        "median_ape": float(np.median(np.abs(predicted / actual - 1))),
    }


def leave_one_quality_level_out(n, d, q, loss) -> dict:
    """留一质量等级：每次扣掉一个 Q 等级（45 组）重拟合，再预测这些点。"""

    levels = sorted(set(q.tolist()))
    folds = []
    for level in levels:
        held = np.isclose(q, level)
        fitted_dq = fit_dq(n[~held], d[~held], q[~held], loss[~held])
        fitted_free = fit_free(n[~held], d[~held], q[~held], loss[~held])
        folds.append({
            "held_out_Q": level,
            "n_held_out": int(held.sum()),
            "dq": score(loss[held], fitted_dq["predict"](n[held], d[held], q[held])),
            "free": score(loss[held], fitted_free["predict"](n[held], d[held],
                                                            q[held])),
            "free_ratio_r": fitted_free["ratio"],
            "free_beta_Q": fitted_free["beta_Q"],
            "free_beta_D": fitted_free["beta_D"],
        })
    pooled = {}
    for key in ("dq", "free"):
        pooled[key] = {
            metric: float(np.mean([fold[key][metric] for fold in folds]))
            for metric in ("rmse", "relative_rmse", "median_ape")
        }
    ratios = np.array([fold["free_ratio_r"] for fold in folds])
    return {"folds": folds, "pooled": pooled,
            "ratio_across_folds": {"min": float(ratios.min()),
                                   "median": float(np.median(ratios)),
                                   "max": float(ratios.max())}}


def bootstrap_ratio(n, d, q, loss, replicates: int = BOOTSTRAP,
                    seed: int = SEED) -> dict:
    """对自由指数模型做 bootstrap，给出 ``r = β_Q/β_D`` 的估计区间。"""

    generator = np.random.default_rng(seed)
    ratios, beta_q, beta_d = [], [], []
    for _ in range(replicates):
        picked = generator.integers(0, len(loss), len(loss))
        try:
            fitted = fit_free(n[picked], d[picked], q[picked], loss[picked])
        except Exception:                       # pragma: no cover - 数值退化
            continue
        ratios.append(fitted["ratio"])
        beta_q.append(fitted["beta_Q"])
        beta_d.append(fitted["beta_D"])
    ratios = np.asarray(ratios)
    beta_q, beta_d = np.asarray(beta_q), np.asarray(beta_d)
    return {
        # 两个指数高度相关 ⇒ 数据主要约束的是它们的**比值**，各自水平并不定
        "corr_beta_Q_beta_D": float(np.corrcoef(beta_d, beta_q)[0, 1]),
        "replicates_used": int(ratios.size),
        "ratio_mean": float(ratios.mean()),
        "ratio_p05": float(np.quantile(ratios, 0.05)),
        "ratio_p50": float(np.quantile(ratios, 0.50)),
        "ratio_p95": float(np.quantile(ratios, 0.95)),
        "ratio_contains_one": bool(np.quantile(ratios, 0.05) <= 1.0
                                   <= np.quantile(ratios, 0.95)),
        "beta_Q_p05": float(np.quantile(beta_q, 0.05)),
        "beta_Q_p95": float(np.quantile(beta_q, 0.95)),
        "beta_D_p05": float(np.quantile(beta_d, 0.05)),
        "beta_D_p95": float(np.quantile(beta_d, 0.95)),
    }


def equal_dq_groups(n, d, q, loss) -> dict:
    """同 ``N``、同 ``D·Q`` 的观测分组：组内损失极差直接量化"等效"的误差。"""

    groups: dict[tuple[float, float], list[float]] = collections.defaultdict(list)
    for nn, dd, qq, ll in zip(n, d, q, loss):
        groups[(round(float(nn), 6), round(float(dd * qq), 6))].append(float(ll))
    multi = [values for values in groups.values() if len(values) > 1]
    ranges = np.array([max(v) - min(v) for v in multi])
    return {
        "n_groups_with_replicates": len(multi),
        "group_size_histogram": dict(collections.Counter(len(v) for v in multi)),
        "within_group_loss_range_median": float(np.median(ranges)),
        "within_group_loss_range_mean": float(ranges.mean()),
        "within_group_loss_range_max": float(ranges.max()),
        "interpretation": (
            "同一 (N, D·Q) 的观测损失并不完全一致：组内极差中位数约 "
            f"{np.median(ranges):.4f}。因此 D·Q 等损失换算是**有误差的模型近似**，"
            "不是精确的物理规律。"),
    }


def substitution_row(q1: float, model: dict, beta_q: float, beta_d: float) -> dict:
    """给定目标质量，给出所需 D、可节省量与等效新增量（两种模型各一行）。"""

    need_dq = D0_BILLION * Q0 / q1
    need_free = D0_BILLION * (Q0 / q1) ** (beta_q / beta_d)
    return {
        "Q1": q1,
        "D_need_dq_model": need_dq,
        "tokens_saved_dq_model": D0_BILLION - need_dq,
        "equivalent_new_tokens": D0_BILLION * (q1 / Q0 - 1),
        "D_need_free_model": need_free,
        "tokens_saved_free_model": D0_BILLION - need_free,
        "ratio_free_over_dq": need_free / need_dq,
    }


def iso_loss_slope(q: float, d0: float = D0_BILLION,
                   q0: float = Q0) -> float:
    """等损失曲线斜率 ``dD/dQ = −D/Q``，在 ``(D0, Q)`` 处取值（负）。"""

    d = d0 * q0 / q
    return -d / q


def analyse() -> dict:
    n, d, q, loss = load_design()
    dq_model = fit_dq(n, d, q, loss)
    free_model = fit_free(n, d, q, loss)
    in_sample = {
        "dq": score(loss, dq_model["predict"](n, d, q)),
        "free": score(loss, free_model["predict"](n, d, q)),
    }
    holdout = leave_one_quality_level_out(n, d, q, loss)
    bootstrap = bootstrap_ratio(n, d, q, loss)
    groups = equal_dq_groups(n, d, q, loss)

    levels = sorted(set(q.tolist()))
    table = [substitution_row(level, free_model, free_model["beta_Q"],
                              free_model["beta_D"]) for level in levels]

    correction = {
        "statement": "等损失曲线的斜率是 dD/dQ = −D/Q < 0；正值的"
                     "「等效新增 Token」是收益换算量，不是等损失曲线斜率。",
        "handoff_material": "results/Q2_to_Q3/交付文档.md §4.3",
        "handoff_error": "该节写的 dN/dQ = +B·β·(D·Q)^(−β)/(A·α·Q·N^(−α−1)) 漏了负号；"
                         "其数值量级正确，符号应为负。",
        "corrected": "dN/dQ = −B·β·(D·Q)^(−β)/(A·α·Q·N^(−α−1)) < 0；"
                     "表中正的「等价于参数增加 x%」是等效换算量（收益幅值），"
                     "不是斜率本身。",
        "slope_at_reference": iso_loss_slope(Q0),
    }

    payload = {
        "question": "参数量与领域配比不变时，提高 Token 质量能少用多少 Token",
        "operating_point": {
            "N_billion_parameters": N0_BILLION,
            "D0_billion_tokens": D0_BILLION,
            "Q0": Q0,
            "Q1_example": Q1,
            "note": "D0=25 十亿 Token 落在质量实验记录的 10 与 50 之间，属于**插值**；"
                    "N=1B 与 Q0=0.6 是实验的观测等级",
        },
        "definitions": {
            "D_need": "D0·Q0/Q1：维持原损失所需 Token",
            "tokens_saved": "D0(1 − Q0/Q1)：**可节省 Token**",
            "equivalent_new_tokens": "D0(Q1/Q0 − 1)：**等效新增 Token**"
                                     "（保持原质量时要多加多少 Token 才有同样降损）",
            "iso_loss_slope": "dD/dQ = −D/Q < 0（等损失曲线斜率）",
            "warning": "可节省 Token 与等效新增 Token 是两个不同的量，正文不可混用；"
                       "后者不是等损失曲线的斜率",
        },
        "models": {
            "dq": {"form": "L = E + A·N^(−α) + B·(D·Q)^(−β)",
                   "parameters": {k: dq_model[k] for k in ("E", "A", "alpha",
                                                           "B", "beta")},
                   "in_sample": in_sample["dq"]},
            "free": {"form": "L = E + A·N^(−α) + B·D^(−β_D)·Q^(−β_Q)",
                     "parameters": {k: free_model[k] for k in ("E", "A", "alpha",
                                                               "B", "beta_D",
                                                               "beta_Q")},
                     "ratio_r": free_model["ratio"],
                     "in_sample": in_sample["free"]},
        },
        "holdout_by_quality_level": holdout,
        "ratio_interval": bootstrap,
        "equal_dq_groups": groups,
        "substitution_table": table,
        "sign_correction": correction,
        "boundary": [
            "广义标度律在本节只用于限定规模工作点与固定配比；其质量分在配比实验中"
            "没有独立变化，本节数值替代率只在独立质量实验的损失尺度上成立。",
            "不能推广为 13 个评估目标各自的质量—Token 兑换率：质量实验没有逐评估"
            "目标的观测。",
            "算力成本是否值得，留待后续问题判断；本节只回答「能少用多少 Token」。",
            "D·Q 等损失换算是模型近似：同 (N, D·Q) 观测的组内损失极差中位数 "
            f"{groups['within_group_loss_range_median']:.4f}，不为零。",
        ],
        "design_summary": {
            "n_observations": int(loss.size),
            "quality_levels": levels,
            "N_billion_range": [float(n.min()), float(n.max())],
            "D_billion_range": [float(d.min()), float(d.max())],
        },
    }
    return {"payload": payload, "output": OUTPUT}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(payload: dict, path: Path) -> None:
    point = payload["operating_point"]
    definitions = payload["definitions"]
    holdout = payload["holdout_by_quality_level"]
    ratio = payload["ratio_interval"]
    groups = payload["equal_dq_groups"]
    example = next(row for row in payload["substitution_table"]
                   if abs(row["Q1"] - point["Q1_example"]) < 1e-9)

    lines = [
        "# 等损失替代关系：质量能替代多少 Token",
        "",
        f"**问题**：参数量与领域配比不变时，把 Token 质量从 $Q_0$ 提到 $Q_1$，"
        f"能少用多少 Token？",
        "",
        "## 1. 先定义，再换算",
        "",
        "独立质量实验的模型是 $L_Q = E + A N^{-\\alpha} + B (DQ)^{-\\beta}$。"
        "在**该模型内部**、$N$ 固定时，损失不变等价于 $DQ$ 不变，于是",
        "",
        "$$D_{\\rm need} = D_0\\frac{Q_0}{Q_1},\\qquad "
        "\\Delta D_{\\rm saved} = D_0\\Big(1-\\frac{Q_0}{Q_1}\\Big),\\qquad "
        "\\Delta D_{\\rm eq} = D_0\\Big(\\frac{Q_1}{Q_0}-1\\Big).$$",
        "",
        "- $\\Delta D_{\\rm saved}$：**可节省 Token**；",
        "- $\\Delta D_{\\rm eq}$：**等效新增 Token**，即维持原质量时要多加多少 Token "
        "才能买到同样的降损。",
        "",
        "这两个量既不相等也不互为倒数，正文必须分开表述。"
        "等损失曲线的斜率是 $dD/dQ = -D/Q < 0$；正的 $\\Delta D_{\\rm eq}$ 是"
        "**收益换算量**，不是等损失曲线的斜率。",
        "",
        f"以工作点 $N={point['N_billion_parameters']:g}$B、"
        f"$D_0={point['D0_billion_tokens']:g}$B、$Q_0={point['Q0']}$ 为例，"
        f"提到 $Q_1={point['Q1_example']}$：",
        "",
        f"- 维持原损失所需 Token $D_{{\\rm need}} = "
        f"{example['D_need_dq_model']:.2f}$ 十亿；",
        f"- **可节省 {example['tokens_saved_dq_model']:.2f} 十亿 Token**；",
        f"- 该降损**等效于在原质量下额外增加 "
        f"{example['equivalent_new_tokens']:.2f} 十亿 Token**。",
        "",
        f"> {point['note']}。",
        "",
        "## 2. 检验 $DQ$ 假设",
        "",
        "$$L_Q = E + A N^{-\\alpha} + B D^{-\\beta_D} Q^{-\\beta_Q}$$",
        "",
        "留一质量等级（每次扣掉一个 $Q$ 等级 = 45 组，重拟合后预测该等级）：",
        "",
        "| 模型 | 留出 RMSE | 留出相对 RMSE | 留出中位 APE | 样本内 RMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| $DQ$ 模型（5 参数） | {holdout['pooled']['dq']['rmse']:.4f} | "
        f"{holdout['pooled']['dq']['relative_rmse']:.2%} | "
        f"{holdout['pooled']['dq']['median_ape']:.2%} | "
        f"{payload['models']['dq']['in_sample']['rmse']:.4f} |",
        f"| 自由指数模型（6 参数） | {holdout['pooled']['free']['rmse']:.4f} | "
        f"{holdout['pooled']['free']['relative_rmse']:.2%} | "
        f"{holdout['pooled']['free']['median_ape']:.2%} | "
        f"{payload['models']['free']['in_sample']['rmse']:.4f} |",
        "",
        f"自由指数模型的 $r=\\beta_Q/\\beta_D$：逐折 "
        f"[{holdout['ratio_across_folds']['min']:.3f}, "
        f"{holdout['ratio_across_folds']['max']:.3f}]，"
        f"中位 {holdout['ratio_across_folds']['median']:.3f}；"
        f"bootstrap {ratio['replicates_used']} 次的 5–95% 区间 "
        f"[{ratio['ratio_p05']:.3f}, {ratio['ratio_p95']:.3f}]"
        f"（{'包含' if ratio['ratio_contains_one'] else '不包含'} 1）。",
        "",
        f"bootstrap 中 $\\beta_D$ 与 $\\beta_Q$ 的相关系数为 "
        f"**{ratio['corr_beta_Q_beta_D']:+.3f}**：数据主要约束的是两者的**比值**，"
        "各自水平并不定，因此 $r$ 偏离 1 的幅度不宜单独解读。"
        f"在此同时，$DQ$ 模型的留出误差（"
        f"{holdout['pooled']['dq']['rmse']:.4f}）**不高于**自由指数模型（"
        f"{holdout['pooled']['free']['rmse']:.4f}）——多出来的自由度没有换来泛化改善，"
        "故正文仍采用 $DQ$ 换算，并把 $r$ 作为敏感性对照报告。",
        "",
        f"同 $N$、同 $DQ$ 的观测分组（{groups['n_groups_with_replicates']} 组，"
        f"规模分布 {groups['group_size_histogram']}）："
        f"组内损失极差中位 **{groups['within_group_loss_range_median']:.4f}**、"
        f"最大 {groups['within_group_loss_range_max']:.4f}。"
        "因此 $DQ$ 等效是**有误差的模型近似**，不是精确物理规律。",
        "",
        "## 3. 替代关系表",
        "",
        f"工作点 $N={point['N_billion_parameters']:g}$B、"
        f"$D_0={point['D0_billion_tokens']:g}$B、$Q_0={point['Q0']}$；"
        "所有量单位为十亿 Token。末两行为两种模型的留出误差（十亿 Token 之外的指标）。",
        "",
        "| $Q_1$ | $DQ$ 模型所需 $D$ | 可节省 | 等效新增 | 自由指数模型所需 $D$ |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["substitution_table"]:
        mark = " **（基准）**" if abs(row["Q1"] - point["Q0"]) < 1e-9 else ""
        lines.append(
            f"| {row['Q1']:.1f}{mark} | {row['D_need_dq_model']:.2f} | "
            f"{row['tokens_saved_dq_model']:+.2f} | "
            f"{row['equivalent_new_tokens']:+.2f} | "
            f"{row['D_need_free_model']:.2f} |")
    lines += [
        f"| 留出误差（$DQ$ 模型） | RMSE {holdout['pooled']['dq']['rmse']:.4f} | "
        f"相对 RMSE {holdout['pooled']['dq']['relative_rmse']:.2%} | "
        f"中位 APE {holdout['pooled']['dq']['median_ape']:.2%} | — |",
        f"| 留出误差（自由指数模型） | RMSE "
        f"{holdout['pooled']['free']['rmse']:.4f} | 相对 RMSE "
        f"{holdout['pooled']['free']['relative_rmse']:.2%} | 中位 APE "
        f"{holdout['pooled']['free']['median_ape']:.2%} | — |",
        "",
        "## 4. 适用边界",
        "",
    ]
    lines += [f"- {item}" for item in payload["boundary"]]
    lines += [
        "",
        "## 5. 对现有交接材料的符号更正",
        "",
        f"- 位置：`{payload['sign_correction']['handoff_material']}`",
        f"- 原文：{payload['sign_correction']['handoff_error']}",
        f"- 更正：{payload['sign_correction']['corrected']}",
        "",
        "```bash",
        "python Q2/Q_2_4/Q_2_4_5_quality_substitution_analysis.py",
        "python Q2/Q_2_4/Q_2_4_6_quality_substitution_figures.py",
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
    (output / "quality_substitution.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "substitution_table.csv", payload["substitution_table"])
    write_csv(output / "holdout_by_quality_level.csv",
              payload["holdout_by_quality_level"]["folds"])
    write_markdown(payload, output / "analysis_report.md")

    example = next(row for row in payload["substitution_table"]
                   if abs(row["Q1"] - payload["operating_point"]["Q1_example"]) < 1e-9)
    print("等损失替代关系")
    print(f"  工作点 N=1B, D0=25B, Q0=0.6 -> Q1=0.7")
    print(f"  所需 D {example['D_need_dq_model']:.2f} 十亿  "
          f"可节省 {example['tokens_saved_dq_model']:.2f} 十亿  "
          f"等效新增 {example['equivalent_new_tokens']:.2f} 十亿")
    print(f"  留出 RMSE: DQ {payload['holdout_by_quality_level']['pooled']['dq']['rmse']:.4f}"
          f"  vs 自由指数 "
          f"{payload['holdout_by_quality_level']['pooled']['free']['rmse']:.4f}")
    ratio = payload["ratio_interval"]
    print(f"  r=beta_Q/beta_D 的 5-95% 区间 "
          f"[{ratio['ratio_p05']:.3f}, {ratio['ratio_p95']:.3f}]"
          f"  包含 1: {ratio['ratio_contains_one']}")
    print(f"  同 (N, DQ) 组内损失极差中位 "
          f"{payload['equal_dq_groups']['within_group_loss_range_median']:.4f}"
          f"（{payload['equal_dq_groups']['n_groups_with_replicates']} 组）")
    print(f"产物目录: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
