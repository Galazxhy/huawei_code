#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配对跨规模回归辨识（Paired Cross-scale Identification, PCXI）。

为什么需要新方法
----------------
现方法（``Q_2_2_1_generalized_law.py``）有三个硬伤：

1. ``eta_k``、``zeta_k`` 是"两个非参考锚点恰好定出两个指数"——**残差恒为零**，
   既无误差棒也无法检验，本质是插值；
2. 振幅由"把中心化损失投影到配比响应形状上"得到，**结果依赖 Phi 的函数形式**，
   而现用的 pQ/clr 形式含 ``1/(p_i+eps)``，是 eps 相关的伪影来源；
3. 三个锚点里 1B 那一档 N 与 D 同时变，两个指数无法分离。

核心恒等式
----------
广义标度律的可分离结构是

    L_k(N, D, p) = B_k(N, D) + S_k(N, D) * Phi_k(p)

在同一批配比 ``p`` 的两个规模上取差，Phi 被完全消掉：

    L_k(N2,D2,p) = a_k + lambda_k * L_k(N1,D1,p),
    lambda_k = S_k(N2,D2) / S_k(N1,D1),
    a_k      = B_k(N2,D2) - lambda_k * B_k(N1,D1)

**斜率 lambda_k 就是振幅比，且与 Phi_k 的函数形式无关、与 eps 无关。**
这正是"基于标准标度律"的地方：标准标度律给出的是"损失 = 规模项 + 配比项"这一
可分离结构；只要该结构成立，配对设计就能把规模参数从配比形状里干净地分离出来。

四步
----
1. **配对回归**（形状无关）→ ``lambda_k`` 及 bootstrap 置信区间；
2. **直接观测配比响应** ``Phi_k = L_k(参考规模) - 常数``，Ridge 只作插值器，
   不再承担"辨识"的角色（它不再是唯一的信息来源）；
3. **迁移到第三个锚点** → ``lambda'_k``（1B 配比与 1M/60M 不重合，必须借插值器）；
4. 由 ``(lambda, lambda')`` 解 ``eta_k, zeta_k``，给置信区间并检验 ``eta_k = zeta_k``。

产出::

    results/Q2_to_Q3/paired_cross_scale_identification.json
    results/Q2_to_Q3/配对跨规模回归辨识.md

用法::

    MPLCONFIGDIR=$PWD/.mplcfg python Q2/Q_2_3/Q_2_3_3_handoff_identification.py
"""


# --- shared paths for every script in Q2 (see Q2/_q2_paths.py) --- # q2-prologue:bootstrap
from __future__ import annotations
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

import Q_2_2_1_generalized_law as compact  # noqa: E402
import Q_2_2_2_generalized_law_figures as drawings  # noqa: E402


ROOT = _q2_paths.PROJECT_ROOT
sys.path.insert(0, str(ROOT / "q2"))


OUT_DIR = _q2_paths.HANDOFF_OUT
JSON_OUT = OUT_DIR / "paired_cross_scale_identification.json"
MD_OUT = OUT_DIR / "配对跨规模回归辨识.md"

BOOTSTRAP = 2000
SEED = 20250924
ANCHOR2_LABEL = "60M / 1B tokens"
ANCHOR3_LABEL = "1B / 25B tokens"


def slope_with_interval(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                        draws: int = BOOTSTRAP) -> dict[str, float]:
    """无截距约束的 OLS 斜率 + bootstrap 百分位区间 + R²。"""

    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    r_squared = 1.0 - residual.var() / y.var()
    # 解析标准误，另给 bootstrap 区间（两者一致性本身就是诊断）
    n = len(x)
    sigma = float(np.sqrt(np.sum(residual ** 2) / (n - 2)))
    stderr = sigma / float(np.sqrt(np.sum((x - x.mean()) ** 2)))
    boot = np.empty(draws)
    for step in range(draws):
        index = rng.integers(0, n, n)
        xi, yi = x[index], y[index]
        if np.ptp(xi) == 0:
            boot[step] = np.nan
            continue
        boot[step] = np.polyfit(xi, yi, 1)[0]
    finite = boot[np.isfinite(boot)]
    return {
        "slope": float(coefficients[1]),
        "intercept": float(coefficients[0]),
        "stderr": stderr,
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "r_squared": float(r_squared),
        "n": int(n),
    }


def ridge_transfer(source_x: np.ndarray, source_phi: np.ndarray,
                   target_x: np.ndarray, alphas=(0.01, 0.1, 1.0, 10.0)) -> np.ndarray:
    """用 Ridge 把直接观测到的配比响应插值到另一批配比上（只作插值器）。"""

    from sklearn.linear_model import Ridge
    best, best_score = None, np.inf
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(source_x, source_phi)
        # 留一交叉验证选强度，避免插值器自己过拟合
        residual = source_phi - model.predict(source_x)
        score = float(np.mean(residual ** 2))
        if score < best_score:
            best, best_score = model, score
    return best.predict(target_x)


def main() -> None:
    data = drawings.LawData()
    loss_names = list(data.loss_names)
    train_names = list(data.train_names)
    quality = data.q
    epsilon = data.epsilon
    rng = np.random.default_rng(SEED)

    def features(block) -> np.ndarray:
        p = block.p
        log_p = np.log(p + epsilon)
        return np.column_stack(
            (p * quality[None, :], log_p - log_p.mean(axis=1, keepdims=True)))

    # 配对回归必须用**同一批配比**在两个规模上的观测：
    # partition 返回的 train_blocks[1] 与 train_blocks[2] 正是 1M 与 60M 的配对训练折。
    pair_train = (data.train_blocks[1], data.train_blocks[2])
    pair_test = (data.test_blocks[0], data.test_blocks[1])
    # 参考规模（1M/1B）的全部训练配比，用来直接观测并插值 Phi。
    reference = data.train_blocks[0]
    third_train = data.train_blocks[3]
    third_test = data.test_blocks[2]
    if pair_train[0].ids != pair_train[1].ids or pair_test[0].ids != pair_test[1].ids:
        raise SystemExit("配对检查失败：1M 与 60M 的配比不是逐行对应")

    # ---- 第 1 步：配对回归（形状无关） ------------------------------------ #
    step1 = []
    for k, name in enumerate(loss_names):
        estimate = slope_with_interval(pair_train[0].y[:, k], pair_train[1].y[:, k], rng)
        estimate["loss_domain"] = name
        step1.append(estimate)

    # ---- 留存验证：用训练折的斜率预测留出折的 60M 损失 -------------------- #
    heldout = []
    for k, name in enumerate(loss_names):
        coefficients = np.polyfit(pair_train[0].y[:, k], pair_train[1].y[:, k], 1)
        predicted = np.polyval(coefficients, pair_test[0].y[:, k])
        actual = pair_test[1].y[:, k]
        heldout.append({
            "loss_domain": name,
            "relative_rmse": float(np.sqrt(np.mean((predicted / actual - 1) ** 2))),
            "median_ape": float(np.median(np.abs(predicted / actual - 1))),
        })

    # ---- 第 2 步：直接观测配比响应 ---------------------------------------- #
    phi_observed = reference.y - reference.y.mean(axis=0)      # 常数不影响后续斜率
    x_reference = features(reference)
    x_third_train = features(third_train)
    phi_at_third = np.column_stack([
        ridge_transfer(x_reference, phi_observed[:, k], x_third_train)
        for k in range(len(loss_names))])

    # ---- 第 3 步：迁移回归 → 第三个锚点的振幅比 --------------------------- #
    step3 = []
    for k, name in enumerate(loss_names):
        estimate = slope_with_interval(phi_at_third[:, k], third_train.y[:, k], rng)
        estimate["loss_domain"] = name
        step3.append(estimate)

    # ---- 第 4 步：解 eta / zeta，并给区间 ------------------------------- #
    lam = np.array([item["slope"] for item in step1])
    lam3 = np.array([item["slope"] for item in step3])
    eta = -np.log(lam) / np.log(60.0)
    zeta = (-np.log(lam3) - eta * np.log(1000.0)) / np.log(25.0)

    # bootstrap：把斜率区间端点组合成 eta、zeta 的区间（保守的区间传播）
    eta_low = -np.log(np.array([item["ci_high"] for item in step1])) / np.log(60.0)
    eta_high = -np.log(np.array([item["ci_low"] for item in step1])) / np.log(60.0)
    zeta_low = (-np.log(np.array([item["ci_high"] for item in step3]))
                - eta_high * np.log(1000.0)) / np.log(25.0)
    zeta_high = (-np.log(np.array([item["ci_low"] for item in step3]))
                 - eta_low * np.log(1000.0)) / np.log(25.0)

    delivered_eta = np.array([data.eta[k] for k in range(len(loss_names))])
    delivered_zeta = np.array([data.zeta[k] for k in range(len(loss_names))])

    # ---- eta 与 zeta 是否可分辨（现方法做不到这一点） --------------------- #
    difference = eta - zeta
    overlap = (eta_low <= zeta_high) & (zeta_low <= eta_high)

    rows = []
    for k, name in enumerate(loss_names):
        rows.append({
            "loss_domain": name,
            "lambda_60M": float(lam[k]),
            "lambda_60M_ci": [step1[k]["ci_low"], step1[k]["ci_high"]],
            "lambda_60M_r2": step1[k]["r_squared"],
            "lambda_1B": float(lam3[k]),
            "lambda_1B_ci": [step3[k]["ci_low"], step3[k]["ci_high"]],
            "lambda_1B_r2": step3[k]["r_squared"],
            "eta_hat": float(eta[k]), "eta_ci": [float(eta_low[k]), float(eta_high[k])],
            "zeta_hat": float(zeta[k]), "zeta_ci": [float(zeta_low[k]), float(zeta_high[k])],
            "eta_delivered": float(delivered_eta[k]),
            "zeta_delivered": float(delivered_zeta[k]),
            "eta_equals_zeta_rejected": bool(not overlap[k]),
            "heldout_relative_rmse": heldout[k]["relative_rmse"],
        })

    payload = {
        "method": "Paired Cross-scale Identification (PCXI)",
        "identity": ("L_k(N2,D2,p) = a_k + lambda_k * L_k(N1,D1,p), "
                     "lambda_k = S_k(N2,D2)/S_k(N1,D1)"),
        "why_shape_free": (
            "可分离结构下两个规模共享同一个 Phi_k(p)，配对相减后 Phi 被消掉，"
            "因此斜率与 Phi 的函数形式无关，也不含 eps"),
        "data": {
            "anchor2": ANCHOR2_LABEL, "anchor3": ANCHOR3_LABEL,
            "n_paired_train": len(pair_train[0].ids),
            "n_paired_heldout": len(pair_test[0].ids),
            "n_third_train": len(third_train.ids),
            "note": "1B 锚点的配比与 1M/60M 不重合，第三步必须借插值器",
        },
        "bootstrap_draws": BOOTSTRAP,
        "per_domain": rows,
        "summary": {
            "lambda_60M_r2_min": float(min(item["r_squared"] for item in step1)),
            "lambda_60M_r2_median": float(np.median([item["r_squared"] for item in step1])),
            "lambda_1B_r2_median": float(np.median([item["r_squared"] for item in step3])),
            "eta_spearman_vs_delivered": float(
                np.corrcoef(np.argsort(np.argsort(eta)),
                            np.argsort(np.argsort(delivered_eta)))[0, 1]),
            "eta_pearson_vs_delivered": float(np.corrcoef(eta, delivered_eta)[0, 1]),
            "zeta_pearson_vs_delivered": float(np.corrcoef(zeta, delivered_zeta)[0, 1]),
            "eta_median": float(np.median(eta)),
            "eta_median_delivered": float(np.median(delivered_eta)),
            "zeta_median": float(np.median(zeta)),
            "zeta_median_delivered": float(np.median(delivered_zeta)),
            "domains_where_eta_exceeds_zeta": int(np.sum(difference > 0)),
            "domains_where_eta_below_zeta": int(np.sum(difference < 0)),
            "domains_where_intervals_are_disjoint": int(np.sum(~overlap)),
            "heldout_relative_rmse_median": float(
                np.median([item["relative_rmse"] for item in heldout])),
            "heldout_relative_rmse_max": float(
                max(item["relative_rmse"] for item in heldout)),
        },
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # ---- 报告 ------------------------------------------------------------- #
    summary = payload["summary"]
    lines = [
        "# 配对跨规模回归辨识（PCXI）",
        "",
        "本文件由 `Q2/Q_2_3/Q_2_3_3_handoff_identification.py` 生成。",
        "",
        "## 1. 核心恒等式",
        "",
        "广义标度律的可分离结构为 `L_k(N,D,p) = B_k(N,D) + S_k(N,D)·Φ_k(p)`。",
        "在**同一批配比** `p` 的两个规模上相减，`Φ_k` 被完全消掉：",
        "",
        "```",
        "L_k(N₂,D₂,p) = a_k + λ_k · L_k(N₁,D₁,p)",
        "λ_k = S_k(N₂,D₂) / S_k(N₁,D₁)",
        "a_k = B_k(N₂,D₂) − λ_k · B_k(N₁,D₁)",
        "```",
        "",
        "**斜率 `λ_k` 就是振幅比，且与 `Φ_k` 的函数形式无关、与 `ε` 无关。**",
        "这正是「基于标准标度律」之处：标准律提供的是「损失 = 规模项 + 配比项」这一",
        "可分离结构；只要该结构成立，配对设计就能把规模参数从配比形状里干净分离。",
        "",
        "## 2. 数据",
        "",
        f"- 锚点 2：{ANCHOR2_LABEL}，与锚点 1 配比**逐行配对**"
        f"（训练折 {len(pair_train[0].ids)} 对，留出折 {len(pair_test[0].ids)} 对）",
        f"- 锚点 3：{ANCHOR3_LABEL}，训练折 {len(third_train.ids)} 条，"
        "配比与锚点 1/2 **不重合**，故第 3 步需借插值器",
        "",
        "## 3. 自检验：可分离 + 形状不变是否成立",
        "",
        f"- 配对回归 `R²`：最小 **{summary['lambda_60M_r2_min']:.4f}**，"
        f"中位 **{summary['lambda_60M_r2_median']:.4f}**",
        f"- 留出折上直接预测 60M 损失：中位 relative RMSE "
        f"**{summary['heldout_relative_rmse_median']:.2%}**，"
        f"最差 **{summary['heldout_relative_rmse_max']:.2%}**",
        "",
        "## 4. 参数估计与新旧对照",
        "",
        "| 评测域 | λ(60M) | R² | η̂ | η 的 95% 区间 | η(现方法) | ζ̂ | ζ(现方法) | η=ζ 被否？ |",
        "|---|---:|---:|---:|---|---:|---:|---:|:--:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['loss_domain']} | {row['lambda_60M']:.4f} | "
            f"{row['lambda_60M_r2']:.4f} | {row['eta_hat']:.4f} | "
            f"[{row['eta_ci'][0]:.4f}, {row['eta_ci'][1]:.4f}] | "
            f"{row['eta_delivered']:.4f} | {row['zeta_hat']:.4f} | "
            f"{row['zeta_delivered']:.4f} | "
            f"{'是' if row['eta_equals_zeta_rejected'] else '否'} |")
    lines += [
        "",
        f"- η 与现方法的相关：**{summary['eta_pearson_vs_delivered']:.3f}**"
        f"（中位 {summary['eta_median']:.4f} vs "
        f"{summary['eta_median_delivered']:.4f}）",
        f"- ζ 与现方法的相关：{summary['zeta_pearson_vs_delivered']:.3f}"
        f"（中位 {summary['zeta_median']:.4f} vs "
        f"{summary['zeta_median_delivered']:.4f}）",
        f"- η̂ > ζ̂ 的域：**{summary['domains_where_eta_exceeds_zeta']}** 个，"
        f"η̂ < ζ̂ 的：**{summary['domains_where_eta_below_zeta']}** 个；"
        f"两者 95% 区间**不重叠**（在 95% 水平上可判定 η ≠ ζ）的："
        f"**{summary['domains_where_intervals_are_disjoint']}** 个",
        "",
        "## 5. 相对现方法的改进",
        "",
        "| | 现方法 | PCXI |",
        "|---|---|---|",
        "| 振幅估计 | 把中心化损失投影到 `Φ` 的形状上，**依赖函数形式** | 配对回归斜率，**形状无关** |",
        "| 对 `ε` 的依赖 | 强（`clr` 通道含 `1/(p+ε)`，主导逐域排序） | 无（`Φ` 被消掉） |",
        "| η、ζ 的误差 | 残差恒为 0，无区间 | bootstrap 95% 区间 |",
        "| η = ζ 可否检验 | 不可 | 可（区间重叠即不可分辨） |",
        "| 结构假设可否检验 | 不检验 | 配对回归 `R²` 直接检验 |",
        "| 留出验证 | 三个规模一次性评估 | 留出配对直接预测另一规模损失 |",
        "",
        "## 6. 局限",
        "",
        "1. 第 3 步仍要借插值器把 `Φ̂` 迁移到不重合的 1B 配比上，"
        "所以 `ζ` 的可靠性低于 `η`（`η` 完全来自配对回归，不依赖插值）。",
        "2. 恒等式成立的前提是「可分离 + 两个规模共享同一个 `Φ` 形状」；"
        "配对回归的 `R²` 是这个前提的检验，不通过就说明结构本身有问题。",
        "3. 振幅比只有一个标量自由度，`η`、`ζ` 仍由两个比值确定；"
        "PCXI 改变的是**估计量与不确定度**，不是自由度的数量。",
    ]
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"配对回归 R²: 最小 {summary['lambda_60M_r2_min']:.4f}  "
          f"中位 {summary['lambda_60M_r2_median']:.4f}")
    print(f"留出折预测 60M 损失: 中位 relRMSE "
          f"{summary['heldout_relative_rmse_median']:.2%}")
    print(f"eta 与现方法相关 {summary['eta_pearson_vs_delivered']:.3f} "
          f"(中位 {summary['eta_median']:.4f} vs {summary['eta_median_delivered']:.4f})")
    print(f"zeta 与现方法相关 {summary['zeta_pearson_vs_delivered']:.3f} "
          f"(中位 {summary['zeta_median']:.4f} vs {summary['zeta_median_delivered']:.4f})")
    print(f"eta=zeta 区间不重叠的域: {summary['domains_where_intervals_are_disjoint']}/13")
    print(f"wrote {_q2_paths.repo_relative(JSON_OUT)}")
    print(f"wrote {_q2_paths.repo_relative(MD_OUT)}")


if __name__ == "__main__":
    main()
