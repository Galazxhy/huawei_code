#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出「可直接写进论文」的广义标度律公式与参数表。

与 ``generalized_law.py`` 的辨识结果逐位一致（偏差 < 1e-14），
只是换了记号：把 ``N_ref``、``D_ref`` 吸收进配比项，并把 ``alpha = beta`` 合并。

产出::

    data_analysis/Q2_to_Q3/广义标度律_论文写法.md
    data_analysis/Q2_to_Q3/generalized_scaling_law_paper_coefficients.csv

用法::

    MPLCONFIGDIR=$PWD/.mplcfg python q2/handoff_paper_form.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "q2"))

import generalized_law as compact  # noqa: E402
import generalized_law_figures as drawings  # noqa: E402

OUT_DIR = ROOT / "data_analysis/Q2_to_Q3"
MARKDOWN = OUT_DIR / "广义标度律_论文写法.md"
COEFFICIENTS = OUT_DIR / "generalized_scaling_law_paper_coefficients.csv"


def collect() -> dict:
    data = drawings.LawData()
    law = json.loads(
        (ROOT / "data_analysis/Q_1_to_2/generalized_scaling_compact.json")
        .read_text(encoding="utf-8"))["compositional_scaling_law"]
    parameters = law["parameters"]
    loss_names = list(data.loss_names)
    train_names = list(data.train_names)

    def per_domain(key: str) -> np.ndarray:
        return np.asarray([parameters[key][name] for name in loss_names])

    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    eta = per_domain("eta")
    zeta = per_domain("zeta")
    # 吸收常数：S_k·Phi_k = N^-eta D^-zeta · Psi_k，Psi_k = c_k · Phi_k
    absorb = data.reference_n ** eta * data.reference_d ** zeta
    pQ = np.asarray([[parameters["effective_weight_pQ"][k][t] for t in train_names]
                     for k in loss_names])
    clr = np.asarray([[parameters["effective_weight_clr"][k][t] for t in train_names]
                      for k in loss_names])
    return {
        "data": data, "loss_names": loss_names, "train_names": train_names,
        "alpha": alpha, "beta": beta, "eta": eta, "zeta": zeta,
        "E": per_domain("E"), "A": per_domain("A"), "B": per_domain("B"),
        "absorb": absorb,
        "psi_pQ": absorb[:, None] * pQ, "psi_clr": absorb[:, None] * clr,
        "mbar": np.asarray(parameters["feature_mean"]),
        "epsilon": float(law["epsilon"]),
        "heldout": law["aggregate_heldout"],
    }


def verify(values: dict) -> float:
    """吸收写法必须与原模型逐位一致。"""

    blocks, _, _, _ = compact.load_data(compact.REGMIX, compact.HANDOFF)
    worst = 0.0
    for block in blocks:
        p = block.p
        log_p = np.log(p + values["epsilon"])
        features = np.column_stack(
            (p * values["data"].q[None, :], log_p - log_p.mean(axis=1, keepdims=True)))
        n_b, d_b = block.n / 1e9, block.d / 1e9
        base = (values["E"][None, :] + n_b ** (-values["alpha"]) * values["A"][None, :]
                + d_b ** (-values["beta"]) * values["B"][None, :])
        psi = ((features[:, :17] - values["mbar"][:17]) @ values["psi_pQ"].T
               + (features[:, 17:] - values["mbar"][17:]) @ values["psi_clr"].T)
        rebuilt = base + (n_b ** (-values["eta"]) * d_b ** (-values["zeta"]))[None, :] * psi
        reference = values["data"].predict(block.n, block.d, block.p)
        worst = max(worst, float(np.abs(rebuilt - reference).max()))
    if worst > 1e-12:
        raise SystemExit(f"吸收写法与原模型不一致：{worst}")
    return worst


def markdown(values: dict, gap: float) -> str:
    v = values
    held = v["heldout"]
    lines: list[str] = []
    add = lines.append

    add("# 广义标度律：可直接写进论文的形式\n")
    add("本文件由 `q2/handoff_paper_form.py` 从 `data_analysis/Q_1_to_2/"
        "generalized_scaling_compact.json` 生成，\n"
        f"与交付求值器逐位一致（最大偏差 {gap:.1e}）。记号做了两处等价整理：\n")
    add("1. **吸收参考规模**：把 `(N/N_ref)^(−η)(D/D_ref)^(−ζ)` 换成 `N^(−η)·D^(−ζ)`，"
        "`N_ref`、`D_ref` 并入配比项系数，正文不再出现；\n"
        "2. **合并 α = β**：交付模型里两者是同一个数（"
        f"{v['alpha']:.6f}），基线可写成「规模因子 × 配比因子」。\n")

    add("\n## 1. 主公式\n")
    add("```latex")
    add(r"L_k(N,D,\boldsymbol p) = E_k"
        r" + N^{-\alpha}\Big[A_k + B_k\Big(\frac{D}{N}\Big)^{-\alpha}\Big]"
        r" + N^{-\eta_k}D^{-\zeta_k}\,\Psi_k(\boldsymbol p),"
        r"\qquad k = 1,\dots,13")
    add("```")
    add("\n其中 $N$ 为参数量、$D$ 为训练 token 数，**均以十亿（$10^9$）为单位**；"
        "$L_k$ 是第 $k$ 个评测域的验证交叉熵；$\\boldsymbol p$ 是 17 维数据配比，"
        "$\\sum_i p_i = 1$。\n")
    add("两项的读法：\n")
    add("- 基线 $N^{-\\alpha}[A_k + B_k (D/N)^{-\\alpha}]$：**只依赖规模 $N$ 与配比 "
        "$D/N$**，与数据配比无关；")
    add("- 配比项 $N^{-\\eta_k}D^{-\\zeta_k}\\Psi_k(\\boldsymbol p)$："
        "$\\Psi_k$ 是配比响应，$N^{-\\eta_k}D^{-\\zeta_k}$ 是它的**振幅随规模的衰减**。\n")

    add("\n## 2. 配比响应 $\\Psi_k$\n")
    add("```latex")
    add(r"\Psi_k(\boldsymbol p) = \sum_{i=1}^{17} w^{pQ}_{k,i}"
        r"\big(p_i Q_i - \bar m_i\big)"
        r" + \sum_{i=1}^{17} w^{clr}_{k,i}\big(\mathrm{clr}_\varepsilon(p)_i"
        r" - \bar m'_i\big)")
    add("```")
    add("\n```latex")
    add(r"\mathrm{clr}_\varepsilon(\boldsymbol p)_i = \ln(p_i+\varepsilon)"
        r" - \frac{1}{17}\sum_{j=1}^{17}\ln(p_j+\varepsilon),"
        rf"\qquad \varepsilon = {v['epsilon']:g}")
    add("```")
    add("\n$Q_i$ 是第 $i$ 个数据域的固定质量分（观测输入），"
        "$\\bar m_i$、$\\bar m'_i$ 是中心化常数 = 参考锚点训练配比上该特征的均值，"
        "作用是让 $\\Psi_k$ 在参考配比上均值为 0"
        "（即 $\\Psi_k$ 读作「相对平均训练配比的偏离」）。"
        "两段系数的完整数值见同目录 "
        "`generalized_scaling_law_paper_coefficients.csv`。\n")

    add("\n## 3. 参数\n")
    add(f"- $\\alpha = \\beta = {v['alpha']:.6f}$（由经典标度律 B1 外生固定，不参与本模型拟合）")
    add(f"- $\\varepsilon = {v['epsilon']:g}$")
    add(f"- 逐评测域参数 $E_k, A_k, B_k, \\eta_k, \\zeta_k$（下表）")
    add("- 配比项系数 $w^{pQ}_{k,i}$、$w^{clr}_{k,i}$：13 × 34 = 442 个，见 CSV\n")
    add("| 评测域 $k$ | $E_k$ | $A_k$ | $B_k$ | $\\eta_k$ | $\\zeta_k$ |")
    add("|---|---:|---:|---:|---:|---:|")
    for index, name in enumerate(v["loss_names"]):
        add(f"| {name} | {v['E'][index]:.4f} | {v['A'][index]:.4f} | {v['B'][index]:.4f} "
            f"| {v['eta'][index]:.4f} | {v['zeta'][index]:.4f} |")

    add("\n### 3.1 中心化常数\n")
    add("| $i$ | 数据域 | $Q_i$ | $\\bar m_i$ | $\\bar m'_i$ |")
    add("|---|---|---:|---:|---:|")
    for index, name in enumerate(v["train_names"]):
        add(f"| {index + 1} | {name} | {v['data'].q[index]:.4f} "
            f"| {v['mbar'][index]:.4f} | {v['mbar'][index + 17]:.4f} |")

    add("\n## 4. 精度与适用区间（论文里应一并给出）\n")
    add("| 项目 | 值 |")
    add("|---|---:|")
    add(f"| 留出集 relative RMSE | {held['relative_rmse']:.2%} |")
    add(f"| 留出集中位 APE | {held['median_ape']:.2%} |")
    add(f"| 留出集配比项 $R^2$ | {held['mixture_r_squared_within_scale']:.4f} |")
    add(f"| 不可约下界 $E_k$ 未被击穿 | "
        f"{'是' if held['irreducible_floor']['fraction_of_predictions_below_E'] == 0 else '否'} |")
    add("\n观测支撑域：$N \\in [10^{-3}, 1]$（十亿参数），"
        "$D \\in [1, 25]$（十亿 token），配比熵 $H(\\boldsymbol p)\\in[0.008, 2.381]$。"
        "超出即外推。\n")

    add("\n## 5. 三条必须写进论文的限定\n")
    add("1. **$\\eta_k$、$\\zeta_k$ 没有留出检验**。每个评测域的振幅只在三个锚点上各有一个标量，"
        "两个非参考锚点恰好定出两个指数，曲线逐点穿过反演值、残差为零——这是插值不是验证。")
    add("2. **$Q_i$ 不可独立辨识**。附件 A 中每个域的 $Q_i$ 固定，自由域系数可以把它的尺度整体吸收，"
        "所以本模型给的是**配比**响应，不是质量效应；质量弹性须取自 B6/B7。")
    add("3. **$w^{pQ}$、$w^{clr}$ 不可按域解读**。69% 的权重为负，"
        "而控制住配比多样性之后逐域归因会崩塌（排序 Spearman ≈ −0.10）；"
        "这些系数只用于预测，不能读成「哪个域质量高」的排序。\n")
    return "\n".join(lines) + "\n"


def write_coefficients(values: dict) -> None:
    rows = []
    for k, domain in enumerate(values["loss_names"]):
        for i, source in enumerate(values["train_names"]):
            rows.append({
                "loss_domain_k": domain, "source_i": source, "feature": f"pQ:{source}",
                "Q_i": float(values["data"].q[i]),
                "mbar_i": float(values["mbar"][i]),
                "w_pQ": float(values["psi_pQ"][k, i]),
                "w_clr": float(values["psi_clr"][k, i]),
                "mbar_clr_i": float(values["mbar"][i + 17]),
            })
    with COEFFICIENTS.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    values = collect()
    gap = verify(values)
    MARKDOWN.write_text(markdown(values, gap), encoding="utf-8")
    write_coefficients(values)
    print(f"吸收写法与原模型最大偏差 {gap:.1e}")
    print(f"wrote {MARKDOWN.relative_to(ROOT)}")
    print(f"wrote {COEFFICIENTS.relative_to(ROOT)} "
          f"({len(values['loss_names'])} x {len(values['train_names'])} rows)")


if __name__ == "__main__":
    main()
