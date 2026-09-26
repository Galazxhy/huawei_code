#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""广义标度律四因素边际效益与弹性的例图（中文标注）。

读 ``data_analysis/factor_marginal_elasticity/`` 的产物，不重算模型。

三张图
------
``factor_elasticity_overview``
    (a) 逐域、逐锚点的 ``∂lnL/∂lnN`` 对 ``∂lnL/∂lnD`` 散点，看两个算力因素谁更敏感；
    (b) 四个因素的弹性汇总（点 = 中位，横线 = 逐域范围），工作点写在图例里。
``mixture_directional_gain``
    配额提高 1 个百分点的**精确**增益热图，13 评测域 × 17 来源，1M 与 1B 两个锚点。
``quality_marginal_curve``
    独立质量律的 ``Q`` 边际效益与弹性随 ``Q`` 的曲线；广义律的隐含值作为反面对照。

用法：
    python q2/factor_elasticity_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generalized_law_figures import chinese_family  # noqa: E402

INPUT = ROOT / "data_analysis" / "factor_marginal_elasticity"
OUTPUT = INPUT / "figures"
PANEL_FIGSIZE = (6.0, 4.0)
#: 上下布局的统一画布：本目录所有图都改为上下排列，尺寸保持一致。
STACKED_FIGSIZE = (6, 7.0)
#: 两张 13 行热图叠起来需要更高一些。
STACKED_TALL_FIGSIZE = (6, 7.0)
BAR_COLOR = "#4C78A8"
BAR_COLOR_ALT = "#7BA7CC"
BAR_COLOR_MUTED = "#BFBFBF"
ACCENT = "#C44E52"
ANCHOR_STYLE = {
    "1M / 1B tokens": ("#4C78A8", "o"),
    "60M / 1B tokens": ("#F58518", "s"),
    "1B / 25B tokens": ("#54A24B", "^"),
}


def configure() -> str:
    """House standard: no titles, sizes 14/12/12, white background."""

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })
    return chinese_family()


def tidy(axis, *, grid_axis: str = "y") -> None:
    axis.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def figure_a_scale_mixture_sensitivity(payload: dict, domain_rows: list[dict],
                                       mixture_rows: list[dict]) -> plt.Figure:
    """图 A：规模敏感性（左）与配比调整方向的收益分布（右）。

    左：``N``、``D`` 弹性的中位数与跨评估目标的**分布范围**（四分位），三个观测规模
    为实心标记；10B/250B 外推以空心标记 + 虚线单独画出，且**不并入**汇总统计。
    右：可行的"某来源份额 +1 个百分点"调整的精确降损分布，零收益线标出，
    正收益方向占比写在图例下方。

    中文读者导向标签，不出现数据附件代号。跨目标区间是分布范围，不是置信区间。
    """

    observed = [row["label"] for row in payload["observed_scales"]]
    extrapolation = payload["extrapolation_scenario"]["label"]
    # 上下布局；上panel 的 x 标题省略（其分类与下panel 不同，不能共用横轴）。
    figure, (left, right) = plt.subplots(2, 1, figsize=STACKED_FIGSIZE,
                                         layout="constrained")

    index = {label: position for position, label in enumerate(observed + [extrapolation])}
    styles = {
        "N": (BAR_COLOR, "o"),
        "D": ("#54A24B", "s"),
    }
    for factor, (colour, marker) in styles.items():
        for horizon, filled in ((observed, True), ([extrapolation], False)):
            medians, lows, highs, xs = [], [], [], []
            for label in horizon:
                picked = [r for r in domain_rows
                          if r["factor"] == factor and r["scale"] == label]
                values = np.array([float(r["elasticity"]) for r in picked])
                xs.append(index[label])
                medians.append(np.median(values))
                lows.append(np.quantile(values, 0.25))
                highs.append(np.quantile(values, 0.75))
            left.vlines(xs, lows, highs, color=colour, linewidth=2.2,
                        alpha=0.55 if filled else 0.35,
                        linestyle="-" if filled else "--", zorder=2)
            left.plot(xs, medians, linestyle="none", marker=marker, markersize=9,
                      markerfacecolor=colour if filled else "white",
                      markeredgecolor=colour, markeredgewidth=1.6, zorder=3,
                      label=(f"{factor}" + ("" if filled else "（外推）")))
    left.axhline(0, color="#666666", linewidth=0.9)
    left.set_xticks(list(index.values()))
    left.set_xticklabels([label.replace("（外推）", "\n（外推）") for label in index],
                         fontsize=13)
    left.set_ylabel("弹性  $\\partial\\ln L/\\partial\\ln x$")
    # 两个 panel 的 x 分类不同（上含 10B 外推、下只有观测规模），刻度要保留；
    # 只省掉与刻度重复的 x 标题。外推的说明由图例的「（外推）」承担。
    tidy(left)
    left.legend(loc="lower right", fontsize=13, frameon=True, framealpha=0.92,
                edgecolor="#CCCCCC")

    # 只纳入内点且扰动前后都在观测支撑域内的方向（feasible 已含该筛选）。
    shares = payload["reference_mixture"]["shares"]
    interior = [name for name, value in shares.items() if value > 0]
    series = {}
    for label in observed:
        # CSV 里 feasible 是字符串 "True"/"False"，直接取真值会把 "False" 也放行。
        values = [float(r["benefit_1pp_exact"]) for r in mixture_rows
                  if r["scale"] == label
                  and str(r["feasible"]).strip().lower() == "true"
                  and r["benefit_1pp_exact"] not in ("", None)]
        series[label] = np.array(values, dtype=float)
    positions = np.arange(len(observed))
    box = right.boxplot([series[label] for label in observed], positions=positions,
                        widths=0.45, showfliers=False, patch_artist=True,
                        medianprops={"color": "#333333", "linewidth": 1.6})
    for patch in box["boxes"]:
        patch.set_facecolor(BAR_COLOR_ALT)
        patch.set_edgecolor("#4C78A8")
        patch.set_alpha(0.85)
    for position, label in zip(positions, observed):
        values = series[label]
        # jitter = np.linspace(-0.15, 0.15, values.size)
        # right.scatter(np.full(values.size, position) + jitter, values, s=7,
        #               color="#333333", alpha=0.35, zorder=3)
        # right.text(position, 0.98, f"正收益 {np.mean(values > 0):.0%}",
        #            transform=right.get_xaxis_transform(), ha="center", va="top",
        #            fontsize=11, color=ACCENT)
    right.axhline(0, color="#666666", linewidth=0.9)
    right.set_xticks(positions)
    right.set_xticklabels(observed, fontsize=13)
    right.set_ylabel("配比 +1 个百分点的降损")
    right.set_xlabel(f"观测规模")
    tidy(right)
    return figure


def mixture_directional_gain(payload: dict, rows: list[dict]) -> plt.Figure:
    anchors = [a["label"] for a in payload["observed_scales"]]
    picked = [anchors[0], anchors[-1]]
    sources = list(payload["reference_mixture"]["shares"])
    domains = sorted({row["eval_domain"] for row in rows})
    # 上下布局：两个面板的来源顺序相同，故共用横轴；评测域顺序也相同，两面各自标注 y。
    figure, axes = plt.subplots(2, 1, figsize=STACKED_TALL_FIGSIZE, sharex=True,
                                layout="constrained")
    values = {}
    for row in rows:
        values[(row["scale"], row["eval_domain"], row["source_domain"])] = (
            None if row["benefit_1pp_exact"] in ("", None)
            else float(row["benefit_1pp_exact"]))
    finite = [v for v in values.values() if v is not None]
    # 最大的一个格子（dm_mathematics @1M，+2.3）会比其余大两个数量级，线性色标
    # 会把其它方向全部洗成白色；改用对称对数色标，并在色标标题里写明。
    limit = float(np.max(np.abs(finite)))
    norm = SymLogNorm(linthresh=1e-3, vmin=-limit, vmax=limit, base=10)
    image = None
    # i = 0
    for axis, anchor in zip(axes, picked):
        grid = np.full((len(domains), len(sources)), np.nan)
        for i, domain in enumerate(domains):
            for j, source in enumerate(sources):
                value = values.get((anchor, domain, source))
                if value is not None:
                    grid[i, j] = value
        image = axis.imshow(grid, aspect="auto", cmap="RdBu_r", norm=norm)
        axis.set_xticks(range(len(sources)))
        axis.set_xticklabels(sources, rotation=45, ha="right", fontsize=9)
        # if i == 0:
        axis.set_yticks(range(len(domains)))
        axis.set_yticklabels(domains, fontsize=11)
        # 横轴共用，只在下方标注来源域；锚点写进各自的 y 轴标题——放在面板里会压住
        # 头两行的格子，而做成面板标题又违反本仓库"不画标题"的约定。
        axis.set_ylabel(f"评测域（{anchor}）")
        # i += 1
    axes[-1].set_xlabel("来源域")        # 横轴共用，只在下方标注一次
    for axis in axes:
        axis.tick_params(labelsize=10)
    if image is not None:
        bar = figure.colorbar(image, ax=list(axes), fraction=0.022, pad=0.015)
        bar.set_label("份额 +1 个百分点的损失降幅",
                      fontsize=13)
        bar.ax.tick_params(labelsize=10)
    return figure


def figure_b_quality_response(payload: dict) -> plt.Figure:
    """图 B：质量响应曲线（独立质量实验）。

    上下两个对齐面板共用横轴（数据质量得分）：上为 ``Q+0.01`` 的**精确**降损，
    下为质量弹性 ``∂lnL/∂lnQ``。实线为主组拟合，细虚线为子集对照组。

    **不画广义标度律对固定 ``Q_i`` 求出的隐含响应**——``Q_i`` 在配比数据里从未独立
    变化，该量不可识别，不能作为有效的质量弹性估计。

    主工作点 N=1B、D=25B、Q=0.6：其中 Q=0.6 与 N=1B 是质量实验的观测等级，
    **D=25B 落在实验的 10B 与 50B 之间，属于插值**。
    """

    curve = payload["quality_curve"]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=STACKED_FIGSIZE, sharex=True,
                                         layout="constrained")
    styles = {
        "primary": ("#4C78A8", "-", 3.0, f"主组拟合（{payload['quality_law_source']['primary']['sample_count']} 组）"),
        "subset_control": ("#C44E52", "--", 2.5, f"子集对照（{payload['quality_law_source']['control']['sample_count']} 组）"),
    }
    for fit, (colour, dashes, width, label) in styles.items():
        picked = [row for row in curve if row["fit"] == fit]
        q = np.array([row["Q"] for row in picked])
        benefit = np.array([row["benefit_for_plus_0_01_Q"]
                            if row["benefit_for_plus_0_01_Q"] is not None else np.nan
                            for row in picked], dtype=float)
        elasticity = np.array([row["elasticity"] for row in picked])
        top.plot(q, benefit, color=colour, linestyle=dashes, linewidth=width,
                 label=label)
        bottom.plot(q, elasticity, color=colour, linestyle=dashes, linewidth=width,
                    label=label)

    point = payload["quality_operating_point"]
    for axis in (top, bottom):
        axis.axvline(point["Q"], color="#999999", linewidth=1.0, linestyle=":")
        axis.set_xlim(0.09, 1.0)
    # top.annotate(f"主工作点 Q={point['Q']:g}", xy=(point["Q"], 0.55),
    #              xycoords=("data", "axes fraction"), xytext=(7, 0),
    #              textcoords="offset points", ha="left", va="center", fontsize=11,
    #              color="#555555")
    top.set_ylabel("$Q+0.01$ 的损失降幅")
    bottom.set_ylabel("弹性  $\\partial\\ln L/\\partial\\ln Q$")
    bottom.set_xlabel("数据质量得分  $Q$")
    # 刻度必须落在 0.1–1.0 这些真实取值上：写成 range(1,11) 会把坐标轴拉到 1–10，
    # 曲线被挤到最左侧一条缝里。
    bottom.set_xticks([0.1 * level for level in range(1, 11)])
    bottom.set_xticklabels([f"{level / 10:.1f}" for level in range(1, 11)], fontsize=13)
    tidy(top, grid_axis="y")
    tidy(bottom, grid_axis="y")
    top.legend(loc="upper right", fontsize=12, frameon=True, framealpha=0.92,
               edgecolor="#CCCCCC")
    # top.text(0.99, 0.05, "工作点 N=1B、D=25B（实验范围内插值）、Q=0.6",
    #          transform=top.transAxes, ha="right", fontsize=9.5, color="#555555")
    return figure


def mixture_marginal_curve(payload: dict, rows: list[dict]) -> plt.Figure:
    """配额份额扫描：全部 13 个评测域的边际效益与弹性随份额的变化。

    与质量曲线图同构（横轴是该因素自身的取值），但配比没有单一标量，所以横轴取
    "该来源域的份额"：把份额从 ``0.05`` 扫到它的记录上限，其余来源按原比例缩小。
    两端都被裁剪，因为两端各有方向本身带来的伪影而非数据支持的效应：``p_i→0`` 时
    ``Φ`` 的 clr 分量导数含 ``1/(p_i+ε)`` 被平滑常数放大；``p_i→1`` 时其余份额趋零，
    定向方向的 ``1/(1-p_i)`` 发散。理由见 JSON 的 ``why_the_range_is_trimmed``。

    左：份额 +1 个百分点的**精确**降幅（13 个评测域的中位）。
    右：对数弹性 ``(p_i/L)·∂L/∂p_i`` 的同一个中位。
    13 条曲线共用一套配色，图例分两列。阴影是同一份额上 13 个评测域之间的四分位距（曲线本身是中位），
    alpha 压到 0.10 以免 13 条带叠成一片。不再逐条标注"转负"份额——13 条标注会互相压住，该值见报告 §5 的表。
    """

    sweep = payload["mixture_share_sweep"]
    anchor = sweep["anchor"]
    sources = list(sweep["sources"])
    colours = plt.get_cmap("tab20")(np.linspace(0, 1, len(sources)))

    # 上下布局：两个面板都以份额为横轴，故共用横轴、上panel 省略 x 标题与刻度。
    figure, (left, right) = plt.subplots(2, 1, figsize=STACKED_FIGSIZE,
                                         sharex=True, layout="constrained")
    for colour, source in zip(colours, sources):
        picked = [r for r in rows
                  if r["source_domain"] == source and r["anchor"] == anchor]
        shares = np.array(sorted({float(r["share"]) for r in picked}))
        benefit, elasticity = [], []
        benefit_lo, benefit_hi, elastic_lo, elastic_hi = [], [], [], []
        for share in shares:
            values = [float(r["exact_benefit_for_plus_1pp"]) for r in picked
                      if abs(float(r["share"]) - share) < 1e-12]
            elastics = [float(r["elasticity_dlnL_dln_share"]) for r in picked
                        if abs(float(r["share"]) - share) < 1e-12]
            benefit.append(np.median(values))
            elasticity.append(np.median(elastics))
            # 误差带 = 同一份额上 13 个评测域之间的四分位距（曲线是它们的中位）。
            benefit_lo.append(np.quantile(values, 0.25))
            benefit_hi.append(np.quantile(values, 0.75))
            elastic_lo.append(np.quantile(elastics, 0.25))
            elastic_hi.append(np.quantile(elastics, 0.75))
        left.fill_between(shares, benefit_lo, benefit_hi, color=colour, alpha=0.10,
                          linewidth=0)
        right.fill_between(shares, elastic_lo, elastic_hi, color=colour, alpha=0.10,
                           linewidth=0)
        left.plot(shares, benefit, color=colour, linewidth=2.5, label=source)
        right.plot(shares, elasticity, color=colour, linewidth=2.5, label=source)

    left.axhline(0, color="#666666", linewidth=0.9)
    right.axhline(0, color="#666666", linewidth=0.9)
    left.set_xscale("log")
    right.set_xscale("log")
    right.set_xlabel(f"来源域份额  $p_i$（{anchor}）")
    left.set_ylabel("边际效益（+1 个百分点的损失降幅）")
    right.set_ylabel("弹性  $(p_i/L)\\,\\partial L/\\partial p_i$")
    tidy(left, grid_axis="y")
    tidy(right, grid_axis="y")
    for axis in (left, right):
        axis.legend(loc="upper right", fontsize=8.5, ncols=2, frameon=True,
                    framealpha=0.92, edgecolor="#CCCCCC", handlelength=1.4,
                    columnspacing=0.9, labelspacing=0.3)
    return figure


def main() -> int:
    family = configure()
    payload = json.loads((INPUT / "marginal_utility.json").read_text(encoding="utf-8"))
    domain_rows = read_rows(INPUT / "scale_domain_detail.csv")
    mixture_rows = read_rows(INPUT / "mixture_direction_detail.csv")
    sweep_rows = read_rows(INPUT / "mixture_share_sweep.csv")
    figures = {
        # 本节交付：图 A、图 B（+ 配比份额扫描作为补充例图）。
        "figure_a_scale_mixture_sensitivity": figure_a_scale_mixture_sensitivity(
            payload, domain_rows, mixture_rows),
        "figure_b_quality_response": figure_b_quality_response(payload),
        "figure_mixture_share_sweep": mixture_marginal_curve(payload, sweep_rows),
        # 详细「来源 × 评估目标」热图留给下一小节。
        "figure_mixture_directional_gain": mixture_directional_gain(payload,
                                                                   mixture_rows),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"\nfont: {family}\noutput: {OUTPUT}")
    for name, figure in figures.items():
        path = OUTPUT / f"{name}.png"
        # 字体族要在**绘制时**生效，而绘制发生在 savefig 里，故在此套用。
        with plt.rc_context({"font.sans-serif": [family]}):
            figure.savefig(path, dpi=300)
        print(f"  wrote {path.relative_to(ROOT)}")
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
