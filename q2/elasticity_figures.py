#!/usr/bin/env python3
"""Scientific figures for the quality and mixture elasticity grid analysis."""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_analysis" / "quality_mixture_elasticity"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "data_analysis" / "Q2_to_Q3"))

import generalized_law as compact  # noqa: E402
from generalized_law_evaluator import GeneralizedLawEvaluator  # noqa: E402

ANCHORS = ("1M / 1B tokens", "60M / 1B tokens", "1B / 25B tokens")
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float:
    return float(value) if value not in (None, "") else np.nan


def register_cjk() -> str | None:
    """Register the bundled Chinese subset; return its family name or None."""

    if not BUNDLED_CJK.exists():
        return None
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(BUNDLED_CJK))
    return font_manager.FontProperties(fname=str(BUNDLED_CJK)).get_name()


def configure() -> None:
    # 中文用随脚本存放的子集字体兜底：DejaVu 在前（拉丁字形不变），CJK 子集在后
    # 供逐字回退。放在全局而不是 panel_style 里，是因为 matplotlib 在**绘制时**
    # 才解析字体，图例一类延迟创建的文字会读不到上下文里设的字体链。
    # 中文字体放在链首：matplotlib 的逐字回退在"第一个 family 存在"时就停住，
    # 把 DejaVu 放前面会让汉字落到 DejaVu 上变成方框。子集里带完整 ASCII/Latin-1，
    # 拉丁字形由 Noto Sans CJK 提供，整份图的排版仍然一致。
    family = register_cjk()
    chain = ([family] if family else []) + ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": chain,
        "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "savefig.facecolor": "white",
        "figure.facecolor": "white", "axes.grid": False,
    })


def quality_curves(rows: list[dict]) -> plt.Figure:
    """Q sweep over all observed D levels, plus N-dependent elasticities."""

    with panel_style():
        # 参考脚本的左右合成图画布是 12x4。
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), layout="constrained")
        left, right = axes
        ds = sorted({number(r["D_billion_tokens"]) for r in rows})
        for color, d in zip(COLORS, ds):
            selected = sorted((r for r in rows
                               if number(r["N_billion_parameters"]) == 1.0
                               and number(r["D_billion_tokens"]) == d),
                              key=lambda r: number(r["Q"]))
            left.plot([number(r["Q"]) for r in selected],
                      [number(r["B7_marginal_benefit_nats_per_unit_Q"])
                       for r in selected],
                      marker="o", markersize=3.3, linewidth=1.5, color=color,
                      label=f"D={d:g}B")
        left.set_xlabel("数据质量得分  Q")
        left.set_ylabel("质量边际收益  −∂L/∂Q")
        left.set_xlim(0.08, 1.02)
        left.grid(axis="y", color="#D8D8D8", linewidth=0.6)
        left.legend(ncol=2, loc="upper right", title="已观测 token 档位",
                    title_fontsize=11)

        for color, n in zip(COLORS[:3], (0.07, 1.0, 11.97)):
            selected = sorted((r for r in rows
                               if number(r["N_billion_parameters"]) == n
                               and number(r["D_billion_tokens"]) == 50.0),
                              key=lambda r: number(r["Q"]))
            right.plot([number(r["Q"]) for r in selected],
                       [-number(r["B7_loss_elasticity_wrt_Q"]) for r in selected],
                       marker="o", markersize=3.3, linewidth=1.5, color=color,
                       label=f"N={n:g}B")
        right.set_xlabel("数据质量得分  Q")
        right.set_ylabel("Q 提高 1% 的损失下降幅度（%）")
        right.set_xlim(0.08, 1.02)
        right.grid(axis="y", color="#D8D8D8", linewidth=0.6)
        right.legend(loc="upper right", title="D = 50B token", title_fontsize=11)
    return fig


def quality_heatmap(rows: list[dict]) -> plt.Figure:
    """One Q slice exposes the full N x D variation rather than one point."""

    ns = sorted({number(r["N_billion_parameters"]) for r in rows})
    ds = sorted({number(r["D_billion_tokens"]) for r in rows})
    values = np.full((len(ns), len(ds)), np.nan)
    for r in rows:
        if np.isclose(number(r["Q"]), 0.6):
            i = ns.index(number(r["N_billion_parameters"]))
            j = ds.index(number(r["D_billion_tokens"]))
            values[i, j] = -number(r["B7_loss_elasticity_wrt_Q"])
    if np.isnan(values).any():
        raise ValueError("Missing Q=0.6 cells in quality factorial grid")
    with panel_style():
        fig, ax = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
        image = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0.025,
                          vmax=0.065)
        ax.set_xticks(range(len(ds)), [f"{d:g}" for d in ds])
        ax.set_yticks(range(len(ns)), [f"{n:g}" for n in ns])
        ax.set_xlabel("训练 token 数  D（十亿）")
        ax.set_ylabel("参数量  N（十亿）")
        for i in range(len(ns)):
            for j in range(len(ds)):
                value = values[i, j]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                        fontsize=10,
                        color="white" if value > 0.048 else "#202020")
        bar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.04)
        bar.set_label("Q 提高 1% 的损失下降幅度（%）", fontsize=12)
        bar.ax.tick_params(labelsize=11)
    return fig


# 随脚本存放的简体中文子集字体（字形由 q2/figure_font_subset.py 生成）。
# 本机没有任何中文字体，不注册的话中文会渲染成方框。
BUNDLED_CJK = ROOT / "q2" / "assets" / "NotoSansSC-Regular-subset.otf"


@contextmanager
def panel_style():
    """Canvas and font sizes of ``classic_law_figures_zh.py``.

    Applied only around the figures that ask for them, so the remaining figures
    in this file keep their own (smaller) global rcParams. Chinese text needs the
    bundled subset font, so the family chain is replaced for the duration.
    """

    keys = {"axes.labelsize": 14, "axes.titlesize": 14, "xtick.labelsize": 12,
            "ytick.labelsize": 12, "legend.fontsize": 12,
            "axes.unicode_minus": False}
    previous = {key: plt.rcParams[key] for key in keys}
    plt.rcParams.update(keys)
    try:
        yield
    finally:
        plt.rcParams.update(previous)


# 配比扰动步长：1 个百分点 = 0.01 份额。
SHARE_STEP = 0.01
# 与 elasticity_analysis.py 一致的支持域筛选口径。
MIN_ACTIVE_SHARE = 0.05
MAX_ACTIVE_SHARE = 0.95


def predicted_gain_estimates(step: float = SHARE_STEP,
                             min_feasible: int = 10) -> list[dict]:
    """Fitted-law loss reduction for a ``step`` proportional upweight.

    Mirrors ``elasticity_analysis.finite_share_gain``: for every
    recorded mixture, raise source ``i`` by ``step`` and scale every other share
    down proportionally, then keep only the moves that stay inside the recorded
    support (active share window, recorded maximum share, recorded entropy
    interval). Reported values are percentiles over the feasible moves and are
    already signed as loss reduction (positive = the loss falls).
    """

    law = GeneralizedLawEvaluator.from_directory(ROOT / "data_analysis" / "Q2_to_Q3")
    blocks, train_names, _, _ = compact.load_data(compact.REGMIX, compact.HANDOFF)
    maximum = np.asarray([
        law.support()["recorded_max_share_per_domain"][name]
        for name in train_names])
    rows = []
    for anchor, block in zip(ANCHORS, (blocks[1], blocks[2], blocks[3])):
        p = block.p / block.p.sum(axis=1, keepdims=True)
        base = law.predict(block.n, block.d, p, warn_outside_support=False)
        for source, name in enumerate(train_names):
            candidate = p * (1 - step / np.maximum(1 - p[:, [source]], 1e-12))
            candidate[:, source] = p[:, source] + step
            allowed = ((p[:, source] >= MIN_ACTIVE_SHARE)
                       & (p[:, source] <= MAX_ACTIVE_SHARE)
                       & (candidate[:, source] <= maximum[source] + 1e-9)
                       & ~law.outside_support(candidate))
            if allowed.sum() < min_feasible:
                continue
            gain = (base[allowed]
                    - law.predict(block.n, block.d, candidate[allowed],
                                  warn_outside_support=False)).mean(axis=1)
            rows.append({
                "anchor": anchor, "source": name,
                "n_active": int(((p[:, source] >= MIN_ACTIVE_SHARE)
                                 & (p[:, source] <= MAX_ACTIVE_SHARE)).sum()),
                "n_total": int(len(p)),
                "n_feasible_plus_1pp": int(allowed.sum()),
                "median_exact_gain_nats_for_plus_1pp": float(np.percentile(gain, 50)),
                "p10_exact_gain_nats_for_plus_1pp": float(np.percentile(gain, 10)),
                "p90_exact_gain_nats_for_plus_1pp": float(np.percentile(gain, 90)),
            })
    return rows


def actual_gain_estimates(step: float = SHARE_STEP, bootstrap: int = 600,
                          seed: int = 20250924) -> list[dict]:
    """Data-side counterpart of the +step gains, from attachment A.

    For each anchor the observed 13-domain mean loss is linearly projected on the
    17 shares. With the simplex constraint, moving ``step`` of share into source
    ``i`` while scaling the others down proportionally has the directional
    derivative

        m_i = b_i - sum_{j != i} p_j b_j / (1 - p_i)

    and the reported loss reduction is ``-step * m_i`` (positive = loss falls).
    The 10th-90th percentile comes from a bootstrap over the recorded mixtures.
    This is a *linear projection* (R^2 ~ 0.3) and it is not robust to controlling
    for mixture diversity, so it is a descriptive estimate, not a causal ranking.

    Anchors with fewer than 100 recorded mixtures are skipped: 17 shares plus an
    intercept cannot be estimated stably from the 64 mixtures of the 1B anchor.
    """

    blocks, train_names, _, _ = compact.load_data(compact.REGMIX, compact.HANDOFF)
    # load_data returns (train_1m, test_1m, test_60m, test_1b) in SCALES order;
    # the test_* tables are the row sets the model-side figure reports on.
    anchor_blocks = [(ANCHORS[0], blocks[1]), (ANCHORS[1], blocks[2])]
    keep = list(range(1, len(train_names)))
    rng = np.random.default_rng(seed)
    rows = []
    for anchor, block in anchor_blocks:
        shares = block.p
        loss = block.y.mean(axis=1)
        count = len(shares)
        if count < 100:
            continue

        def coefficients(index: np.ndarray) -> np.ndarray:
            design = np.column_stack([np.ones(len(index)), shares[index][:, keep]])
            fitted, *_ = np.linalg.lstsq(design, loss[index], rcond=None)
            full = np.zeros(len(train_names))
            full[keep] = fitted[1:]
            return full

        def directional(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
            out = np.empty(len(train_names))
            for i in range(len(train_names)):
                out[i] = (estimate[i]
                          - float(np.delete(reference, i) @ np.delete(estimate, i))
                          / (1.0 - reference[i]))
            return out

        draws = np.empty((bootstrap, len(train_names)))
        for index in range(bootstrap):
            resampled = rng.integers(0, count, count)
            draws[index] = directional(coefficients(resampled),
                                       shares[resampled].mean(axis=0))
        draws *= -step                                  # -> loss reduction
        for i, source in enumerate(train_names):
            # 点估计与区间取自同一个 bootstrap 分布，保证 p10 <= 中位 <= p90。
            rows.append({
                "anchor": anchor, "source": source, "n_mixtures": count,
                "median_exact_gain_nats_for_plus_1pp":
                    float(np.percentile(draws[:, i], 50)),
                "p10_exact_gain_nats_for_plus_1pp":
                    float(np.percentile(draws[:, i], 10)),
                "p90_exact_gain_nats_for_plus_1pp":
                    float(np.percentile(draws[:, i], 90)),
            })
    return rows


def mixture_gain_intervals(predicted: list[dict],
                           actual: list[dict] | None = None,
                           step: float = SHARE_STEP) -> plt.Figure:
    """Predicted and data-side loss reduction for +``step`` of a source's share.

    Two stacked panels share the source axis. Top: the fitted generalized law,
    one dodged series per compute anchor. Bottom: the same quantity estimated
    from the recorded losses of attachment A. Positive always means the loss
    falls, matching the sign convention of the source tables.
    """

    sources = [name for name, r in
               ((r["source"], r) for r in predicted if r["anchor"] == ANCHORS[0])
               if int(r["n_feasible_plus_1pp"]) >= 10]
    sources.sort(key=lambda name: next(
        number(r["median_exact_gain_nats_for_plus_1pp"]) for r in predicted
        if r["anchor"] == ANCHORS[0] and r["source"] == name), reverse=True)
    by_key = {(r["anchor"], r["source"]): r for r in predicted}
    actual = actual or []
    actual_by_key = {(r["anchor"], r["source"]): r for r in actual}
    actual_anchors = [a for a in ANCHORS if any(key[0] == a for key in actual_by_key)]

    with panel_style():
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(6.0, 4.0), sharex=True, layout="constrained")
        x = np.arange(len(sources))
        offsets = np.linspace(-0.26, 0.26, len(ANCHORS))

        def draw(axis, table, anchors, filled):
            plotted = []
            for anchor, color, offset in zip(anchors, COLORS, offsets):
                xs, mids, lows, highs = [], [], [], []
                for i, source in enumerate(sources):
                    r = table.get((anchor, source))
                    if r is None or int(r.get("n_feasible_plus_1pp", 10)) < 10:
                        continue
                    low = number(r["p10_exact_gain_nats_for_plus_1pp"])
                    high = number(r["p90_exact_gain_nats_for_plus_1pp"])
                    mid = number(r["median_exact_gain_nats_for_plus_1pp"])
                    xs.append(i + offset)
                    mids.append(mid)
                    lows.append(mid - low)
                    highs.append(high - mid)
                    plotted.extend((low, high))
                axis.errorbar(xs, mids, yerr=[lows, highs],
                              fmt="o" if filled else "s", color=color,
                              markersize=4.5, elinewidth=1.7, capsize=0,
                              linewidth=0,
                              markerfacecolor=color if filled else "white",
                              markeredgewidth=0 if filled else 1.2,
                              label=anchor)
            axis.axhline(0, color="#444444", linewidth=0.9)
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
            if plotted:
                extent = 1.12 * max(abs(min(plotted)), abs(max(plotted)))
                axis.set_ylim(-extent, extent)

        draw(top, by_key, ANCHORS, filled=True)
        draw(bottom, actual_by_key, actual_anchors, filled=False)

        # 单位已由面板内的横排标签给出，这里只留面板身份，避免重复。
        top.set_ylabel("预测")
        bottom.set_ylabel("实测")
        bottom.set_xticks(x)
        bottom.set_xticklabels(sources, rotation=30, ha="right",
                               rotation_mode="anchor")
        # 30 度时相邻标签的垂直净距只有约 11.8 pt，12 pt 的 tick 字号会压字，
        # 这里单独降到 10 pt（其余面板仍用参考脚本的 12 pt）。
        bottom.tick_params(axis="x", labelsize=10)
        bottom.set_xlim(-0.7, len(sources) - 0.3)
        bottom.set_xlabel("训练来源")
        top.tick_params(labelbottom=False)
        top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3,
                   handletextpad=0.4, columnspacing=1.2)

        points = int(round(step * 100))
        # 纵轴数值就是附件原始 val_loss 列的量纲（原始表未声明单位，全项目统一按
        # 「验证集交叉熵损失」标注），因此不写单位。
        top.text(0.98, 0.96, f"预测损失下降（份额 +{points} 个百分点）",
                 transform=top.transAxes, ha="right", va="top", fontsize=11)
        bottom.text(0.98, 0.96, f"实测损失下降（份额 +{points} 个百分点）",
                    transform=bottom.transAxes, ha="right", va="top", fontsize=11)
    return fig


def mixture_quantile_slices(grid: list[dict], summary: list[dict]) -> plt.Figure:
    """Show local elasticity at five observed share quantiles for four sources."""

    rows = [r for r in summary if r["anchor"] == ANCHORS[0]
            and int(r["n_feasible_plus_1pp"]) >= 10]
    ranked = sorted(rows, key=lambda r: number(r["median_exact_gain_nats_for_plus_1pp"]))
    candidates = [ranked[0]["source"], ranked[len(ranked) // 2]["source"],
                  ranked[-2]["source"], ranked[-1]["source"]]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.15,
                        wspace=0.25, hspace=0.4)
    for ax, source, color in zip(axes.flat, candidates, COLORS):
        selected = sorted((r for r in grid if r["anchor"] == ANCHORS[0]
                           and r["source"] == source),
                          key=lambda r: number(r["p_i"]))
        x = [100 * number(r["p_i"]) for r in selected]
        y = [number(r["mean_loss_elasticity_relative_share"]) for r in selected]
        ax.scatter(x, y, s=46, color=color, zorder=3)
        for r, xv, yv in zip(selected, x, y):
            quantile = number(r["share_quantile"])
            if any(np.isclose(quantile, v) for v in (0.1, 0.5, 0.9)):
                offset = (-12, 8) if np.isclose(quantile, 0.1) else (4, 5)
                ax.annotate(f"{100*quantile:g}%", (xv, yv),
                            xytext=offset, textcoords="offset points", fontsize=7)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_title(source, fontsize=10)
        ax.set_xlabel("Observed source share (%)")
        ax.set_ylabel("Local loss elasticity")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.text(0.1, 0.035,
             "Five actual 1M mixtures at share quantiles (10/25/50/75/90%). "
             "Other shares also differ, so these points are not a one-variable sweep.",
             fontsize=8, ha="left")
    return fig


def representative_examples() -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, tuple[str, ...], list[int]]:
    """Three real 1M mixtures ordered by observed composition entropy."""

    law = GeneralizedLawEvaluator.from_directory(ROOT / "data_analysis" / "Q2_to_Q3")
    blocks, train_names, _, _ = compact.load_data(compact.REGMIX, compact.HANDOFF)
    block = next(b for b in blocks if b.name == "test_1m")
    p = block.p / block.p.sum(axis=1, keepdims=True)
    entropy = law.entropy(p)
    order = np.argsort(entropy)
    indices = [int(order[round(q * (len(order) - 1))]) for q in (0.1, 0.5, 0.9)]
    selected = p[indices]
    predicted = law.predict(block.n, block.d, selected,
                            warn_outside_support=False).mean(axis=1)
    observed = block.y[indices].mean(axis=1)
    top = np.argsort(-selected.sum(axis=0))[:6]
    return selected, entropy[indices], predicted, observed, train_names, list(top)


def representative_pies() -> plt.Figure:
    """Three observed mixtures; pies encode composition, not marginal benefit."""

    selected, entropy, _, _, train_names, top = representative_examples()
    names = [train_names[i] for i in top] + ["其他来源"]
    palette = [plt.get_cmap("tab10")(i) for i in range(6)] + ["#BDBDBD"]
    with panel_style():
        # 参考脚本的左右合成图画布是 12x4；三联饼图沿用同一画布。
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), layout="constrained")
        for ax, shares, h, label in zip(axes, selected, entropy,
                                        ("低熵配比", "中位熵配比", "高熵配比")):
            values = list(shares[top]) + [float(1 - shares[top].sum())]
            ax.pie(values, colors=palette, startangle=90,
                   wedgeprops={"width": 0.43, "edgecolor": "white",
                               "linewidth": 0.8},
                   autopct=lambda v: f"{v:.0f}%" if v >= 10 else "",
                   pctdistance=0.78, textprops={"fontsize": 10})
            ax.text(0, 0, f"H={h:.2f}", ha="center", va="center", fontsize=12)
            ax.set_title(label, fontsize=14)
        handles = [Patch(facecolor=color, label=name)
                   for color, name in zip(palette, names)]
        fig.legend(handles=handles, loc="outside lower center", ncol=4,
                   frameon=False, fontsize=12)
    return fig


def _darken(color: object, factor: float = 0.72) -> tuple[float, float, float]:
    return tuple(np.asarray(to_rgb(color)) * factor)


def _draw_loss_cylinder(ax: plt.Axes, values: list[float],
                        colors: list[object], height: float) -> None:
    """Extrude a pie to a zero-based height proportional to mixture-level loss."""

    radius = 0.95
    angle = np.pi / 2
    # 相机方向（与 view_init 一致）。mplot3d 不做跨 collection 的深度排序，
    # 所以侧壁必须自己剔除背面：否则一个扇形绕到背后那半圈也会被画出来，
    # 叠在前壁上形成"里外两层"的错色带（最左那根柱子上的浅蓝块就是这么来的）。
    elevation, azimuth = 28.0, -62.0
    view = np.array([np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)),
                     np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth)),
                     np.sin(np.radians(elevation))])
    for fraction, color in zip(values, colors):
        if fraction <= 0:
            continue
        end = angle + 2 * np.pi * fraction
        angles = np.linspace(angle, end, max(4, int(np.ceil(120 * fraction))))
        xy = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
        dark = _darken(color)
        sides = []
        for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
            normal = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0])
            norm = float(np.linalg.norm(normal))
            if norm == 0 or float(normal @ view) / norm <= 0.0:
                continue                      # 背面，跳过
            sides.append([(x0, y0, 0), (x1, y1, 0),
                          (x1, y1, height), (x0, y0, height)])
        if sides:
            # edgecolors 与面色相同：相邻薄片之间不留抗锯齿缝（原来 edgecolors="none"
            # 会让侧壁出现细竖条纹）。
            ax.add_collection3d(Poly3DCollection(
                sides, facecolors=dark, edgecolors=dark, linewidths=0.4))
        top = [(0, 0, height)] + [(x, y, height) for x, y in xy]
        ax.add_collection3d(Poly3DCollection(
            [top], facecolors=color, edgecolors="white", linewidths=0.45))
        angle = end
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(0, 6.0)
    ax.set_box_aspect((2.3, 2.3, 3.3))
    ax.view_init(elev=28, azim=-62)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([0, 2, 4, 6])
    ax.tick_params(axis="z", labelsize=11, pad=0)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")


def representative_loss_cylinders() -> plt.Figure:
    """Top sectors show p; cylinder height shows whole-mixture predicted loss."""

    selected, entropy, predicted, observed, train_names, top = representative_examples()
    names = [train_names[i] for i in top] + ["其他来源"]
    with panel_style():
        colors = [plt.get_cmap("tab10")(i) for i in range(6)] + ["#BDBDBD"]
        # 9:4（三幅并排）。保存时 bbox_inches="tight" 还会再加 pad_inches=0.1，
        # 而下面那个整画布矩形让紧包围盒恒等于画布，所以画布本身取 8.8x3.8，
        # 输出正好 9.0x4.0 in。
        fig = plt.figure(figsize=(8.8, 3.8))
        axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
        fig.subplots_adjust(left=0.005, right=0.995, top=0.87, bottom=0.38,
                            wspace=0.02)
        baseline = float(predicted[0])
        observed_baseline = float(observed[0])
        for index, (ax, shares, h, model_loss, measured_loss, label) in enumerate(
                zip(axes, selected, entropy, predicted, observed,
                    ("低熵配比", "中位熵配比", "高熵配比"))):
            values = list(shares[top]) + [float(1 - shares[top].sum())]
            _draw_loss_cylinder(ax, values, colors, float(model_loss))
            gain = baseline - model_loss
            observed_gain = observed_baseline - measured_loss
            ax.set_title(f"{label}（H={h:.2f}）", fontsize=14, pad=2)
            fig.text((index + 0.5) / 3, 0.235,
                     f"预测 L = {model_loss:.2f}　实测 L = {measured_loss:.2f}\n"
                     f"相对低熵的收益　预测 {gain:+.2f}　实测 {observed_gain:+.2f}",
                     ha="center", va="center", fontsize=11)
        handles = [Patch(facecolor=color, label=name)
                   for color, name in zip(colors, names)]
        # 9 in 宽放不下 7 项一行（会撑破画布、bbox tight 再把宽度放大），
        # 所以排成两行；行高与下面的数值说明留出间隔。
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                   fontsize=12, bbox_to_anchor=(0.5, 0.005),
                   columnspacing=1.2, handletextpad=0.4)
        # fig.text(0.01, 0.985,
        #          "扇区 = 训练来源份额；柱高 = 该配比在 N=1M、D=1B token 下的预测平均"
        #          "验证损失（所有柱子共用同一零点）。",
        #          va="top", ha="left", fontsize=10)
        # 保存用的是 bbox_inches="tight"：紧包围盒随内容增减而变（上面的说明被注释掉
        # 之后输出高度就少了 60 多像素），9:4 会守不住。放一个整画布的不可见矩形，
        # 让紧包围盒恒等于画布本身。
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 facecolor="none", edgecolor="none"))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", default="png,svg")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    configure()
    out = args.output_dir or args.input_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    quality = read_csv(args.input_dir / "quality_factorial_grid.csv")
    mixture = read_csv(args.input_dir / "mixture_source_summary.csv")
    p_grid = read_csv(args.input_dir / "mixture_share_quantile_grid.csv")
    figures = {
        "quality_curves": quality_curves(quality),
        "quality_heatmap": quality_heatmap(quality),
        "mixture_gain_intervals": mixture_gain_intervals(
            predicted_gain_estimates(), actual_gain_estimates()),
        "mixture_quantile_slices": mixture_quantile_slices(p_grid, mixture),
        "representative_mixtures": representative_pies(),
        "representative_mixtures_loss_cylinders": representative_loss_cylinders(),
    }
    for name, fig in figures.items():
        for extension in args.formats.split(","):
            extension = extension.strip().lstrip(".")
            path = out / f"{name}.{extension}"
            fig.savefig(path, dpi=args.dpi if extension == "png" else None,
                        bbox_inches="tight")
            print(f"wrote {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
