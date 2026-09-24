#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单独输出论文用图（单图统一 6×4 英寸，左右合成图 12×4，标注统一中文）。

八张图分别对应原组合图里的一个面板：

``extra-a``
    ``traditional_scaling_law_extrapolation`` 的 (a)：经典标度律在 10B 以上
    参数规模的外推曲线与 B10 估算点。

``steps-b``
    ``traditional_scaling_law_loss_vs_steps_law`` 的 (b)：把横轴换成计算量后，
    8 条训练曲线与一条托住它们右下边界的直线（原黑色阶梯状前沿线已移除）。

``steps-b-b2``
    同上，但数据源换成附件 B2（Cerebras 训练日志），7 个模型规模。

``validation-a``
    ``traditional_scaling_law_validation`` 的 (a)：归一化尺度下多来源观测与
    经典标度律预测的对照。

``selection-a``
    ``traditional_scaling_law_selection`` 的 (a)：按当前的自适应结构早停策略
    重绘的参数释放阶梯。

``quad-a``
    ``traditional_scaling_law`` 的 (a)：固定 N 的数据标度。

``quad-b``
    ``traditional_scaling_law`` 的 (b)：固定 D 的参数标度。

``quad-ab``
    把 ``quad-a`` 与 ``quad-b`` 按上下布局合成一张（画布 6×4，(a) 在上、(b) 在下）。

两张四联图面板已拆成独立单图，画布与其余单图一致（均为 6×4）。

中文字形由脚本自带的子集字体提供（``assets/NotoSansSC-Regular-subset.otf`` 与
``...-Bold-subset.otf``，合计约 0.3 MB，启动时自动注册进 matplotlib），因此
本机无需预装中文字体。两个字重都注册，加粗标题才不会静默降级为常规体。

子集只覆盖**当前文案用到的字形**：改动图内文字后若新增了汉字，脚本会在启动时
报错并列出缺失字形，而不是画出一堆方框。此时重跑
``python3 q2/figure_font_subset.py`` 重新生成子集即可。

用法::

    python3 q2/classic_law_figures_zh.py
    python3 q2/classic_law_figures_zh.py --figures extra-a,validation-a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classic_law_figures_en as base  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 随脚本一起存放的简体中文子集字体（只覆盖本文件用到的字形，合计约 0.3 MB）。
# 本机通常没有任何中文字体，缺字体时 PNG 会渲染成方框，因此这里直接自带。
# 两个字重都要注册：只带 Regular 时 matplotlib 找不到粗体，会把加粗中文
# 悄悄降级为常规体（并打印 findfont 警告），图上的层级就分不出来了。
# 由 q2/figure_font_subset.py 生成，可复现。
BUNDLED_FONTS = (
    Path(__file__).resolve().parent / "assets" / "NotoSansSC-Regular-subset.otf",
    Path(__file__).resolve().parent / "assets" / "NotoSansSC-Bold-subset.otf",
)
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT / "data_analysis" / "traditional_scaling_law"
)
DEFAULT_FULL_DIR = PROJECT_ROOT / "data_analysis" / "scaling_law_full"

# 统一画布：所有单图都用这个尺寸。
PANEL_FIGSIZE = (6.0, 4.0)
# 四联图拆出的 (a)(b) 单图，与其余单图同尺寸。
SIG_FIGSIZE = (6.0, 4.0)
# 上下合成图两格之间的相对间距（gridspec hspace，相对单格高度）。
# 要能容纳上格的横轴标题，由 _assert_row_gap 核对，不要凭感觉调小。
HSPACE = 0.62

# 统一柱色：凡是「同一系列的柱」都用同一色，避免读者以为颜色有含义。
# 注：改画逐层释放流程图后，本文件已没有柱状图，这三个常量目前无引用，
# 保留供后续加柱状图时沿用同一套配色。
BAR_COLOR = "#4C78A8"
BAR_COLOR_ALT = "#7BA7CC"
BAR_COLOR_MUTED = "#BFBFBF"

# 中文字体回退链（按常见度排序，含开源与商用字体）。
CJK_FONTS = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Droid Sans Fallback",
    "Arial Unicode MS")


def configure_chinese() -> tuple[str, bool]:
    """配置中文字体；返回 (实际字体名, 是否真的是中文字体)。

    优先使用脚本自带的子集字体，其次找系统里已装的常见中文字体。两者都没有时
    才退化为 DejaVu，并在调用方打印警告。
    """

    registered = 0
    bundled_name: str | None = None
    for font_path in BUNDLED_FONTS:
        if not font_path.exists():
            continue
        font_manager.fontManager.addfont(str(font_path))
        registered += 1
        if bundled_name is None:
            bundled_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    # 只带 Regular 时加粗中文会被静默降级，图上的层级就没了——明确报出来。
    if registered == 1:
        print(
            "警告：只找到 1 个字重的自带字体，加粗中文将降级为常规体。\n"
            "      请运行 python3 q2/figure_font_subset.py 生成两个字重。",
            file=sys.stderr)

    available = {item.name for item in font_manager.fontManager.ttflist}
    matched = bundled_name if bundled_name in available else None
    if matched is None:
        matched = next((name for name in CJK_FONTS if name in available), None)
    chain = [matched] if matched else []
    chain += list(CJK_FONTS) + ["DejaVu Sans"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": chain,
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
            "grid.alpha": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return (matched or "DejaVu Sans"), matched is not None


def assert_legend_clear(axis: plt.Axes, *, max_hits: int = 0) -> None:
    """核对坐标轴内的图例没有压住曲线或数据点。

    图例遮挡是这类图最常见的缺陷，而它完全取决于数据落点——挪一次数据或改一次
    轴范围就可能重新压上，靠肉眼每次核对不可靠。这里把图例的窗口包围盒与
    **曲线采样点、散点**逐一比对，落在框内的点数超过 ``max_hits`` 就报错。

    只统计坐标轴内的图例：放到轴外的图例不可能压住轴内数据。
    """

    legend = axis.get_legend()
    if legend is None:
        return
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    box = legend.get_window_extent(renderer=renderer)

    def inside(points) -> int:
        if len(points) == 0:
            return 0
        display = axis.transData.transform(np.asarray(points))
        return int(np.sum(
            (display[:, 0] >= box.x0) & (display[:, 0] <= box.x1)
            & (display[:, 1] >= box.y0) & (display[:, 1] <= box.y1)))

    hits = 0
    worst = ""
    for line in axis.lines:
        x_data, y_data = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        if x_data.size < 2:
            continue
        # 曲线在相邻采样点之间是直线，细采样后再判，避免漏掉穿过图例的线段
        dense_x, dense_y = [], []
        for index in range(x_data.size - 1):
            t = np.linspace(0.0, 1.0, 16)
            dense_x.append(x_data[index] + (x_data[index + 1] - x_data[index]) * t)
            dense_y.append(y_data[index] + (y_data[index + 1] - y_data[index]) * t)
        if not dense_x:
            continue
        count = inside(np.column_stack([np.concatenate(dense_x),
                                        np.concatenate(dense_y)]))
        if count:
            worst = f"曲线「{line.get_label()}」{count} 点"
        hits += count
    for collection in axis.collections:
        offsets = collection.get_offsets()
        if len(offsets):
            count = inside(np.asarray(offsets))
            if count:
                worst = f"散点 {count} 点"
            hits += count

    if hits > max_hits:
        raise RuntimeError(
            f"图例压住了数据：{hits} 个采样点落在图例框内（{worst}）。\n"
            f"请把图例换到空白区，或缩小图例（改标签/字号）。"
        )

    # 图例与轴内注释框（公式框、结论框等）也会互相压住——这类冲突不涉及数据，
    # 只查数据是查不出来的，所以单独再比一遍矩形的交叠面积。
    for item in axis.texts:
        if not item.get_text().strip():
            continue
        other = item.get_window_extent(renderer=renderer)
        overlap_w = min(box.x1, other.x1) - max(box.x0, other.x0)
        overlap_h = min(box.y1, other.y1) - max(box.y0, other.y0)
        if overlap_w > 0 and overlap_h > 0:
            raise RuntimeError(
                f"图例与轴内注释框重叠 {overlap_w:.0f}×{overlap_h:.0f} 像素："
                f"「{item.get_text()[:40]}」。\n请把两者分开到不同空白区。"
            )


def assert_glyphs_available(font_paths: tuple[Path, ...]) -> None:
    """核对自带子集字体覆盖了本文件全部字符串字面量里的非 ASCII 字形。

    子集字体是为了避免方框而自带的，但它只覆盖生成时的文案。改动图内文字后
    新增的汉字不会有字形——那种情况 matplotlib 只会安静地画出空白/方框，
    所以这里在开画之前先查一遍，缺字形就报错并指出该重跑哪个脚本。
    """

    import ast as _ast

    from fontTools.ttLib import TTFont

    # 与生成脚本共用同一份「运行时符号」集合，避免两边口径不一致导致漏字形。
    import figure_font_subset as font_builder

    source = Path(__file__).read_text(encoding="utf-8")
    needed: set[str] = set(font_builder.RUNTIME_SYMBOLS)
    for node in _ast.walk(_ast.parse(source)):
        if isinstance(node, _ast.Constant) and isinstance(node.value, str):
            needed.update(node.value)
        elif isinstance(node, _ast.JoinedStr):
            for piece in node.values:
                if isinstance(piece, _ast.Constant) and isinstance(piece.value, str):
                    needed.update(piece.value)
    needed = {c for c in needed if c.isprintable() and ord(c) > 0x7F}
    if not needed:
        return

    covered: set[int] = set()
    for path in font_paths:
        if not path.exists():
            continue
        font = TTFont(str(path))
        for table in font["cmap"].tables:
            covered.update(table.cmap.keys())

    missing = sorted(c for c in needed if ord(c) not in covered)
    if missing:
        raise RuntimeError(
            f"自带子集字体缺 {len(missing)} 个字形：{''.join(missing)}\n"
            f"请重新运行 python3 q2/figure_font_subset.py 生成子集。"
        )


def style_axis(axis: plt.Axes, grid_axis: str = "both") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, axis=grid_axis, which="major")
    axis.set_axisbelow(True)


def _assert_text_fits(figure: plt.Figure, *, margin: float = 0.01) -> None:
    """检查图级文字（``figure.text``）有没有横向溢出画布。

    这类溢出不报错、不警告，只会安静地把文字裁掉——``savefig`` 的 tight 包围盒
    只统计坐标轴与图形元素，管不到画布外的手工文字。所以这里用渲染器的真实
    包围盒做确定性检查，超界直接抛错而不是输出一张缺字的图。

    ``margin`` 是允许的贴边余量（英寸），小于它会视作过挤。
    """

    renderer = figure.canvas.get_renderer()
    canvas_width = figure.get_size_inches()[0]
    for item in figure.texts:
        if not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer=renderer)
        left = box.x0 / figure.dpi
        right = box.x1 / figure.dpi
        if left < margin or right > canvas_width - margin:
            raise RuntimeError(
                f"图级文字横向溢出画布：占用 {left:.2f}–{right:.2f} 英寸，"
                f"画布宽 {canvas_width:.2f} 英寸。\n文字：{item.get_text()[:80]}"
            )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #
# 图 1：extrapolation 的 (a)
# --------------------------------------------------------------------------- #
def figure_extrapolation_a() -> plt.Figure:
    """经典标度律在 10B 以上参数规模的外推。"""

    rows = _read_csv_rows(DEFAULT_FULL_DIR / "large_model_extrapolation.csv")
    n_values = np.asarray([float(r["N_params_B"]) for r in rows])
    d_values = np.asarray([float(r["D_tokens_B"]) for r in rows])
    estimates = np.asarray([float(r["B10_estimated_loss"]) for r in rows])

    import json

    payload = json.loads(
        (DEFAULT_FULL_DIR / "scaling_law_full_results.json").read_text(
            encoding="utf-8"
        )
    )
    parameters = payload["B1_main_fit"]["parameters"]

    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE)
    axis.scatter(
        n_values, estimates, s=34, facecolors="none", edgecolors="#555555",
        linewidths=1, zorder=4,
        label=f"大模型样本估算损失（{n_values.size} 个模型，$N\\geq$10B）")

    reference_d = float(np.median(d_values))
    n_curve = np.geomspace(n_values.min() * 0.8, n_values.max() * 1.2, 400)
    curve = (
        float(parameters["irreducible_loss"])
        + float(parameters["parameter_coefficient"])
        * np.power(n_curve, -float(parameters["parameter_exponent"]))
        + float(parameters["data_coefficient"])
        * np.power(reference_d, -float(parameters["data_exponent"]))
    )
    axis.plot(
        n_curve, curve, color="#1F4E79", linewidth=4, zorder=5,
        label=f"经典标度律外推（中位 $D$ = {base._format_location(reference_d)}）")
    axis.set_xscale("log")
    axis.set_xlabel("模型参数量 $N$（B）")
    axis.set_ylabel("验证集交叉熵损失")
    # 图例放到坐标轴外：图内左上角是高空散点最密的地方，压上去必然遮数据。
    # （字号由 rcParams 的 legend.fontsize=12 统一给，不再逐处写死。）
    axis.legend(loc="upper center", ncol=1)
    style_axis(axis)
    # axis.text(
    #     0.97, 0.60,
    #     "每个 B10 点对应各自的 $D$，\n曲线只画在中位 $D$ 处作参照。",
    #     transform=axis.transAxes, color="#1F4E79",
    #     va="top", ha="right",
    #     bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
    #           "edgecolor": "#1F4E79", "linewidth": 0.9, "alpha": 0.95},
    # )
    # figure.subplots_adjust(bottom=0.28)
    return figure


# --------------------------------------------------------------------------- #
# 图 2：steps-law 的 (b) —— 去掉阶梯线，改为包住右上边界的直线
# --------------------------------------------------------------------------- #
def envelope_line(
    x_values: np.ndarray, y_values: np.ndarray, headroom: float = 0.0
) -> tuple[float, float, np.ndarray]:
    """在 log-log 平面求包住整族曲线**下沿**的直线。

    下沿 = 每个计算量档位上的最优（最小）损失，也就是原图那条阶梯状前沿所表示的
    量：预算不超过 C 时所达到的最低损失。对 Pythia 这批曲线来说，最大模型
    在每个档位上都最优，所以下沿就是它的轨迹。

    做法：

    1. 按计算量排序，取**运行最小值**作为下沿；
    2. 斜率用一维数值搜索：对每个候选斜率，把截距压到「直线位于全部点之下」的
       最大值，再取「线到下沿的最大间隙」最小的那个斜率。用最大间隙而不是中位
       间隙作为目标，是因为下沿在 log-log 下并不严格是直线——按中位拟合会让线
       在一端贴死、另一端张开（实测最大间隙 0.125 log10），按最大间隙拟合则让
       线均匀贴着下沿（最大间隙降到 0.051）；
    3. 不留余量（``headroom=0``），直线正好与下沿相切于一点；这样它既压住全部
       曲线，又不会有肉眼可见的悬浮。

    返回 ``(斜率, 截距, 下沿点)``，三者坐标均为 log10。
    """

    order = np.argsort(x_values)
    sorted_log_x = np.log10(np.maximum(x_values[order], 1e-12))
    sorted_log_y = np.log10(y_values[order])
    running_min = np.minimum.accumulate(sorted_log_y)

    # 一维搜索斜率：截距取「全部点都不低于直线」的上界（即最大残差的反向），
    # 使直线成为整族曲线的下界，同时尽可能贴近下沿。
    best_slope = None
    best_gap = np.inf
    for candidate in np.linspace(-0.30, 0.02, 641):
        # 让所有点都在直线上方：intercept <= log_y - slope*log_x 对每个点成立
        candidate_intercept = float(
            np.min(sorted_log_y - candidate * sorted_log_x)
        ) - headroom
        gap = running_min - (candidate_intercept + candidate * sorted_log_x)
        # 目标：最小化线到下沿的最大间隙，使整条线均匀贴合
        worst_gap = float(np.max(gap))
        if worst_gap < best_gap:
            best_gap = worst_gap
            best_slope = float(candidate)
    if best_slope is None:
        raise RuntimeError("斜率搜索失败。")

    intercept = float(np.min(sorted_log_y - best_slope * sorted_log_x)) - headroom
    return best_slope, intercept, np.column_stack([sorted_log_x, running_min])


def _steps_law_axis(
    log: base.TrainingLog, axis: plt.Axes, *, label: str
) -> tuple[float, float]:
    """在给定坐标轴上画曲线族与下沿包络直线，返回 (斜率, 截距)。"""

    losses = log.loss("val_loss")
    compute = log.flops_1e21
    unique_n = np.unique(log.n_params_b)
    cmap = plt.get_cmap("viridis")
    norm = LogNorm(vmin=float(unique_n.min()), vmax=float(unique_n.max()))

    for n_value in unique_n:
        mask = np.isclose(log.n_params_b, n_value, rtol=0.0, atol=1e-12)
        order = np.argsort(compute[mask])
        axis.plot(
            compute[mask][order], losses[mask][order],
            color=cmap(norm(n_value)), linewidth=1.8, alpha=0.9, zorder=3)

    slope, intercept, _envelope = envelope_line(compute, losses)
    x_line = np.geomspace(compute.min() * 0.95, compute.max() * 1.15, 200)
    y_line = np.power(10.0, intercept) * np.power(x_line, slope)
    axis.plot(
        x_line, y_line, color="#111111", linewidth=2.2, linestyle="--",
        zorder=6, label="下沿包络直线")

    axis.set_xscale("log")
    # axis.set_yscale("log")
    # 纵轴同时容纳曲线族与直线（直线在右端会落到曲线下方）。
    axis.set_ylim(
        min(float(losses.min()) * 0.92, float(y_line.min()) * 0.92),
        float(losses.max()) * 1.10)
    axis.set_xlabel("计算量 $C$（$10^{21}$ FLOPs）")
    axis.set_ylabel("验证集交叉熵损失")
    style_axis(axis)

    size_handles = [
        Line2D([0], [0], color=cmap(norm(v)), linewidth=1.8,
               label=base._format_size(float(v)))
        for v in unique_n
    ]
    # 图例位置由实测决定，不要凭观感挪：曲线族只占画布中段，包络直线则横贯整个
    # 横轴的左下部分，因此轴内没有横贯全宽的空白带。穷举后只有右上角是零遮挡：
    # 8 条曲线里最高的一条（12B）在框左边界处仍低于框底，最靠右的一条（71M）
    # 最高只到 x≈0.68 的轴位置，够不着框。左上角会压住 257 个曲线采样点。
    # 改数据或轴范围后请重跑 assert_legend_clear 复验。
    size_legend = axis.legend(
        handles=size_handles, title="参数量 $N$", loc="upper right",
        frameon=True, ncol=2, handlelength=1.5)
    size_legend.get_frame().set_facecolor("white")
    size_legend.get_frame().set_edgecolor("#CCCCCC")
    size_legend.get_frame().set_alpha(0.92)
    axis.add_artist(size_legend)

    # 公式框放左下：那里在包络直线之下、曲线族之外，实测 0 个采样点（B1 与 B2
    # 都是）。原先放右上会和图例抢同一块地方——图例挪到右上后两者必然相撞。
    axis.text(
        0.03, 0.03,
        f"下沿包络直线：$L = {np.power(10.0, intercept):.4g}\\,C^{{{slope:.4f}}}$\n"
        f"（$C$ 以 $10^{{21}}$ FLOPs 计，$C=6ND$）",
        transform=axis.transAxes, color="#111111",
        ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": "#111111", "linewidth": 0.9, "alpha": 0.95})
    # 注意：图例遮挡必须在 tight_layout 之后校验（调用方负责）。这里轴还是
    # subplots 的默认宽度，图例按它定位；tight_layout 拉宽轴之后图例位置会变。
    return slope, intercept


def figure_steps_b() -> plt.Figure:
    """B1：计算量横轴下的 8 条训练曲线与下沿包络直线。"""

    log = base.load_training_log(base.DEFAULT_TRAINING_LOG)
    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE)
    _steps_law_axis(log, axis, label="B1")
    figure.tight_layout()
    assert_legend_clear(axis)
    return figure


def figure_steps_b_b2() -> plt.Figure:
    """B2：计算量横轴下的 7 条训练曲线与下沿包络直线。

    B2（Cerebras）与 B1 结构相同：7 个模型规模 × 147 个检查点，字段一致，
    累计计算量同样满足 ``C = 6ND``。区别在于 B2 的计算量跨度大得多
    （约 1.7×10^4 倍，B1 为 2×10^5 倍但下限更低），且损失整体更高。
    """

    log = base.load_training_log(base.DEFAULT_B2_TRAINING_LOG)
    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE)
    _steps_law_axis(log, axis, label="B2")
    figure.tight_layout()
    assert_legend_clear(axis)
    return figure



# --------------------------------------------------------------------------- #
# 图 3：selection 的 (a) —— 按当前自适应结构早停策略重绘
# --------------------------------------------------------------------------- #
def figure_selection_a() -> plt.Figure:
    """按论文正文描述的逐层释放策略绘制早停结构。

    正文口径（三层、由简到繁、以外部验证收益为准）：

    * 第 1 层：令 ``alpha = beta = s``，对共享指数 ``s`` 做一维搜索；
    * 第 2 层：以第 1 层最优值 ``s*`` 为锚点，并列考察「固定 ``beta = s*``、
      释放 ``alpha``」与「固定 ``alpha = s*``、释放 ``beta``」两个候选；
    * 第 3 层：只有当单方向释放产生**稳定的外部收益**时，才进入 ``(alpha, beta)``
      均自由的二维搜索。

    图的内容是「策略结构 + 门控条件」；本次运行的分数与门控结论放在图下方两行
    脚注里，不占用流程图版面。
    """

    analysis = base.load_full_analysis(DEFAULT_FULL_DIR)
    by_model = {row["model"]: row for row in analysis.history}

    anchor = float(
        analysis.payload["adaptive_early_stopping"]["chosen_model"]["parameters"]["alpha"]
    )
    delta = float(
        analysis.payload["adaptive_early_stopping"]["settings"]["minimum_delta"]
    )
    chosen_model = str(
        (analysis.payload["adaptive_early_stopping"]["chosen_model"] or {}).get("model", "")
    )
    scores = {name: float(row["selection_score"]) for name, row in by_model.items()}
    best_shared = scores.get("shared_exponent", float("nan"))
    layer2 = [
        ("release_parameter_exponent", "固定 $\\beta=s^*$，释放 $\\alpha$"),
        ("release_data_exponent", "固定 $\\alpha=s^*$，释放 $\\beta$"),
    ]
    gains = [best_shared - scores.get(name, float("nan")) for name, _ in layer2]

    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    def box(x, y, w, h, title, body, color, dashed=False):
        axis.add_patch(
            Rectangle((x, y), w, h, facecolor=color, alpha=0.10,
                      edgecolor=color, linewidth=1.4,
                      linestyle=(0, (4, 2)) if dashed else "solid", zorder=2)
        )
        axis.text(x + w / 2, y + h - 0.055, title, ha="center", va="center",
                  color=color, fontweight="bold", zorder=3)
        for index, line in enumerate(body):
            axis.text(x + w / 2, y + h - 0.112 - index * 0.056, line,
                      ha="center", va="center", color="#222222",
                      zorder=3)

    def arrow(x0, y0, x1, y1, color="#5A5A5A"):
        axis.annotate("", xy=(x1, y1), xytext=(x0, y0),
                      arrowprops={"arrowstyle": "-|>", "color": color,
                                  "linewidth": 1.6}, zorder=4)

    box(0.05, 0.746, 0.90, 0.165, "第 1 层：共享指数一维搜索",
        ["令 $\\alpha=\\beta=s$，只对 $s$ 做一维搜索",
         f"本次最优 $s^*$ = {anchor:.6f}"], "#1F6FB4")
    arrow(0.50, 0.746, 0.50, 0.692)

    axis.text(0.50, 0.640, "第 2 层：以 $s^*$ 为锚点，两个并列候选",
              ha="center", va="center", color="#4A3D80",
              fontweight="bold", zorder=3)
    box(0.04, 0.460, 0.43, 0.150, "候选 A",
        ["固定 $\\beta=s^*$", "释放 $\\alpha$"], "#2E86C1")
    box(0.53, 0.460, 0.43, 0.150, "候选 B",
        ["固定 $\\alpha=s^*$", "释放 $\\beta$"], "#2E86C1")

    arrow(0.255, 0.460, 0.255, 0.410)
    arrow(0.745, 0.460, 0.745, 0.410)
    axis.plot([0.255, 0.745], [0.410, 0.410], color="#5A5A5A",
              linewidth=1.6, zorder=4)
    arrow(0.50, 0.410, 0.50, 0.392)

    axis.add_patch(
        Rectangle((0.18, 0.275), 0.64, 0.115, facecolor="white",
                  edgecolor=base.WARN_COLOR, linewidth=1.6,
                  linestyle=(0, (5, 2)), zorder=3)
    )
    axis.text(0.50, 0.3325, "单方向释放是否产生稳定的外部收益？",
              ha="center", va="center", color=base.WARN_COLOR, fontweight="bold", zorder=4)

    axis.annotate("", xy=(0.20, 0.195), xytext=(0.33, 0.275),
                  arrowprops={"arrowstyle": "-|>", "color": base.WARN_COLOR,
                              "linewidth": 1.6}, zorder=4)
    axis.text(0.10, 0.258, "不通过", ha="center", va="center", color=base.WARN_COLOR, zorder=4)
    axis.text(0.045, 0.150, "停止释放\n回滚到当前最优", ha="left", va="center",
              color=base.WARN_COLOR, zorder=4)

    box(0.56, 0.120, 0.40, 0.140, "第 3 层：二维搜索",
        ["$(\\alpha,\\beta)$ 均自由"], "#1E8449", dashed=True)
    axis.annotate("", xy=(0.76, 0.265), xytext=(0.70, 0.275),
                  arrowprops={"arrowstyle": "-|>", "color": "#1E8449",
                              "linewidth": 1.6}, zorder=4)
    axis.text(0.83, 0.293, "通过", ha="center", va="center", color="#1E8449", zorder=4)

    figure.text(
        0.5, 0.052,
        f"本次运行：第 1 层 {chosen_model} {best_shared:.6f}　|　"
        f"候选 A {scores.get(layer2[0][0], float('nan')):.6f}　|　"
        f"候选 B {scores.get(layer2[1][0], float('nan')):.6f}",
        ha="center", va="center", color="#333333")
    figure.text(
        0.5, 0.014,
        f"改善 {gains[0]:+.6f} / {gains[1]:+.6f}，未达门槛 {delta:g} → "
        f"门控未通过，回滚到 {chosen_model}",
        ha="center", va="center", color=base.WARN_COLOR)

    figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.105)
    _assert_text_fits(figure)
    return figure



# --------------------------------------------------------------------------- #
# 图：验证图的 (a)
# --------------------------------------------------------------------------- #
def figure_validation_a() -> plt.Figure:
    """归一化尺度下多来源观测与经典标度律预测的对照。

    分析脚本对多来源损失做仿射尺度归一化：先用留一簇拟合 ``L_source = a·L_B1 + b``，
    再把各数据集的实测损失搬回 B1 尺度 ``(L_obs - b) / a``。因此这里画的坐标轴
    与 B1 预测同尺度，``y=x`` 才是正确的参考线。

    归一化修掉的是尺度（水平）差异，修不掉散布——B2 归一化后仍会偏离对角线，
    中位相对误差从 31.89% 降到 6.10% 而不是 0，剩下的部分来自曲线形状差异。
    """

    analysis = base.load_full_analysis(DEFAULT_FULL_DIR)
    datasets = analysis.datasets
    # 图例用中文简称：英文全称＋括号里的 n 与误差会把图例撑到近半幅宽，
    # 放哪儿都会压住对角参考线或数据。这里改成「名称 + 两个数字列」，
    # 宽度减半，才能塞进右下角那块空白。
    source_labels = {
        "B2": "独立模型族端点",
        "B3": "插值训练轨迹",
        "B4": "跨族模型",
        "B5": "文献观测值",
    }

    # 归一化后的 (预测, 观测)：观测按 (L - b)/a 搬回 B1 尺度。
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in datasets:
        item = analysis.predictions[name]
        record = analysis.normalized_for(name)
        pairs[name] = (
            item.predicted,
            (item.actual - record.affine_intercept) / record.affine_slope)

    values: list[float] = []
    for x_values, y_values in pairs.values():
        values.extend(x_values.tolist())
        values.extend(y_values.tolist())
    low, high = min(values), max(values)
    padding = 0.05 * (high - low)
    limits = (low - padding, high + padding)

    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE)
    diagonal = np.asarray(limits)
    axis.fill_between(diagonal, diagonal * 0.9, diagonal * 1.1,
                      color="#999999", alpha=0.15, linewidth=0, zorder=1,
                      label="±10% 带")
    axis.plot(diagonal, diagonal, color="#333333", linestyle="--",
              linewidth=1.3, zorder=3, label="理想迁移 $y=x$")

    limits_per_dataset = {"B2": 320, "B4": 57, "B5": 44, "B3": 300}
    for name in datasets:
        x_values, y_values = pairs[name]
        x_show, y_show = base._display_subsample(
            x_values, y_values, limits_per_dataset.get(name, 320)
        )
        record = analysis.normalized_for(name)
        # 点少的数据集必须画得醒目：B2 只有 7 个点，若按密集数据的淡色小点画，
        # 图上基本看不见；B3 有 4000 个点，才需要抽稀 + 低透明度。
        sparse = x_values.size <= 100
        axis.scatter(
            x_show, y_show,
            s=30 if sparse else 10,
            alpha=0.9 if sparse else 0.35,
            linewidths=0, color=base.DATASET_COLORS[name], rasterized=True,
            zorder=5 if sparse else 4,
            label=(
                f"{source_labels.get(name, name)}　n={x_values.size}，"
                f"{record.normalized_mape:.2%}"
            ))

    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("经典标度律预测值 $L$")
    axis.set_ylabel("实测 $L$（尺度归一化）")
    # 数据是一条沿对角线铺开的带，把左上和右下角都占了；整块空白只有右下角
    # 对角线下方的三角区。改中文简称 + 去掉括号已经减了宽度，但仍不够——
    # 参数扫描结果是：只有单列 + 字号 10 能做到零遮挡，所以这里显式给 fontsize。
    axis.legend(loc="lower right", title="样本数，中位相对误差",
                framealpha=0.95, borderaxespad=0.5, handlelength=1.6,
                fontsize=10, title_fontsize=10)
    style_axis(axis)

    b2 = analysis.normalized_for("B2")
    # axis.text(
    #     0.97, 0.06,
    #     f"归一化已对齐尺度：B2 的中位相对误差\n"
    #     f"由 {b2.raw_mape:.2%} 降到 {b2.normalized_mape:.2%}，\n"
    #     f"剩余偏差来自曲线形状而非尺度。",
    #     transform=axis.transAxes, color=base.DATASET_COLORS["B2"],
    #     ha="right", va="bottom", zorder=7,
    #     bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
    #           "edgecolor": base.DATASET_COLORS["B2"], "linewidth": 0.9,
    #           "alpha": 0.95},
    # )
    figure.tight_layout()
    assert_legend_clear(axis)
    return figure



# --------------------------------------------------------------------------- #
# 图 4：四联图的 (a) 与 (b)
# --------------------------------------------------------------------------- #
def _quad_inputs():
    """四联图 (a)(b) 共用的数据与配色。"""

    parameters, _metrics = base.load_fit_result(base.DEFAULT_FIT_JSON)
    data = base.load_predictions(base.DEFAULT_PREDICTIONS)
    unique_n = np.unique(data.n_params_b)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.08, 0.92, unique_n.size))
    return parameters, data, unique_n, {n: colors[i] for i, n in enumerate(unique_n)}


def _quad_a_into(axis: plt.Axes) -> None:
    """在给定坐标轴上画 (a)：固定模型规模的数据标度。"""

    parameters, data, unique_n, color_by_n = _quad_inputs()
    d_curve = np.geomspace(data.d_tokens_b.min(), data.d_tokens_b.max(), 360)
    handles: list[Line2D] = []
    for n_value in unique_n:
        mask = np.isclose(data.n_params_b, n_value, rtol=0.0, atol=1e-12)
        order = np.argsort(data.d_tokens_b[mask])
        color = color_by_n[n_value]
        axis.scatter(
            data.d_tokens_b[mask][order], data.actual_loss[mask][order],
            s=14, color=color, alpha=0.45, linewidths=0, rasterized=True)
        axis.plot(
            d_curve, parameters.predict(np.full_like(d_curve, n_value), d_curve),
            color=color, linewidth=1.7)
        handles.append(
            Line2D([0], [0], color=color, linewidth=1.8, marker="o",
                   markersize=3.6, label=base._format_size(float(n_value)))
        )
    axis.set_xscale("log")
    axis.set_xlabel("训练数据量 $D$（十亿 token）")
    axis.set_ylabel("验证集交叉熵损失")
    axis.legend(handles=handles, title="参数量 $N$", ncol=2,
                loc="upper right", handlelength=1.6)
    style_axis(axis)


def _quad_b_into(axis: plt.Axes) -> None:
    """在给定坐标轴上画 (b)：固定 token 预算的参数标度。"""

    parameters, data, unique_n, _ = _quad_inputs()
    log_d = np.log(data.d_tokens_b)
    levels = np.exp(np.quantile(log_d, [0.15, 0.50, 0.85]))
    n_curve = np.geomspace(unique_n.min(), unique_n.max(), 360)
    for d_value, color, style in zip(
        levels, ("#0072B2", "#D55E00", "#009E73"), ("-", "--", "-.")
    ):
        axis.plot(
            n_curve,
            parameters.predict(n_curve, np.full_like(n_curve, d_value)),
            color=color, linestyle=style, linewidth=2.0,
            label=f"$D$ ≈ {d_value:.1f}B")
        observed_n, observed_loss = base._nearest_observations_at_d(
            data, d_value, unique_n
        )
        axis.scatter(
            observed_n, observed_loss, s=28, facecolors="white",
            edgecolors=color, linewidths=1.0, zorder=3)
    axis.set_xscale("log")
    axis.set_xlabel("模型参数量 $N$（十亿）")
    # axis.set_ylabel("验证集交叉熵损失")
    axis.legend(loc="upper right")
    style_axis(axis)


def figure_quad_a() -> plt.Figure:
    """(a) 单图：固定模型规模的数据标度。"""

    figure, axis = plt.subplots(figsize=SIG_FIGSIZE)
    _quad_a_into(axis)
    figure.tight_layout()
    return figure


def figure_quad_b() -> plt.Figure:
    """(b) 单图：固定 token 预算的参数标度。"""

    figure, axis = plt.subplots(figsize=SIG_FIGSIZE)
    _quad_b_into(axis)
    figure.tight_layout()
    return figure


def _assert_row_gap(axis_upper: plt.Axes, axis_lower: plt.Axes,
                    *, clearance: float = 0.012) -> None:
    """检查上下两格之间有没有足够空间容纳上格的横轴标题。

    这是上下布局最容易出问题的地方：上格的 xlabel 画在两格之间的空隙里，
    空隙不够就会被下格的绘图区盖住或贴上去（``tight_layout`` 也救不了，
    它只按包围盒估，不看文字实际落点）。这里直接量文字包围盒到下一格顶端
    的垂直净距，不够就报错，而不是输出一张标题被压住的图。

    ``clearance`` 是要求的最小净距（英寸），留出刻度文字的一点呼吸空间。
    """

    figure = axis_upper.figure
    # 必须先真正绘制一次：在此之前坐标轴标签的窗口包围盒还是默认值，
    # 量出来的净距会是负数（假的失败）。
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi

    upper_label = axis_upper.xaxis.get_label()
    if not upper_label.get_text().strip():
        return
    label_bottom = upper_label.get_window_extent(renderer=renderer).y0
    lower_top = axis_lower.get_window_extent(renderer=renderer).y1
    gap = (label_bottom - lower_top) / dpi

    if gap < clearance:
        raise RuntimeError(
            f"上下布局的间隙不足以容纳上格横轴标题：净距仅 {gap:.3f} 英寸，"
            f"需要 ≥ {clearance:.3f} 英寸（标题“{upper_label.get_text()}”）。\n"
            f"请调大 gridspec 的 hspace 或缩小纵轴预算。"
        )


def figure_quad_ab() -> plt.Figure:
    """把 (a)(b) 按上下布局合成一张（画布 6×4，(a) 在上、(b) 在下）。

    两格各占约 1.3 英寸高。上下布局的关键是两格之间要留出容纳 (a) 横轴标题
    （“训练数据量 $D$（十亿 token）”）的空隙——由 ``HSPACE`` 控制，
    并用 ``_assert_row_gap`` 在渲染前核对，避免标题被下格压住。
    """

    figure, (axis_a, axis_b) = plt.subplots(
        2, 1, figsize=PANEL_FIGSIZE, gridspec_kw={"hspace": HSPACE}
    )
    _quad_a_into(axis_a)
    _quad_b_into(axis_b)
    # 不用 tight_layout：matplotlib 3.11 对多轴图会误报「包围盒重叠」，
    # 且它不保证文字不撞。显式布局 + 显式净距检查更可预期。
    figure.subplots_adjust(left=0.115, right=0.975, top=0.975, bottom=0.085)
    _assert_row_gap(axis_a, axis_b)
    return figure



BUILDERS = {
    "extra-a": (figure_extrapolation_a, "traditional_scaling_law_extrapolation_a"),
    "steps-b": (figure_steps_b, "traditional_scaling_law_loss_vs_steps_law_b"),
    "steps-b-b2": (figure_steps_b_b2, "traditional_scaling_law_loss_vs_steps_law_b_b2"),
    "validation-a": (figure_validation_a, "traditional_scaling_law_validation_a"),
    "selection-a": (figure_selection_a, "traditional_scaling_law_selection_a"),
    "quad-a": (figure_quad_a, "traditional_scaling_law_quad_a"),
    "quad-b": (figure_quad_b, "traditional_scaling_law_quad_b"),
    "quad-ab": (figure_quad_ab, "traditional_scaling_law_quad_ab"),
}


def parse_formats(raw: str) -> list[str]:
    allowed = {"png", "pdf", "svg"}
    formats: list[str] = []
    for item in raw.split(","):
        extension = item.strip().lower().lstrip(".")
        if not extension:
            continue
        if extension not in allowed:
            raise ValueError(f"不支持输出格式 {extension!r}；可选 {sorted(allowed)}。")
        if extension not in formats:
            formats.append(extension)
    if not formats:
        raise ValueError("至少需要指定一种输出格式。")
    return formats


def parse_figures(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(BUILDERS)
    names: list[str] = []
    for item in raw.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in BUILDERS:
            raise ValueError(
                f"不支持的单图名 {name!r}；可选值 {sorted(BUILDERS)}，或 all。"
            )
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("至少需要指定一张单图。")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="单独输出论文用单图（单图 6×4，左右合成图 12×4，中文标注）。"
    )
    parser.add_argument("--figures", default="all",
                        help=f"逗号分隔：{', '.join(BUILDERS)}，或 all。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR,
                        help="图片输出目录。")
    parser.add_argument("--formats", default="png",
                        help="逗号分隔的输出格式：png、pdf、svg；默认只输出 png。")
    parser.add_argument("--dpi", type=int, default=300, help="PNG 输出分辨率。")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.formats)
    names = parse_figures(args.figures)

    font_name, has_cjk = configure_chinese()
    if not has_cjk:
        print(
            "警告：本机未安装中文字体，PNG 中的汉字会显示为方框。\n"
            "      已把常见中文字体名写入回退链，请在装有中文字体的机器上运行，\n"
            "      或安装任一：Noto Sans CJK SC / 文泉驿 / 微软雅黑 / SimHei。",
            file=sys.stderr)
    else:
        # 自带的子集字体只覆盖生成时的文案，改动文字后先查字形再开画。
        assert_glyphs_available(BUNDLED_FONTS)

    outputs: list[Path] = []
    for name in names:
        builder, basename = BUILDERS[name]
        figure = builder()
        allowed_sizes = (PANEL_FIGSIZE, SIG_FIGSIZE)
        actual_size = tuple(round(v, 3) for v in figure.get_size_inches().tolist())
        if actual_size not in allowed_sizes:
            raise RuntimeError(
                f"{name} 的画布 {actual_size} 不在允许范围 {allowed_sizes} 内"
            )
        for extension in formats:
            path = output_dir / f"{basename}.{extension}"
            figure.savefig(
                path,
                dpi=args.dpi if extension == "png" else None,
                format=extension,
                metadata={
                    "Title": basename,
                    "Creator": "classic_law_figures_zh.py",
                })
            outputs.append(path)
        plt.close(figure)

    print(f"字体: {font_name}（中文字体={'是' if has_cjk else '否'}）")
    print(f"画布: {PANEL_FIGSIZE[0]:g} × {PANEL_FIGSIZE[1]:g} 英寸")
    print()
    for path in outputs:
        print(f"已生成: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
