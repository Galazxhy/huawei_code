#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""联合等损失替代图（一图）：配比如何移动"质量—Token"替代边界。

横轴为整体质量等级 ``Q``，纵轴为维持同一目标损失所需的 Token 数（B）。
在同一目标损失下画三条**来自实际观测配比**的曲线（按熵取分位），
另用一条虚线给出敏感性耦合（质量只作用于 ``B_k D^(−β)``）的对照。

中文读者导向标签，不出现数据附件代号；PNG，供 LaTeX 使用。

用法：
    python q2/joint_substitution_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generalized_law_figures import chinese_family  # noqa: E402

INPUT = ROOT / "data_analysis" / "joint_quality_mixture_substitution"
OUTPUT = INPUT / "figures"
#: 画布与配色：六条曲线用的颜色与三维柱图的配比配色共用一套。
ACCENT = "#333333"


def configure() -> str:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
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


def _iso_loss_setup(payload: dict, domain: str | None):
    """取某个评估域的等损失曲线数据（网格、目标信息），并校验网格含 ``Q_0``。"""

    domain = domain or payload["iso_loss_curves_main_domain"]
    block = payload["iso_loss_curves_by_domain"][domain]
    grid = np.asarray(block["grid_Q"], dtype=float)
    q0 = payload["operating_point"]["Q0"]
    anchor = int(np.argmin(np.abs(grid - q0)))
    assert abs(grid[anchor] - q0) < 1e-12, "网格必须精确包含 Q0，读数才对得上表"
    return domain, block, grid


def _draw_iso_loss(axis, payload: dict, domain: str | None, *,
                   show_ylabel: bool = True) -> np.ndarray:
    """把一个评估域的六条等损失曲线画进给定坐标轴；返回 ``(6, n)`` 曲线矩阵。

    单图与「两域并排图」共用这个函数，保证并排图里的每一格与单图**出自同一段代码**，
    不会出现两处画法不一致。
    """

    point = payload["operating_point"]
    q0, q1 = point["Q0"], point["Q1"]
    _, block, grid = _iso_loss_setup(payload, domain)
    curves, target = block["curves"], block["target"]

    pooled = []
    for colour, curve in zip(PALETTE, curves):
        values = np.asarray([np.nan if v is None else v
                             for v in curve["tokens_required_full"]], dtype=float)
        pooled.append(values)
        axis.plot(grid, values, color=colour, linewidth=2.4,
                  label=curve.get("label", "记录配比"), solid_capstyle="round")
        at_q1 = float(np.interp(q1, grid, values))
        axis.plot([q1], [at_q1], marker="o", markersize=5.5, color=colour,
                  markeredgecolor="white", markeredgewidth=0.9, zorder=5)

    axis.axvline(q0, color="#BBBBBB", linewidth=0.9, linestyle=":", zorder=1)
    axis.axvline(q1, color="#BBBBBB", linewidth=0.9, linestyle=":", zorder=1)
    tidy(axis)
    axis.set_yscale("log")
    # 与联合替代分析保持一致：仅展示 Q=0.5–1.0 的情景范围。
    axis.set_xlim(float(grid[0]) - 0.01, float(grid[-1]) + 0.01)
    axis.set_xticks([round(0.1 * step, 1) for step in range(1, 11)])
    axis.set_xlabel("整体质量等级  $Q$")
    if show_ylabel:
        axis.set_ylabel("维持同一损失\n所需 Token（B）")
    axis.text(0.015, 0.97,
              f"目标评估域：{target['eval_domain']}；$N$="
              f"{point['N_parameters'] / 1e9:g}B；"
              f"点线 $Q_0$={q0:g}、$Q_1$={q1:g}",
              transform=axis.transAxes, fontsize=9.5, color="#555555",
              ha="left", va="top")
    return np.asarray(pooled, dtype=float)


def _iso_loss_ylim(*pooled: np.ndarray) -> tuple[float, float]:
    """对数纵轴范围：单图按自己、并排图按两个域合起来取，上方留注释的空间。"""

    low = min(float(np.nanmin(values)) for values in pooled)
    high = max(float(np.nanmax(values)) for values in pooled)
    return low * 0.85, high * 1.55


def iso_loss_figure(payload: dict, domain: str | None = None,
                    filename: str = "joint_iso_loss_curves",
                    ) -> tuple[str, plt.Figure]:
    """六条记录配比的等损失曲线：横轴质量 ``Q``，纵轴维持同一目标损失所需 Token。

    只画一格。原来下面还有一格「每条曲线除以自己在 ``Q_0`` 处的值」的归一化图，
    那张图的主语其实是**配比**（用来证明配比不改变相对收益），与本节
    「提高质量能省多少 Token」的提问不是同一件事，已去掉，相关结论放在 §4 的文字里。

    纵轴用对数刻度：六条曲线的水平相差一个量级，线性轴上低的两条会被压扁成
    贴地的一条线，看不出形状；对数轴上曲线的斜率可直接读作相对变化率。

    ``domain`` 选择目标评估域（默认主图那个）。工作点、配比、$Q$ 网格与两种耦合
    口径都不随域改变，所以换域得到的是**样式完全相同**的一张图，可以直接并排看。
    返回 ``(文件名, 图)``，由 ``main`` 决定写到哪里。
    """

    figure, axis = plt.subplots(figsize=(7.0, 4.6), layout="constrained")
    axis.set_ylim(*_iso_loss_ylim(_draw_iso_loss(axis, payload, domain)))

    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3,
                  fontsize=9.5, frameon=False, handlelength=1.8,
                  columnspacing=1.6)
    return filename, figure


def iso_loss_pair_figure(payload: dict, domains: tuple[str, str],
                         filename: str = "joint_iso_loss_curves_two_domains",
                         ) -> tuple[str, plt.Figure]:
    """两个评估域并排：**共用纵轴与图例**，左格带纵轴刻度与标签，右格只留曲线。

    共用一个图例是因为六条曲线的含义与配色在两个域里完全相同；共用纵轴是因为
    两个域的纵轴是同一个量（维持同一目标损失所需 Token），分开画刻度会让人误以为
    量纲不同——共用后可以直接比较两个域的曲线高度。
    """

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True,
                                layout="constrained")
    pooled = [_draw_iso_loss(axes[0], payload, domains[0]),
              _draw_iso_loss(axes[1], payload, domains[1], show_ylabel=False)]
    limits = _iso_loss_ylim(*pooled)
    for axis in axes:
        axis.set_ylim(*limits)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=6,
                  fontsize=9.5, frameon=False, handlelength=1.8,
                  columnspacing=1.4)
    # 右格不再有自己的纵轴刻度，两格之间只需要一条窄缝。
    figure.get_layout_engine().set(wspace=0.02)
    return filename, figure



def _draw_mixture_effect(axis, payload: dict, domain: str | None,
                         row_order: list[int] | None, *,
                         show_ylabels: bool = True) -> None:
    """把一个评估域的配比影响画进给定坐标轴（单图与并排图共用）。"""

    point = payload["operating_point"]
    domain = domain or payload["iso_loss_curves_main_domain"]
    block = payload["iso_loss_curves_by_domain"][domain]
    target = block["target"]
    curves = block["curves"]
    population = [value for value
                  in payload["mixture_levels_by_domain"][domain]["level_Q0_billion"]
                  if value is not None]
    low, high = min(population), max(population)

    # 配色沿用等损失曲线图的顺序（按熵从高到低），三张图可逐条对照。
    number = {item["row"]: index + 1 for index, item in enumerate(curves)}
    if row_order is None:
        ordered = sorted(curves, key=lambda item: item["tokens_at_Q0_billion"])
    else:
        by_row = {item["row"]: item for item in curves}
        ordered = [by_row[row] for row in row_order]

    rows = len(ordered)
    for y, item in enumerate(ordered):
        colour = PALETTE[(number[item["row"]] - 1) % len(PALETTE)]
        q0 = item["tokens_at_Q0_billion"]
        q1 = item["tokens_at_Q1_billion"]
        axis.plot([0.0, q0], [y, y], color="#D6D6D6", linewidth=1.1, zorder=1)
        axis.plot([q1], [y], marker="o", markersize=4.6, markerfacecolor="white",
                  markeredgecolor=colour, markeredgewidth=1.5, zorder=4)
        axis.plot([q0], [y], marker="o", markersize=7.6, color=colour, zorder=3)
        axis.text(q0 + 0.9, y,
                  f"省 {item['saved_billion']:.2f} B"
                  f"（{item['relative_saving_percent']:.2f}%）",
                  va="center", fontsize=9.5, color="#555555")

    axis.set_yticks(range(rows))
    if show_ylabels:
        axis.set_yticklabels([f"{item['label']}" for item in ordered], fontsize=9)
    axis.set_xlim(0.0, max(item["tokens_at_Q0_billion"] for item in ordered) + 7.0)
    axis.set_ylim(-0.75, rows - 0.02)
    axis.set_xlabel("维持同一目标损失所需 Token（B）")
    tidy(axis, grid_axis="x")
    axis.text(0.995, 0.03,
              f"目标：{target['eval_domain']}；$N$="
              f"{point['N_parameters'] / 1e9:g}B；$Q_0$={point['Q0']:g}$\\to$"
              f"$Q_1$={point['Q1']:g}",
              transform=axis.transAxes, fontsize=9.5, color="#555555",
              ha="right", va="bottom")


def _mixture_legend(figure, ncol: int = 2) -> None:
    fills = Line2D([], [], marker="o", linestyle="none", markersize=7.6,
                   color="#4C78A8", label="$Q_0$ 处所需 Token")
    hollows = Line2D([], [], marker="o", linestyle="none", markersize=4.6,
                     markerfacecolor="white", markeredgecolor="#4C78A8",
                     markeredgewidth=1.5, label="$Q_1$ 处所需 Token（提高质量后）")
    figure.legend(handles=[fills, hollows], loc="outside lower center", ncol=ncol,
                  fontsize=9.5, frameon=False, handletextpad=0.5)


def mixture_effect_figure(payload: dict, domain: str | None = None,
                          filename: str = "joint_mixture_effect",
                          row_order: list[int] | None = None,
                          ) -> tuple[str, plt.Figure]:
    """配比的影响：同一目标损失下，不同配比需要的 Token 相差多少。

    与前两张图的区别在**主语**：那两张画的是「提高质量能省多少」（横轴是质量 ``Q``），
    这张画的是「不同配比把所需 Token 的水平挪多远」。每行一个代表性配比：

    - 实心点 = ``Q_0`` 处维持同一目标损失所需的 Token；
    - 空心点 = 提高到 ``Q_1`` 之后。两点几乎重合，说明质量挪不动它；
    - 行与行之间相差数倍，那才是配比的量级。

    ``domain`` 选择目标评估域（默认主图那个）：工作点、六个配比、``Q_0``/``Q_1`` 全都不动，
    只换目标损失，所以换域得到的图与主图**样式完全相同**，可以直接并排。
    ``row_order`` 固定行的顺序（传主域的排序），两张图的行就能左右对齐读；
    不传则按本域所需 Token 排序。
    返回 ``(文件名, 图)``，由 ``main`` 决定写到哪里。
    """

    figure, axis = plt.subplots(figsize=(6, 4), layout="constrained")
    _draw_mixture_effect(axis, payload, domain, row_order)
    _mixture_legend(figure)
    return filename, figure


def mixture_effect_pair_figure(payload: dict, domains: tuple[str, str],
                               filename: str = "joint_mixture_effect_two_domains",
                               row_order: list[int] | None = None,
                               ) -> tuple[str, plt.Figure]:
    """两个评估域并排：**共用纵轴与图例**，右格去掉配比名（行标签）只留点与读数。

    行顺序由 ``row_order`` 固定成同一个，加上共用纵轴，左右两格同一行就是同一个配比，
    可以直接横着读「同一个配比在两个评估目标下各要多少 Token」。
    """

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4), sharey=True,
                                layout="constrained")
    _draw_mixture_effect(axes[0], payload, domains[0], row_order)
    _draw_mixture_effect(axes[1], payload, domains[1], row_order,
                         show_ylabels=False)
    _mixture_legend(figure)
    figure.get_layout_engine().set(wspace=0.02)
    return filename, figure



def main() -> int:
    family = configure()
    payload = json.loads((INPUT / "joint_substitution.json").read_text(
        encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    main_domain = payload["iso_loss_curves_main_domain"]
    # 等损失曲线画两张：主图那个评估域，以及对照域（默认 arxiv）。两张图除评估域
    # 外口径完全一致，所以并排看就是「换一个评估目标，结论的形状是否还一样」。
    alternate = next(domain for domain in payload["iso_loss_curves_by_domain"]
                     if domain != main_domain)
    # 三张配比图共用同一行顺序（按主域所需 Token 从小到大），拼图左右两格因此能对齐读。
    mixture_row_order = [curve["row"] for curve in sorted(
        payload["iso_loss_curves_by_domain"][main_domain]["curves"],
        key=lambda curve: curve["tokens_at_Q0_billion"])]
    builders = (
        iso_loss_figure(payload, main_domain, "joint_iso_loss_curves"),
        iso_loss_figure(payload, alternate,
                        f"joint_iso_loss_curves_{alternate}"),
        ("joint_iso_loss_cylinders", cylinder_figure(payload)),
        mixture_effect_figure(payload, main_domain, "joint_mixture_effect",
                              mixture_row_order),
        mixture_effect_figure(payload, alternate,
                              f"joint_mixture_effect_{alternate}",
                              mixture_row_order),
        # 两域并排的两张：不是把 PNG 拼起来，而是一张画布两个共享坐标轴的面板，
        # 因此可以共用纵轴与图例、右格不重复画刻度和标签。
        iso_loss_pair_figure(payload, (main_domain, alternate)),
        mixture_effect_pair_figure(payload, (main_domain, alternate),
                                   row_order=mixture_row_order),
    )
    for name, figure in builders:
        path = OUTPUT / f"{name}.png"
        # 字体族要在绘制时生效，而绘制发生在 savefig 里。
        with plt.rc_context({"font.sans-serif": [family]}):
            figure.savefig(path, dpi=300)
        plt.close(figure)
        print(f"  wrote {path.relative_to(ROOT)}")

    print(f"font: {family}")
    return 0


# --------------------------------------------------------------------------- #
# 三维柱图：6 个代表性配比（按组成选取），Q0 与 Q1 并列
# --------------------------------------------------------------------------- #
PALETTE = ("#4C78A8", "#54A24B", "#C44E52", "#8C6BB1", "#F58518", "#4C9F9F")
OTHER_COLOR = "#BDBDBD"
ELEVATION, AZIMUTH = 22.0, -58.0


def _darken(colour, factor: float = 0.72):
    from matplotlib.colors import to_rgb
    return tuple(np.asarray(to_rgb(colour)) * factor)


def _draw_cylinder(axis, values, colours, height: float, centre_x: float,
                   radius: float) -> None:
    """把一张饼挤出成柱：顶面扇形 = 配比组成，柱高 = 所需 Token。

    与既有图集同法：mplot3d 不跨 collection 做深度排序，侧壁必须按相机方向自己
    剔除背面，否则背面那半圈会叠在前壁上形成"里外两层"的错色带；侧壁的
    ``edgecolors`` 取与面色相同，避免相邻薄片之间留下抗锯齿竖缝。
    """

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    view = np.array([np.cos(np.radians(ELEVATION)) * np.cos(np.radians(AZIMUTH)),
                     np.cos(np.radians(ELEVATION)) * np.sin(np.radians(AZIMUTH)),
                     np.sin(np.radians(ELEVATION))])
    angle = np.pi / 2
    for fraction, colour in zip(values, colours):
        if fraction <= 0:
            continue
        end = angle + 2 * np.pi * fraction
        angles = np.linspace(angle, end, max(4, int(np.ceil(120 * fraction))))
        xy = [(centre_x + radius * np.cos(a), radius * np.sin(a)) for a in angles]
        dark = _darken(colour)
        sides = []
        for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
            outward = np.array([(x0 + x1) / 2 - centre_x, (y0 + y1) / 2, 0.0])
            norm = float(np.linalg.norm(outward))
            if norm == 0 or float(outward @ view) / norm <= 0.0:
                continue                      # 背面，跳过
            sides.append([(x0, y0, 0), (x1, y1, 0),
                          (x1, y1, height), (x0, y0, height)])
        if sides:
            axis.add_collection3d(Poly3DCollection(
                sides, facecolors=dark, edgecolors=dark, linewidths=0.3))
        top = [(centre_x, 0, height)] + [(x, y, height) for x, y in xy]
        axis.add_collection3d(Poly3DCollection(
            [top], facecolors=colour, edgecolors="white", linewidths=0.4))
        angle = end


def _draw_shared_zticks(axis, ticks, pad: float = 0.032) -> None:
    """在每个面板左侧手绘 z 刻度数字。

    mplot3d 不为所有面板画 z 刻度标签：本图 3 列布局下，左列完全不画数字
    （只有刻度线），右列的数字会压到相邻面板标题上。所以刻度自己画——先把四条
    竖棱投影到屏幕，取最靠左的一条（相机会把 z 轴画在那里），再沿它线性插值，
    得到每个刻度值对应的屏幕高度，最后换算成轴坐标写文字。
    """

    from mpl_toolkits.mplot3d import proj3d

    x0, x1 = axis.get_xlim3d()
    y0, y1 = axis.get_ylim3d()
    best = None
    for x in (x0, x1):
        for y in (y0, y1):
            px, py, _ = proj3d.proj_transform(x, y, 0.0, axis.get_proj())
            point = axis.transData.transform((px, py))
            if best is None or point[0] < best[0]:
                best = (float(point[0]), x, y)
    _, x, y = best
    z0, z1 = axis.get_zlim3d()
    screen = []
    for z in (z0, z1):
        px, py, _ = proj3d.proj_transform(x, y, z, axis.get_proj())
        screen.append(axis.transData.transform((px, py)))
    (sx0, sy0), (sx1, sy1) = screen
    to_axes = axis.transAxes.inverted()
    for value in ticks:
        ratio = (value - z0) / (z1 - z0)
        screen_x = sx0 + (sx1 - sx0) * ratio
        screen_y = sy0 + (sy1 - sy0) * ratio
        fx, fy = to_axes.transform((screen_x, screen_y))
        axis.text2D(fx - pad, fy, f"{value:.1f}", transform=axis.transAxes,
                    ha="right", va="center", fontsize=11, color="#333333")


def cylinder_figure(payload: dict) -> plt.Figure:
    """6 个代表性配比的三维柱图：柱高 = 提升质量**可节省的 Token**。

    早期版本每个面板并排两根柱（左 ``Q_0``、右 ``Q_1``），想用高度差表示节省量。
    但相对节省在六个配比上几乎相同（5.4%–5.6%），两根柱的高度差在 5% 以内，
    图上看不出区别——把"效果"本身画成柱高才读得出来。
    柱顶扇形给出该配比的**实际组成**（占比前四的来源 + 其他），
    因此每个配比按组成命名，而不是只标一个熵值。
    """

    from matplotlib.patches import Patch

    items = payload["representative_mixtures"]
    blocks = payload["iso_loss_curves_by_domain"]
    target = blocks[payload["iso_loss_curves_main_domain"]]["target"]
    point = payload["operating_point"]

    # 图例：6 个代表配比的前四来源里出现最频繁的若干个，其余归入"其他来源"。
    counts: dict[str, int] = {}
    for item in items:
        for entry in item["top_sources"]:
            counts[entry["domain"]] = counts.get(entry["domain"], 0) + 1
    ordered = sorted(counts, key=lambda name: (-counts[name], name))
    legend_sources = ordered[:len(PALETTE)]
    colour_of = {name: PALETTE[i] for i, name in enumerate(legend_sources)}

    savings = [item["saved_billion"] or 0.0 for item in items]
    ceiling = float(np.ceil(max(savings) * 1.1 / 0.5) * 0.5)
    zticks = np.arange(0.0, ceiling + 1e-9, 0.5)

    figure = plt.figure(figsize=(8.4, 6.4))
    axes = [figure.add_subplot(2, 3, index + 1, projection="3d")
            for index in range(len(items))]
    # left 留出边距：3D 面板的 z 刻度标签画在各自面板左侧，左列若不留边距
    # 会被裁到画布之外（前一版左列就完全没有 z 刻度）。
    figure.subplots_adjust(left=0.055, right=0.985, top=0.895, bottom=0.195,
                           wspace=0.0, hspace=0.26)

    for index, (axis, item) in enumerate(zip(axes, items)):
        top_four = item["top_sources"]
        values = [entry["share"] for entry in top_four]
        values.append(max(0.0, 1.0 - sum(values)))
        colours = [colour_of.get(entry["domain"], OTHER_COLOR) for entry in top_four]
        colours.append(OTHER_COLOR)

        high_q0 = item["tokens_required_Q0_billion"]
        high_q1 = item["tokens_required_Q1_billion"]
        _draw_cylinder(axis, values, colours, item["saved_billion"], 0.0, 0.60)

        axis.set_xlim(-1.06, 1.06)
        axis.set_ylim(-0.58, 0.58)
        axis.set_zlim(0, ceiling)
        axis.set_box_aspect((2.1, 1.15, 2.7))
        axis.view_init(elev=ELEVATION, azim=AZIMUTH)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks(zticks)
        axis.set_zticklabels([])
        _draw_shared_zticks(axis, zticks)
        axis.grid(False)
        for component in (axis.xaxis, axis.yaxis, axis.zaxis):
            component.pane.fill = False
            component.pane.set_edgecolor("white")
        # 两行标题：第一行是配比组成，第二行是该配比的替换结果。
        # 之前用 figure.text 摆在图下方，第二排会与图注、图例互相压住。
        if high_q0 is not None and high_q1 is not None:
            # 面板编号 (1)–(6)：排序依据（熵）按约定不出现在图里，
            # 编号让"从左到右、从上往下"的阅读顺序显式可查。
            # 编号放在**第二行**：第一行是配比组成（较长），编号加到它前面会让
            # 相邻面板的标题撞在一起。
            axis.set_title(f"{item['label']}\n({index + 1})  省 "
                           f"{item['saved_billion']:.2f} B"
                           f"（相对 {100 * item['saved_billion'] / high_q0:.2f}%）",
                           fontsize=12, pad=3, linespacing=1.45)
        else:
            axis.set_title(item["label"], fontsize=12, pad=3)

    handles = [Patch(facecolor=colour_of[name], label=name)
               for name in legend_sources]
    handles.append(Patch(facecolor=OTHER_COLOR, label="其他来源"))
    figure.legend(handles=handles, loc="lower center", ncols=4, fontsize=10.5,
                  bbox_to_anchor=(0.5, 0.005))
    figure.text(0.5, 0.145,
                f"柱顶扇形 = 该配比的组成；柱高 = 质量从 $Q_0$="
                f"{point['Q0']:g} 提到 $Q_1$={point['Q1']:g} 可节省的 Token（B，"
                f"$N$={point['N_parameters'] / 1e9:g}B，目标 {target['eval_domain']}）\n",
                ha="center", va="center", fontsize=11, color="#555555",
                linespacing=1.5)
    return figure

if __name__ == "__main__":
    raise SystemExit(main())
