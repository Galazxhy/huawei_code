#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等损失替代关系图（一图）：维持参考损失所需的 Token 数随质量的变化。

横轴为数据质量得分 ``Q``，纵轴为在 ``N=1B`` 下维持参考损失所需的 Token 数
（十亿）。两条曲线：

* 实线：``D·Q`` 模型，``D_need = D0·Q0/Q``；
* 虚线：自由指数模型 ``L = E + A·N^(−α) + B·D^(−β_D)·Q^(−β_Q)`` 的等损失曲线
  ``D_need = D0·(Q0/Q)^r``，``r = β_Q/β_D``。

并标注 ``Q: 0.6 → 0.7`` 的 Token 节省量。

中文读者导向标签，不出现数据附件代号；PNG，供 LaTeX 使用。

用法：
    python Q2/Q_2_4/Q_2_4_6_quality_substitution_figures.py
"""


# --- shared paths for every script in Q2 (see Q2/_q2_paths.py) --- # q2-prologue:bootstrap
from __future__ import annotations
import json
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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

from Q_2_2_2_generalized_law_figures import chinese_family  # noqa: E402


matplotlib.use("Agg")


ROOT = _q2_paths.PROJECT_ROOT


INPUT = _q2_paths.ANALYSIS_DIR / "quality_token_substitution"
OUTPUT = INPUT / "figures"
PANEL_FIGSIZE = (6.0, 4.0)
MAIN_COLOR = "#4C78A8"
ALT_COLOR = "#C44E52"
ACCENT = "#333333"


def configure() -> str:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
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


def iso_loss_figure(payload: dict) -> plt.Figure:
    point = payload["operating_point"]
    d0, q0, q1 = (point["D0_billion_tokens"], point["Q0"], point["Q1_example"])
    ratio = payload["models"]["free"]["ratio_r"]
    grid = np.linspace(0.30, 1.0, 400)

    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    axis.plot(grid, d0 * q0 / grid, color=MAIN_COLOR, linewidth=2.0,
              label="$D\\cdot Q$ 模型：" + "$D_0Q_0/Q$")
    axis.plot(grid, d0 * (q0 / grid) ** ratio, color=ALT_COLOR, linewidth=1.6,
              linestyle="--",
              label=f"自由指数模型：$D_0(Q_0/Q)^r$，$r={ratio:.2f}$")

    need = d0 * q0 / q1
    # 只在有意义的区间内画：Q→0 时所需 Token 发散（Q=0.1 处要 150 十亿），
    # 把它画进来只会把 Q∈[0.5,1] 压成一条线。
    axis.axvline(q1, color="#BBBBBB", linewidth=0.9, linestyle=":", zorder=1)
    axis.plot([q0, q1], [d0, d0], color="#999999", linewidth=0.9, linestyle=":",
              zorder=2)
    axis.plot([q0], [d0], marker="o", markersize=9, color=ACCENT, zorder=5)
    axis.plot([q1, q1], [need, d0], color=ACCENT, linewidth=1.8, zorder=4)
    axis.plot([q1], [need], marker="v", markersize=8, color=ACCENT, zorder=5)
    # 基准点标签朝上、节省量标签朝右，避免两段文字互相压住。
    axis.annotate(f"基准点 $Q_0$={q0:g}、$D_0$={d0:g}", xy=(q0, d0),
                  xytext=(11, 12), textcoords="offset points", fontsize=11,
                  color=ACCENT, ha="left")
    axis.annotate(f"可节省 {d0 - need:.2f} 十亿 Token\n"
                  f"（等效新增 {d0 * (q1 / q0 - 1):.2f} 十亿）",
                  xy=(q1, (need + d0) / 2), xytext=(13, 0),
                  textcoords="offset points", fontsize=11, color=ACCENT,
                  va="center", ha="left", linespacing=1.5,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85,
                        "boxstyle": "round,pad=0.2"})

    axis.set_xlim(0.30, 1.02)
    axis.set_ylim(0, d0 * q0 / 0.30 * 1.06)
    axis.set_xticks([0.1 * level for level in range(3, 11)])
    axis.set_xticklabels([f"{level / 10:.1f}" for level in range(3, 11)])
    axis.set_xlabel("数据质量得分  $Q$")
    axis.set_ylabel("维持参考损失所需 Token（十亿）")
    tidy(axis)
    axis.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.92,
                edgecolor="#CCCCCC")
    axis.text(0.02, 0.04,
              f"工作点 $N$={point['N_billion_parameters']:g}B；"
              f"$D_0$={d0:g} 十亿为质量实验范围内的插值",
              transform=axis.transAxes, fontsize=10, color="#555555")
    return figure


def main() -> int:
    family = configure()
    payload = json.loads((INPUT / "quality_substitution.json").read_text(
        encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure = iso_loss_figure(payload)
    path = OUTPUT / "quality_token_substitution.png"
    # 字体族要在绘制时生效，而绘制发生在 savefig 里。
    with plt.rc_context({"font.sans-serif": [family]}):
        figure.savefig(path, dpi=300)
    plt.close(figure)
    print(f"\nfont: {family}")
    print(f"  wrote {_q2_paths.repo_relative(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
