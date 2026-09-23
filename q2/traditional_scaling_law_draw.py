#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制经典标度律拟合图与拐点诊断图。

本脚本读取 ``traditional_scaling_law.py`` 生成的参数 JSON 和逐样本预测 CSV，
输出三张适合论文使用的图（可用 ``--figures`` 选择生成哪些）：

``quad``
    四联诊断图：数据标度、参数标度、观测-预测一致性、残差。

``knee-n``
    **不同参数规模 N 下的 Loss 曲线族**，并直观展示 Loss 的拐点与 N 的关系。
    左图：每个 N 一条 Loss-D 曲线（对数横轴），标出两类拐点；
    右图：两类拐点的位置随 N 的变化。

``knee-d``
    **不同 Token 数 D 下的 Loss 曲线族**，同样展示拐点位置随 D 的变化。

``steps``
    **训练曲线图**：横轴为训练步数，纵轴为交叉熵损失，**曲线颜色编码参数量 N**。
    数据取自原始训练日志（``--training-log``），因为拟合结果里没有 ``steps`` 列。

拐点的两种定义（两者都在脚本内解析求解并与数值解交叉验证）：

1. **几何拐点**：曲线在 (ln x, L) 坐标下曲率最大的位置。
   对 ``L = C + B·x^-β``，令 ``v = β·B·x^-β`` 可解析得到 ``v* = 1/√2``，故

       x_κ = (√2 · β · B)^(1/β)

   注意该位置**只取决于衰减项的系数与指数，与常数项 C 无关**——因此在
   Chinchilla 加法可分离形式下，不同 N 的 Loss-D 曲线互为垂直平移，
   几何拐点落在同一个 D 上。

2. **等贡献拐点**：数据项与竞争项（此处为参数项 ``A·N^-α``）贡献相等的位置。
   等价于「可约损失的局部双对数斜率由 −β 衰减到 −β/2」的位置：

       D_eq = ( B / (A·N^-α) )^(1/β)  ∝  N^(α/β)

   该位置**随 N 单调增大**，衡量「多大的数据量之后，继续加数据的边际收益
   降到与换更大模型相当的水平」。

用法::

    conda run -n GNN python q1/traditional_scaling_law_draw.py
    conda run -n GNN python q1/traditional_scaling_law_draw.py --figures knee-n,knee-d
    conda run -n GNN python q1/traditional_scaling_law_draw.py \
        --formats png,pdf,svg --dpi 400
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

# 脚本用于批量生成论文图片，使用非交互式后端，避免无显示器环境报错。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LogNorm
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "data_analysis" / "traditional_scaling_law"
DEFAULT_FIT_JSON = DEFAULT_RESULT_DIR / "traditional_scaling_fit.json"
DEFAULT_PREDICTIONS = DEFAULT_RESULT_DIR / "traditional_scaling_predictions.csv"
DEFAULT_TRAINING_LOG = (
    PROJECT_ROOT
    / "data"
    / "real_attachments"
    / "B_scaling_laws"
    / "pythia_training_log_existing.csv"
)
DEFAULT_BASELINE_CSV = (
    PROJECT_ROOT / "data" / "real_attachments" / "B_scaling_laws" / "scaling_baseline.csv"
)
DEFAULT_PUBLISHED_CSV = (
    PROJECT_ROOT
    / "data"
    / "real_attachments"
    / "B_scaling_laws"
    / "published_scaling_data.csv"
)

# 拐点标记与配色（全局统一，便于跨图对照）。
GEOMETRIC_COLOR = "#C0392B"
EQUAL_CONTRIBUTION_COLOR = "#1F4E79"
EXTRAPOLATION_SHADE = "#B0B0B0"


@dataclass(frozen=True)
class ScalingParameters:
    irreducible_loss: float
    parameter_coefficient: float
    parameter_exponent: float
    data_coefficient: float
    data_exponent: float

    def predict(self, n_params_b: np.ndarray, d_tokens_b: np.ndarray) -> np.ndarray:
        return (
            self.irreducible_loss
            + self.parameter_coefficient
            * np.power(n_params_b, -self.parameter_exponent)
            + self.data_coefficient * np.power(d_tokens_b, -self.data_exponent)
        )

    def parameter_term(self, n_params_b: np.ndarray | float) -> np.ndarray:
        """参数项 A·N^-α。"""

        return self.parameter_coefficient * np.power(
            np.asarray(n_params_b, dtype=float), -self.parameter_exponent
        )

    def data_term(self, d_tokens_b: np.ndarray | float) -> np.ndarray:
        """数据项 B·D^-β。"""

        return self.data_coefficient * np.power(
            np.asarray(d_tokens_b, dtype=float), -self.data_exponent
        )

    def geometric_knee_data(self) -> float:
        """Loss-D 曲线的几何拐点位置 D_κ = (√2·β·B)^(1/β)，与 N 无关。"""

        return geometric_knee_location(self.data_coefficient, self.data_exponent)

    def geometric_knee_params(self) -> float:
        """Loss-N 曲线的几何拐点位置 N_κ = (√2·α·A)^(1/α)，与 D 无关。"""

        return geometric_knee_location(
            self.parameter_coefficient, self.parameter_exponent
        )

    def equal_contribution_data(self, n_params_b: np.ndarray | float) -> np.ndarray:
        """给定 N，数据项与参数项相等处的 D_eq = (B / (A·N^-α))^(1/β)。"""

        competing = self.parameter_term(n_params_b)
        return np.power(
            self.data_coefficient / competing, 1.0 / self.data_exponent
        )

    def equal_contribution_params(self, d_tokens_b: np.ndarray | float) -> np.ndarray:
        """给定 D，参数项与数据项相等处的 N_eq = (A / (B·D^-β))^(1/α)。"""

        competing = self.data_term(d_tokens_b)
        return np.power(
            self.parameter_coefficient / competing, 1.0 / self.parameter_exponent
        )


@dataclass(frozen=True)
class PredictionData:
    n_params_b: np.ndarray
    d_tokens_b: np.ndarray
    actual_loss: np.ndarray
    predicted_loss: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class TrainingLog:
    """原始训练日志中的 (N, steps, train_loss, val_loss) 四元组。"""

    n_params_b: np.ndarray
    steps: np.ndarray
    train_loss: np.ndarray
    val_loss: np.ndarray

    def loss(self, column: str) -> np.ndarray:
        if column == "train_loss":
            return self.train_loss
        if column == "val_loss":
            return self.val_loss
        raise ValueError(f"不支持的损失列 {column!r}；可选 val_loss / train_loss。")


@dataclass(frozen=True)
class ReferencePoints:
    """跨族/文献观测点，用于把 Loss-N 图延伸到 Pythia 之外（最高到 540B）。"""

    label: str
    n_params_b: np.ndarray
    d_tokens_b: np.ndarray
    val_loss: np.ndarray

    @property
    def max_params(self) -> float:
        return float(self.n_params_b.max()) if self.n_params_b.size else 0.0


@dataclass(frozen=True)
class KneeSummary:
    """一次拐点分析的结果，用于控制台汇报。"""

    direction: str
    geometric_knee: float
    geometric_knee_numeric: float
    equal_contribution: np.ndarray
    driving_values: np.ndarray
    power_law_exponent: float


def configure_console() -> None:
    """确保 Windows 终端能够输出中文路径和特殊字符。"""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def configure_matplotlib() -> str:
    """配置论文图样式，并选择可用的中文字体。"""

    available_fonts = {item.name for item in font_manager.fontManager.ttflist}
    font_candidates = (
        "DejaVu Sans",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    )
    selected_font = next(
        (font_name for font_name in font_candidates if font_name in available_fonts),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected_font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
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
    return selected_font


def load_fit_result(path: Path) -> tuple[ScalingParameters, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到拟合结果：{path}\n"
            "请先运行 q1/traditional_scaling_law.py。"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_parameters = payload.get("parameters") or {}
    required_parameters = {
        "irreducible_loss",
        "parameter_coefficient",
        "parameter_exponent",
        "data_coefficient",
        "data_exponent",
    }
    missing = required_parameters.difference(raw_parameters)
    if missing:
        raise ValueError(f"拟合结果缺少参数：{sorted(missing)}")
    parameters = ScalingParameters(
        **{name: float(raw_parameters[name]) for name in required_parameters}
    )
    metrics = {
        name: float(value)
        for name, value in (payload.get("metrics") or {}).items()
        if isinstance(value, (int, float))
    }
    return parameters, metrics


def load_predictions(path: Path) -> PredictionData:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到逐样本预测：{path}\n"
            "请先运行 q1/traditional_scaling_law.py。"
        )

    n_values: list[float] = []
    d_values: list[float] = []
    actual_values: list[float] = []
    predicted_values: list[float] = []
    residual_values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "N_params_B",
            "D_tokens_B",
            "actual_loss",
            "predicted_loss",
            "residual",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"预测文件缺少字段：{sorted(missing)}")
        for row in reader:
            values = [
                float(row["N_params_B"]),
                float(row["D_tokens_B"]),
                float(row["actual_loss"]),
                float(row["predicted_loss"]),
                float(row["residual"]),
            ]
            if not all(math.isfinite(value) for value in values):
                continue
            n_values.append(values[0])
            d_values.append(values[1])
            actual_values.append(values[2])
            predicted_values.append(values[3])
            residual_values.append(values[4])

    if not n_values:
        raise ValueError("预测文件中没有可用数据。")
    return PredictionData(
        n_params_b=np.asarray(n_values, dtype=float),
        d_tokens_b=np.asarray(d_values, dtype=float),
        actual_loss=np.asarray(actual_values, dtype=float),
        predicted_loss=np.asarray(predicted_values, dtype=float),
        residual=np.asarray(residual_values, dtype=float),
    )


def load_training_log(path: Path) -> TrainingLog:
    """读取原始训练日志，取出 steps 与两种损失。

    拟合结果（fit JSON / predictions CSV）里没有 ``steps`` 列，因此训练曲线图
    必须回到原始日志取数。
    """

    if not path.exists():
        raise FileNotFoundError(
            f"找不到训练日志：{path}\n"
            "请用 --training-log 指定 pythia_training_log_existing.csv 的路径。"
        )

    n_values: list[float] = []
    step_values: list[float] = []
    train_values: list[float] = []
    val_values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"N_params_B", "steps", "train_loss", "val_loss"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"训练日志缺少字段：{sorted(missing)}")
        for row in reader:
            values = [
                float(row["N_params_B"]),
                float(row["steps"]),
                float(row["train_loss"]),
                float(row["val_loss"]),
            ]
            if not all(math.isfinite(value) for value in values):
                continue
            n_values.append(values[0])
            step_values.append(values[1])
            train_values.append(values[2])
            val_values.append(values[3])

    if not n_values:
        raise ValueError("训练日志中没有可用数据。")
    return TrainingLog(
        n_params_b=np.asarray(n_values, dtype=float),
        steps=np.asarray(step_values, dtype=float),
        train_loss=np.asarray(train_values, dtype=float),
        val_loss=np.asarray(val_values, dtype=float),
    )


def load_reference_points(path: Path, label: str) -> ReferencePoints:
    """读取跨族/文献标度律数据（B4/B5），用于扩展 Loss-N 图的参数规模覆盖。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到参考数据：{path}")

    n_values: list[float] = []
    d_values: list[float] = []
    loss_values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"N_params_B", "D_tokens_B", "val_loss"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"参考数据缺少字段：{sorted(missing)}")
        for row in reader:
            values = [
                float(row["N_params_B"]),
                float(row["D_tokens_B"]),
                float(row["val_loss"]),
            ]
            if not all(math.isfinite(value) for value in values):
                continue
            n_values.append(values[0])
            d_values.append(values[1])
            loss_values.append(values[2])

    if not n_values:
        raise ValueError(f"参考数据没有可用记录：{path}")
    return ReferencePoints(
        label=label,
        n_params_b=np.asarray(n_values, dtype=float),
        d_tokens_b=np.asarray(d_values, dtype=float),
        val_loss=np.asarray(loss_values, dtype=float),
    )


def _style_axis(axis: plt.Axes, grid_axis: str = "both") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, axis=grid_axis, which="major")
    axis.set_axisbelow(True)


def _format_size(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.0f}M"
    if value < 10:
        return f"{value:.2g}B"
    return f"{value:.3g}B"


def _format_location(value: float) -> str:
    """按量级选择 M / B / T 单位，用于标注拐点位置。"""

    if value < 1e-3:
        return f"{value * 1e6:.0f}M"
    if value < 1.0:
        return f"{value * 1000:.0f}M"
    if value < 1000.0:
        return f"{value:.3g}B"
    return f"{value / 1000.0:.3g}T"


def _nearest_observations_at_d(
    data: PredictionData, d_target: float, unique_n: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected_n: list[float] = []
    selected_loss: list[float] = []
    for n_value in unique_n:
        mask = np.isclose(data.n_params_b, n_value, rtol=0.0, atol=1e-12)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        nearest = indices[np.argmin(np.abs(np.log(data.d_tokens_b[indices] / d_target)))]
        selected_n.append(data.n_params_b[nearest])
        selected_loss.append(data.actual_loss[nearest])
    return np.asarray(selected_n), np.asarray(selected_loss)


# --------------------------------------------------------------------------- #
# 拐点分析
# --------------------------------------------------------------------------- #
def geometric_knee_location(coefficient: float, exponent: float) -> float:
    """解析求曲线 ``L = C + coefficient · x^-exponent`` 的最大曲率点。

    对 ``x`` 取对数坐标 ``u = ln x``，有 ``L(u) = C + coefficient·e^{-exponent·u}``。
    令 ``v = exponent · coefficient · e^{-exponent·u}``，可推出

        κ(v) ∝ v / (1 + v²)^{3/2},

    其最大值出现在 ``v = 1/√2``，与常数项 C 无关（垂直平移不改变曲率）。
    代回得 ``x_κ = (√2 · exponent · coefficient)^(1/exponent)``。
    """

    return (math.sqrt(2.0) * exponent * coefficient) ** (1.0 / exponent)


def equal_contribution_location(
    coefficient: float, exponent: float, competing_term: float
) -> float:
    """解析求两大项贡献相等的位置。

    ``coefficient · x^-exponent = competing_term``
    ⟹ ``x = (coefficient / competing_term)^(1/exponent)``。

    这也是可约损失的局部双对数斜率 ``d ln(L-E)/d ln x`` 由 ``-exponent``
    衰减到 ``-exponent/2`` 的位置。
    """

    return (coefficient / competing_term) ** (1.0 / exponent)


def numeric_curvature_knee(
    x_grid: np.ndarray, loss_grid: np.ndarray
) -> tuple[float, np.ndarray]:
    """在 (ln x, L) 坐标下数值求曲率最大点，用于与解析解交叉验证。"""

    log_x = np.log(x_grid)
    first = np.gradient(loss_grid, log_x)
    second = np.gradient(first, log_x)
    curvature = np.abs(second) / np.power(1.0 + first * first, 1.5)
    return float(x_grid[int(np.argmax(curvature))]), curvature


def analyse_knee_over_tokens(
    parameters: ScalingParameters, data: PredictionData
) -> KneeSummary:
    """沿 D 方向做拐点分析（固定各个 N）。"""

    unique_n = np.unique(data.n_params_b)
    driving = unique_n
    equal = parameters.equal_contribution_data(unique_n)
    geometric = parameters.geometric_knee_data()

    # 采样区间必须把解析拐点包住，否则数值解会退化为区间端点。
    span = np.geomspace(
        min(data.d_tokens_b.min() * 0.5, geometric / 4.0),
        max(data.d_tokens_b.max() * 8.0, geometric * 4.0),
        6000,
    )
    median_n = float(np.median(unique_n))
    numeric, _ = numeric_curvature_knee(span, parameters.predict(np.full_like(span, median_n), span))

    return KneeSummary(
        direction="D",
        geometric_knee=geometric,
        geometric_knee_numeric=numeric,
        equal_contribution=np.asarray(equal, dtype=float),
        driving_values=driving,
        power_law_exponent=parameters.parameter_exponent / parameters.data_exponent,
    )


def analyse_knee_over_params(
    parameters: ScalingParameters, data: PredictionData
) -> KneeSummary:
    """沿 N 方向做拐点分析（固定各个 D）。"""

    unique_d = np.unique(data.d_tokens_b)
    driving = unique_d
    equal = parameters.equal_contribution_params(unique_d)
    geometric = parameters.geometric_knee_params()

    # 采样区间必须把解析拐点包住，否则数值解会退化为区间端点。
    span = np.geomspace(
        min(data.n_params_b.min() * 0.4, geometric / 4.0),
        max(data.n_params_b.max() * 8.0, geometric * 4.0),
        6000,
    )
    median_d = float(np.median(unique_d))
    numeric, _ = numeric_curvature_knee(span, parameters.predict(span, np.full_like(span, median_d)))

    return KneeSummary(
        direction="N",
        geometric_knee=geometric,
        geometric_knee_numeric=numeric,
        equal_contribution=np.asarray(equal, dtype=float),
        driving_values=driving,
        power_law_exponent=parameters.data_exponent / parameters.parameter_exponent,
    )


def _draw_knee_legend(axis: plt.Axes, equal_label: str, loc: str = "lower left") -> None:
    """在坐标轴内单独放一个拐点图例，避免与曲线图例互相挤占。

    注意：这里**不能**用 ``axis.legend(...)``——第二次调用会替换掉先前的图例，
    使曲线图例消失。改为直接构造 ``Legend`` 并用 ``add_artist`` 挂上。
    """

    handles = [
        Line2D(
            [0],
            [0],
            color=GEOMETRIC_COLOR,
            linestyle="none",
            marker="X",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label="Geometric knee (max curvature)",
        ),
        Line2D(
            [0],
            [0],
            color=EQUAL_CONTRIBUTION_COLOR,
            linestyle="none",
            marker="D",
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=equal_label,
        ),
    ]
    legend = Legend(
        axis,
        handles,
        [handle.get_label() for handle in handles],
        loc=loc,
        frameon=True,
        handlelength=1.2,
        borderpad=0.5,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#CCCCCC")
    legend.get_frame().set_alpha(0.92)
    axis.add_artist(legend)


def draw_loss_vs_tokens_by_params(
    parameters: ScalingParameters,
    metrics: dict[str, float],
    data: PredictionData,
) -> plt.Figure:
    """不同参数规模 N 下的 Loss-D 曲线族，并展示拐点与 N 的关系。"""

    figure, (axis_curve, axis_knee) = plt.subplots(
        1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    unique_n = np.unique(data.n_params_b)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, unique_n.size))
    color_by_n = {n_value: colors[index] for index, n_value in enumerate(unique_n)}

    d_min_obs = float(data.d_tokens_b.min())
    d_max_obs = float(data.d_tokens_b.max())
    d_curve = np.geomspace(d_min_obs * 0.35, d_max_obs * 6.0, 720)

    # 超出观测范围的区间用浅色底纹标出，避免把外推段误读为观测。
    axis_curve.axvspan(d_curve[0], d_min_obs, color=EXTRAPOLATION_SHADE, alpha=0.16, linewidth=0)
    axis_curve.axvspan(d_max_obs, d_curve[-1], color=EXTRAPOLATION_SHADE, alpha=0.16, linewidth=0)

    geometric_knee = parameters.geometric_knee_data()
    axis_curve.axvline(
        geometric_knee,
        color=GEOMETRIC_COLOR,
        linestyle="--",
        linewidth=1.5,
        zorder=2,
    )

    # 等贡献拐点先收集、循环结束后再画引线，避免依赖循环中途尚未稳定的 y 轴范围。
    equal_points: list[tuple[float, float]] = []

    for n_value in unique_n:
        color = color_by_n[n_value]
        mask = np.isclose(data.n_params_b, n_value, rtol=0.0, atol=1e-12)
        order = np.argsort(data.d_tokens_b[mask])
        axis_curve.scatter(
            data.d_tokens_b[mask][order],
            data.actual_loss[mask][order],
            s=7,
            color=color,
            alpha=0.34,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        axis_curve.plot(
            d_curve,
            parameters.predict(np.full_like(d_curve, n_value), d_curve),
            color=color,
            linewidth=1.5,
            zorder=4,
        )
        # 几何拐点：所有 N 落在同一个 D 上，因此退化为一条竖线。
        knee_loss = float(parameters.predict(np.array([n_value]), np.array([geometric_knee]))[0])
        axis_curve.plot(
            [geometric_knee],
            [knee_loss],
            marker="X",
            markersize=8.5,
            color=GEOMETRIC_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linestyle="none",
            zorder=6,
        )
        # 等贡献拐点：随 N 右移。
        d_equal = float(parameters.equal_contribution_data(n_value))
        if d_curve[0] <= d_equal <= d_curve[-1]:
            loss_equal = float(parameters.predict(np.array([n_value]), np.array([d_equal]))[0])
            axis_curve.plot(
                [d_equal],
                [loss_equal],
                marker="D",
                markersize=5.5,
                color=EQUAL_CONTRIBUTION_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linestyle="none",
                zorder=6,
            )
            equal_points.append((d_equal, loss_equal))

    # 等贡献拐点的引线：在坐标范围确定后再画，末端止于曲线上的拐点。
    axis_curve.set_xscale("log")
    bottom, _ = axis_curve.get_ylim()
    if equal_points:
        axis_curve.vlines(
            [point[0] for point in equal_points],
            bottom,
            [point[1] for point in equal_points],
            color=EQUAL_CONTRIBUTION_COLOR,
            linestyle=":",
            linewidth=1.0,
            alpha=0.55,
            zorder=1,
        )

    axis_curve.set_xlabel("Training tokens, D (billions)")
    axis_curve.set_ylabel("Validation cross-entropy loss")
    axis_curve.set_title(
        "(a) Loss-token curves at different model sizes, with knees", loc="left"
    )
    _style_axis(axis_curve)

    curves_legend = [
        Line2D(
            [0],
            [0],
            color=color_by_n[n_value],
            linewidth=1.7,
            marker="o",
            markersize=3.2,
            label=_format_size(float(n_value)),
        )
        for n_value in unique_n
    ]
    curves_legend.append(
        Line2D([0], [0], color=EXTRAPOLATION_SHADE, linewidth=7, alpha=0.35,
               label="Outside observed D range")
    )
    axis_curve.legend(
        handles=curves_legend,
        title="Parameters, N",
        ncol=2,
        frameon=False,
        loc="upper right",
        columnspacing=0.9,
        handlelength=1.5,
    )
    _draw_knee_legend(axis_curve, "Equal-contribution knee", loc="lower left")

    # ---- 右图：拐点位置随 N 的变化 ----
    n_grid = np.geomspace(unique_n.min() * 0.5, unique_n.max() * 2.0, 320)
    axis_knee.plot(
        n_grid,
        parameters.equal_contribution_data(n_grid),
        color=EQUAL_CONTRIBUTION_COLOR,
        linewidth=2.0,
        label="Equal-contribution knee  $D_{eq}\\propto N^{\\alpha/\\beta}$",
    )
    axis_knee.plot(
        unique_n,
        parameters.equal_contribution_data(unique_n),
        marker="D",
        markersize=5.0,
        color=EQUAL_CONTRIBUTION_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.6,
        linestyle="none",
    )
    axis_knee.axhline(
        geometric_knee,
        color=GEOMETRIC_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Geometric knee  $D_\\kappa$ (N-independent)",
    )
    axis_knee.plot(
        unique_n,
        np.full_like(unique_n, geometric_knee),
        marker="X",
        markersize=8.0,
        color=GEOMETRIC_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.7,
        linestyle="none",
    )
    axis_knee.axhspan(d_min_obs, d_max_obs, color="#4C9F70", alpha=0.10, linewidth=0)
    axis_knee.set_xscale("log")
    axis_knee.set_yscale("log")
    axis_knee.set_xlabel("Model parameters, N (billions)")
    axis_knee.set_ylabel("Knee location in training tokens, $D^*$ (billions)")
    axis_knee.set_title("(b) Knee location versus model size", loc="left")
    axis_knee.legend(frameon=False, loc="upper left")
    _style_axis(axis_knee)

    ratio = parameters.parameter_exponent / parameters.data_exponent
    ratio_text = (
        f"$\\alpha/\\beta$ = {ratio:.3f}\n"
        f"$D_{{eq}}$ = {_format_location(float(parameters.equal_contribution_data(1.0)))} at N = 1B\n"
        f"$D_\\kappa$ = {_format_location(geometric_knee)}"
    )
    axis_knee.text(
        0.97,
        0.06,
        ratio_text,
        transform=axis_knee.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": "#CCCCCC", "alpha": 0.92},
    )

    equation = (
        r"$L(N,D)="
        f"{parameters.irreducible_loss:.4f}"
        f"+{parameters.parameter_coefficient:.4f}N^{{-{parameters.parameter_exponent:.4f}}}"
        f"+{parameters.data_coefficient:.4f}D^{{-{parameters.data_exponent:.4f}}}$"
    )
    figure.suptitle(
        "Loss curves across model sizes and the location of their knees",
        fontsize=14,
        fontweight="semibold",
    )
    figure.text(0.5, 0.925, equation, ha="center", va="center", fontsize=10.5)
    figure.text(
        0.5,
        0.012,
        "Coloured lines: fitted curves at the eight Pythia model sizes. "
        "Shaded D ranges lie outside the observed data.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#555555",
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.145, top=0.865, wspace=0.24)
    return figure


def draw_loss_vs_params_by_tokens(
    parameters: ScalingParameters,
    metrics: dict[str, float],
    data: PredictionData,
    references: Sequence[ReferencePoints] | None = None,
    max_params: float | None = None,
) -> plt.Figure:
    """不同 Token 数 D 下的 Loss-N 曲线族，并展示拐点与 D 的关系。

    ``references`` 中的跨族/文献观测点会叠加到左图，使参数规模覆盖延伸到
    Pythia 之外（B4 到 72B、B5 到 540B）。这些点各自对应不同的 D，因此在
    图中只作为规模覆盖的参照，不与某一条固定 D 的曲线严格对齐。
    """

    figure, (axis_curve, axis_knee) = plt.subplots(
        1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    unique_n = np.unique(data.n_params_b)
    log_d = np.log(data.d_tokens_b)
    # 取对数分布的分位数作为代表 Token 水平，覆盖低、中、高预算。
    token_levels = np.exp(np.quantile(log_d, [0.10, 0.30, 0.50, 0.70, 0.90]))
    palette = plt.get_cmap("plasma")(np.linspace(0.10, 0.86, token_levels.size))

    n_min_obs = float(unique_n.min())
    n_max_obs = float(unique_n.max())
    # 参数规模覆盖：至少画到 Pythia 观测范围的 4 倍，若给定了参考数据或
    # max_params，则延伸到能容纳最大模型（B5 到 540B）。
    n_upper = max(n_max_obs * 4.0, float(max_params) if max_params else 0.0)
    if references:
        for reference in references:
            if reference.max_params > 0:
                n_upper = max(n_upper, reference.max_params)
    n_curve = np.geomspace(n_min_obs * 0.25, n_upper * 1.15, 900)

    axis_curve.axvspan(n_curve[0], n_min_obs, color=EXTRAPOLATION_SHADE, alpha=0.16, linewidth=0)
    axis_curve.axvspan(n_max_obs, n_curve[-1], color=EXTRAPOLATION_SHADE, alpha=0.16, linewidth=0)

    geometric_knee = parameters.geometric_knee_params()
    axis_curve.axvline(
        geometric_knee, color=GEOMETRIC_COLOR, linestyle="--", linewidth=1.5, zorder=2
    )

    legend_handles: list[Line2D] = []
    for d_value, color in zip(token_levels, palette):
        axis_curve.plot(
            n_curve,
            parameters.predict(n_curve, np.full_like(n_curve, d_value)),
            color=color,
            linewidth=1.7,
            zorder=4,
        )
        observed_n, observed_loss = _nearest_observations_at_d(data, float(d_value), unique_n)
        axis_curve.scatter(
            observed_n,
            observed_loss,
            s=20,
            facecolors="white",
            edgecolors=color,
            linewidths=0.9,
            zorder=5,
        )
        # 几何拐点：所有 D 落在同一个 N 上。
        knee_loss = float(parameters.predict(np.array([geometric_knee]), np.array([d_value]))[0])
        axis_curve.plot(
            [geometric_knee],
            [knee_loss],
            marker="X",
            markersize=8.5,
            color=GEOMETRIC_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linestyle="none",
            zorder=6,
        )
        # 等贡献拐点：随 D 右移。
        n_equal = float(parameters.equal_contribution_params(d_value))
        if n_curve[0] <= n_equal <= n_curve[-1]:
            loss_equal = float(parameters.predict(np.array([n_equal]), np.array([d_value]))[0])
            axis_curve.plot(
                [n_equal],
                [loss_equal],
                marker="D",
                markersize=5.5,
                color=EQUAL_CONTRIBUTION_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linestyle="none",
                zorder=6,
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.7,
                marker="o",
                markersize=3.4,
                markerfacecolor="white",
                label=f"D ≈ {d_value:.3g}B",
            )
        )

    # ---- 叠加跨族/文献观测点，把参数规模覆盖延伸到 540B ----
    reference_markers = ("^", "s", "P", "v")
    for index, reference in enumerate(references or []):
        marker = reference_markers[index % len(reference_markers)]
        axis_curve.scatter(
            reference.n_params_b,
            reference.val_loss,
            s=26,
            marker=marker,
            facecolors="none",
            edgecolors="#333333",
            linewidths=0.8,
            alpha=0.75,
            zorder=5,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle="none",
                marker=marker,
                markersize=5.5,
                markerfacecolor="none",
                label=reference.label,
            )
        )

    axis_curve.set_xscale("log")
    axis_curve.set_xlabel("Model parameters, N (billions)")
    axis_curve.set_ylabel("Validation cross-entropy loss")
    axis_curve.set_title("(a) Loss-parameter curves at different token budgets", loc="left")
    legend_handles.append(
        Line2D([0], [0], color=EXTRAPOLATION_SHADE, linewidth=7, alpha=0.35,
               label="Outside observed N range")
    )
    axis_curve.legend(
        handles=legend_handles, title="Tokens, D", frameon=False, loc="upper right",
        handlelength=1.8,
    )
    _draw_knee_legend(axis_curve, "Equal-contribution knee", loc="lower left")
    _style_axis(axis_curve)

    # ---- 右图：拐点位置随 D 的变化 ----
    d_grid = np.geomspace(float(data.d_tokens_b.min()) * 0.5, float(data.d_tokens_b.max()) * 2.0, 320)
    axis_knee.plot(
        d_grid,
        parameters.equal_contribution_params(d_grid),
        color=EQUAL_CONTRIBUTION_COLOR,
        linewidth=2.0,
        label="Equal-contribution knee  $N_{eq}\\propto D^{\\beta/\\alpha}$",
    )
    axis_knee.plot(
        token_levels,
        parameters.equal_contribution_params(token_levels),
        marker="D",
        markersize=5.0,
        color=EQUAL_CONTRIBUTION_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.6,
        linestyle="none",
    )
    axis_knee.axhline(
        geometric_knee,
        color=GEOMETRIC_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Geometric knee  $N_\\kappa$ (D-independent)",
    )
    axis_knee.plot(
        token_levels,
        np.full_like(token_levels, geometric_knee),
        marker="X",
        markersize=8.0,
        color=GEOMETRIC_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.7,
        linestyle="none",
    )
    axis_knee.axhspan(n_min_obs, n_max_obs, color="#4C9F70", alpha=0.10, linewidth=0)
    axis_knee.set_xscale("log")
    axis_knee.set_yscale("log")
    axis_knee.set_xlabel("Training tokens, D (billions)")
    axis_knee.set_ylabel("Knee location in parameters, $N^*$ (billions)")
    axis_knee.set_title("(b) Knee location versus token budget", loc="left")
    axis_knee.legend(frameon=False, loc="upper left")
    _style_axis(axis_knee)

    ratio = parameters.data_exponent / parameters.parameter_exponent
    ratio_text = (
        f"$\\beta/\\alpha$ = {ratio:.3f}\n"
        f"$N_{{eq}}$ = {_format_location(float(parameters.equal_contribution_params(100.0)))} at D = 100B\n"
        f"$N_\\kappa$ = {_format_location(geometric_knee)}"
    )
    axis_knee.text(
        0.97,
        0.06,
        ratio_text,
        transform=axis_knee.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": "#CCCCCC", "alpha": 0.92},
    )

    figure.suptitle(
        "Loss curves across token budgets and the location of their knees",
        fontsize=14,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.012,
        "Hollow circles: Pythia observations nearest each token level. "
        "Triangles and squares: cross-family (B4, ≤72B) and published (B5, ≤540B) models, "
        "each at its own D. Shaded N ranges lie outside the observed Pythia data.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#555555",
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.145, top=0.90, wspace=0.24)
    return figure


def draw_loss_vs_steps(
    parameters: ScalingParameters,
    metrics: dict[str, float],
    log: TrainingLog,
    loss_column: str = "val_loss",
    x_log: bool = True,
) -> plt.Figure:
    """训练曲线图：横轴训练步数，纵轴交叉熵损失，颜色编码参数量 N。"""

    figure, axis = plt.subplots(figsize=(9.0, 5.8))

    unique_n = np.unique(log.n_params_b)
    norm = LogNorm(vmin=float(unique_n.min()), vmax=float(unique_n.max()))
    cmap = plt.get_cmap("viridis")
    losses = log.loss(loss_column)

    for n_value in unique_n:
        mask = np.isclose(log.n_params_b, n_value, rtol=0.0, atol=1e-12)
        order = np.argsort(log.steps[mask])
        axis.plot(
            log.steps[mask][order],
            losses[mask][order],
            color=cmap(norm(n_value)),
            linewidth=1.6,
            marker="o",
            markersize=2.6,
            markevery=8,
            markeredgewidth=0,
            zorder=3,
        )

    if x_log:
        axis.set_xscale("log")
    axis.set_xlabel("Training steps")
    axis.set_ylabel(
        "Validation cross-entropy loss" if loss_column == "val_loss"
        else "Training cross-entropy loss"
    )
    axis.set_title("Training curves coloured by model size", loc="left")
    _style_axis(axis)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(unique_n)
    colorbar = figure.colorbar(scalar_mappable, ax=axis, pad=0.02, fraction=0.046)
    colorbar.set_label("Parameters, N (billions)")
    colorbar.set_ticks(unique_n)
    # 刻度标签同时给出最终损失，避免在曲线末端加注释导致低位曲线互相压字。
    final_loss_by_n = {}
    for n_value in unique_n:
        mask = np.isclose(log.n_params_b, n_value, rtol=0.0, atol=1e-12)
        final_loss_by_n[float(n_value)] = float(losses[mask][np.argmax(log.steps[mask])])
    colorbar.set_ticklabels(
        [
            f"{_format_size(float(value))}  ·  {final_loss_by_n[float(value)]:.2f}"
            for value in unique_n
        ]
    )
    colorbar.ax.tick_params(labelsize=8)

    axis.margins(x=0.06)

    figure.text(
        0.5,
        0.012,
        f"Source: pythia_training_log_existing.csv ({log.steps.size} checkpoints, "
        f"{unique_n.size} model sizes). "
        "The second number on each colourbar tick is that size's final loss.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#555555",
    )
    figure.subplots_adjust(left=0.10, right=0.99, bottom=0.145, top=0.93)
    return figure


def draw_scaling_law(
    parameters: ScalingParameters,
    metrics: dict[str, float],
    data: PredictionData,
) -> plt.Figure:
    """创建四联标度律拟合与诊断图。"""

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), constrained_layout=False)
    axis_tokens, axis_params, axis_parity, axis_residual = axes.ravel()

    unique_n = np.unique(data.n_params_b)
    color_map = plt.get_cmap("viridis")
    colors = color_map(np.linspace(0.08, 0.92, unique_n.size))
    color_by_n = {n_value: colors[index] for index, n_value in enumerate(unique_n)}

    # (a) 固定 N 的训练轨迹：Loss-D。
    d_curve = np.geomspace(data.d_tokens_b.min(), data.d_tokens_b.max(), 360)
    legend_handles: list[Line2D] = []
    for n_value in unique_n:
        mask = np.isclose(data.n_params_b, n_value, rtol=0.0, atol=1e-12)
        order = np.argsort(data.d_tokens_b[mask])
        d_observed = data.d_tokens_b[mask][order]
        loss_observed = data.actual_loss[mask][order]
        color = color_by_n[n_value]
        axis_tokens.scatter(
            d_observed,
            loss_observed,
            s=9,
            color=color,
            alpha=0.42,
            linewidths=0,
            rasterized=True,
        )
        predicted_curve = parameters.predict(
            np.full_like(d_curve, n_value), d_curve
        )
        axis_tokens.plot(d_curve, predicted_curve, color=color, linewidth=1.45)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.6,
                marker="o",
                markersize=3.5,
                label=_format_size(float(n_value)),
            )
        )
    axis_tokens.set_xscale("log")
    axis_tokens.set_xlabel("Training tokens, D (billions)")
    axis_tokens.set_ylabel("Validation cross-entropy loss")
    axis_tokens.set_title("(a) Data scaling at fixed model sizes", loc="left")
    axis_tokens.legend(
        handles=legend_handles,
        title="Parameters, N",
        ncol=2,
        frameon=False,
        loc="upper right",
        columnspacing=0.9,
        handlelength=1.6,
    )
    _style_axis(axis_tokens)

    # (b) 固定 D 的 Loss-N 曲线。Token 水平取对数分布的低、中、高分位数。
    log_d = np.log(data.d_tokens_b)
    token_levels = np.exp(np.quantile(log_d, [0.15, 0.50, 0.85]))
    n_curve = np.geomspace(unique_n.min(), unique_n.max(), 360)
    line_colors = ("#0072B2", "#D55E00", "#009E73")
    line_styles = ("-", "--", "-.")
    for d_value, color, line_style in zip(
        token_levels, line_colors, line_styles
    ):
        predicted_curve = parameters.predict(
            n_curve, np.full_like(n_curve, d_value)
        )
        axis_params.plot(
            n_curve,
            predicted_curve,
            color=color,
            linestyle=line_style,
            linewidth=1.8,
            label=f"D ≈ {d_value:.1f}B",
        )
        observed_n, observed_loss = _nearest_observations_at_d(
            data, float(d_value), unique_n
        )
        axis_params.scatter(
            observed_n,
            observed_loss,
            s=22,
            facecolors="white",
            edgecolors=color,
            linewidths=0.9,
            zorder=3,
        )
    axis_params.set_xscale("log")
    axis_params.set_xlabel("Model parameters, N (billions)")
    axis_params.set_ylabel("Validation cross-entropy loss")
    axis_params.set_title("(b) Parameter scaling at fixed token budgets", loc="left")
    axis_params.legend(frameon=False, loc="upper right")
    _style_axis(axis_params)

    # (c) 观测值-预测值一致性图。
    lower = min(data.actual_loss.min(), data.predicted_loss.min())
    upper = max(data.actual_loss.max(), data.predicted_loss.max())
    padding = 0.035 * (upper - lower)
    limits = (lower - padding, upper + padding)
    axis_parity.scatter(
        data.actual_loss,
        data.predicted_loss,
        s=11,
        color="#2F6B8A",
        alpha=0.52,
        linewidths=0,
        rasterized=True,
    )
    axis_parity.plot(limits, limits, color="#333333", linewidth=1.1, linestyle="--")
    axis_parity.set_xlim(limits)
    axis_parity.set_ylim(limits)
    axis_parity.set_aspect("equal", adjustable="box")
    axis_parity.set_xlabel("Observed loss")
    axis_parity.set_ylabel("Predicted loss")
    axis_parity.set_title("(c) Observed versus predicted loss", loc="left")
    r_squared = metrics.get("r_squared", float("nan"))
    rmse = metrics.get("rmse", float("nan"))
    axis_parity.text(
        0.04,
        0.94,
        f"$R^2$ = {r_squared:.8f}\nRMSE = {rmse:.2e}",
        transform=axis_parity.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    _style_axis(axis_parity)

    # (d) 残差图。颜色与 (a) 中的参数规模保持一致。
    for n_value in unique_n:
        mask = np.isclose(data.n_params_b, n_value, rtol=0.0, atol=1e-12)
        axis_residual.scatter(
            data.d_tokens_b[mask],
            data.residual[mask],
            s=9,
            color=color_by_n[n_value],
            alpha=0.48,
            linewidths=0,
            rasterized=True,
        )
    axis_residual.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    axis_residual.set_xscale("log")
    axis_residual.set_xlabel("Training tokens, D (billions)")
    axis_residual.set_ylabel("Residual (observed − predicted)")
    axis_residual.set_title("(d) Residual diagnostics", loc="left")
    axis_residual.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axis_residual.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    _style_axis(axis_residual)

    equation = (
        r"$L(N,D)="
        f"{parameters.irreducible_loss:.4f}"
        f"+{parameters.parameter_coefficient:.4f}N^{{-{parameters.parameter_exponent:.4f}}}"
        f"+{parameters.data_coefficient:.4f}D^{{-{parameters.data_exponent:.4f}}}$"
    )
    figure.suptitle("Classical scaling law fit", fontsize=15, fontweight="semibold", y=0.985)
    figure.text(0.5, 0.948, equation, ha="center", va="center", fontsize=11)
    figure.text(
        0.5,
        0.018,
        "Points denote observations; lines denote fitted scaling-law predictions. "
        "N and D are measured in billions.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    figure.subplots_adjust(left=0.075, right=0.975, bottom=0.09, top=0.89, wspace=0.25, hspace=0.29)
    return figure


def parse_formats(raw_formats: str) -> list[str]:
    allowed = {"png", "pdf", "svg"}
    formats: list[str] = []
    for item in raw_formats.split(","):
        extension = item.strip().lower().lstrip(".")
        if not extension:
            continue
        if extension not in allowed:
            raise ValueError(
                f"不支持输出格式 {extension!r}；可选格式为 {sorted(allowed)}。"
            )
        if extension not in formats:
            formats.append(extension)
    if not formats:
        raise ValueError("至少需要指定一种输出格式。")
    return formats


def parse_figures(raw_figures: str) -> list[str]:
    allowed = {"quad", "knee-n", "knee-d", "steps"}
    figures: list[str] = []
    for item in raw_figures.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in allowed:
            raise ValueError(
                f"不支持图类型 {name!r}；可选值为 {sorted(allowed)}。"
            )
        if name not in figures:
            figures.append(name)
    if not figures:
        raise ValueError("至少需要指定一种图类型。")
    return figures


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    basename: str,
    formats: Sequence[str],
    dpi: int,
    title: str = "Classical scaling law fit",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for extension in formats:
        output_path = output_dir / f"{basename}.{extension}"
        figure.savefig(
            output_path,
            dpi=dpi if extension == "png" else None,
            format=extension,
            metadata={
                "Title": title,
                "Creator": "traditional_scaling_law_draw.py",
            },
        )
        output_paths.append(output_path)
    return output_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="绘制经典标度律拟合与拐点诊断图。")
    parser.add_argument(
        "--fit-json", type=Path, default=DEFAULT_FIT_JSON, help="拟合参数 JSON。"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="逐样本预测 CSV。",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULT_DIR, help="图片输出目录。"
    )
    parser.add_argument("--basename", default="traditional_scaling_law", help="输出文件基础名。")
    parser.add_argument(
        "--figures",
        default="all",
        help=(
            "要生成的图，逗号分隔：quad（四联诊断）、knee-n（不同 N 的 Loss-D 曲线族与拐点）、"
            "knee-d（不同 D 的 Loss-N 曲线族与拐点）、steps（训练步数-损失训练曲线），或 all。"
        ),
    )
    parser.add_argument(
        "--training-log",
        type=Path,
        default=DEFAULT_TRAINING_LOG,
        help="原始训练日志 CSV（steps 图需要，取自 B1）。",
    )
    parser.add_argument(
        "--loss-column",
        choices=("val_loss", "train_loss"),
        default="val_loss",
        help="steps 图使用的损失列，默认 val_loss（验证集交叉熵）。",
    )
    parser.add_argument(
        "--steps-xscale",
        choices=("log", "linear"),
        default="linear",
        help="steps 图的横轴刻度，默认 linear（如需看清早期下降可改用 log）。",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=DEFAULT_BASELINE_CSV,
        help="跨族收敛基准 B4，用于把 Loss-N 图的参数规模延伸到 72B。",
    )
    parser.add_argument(
        "--published-csv",
        type=Path,
        default=DEFAULT_PUBLISHED_CSV,
        help="已发表标度律数据 B5，用于把 Loss-N 图的参数规模延伸到 540B。",
    )
    parser.add_argument(
        "--max-params",
        type=float,
        default=540.0,
        help="Loss-N 图横轴（参数量）的上界，默认 540B。",
    )
    parser.add_argument(
        "--no-references",
        action="store_true",
        help="不在 Loss-N 图上叠加 B4/B5 跨族观测点。",
    )
    parser.add_argument(
        "--formats", default="png,pdf", help="逗号分隔的输出格式：png,pdf,svg。"
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG 输出分辨率。")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只输出生成的文件路径，不打印拐点分析摘要。",
    )
    return parser


def _report_knee(summary: KneeSummary, reference_label: str) -> None:
    print()
    print(f"[拐点分析] 沿 {summary.direction} 方向")
    print(
        f"  几何拐点（最大曲率，解析解）      = {_format_location(summary.geometric_knee)}"
        f"  （数值解 {_format_location(summary.geometric_knee_numeric)}）"
    )
    delta = abs(summary.geometric_knee_numeric - summary.geometric_knee) / summary.geometric_knee
    print(f"  解析解与数值解相对偏差            = {delta:.3%}")
    print(
        f"  等贡献拐点（数据项 = 参数项）随 {reference_label} 的变化："
    )
    exponent = summary.power_law_exponent
    order = np.argsort(summary.driving_values)
    values = summary.driving_values[order]
    locations = summary.equal_contribution[order]
    # 被驱动量可能有上百个取值（例如 147 个 Token 水平），只抽样汇报若干代表点。
    max_rows = 8
    if values.size > max_rows:
        picked = np.unique(np.linspace(0, values.size - 1, max_rows).round().astype(int))
    else:
        picked = np.arange(values.size)
    for index in picked:
        print(
            f"      {reference_label} = {_format_size(float(values[index])):>7s}"
            f"  ->  {_format_location(float(locations[index]))}"
        )
    if values.size > picked.size:
        print(
            f"      （共 {values.size} 个 {reference_label} 取值，此处仅列 {picked.size} 个代表点）"
        )
    print(
        f"  等贡献拐点区间                    = "
        f"{_format_location(float(locations.min()))} ~ {_format_location(float(locations.max()))}"
    )
    print(f"  等贡献拐点的幂律关系              = 正比于 {reference_label}^{exponent:.3f}")


def main() -> int:
    configure_console()
    args = build_argument_parser().parse_args()
    if args.dpi < 100:
        raise ValueError("dpi 不应低于 100。")

    fit_json_path = args.fit_json.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    formats = parse_formats(args.formats)
    figure_names = (
        ["quad", "knee-n", "knee-d", "steps"]
        if args.figures.strip().lower() == "all"
        else parse_figures(args.figures)
    )

    selected_font = configure_matplotlib()
    parameters, metrics = load_fit_result(fit_json_path)
    predictions = load_predictions(predictions_path)

    # steps 图取自原始训练日志（拟合结果里没有 steps 列），按需加载。
    training_log = (
        load_training_log(args.training_log.expanduser().resolve())
        if "steps" in figure_names
        else None
    )

    # Loss-N 图的跨族参照点（B4/B5），把参数规模覆盖延伸到 540B。
    references: list[ReferencePoints] = []
    if "knee-d" in figure_names and not args.no_references:
        for path, label in (
            (args.baseline_csv, "Cross-family, B4 (≤72B)"),
            (args.published_csv, "Published, B5 (≤540B)"),
        ):
            resolved = path.expanduser().resolve()
            if resolved.exists():
                references.append(load_reference_points(resolved, label))
            else:
                print(f"提示: 跳过不存在的参考数据 {resolved}")

    builders = {
        "quad": (draw_scaling_law, "", "Classical scaling law fit"),
        "knee-n": (
            draw_loss_vs_tokens_by_params,
            "_loss_vs_tokens",
            "Loss curves across model sizes and knees",
        ),
        "knee-d": (
            draw_loss_vs_params_by_tokens,
            "_loss_vs_params",
            "Loss curves across token budgets and knees",
        ),
        "steps": (
            draw_loss_vs_steps,
            "_loss_vs_steps",
            "Training curves coloured by model size",
        ),
    }

    all_outputs: list[Path] = []
    for name in figure_names:
        builder, suffix, title = builders[name]
        if name == "steps":
            figure = builder(
                parameters,
                metrics,
                training_log,
                loss_column=args.loss_column,
                x_log=args.steps_xscale == "log",
            )
        elif name == "knee-d":
            figure = builder(
                parameters,
                metrics,
                predictions,
                references=references,
                max_params=args.max_params,
            )
        else:
            figure = builder(parameters, metrics, predictions)
        all_outputs.extend(
            save_figure(
                figure,
                output_dir=output_dir,
                basename=f"{args.basename}{suffix}",
                formats=formats,
                dpi=args.dpi,
                title=title,
            )
        )
        plt.close(figure)

    print(f"字体: {selected_font}")
    print(f"样本数: {predictions.actual_loss.size}")
    print(
        "拟合参数: "
        f"E={parameters.irreducible_loss:.6f}  "
        f"A={parameters.parameter_coefficient:.6f}  alpha={parameters.parameter_exponent:.6f}  "
        f"B={parameters.data_coefficient:.6f}  beta={parameters.data_exponent:.6f}"
    )
    if training_log is not None:
        print(
            f"训练日志: {training_log.steps.size} 个检查点, "
            f"{np.unique(training_log.n_params_b).size} 个模型规模, "
            f"steps {int(training_log.steps.min())}–{int(training_log.steps.max())}"
        )
    if not args.quiet:
        if "knee-n" in figure_names:
            _report_knee(analyse_knee_over_tokens(parameters, predictions), "N")
        if "knee-d" in figure_names:
            _report_knee(analyse_knee_over_params(parameters, predictions), "D")
        print()
    for path in all_outputs:
        print(f"已生成: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
