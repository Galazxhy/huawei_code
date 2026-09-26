#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""经典大语言模型标度律的参数估计与验证。

拟合模型

    L(N, D) = E + A * N**(-alpha) + B * D**(-beta)

其中 N 为参数量、D 为训练 Token 数、L 为验证集交叉熵损失。数据集 B
使用的 N 和 D 单位均为十亿（1e9），本程序保持该单位不变。

程序只依赖 Python 标准库。对给定的 (alpha, beta)，E、A、B 是线性参数，
因此先用最小二乘解析估计 E、A、B，再对 alpha、beta 做粗网格搜索和局部
模式搜索。这一“可分离非线性最小二乘”方法比同时盲目搜索五个参数更稳定。

默认输入为数据集 B1：
data/real_attachments/B_scaling_laws/pythia_training_log_existing.csv

示例：

    python Q2/Q_2_1/Q_2_1_1_classic_law.py
    python Q2/Q_2_1/Q_2_1_1_classic_law.py --cv --bootstrap 200
    python Q2/Q_2_1/Q_2_1_1_classic_law.py --input other.csv --target val_loss
"""


# --- shared paths for every script in Q2 (see Q2/_q2_paths.py) --- # q2-prologue:bootstrap
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

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


DEFAULT_INPUT = (
    _q2_paths.B_DATA
    / "pythia_training_log_existing.csv"
)
DEFAULT_OUTPUT_DIR = _q2_paths.ANALYSIS_DIR / "traditional_scaling_law"


@dataclass(frozen=True)
class Observation:
    """一条用于拟合的 (N, D, Loss) 观测。"""

    row_id: str
    n_params_b: float
    d_tokens_b: float
    loss: float


@dataclass(frozen=True)
class ScalingParameters:
    """经典标度律参数。"""

    irreducible_loss: float
    parameter_coefficient: float
    parameter_exponent: float
    data_coefficient: float
    data_exponent: float

    def predict(self, n_params_b: float, d_tokens_b: float) -> float:
        if n_params_b <= 0 or d_tokens_b <= 0:
            raise ValueError("N_params_B 和 D_tokens_B 必须为正数。")
        return (
            self.irreducible_loss
            + self.parameter_coefficient
            * n_params_b ** (-self.parameter_exponent)
            + self.data_coefficient * d_tokens_b ** (-self.data_exponent)
        )


@dataclass(frozen=True)
class FitMetrics:
    sample_count: int
    sse: float
    rmse: float
    mae: float
    mape: float
    r_squared: float
    adjusted_r_squared: float


@dataclass(frozen=True)
class FitResult:
    parameters: ScalingParameters
    metrics: FitMetrics


def _positive_finite(value: str | None) -> float | None:
    """将 CSV 字段解析为正有限浮点数；不合法时返回 None。"""

    try:
        number = float((value or "").strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def load_observations(
    path: Path,
    n_column: str = "N_params_B",
    d_column: str = "D_tokens_B",
    target_column: str = "val_loss",
) -> tuple[list[Observation], int]:
    """读取并清洗 CSV，返回有效观测和被剔除的行数。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    observations: list[Observation] = []
    dropped = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {n_column, d_column, target_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"输入文件缺少字段 {sorted(missing)}；实际字段为 {reader.fieldnames}"
            )

        for index, row in enumerate(reader, start=2):
            n_value = _positive_finite(row.get(n_column))
            d_value = _positive_finite(row.get(d_column))
            loss = _positive_finite(row.get(target_column))
            if n_value is None or d_value is None or loss is None:
                dropped += 1
                continue
            observations.append(
                Observation(
                    row_id=(row.get("run_id") or str(index)).strip(),
                    n_params_b=n_value,
                    d_tokens_b=d_value,
                    loss=loss,
                )
            )

    if len(observations) < 6:
        raise ValueError("有效样本少于 6 条，无法稳定估计五个标度律参数。")
    if len({item.n_params_b for item in observations}) < 2:
        raise ValueError("N_params_B 至少需要两个不同取值。")
    if len({item.d_tokens_b for item in observations}) < 2:
        raise ValueError("D_tokens_B 至少需要两个不同取值。")
    return observations, dropped


def _solve_3x3(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> tuple[float, float, float] | None:
    """带主元选择的 3x3 高斯消元。奇异时返回 None。"""

    augmented = [list(matrix[row]) + [float(vector[row])] for row in range(3)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        scale = max(abs(value) for row in augmented for value in row[:3])
        if abs(augmented[pivot][column]) <= max(1.0, scale) * 1e-14:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        pivot_value = augmented[column][column]
        for item in range(column, 4):
            augmented[column][item] /= pivot_value

        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            for item in range(column, 4):
                augmented[row][item] -= factor * augmented[column][item]

    return tuple(augmented[row][3] for row in range(3))  # type: ignore[return-value]


def _linear_fit_for_exponents(
    observations: Sequence[Observation],
    alpha: float,
    beta: float,
    enforce_physical_constraints: bool = True,
) -> tuple[ScalingParameters, float] | None:
    """固定非线性指数后，在线性子问题中估计 ``(E, A, B)``。"""

    count = len(observations)
    sum_x = sum_z = sum_xx = sum_zz = sum_xz = 0.0
    sum_y = sum_xy = sum_zy = 0.0
    min_loss = math.inf
    transformed: list[tuple[float, float, float]] = []

    for item in observations:
        x_value = item.n_params_b ** (-alpha)
        z_value = item.d_tokens_b ** (-beta)
        y_value = item.loss
        transformed.append((x_value, z_value, y_value))
        sum_x += x_value
        sum_z += z_value
        sum_xx += x_value * x_value
        sum_zz += z_value * z_value
        sum_xz += x_value * z_value
        sum_y += y_value
        sum_xy += x_value * y_value
        sum_zy += z_value * y_value
        min_loss = min(min_loss, y_value)

    # 设计矩阵的三列为 1、N^(-alpha)、D^(-beta)；这里只解三维正规方程。
    coefficients = _solve_3x3(
        (
            (float(count), sum_x, sum_z),
            (sum_x, sum_xx, sum_xz),
            (sum_z, sum_xz, sum_zz),
        ),
        (sum_y, sum_xy, sum_zy),
    )
    if coefficients is None:
        return None

    irreducible_loss, parameter_coefficient, data_coefficient = coefficients
    # 正系数与 E<min(L) 保证规模增大时损失下降，且不可约损失低于观测值。
    if enforce_physical_constraints and (
        irreducible_loss < 0
        or irreducible_loss >= min_loss
        or parameter_coefficient <= 0
        or data_coefficient <= 0
    ):
        return None

    parameters = ScalingParameters(
        irreducible_loss=irreducible_loss,
        parameter_coefficient=parameter_coefficient,
        parameter_exponent=alpha,
        data_coefficient=data_coefficient,
        data_exponent=beta,
    )
    sse = 0.0
    for x_value, z_value, y_value in transformed:
        prediction = (
            irreducible_loss
            + parameter_coefficient * x_value
            + data_coefficient * z_value
        )
        residual = y_value - prediction
        sse += residual * residual
    return parameters, sse


def _grid_values(lower: float, upper: float, count: int) -> Iterable[float]:
    if count < 2:
        yield (lower + upper) / 2.0
        return
    step = (upper - lower) / (count - 1)
    for index in range(count):
        yield lower + index * step


def fit_scaling_law(
    observations: Sequence[Observation],
    exponent_lower: float = 0.01,
    exponent_upper: float = 1.50,
    grid_size: int = 31,
    tolerance: float = 1e-7,
    enforce_physical_constraints: bool = True,
) -> FitResult:
    """先消去线性参数，再在 ``(alpha, beta)`` 平面搜索最小残差。

    每个网格点均重新最小二乘拟合 ``E,A,B``；粗搜后以八邻域逐级细化，
    因而非线性搜索始终只有两个指数维度。
    """

    if exponent_lower <= 0 or exponent_upper <= exponent_lower:
        raise ValueError("指数搜索范围必须满足 0 < lower < upper。")
    if grid_size < 5:
        raise ValueError("grid_size 至少为 5。")

    # 第一阶段：指数粗网格；不可行的线性解不参加比较。
    best: tuple[ScalingParameters, float] | None = None
    for alpha in _grid_values(exponent_lower, exponent_upper, grid_size):
        for beta in _grid_values(exponent_lower, exponent_upper, grid_size):
            candidate = _linear_fit_for_exponents(
                observations, alpha, beta, enforce_physical_constraints
            )
            if candidate is not None and (best is None or candidate[1] < best[1]):
                best = candidate

    if best is None:
        raise RuntimeError("在给定指数范围内没有找到满足约束的可行解。")

    # 第二阶段：围绕当前最优指数作八邻域搜索，停滞时将步长减半。
    step = (exponent_upper - exponent_lower) / (grid_size - 1)
    while step > tolerance:
        current_parameters, _ = best
        improved = False
        for alpha_delta in (-step, 0.0, step):
            for beta_delta in (-step, 0.0, step):
                if alpha_delta == 0.0 and beta_delta == 0.0:
                    continue
                alpha = current_parameters.parameter_exponent + alpha_delta
                beta = current_parameters.data_exponent + beta_delta
                if not (
                    exponent_lower <= alpha <= exponent_upper
                    and exponent_lower <= beta <= exponent_upper
                ):
                    continue
                candidate = _linear_fit_for_exponents(
                    observations, alpha, beta, enforce_physical_constraints
                )
                if candidate is not None and candidate[1] < best[1] - 1e-18:
                    best = candidate
                    improved = True
        if not improved:
            step /= 2.0

    parameters, _ = best
    return FitResult(parameters=parameters, metrics=calculate_metrics(observations, parameters))


def calculate_metrics(
    observations: Sequence[Observation], parameters: ScalingParameters
) -> FitMetrics:
    predictions = [
        parameters.predict(item.n_params_b, item.d_tokens_b)
        for item in observations
    ]
    residuals = [
        item.loss - prediction
        for item, prediction in zip(observations, predictions)
    ]
    sse = sum(value * value for value in residuals)
    mean_loss = statistics.fmean(item.loss for item in observations)
    total_sum_squares = sum((item.loss - mean_loss) ** 2 for item in observations)
    r_squared = 1.0 - sse / total_sum_squares if total_sum_squares > 0 else math.nan
    count = len(observations)
    parameter_count = 5
    adjusted_r_squared = (
        1.0
        - (1.0 - r_squared) * (count - 1) / (count - parameter_count - 1)
        if count > parameter_count + 1 and math.isfinite(r_squared)
        else math.nan
    )
    return FitMetrics(
        sample_count=count,
        sse=sse,
        rmse=math.sqrt(sse / count),
        mae=statistics.fmean(abs(value) for value in residuals),
        mape=statistics.fmean(
            abs(value) / item.loss
            for item, value in zip(observations, residuals)
        ),
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
    )


def grouped_residual_summary(
    observations: Sequence[Observation], parameters: ScalingParameters
) -> list[dict[str, float | int]]:
    """按参数规模汇总残差，用于识别系统性偏差。"""

    groups: dict[float, list[float]] = {}
    for item in observations:
        residual = item.loss - parameters.predict(item.n_params_b, item.d_tokens_b)
        groups.setdefault(item.n_params_b, []).append(residual)

    summary: list[dict[str, float | int]] = []
    for n_value in sorted(groups):
        residuals = groups[n_value]
        summary.append(
            {
                "N_params_B": n_value,
                "sample_count": len(residuals),
                "mean_residual": statistics.fmean(residuals),
                "rmse": math.sqrt(statistics.fmean(value * value for value in residuals)),
                "max_abs_residual": max(abs(value) for value in residuals),
            }
        )
    return summary


def leave_one_size_out_cv(
    observations: Sequence[Observation],
    exponent_lower: float,
    exponent_upper: float,
    grid_size: int,
) -> dict[str, object]:
    """按 N 取值留一交叉验证，避免随机拆分同一训练轨迹造成信息泄漏。"""

    all_predictions: list[tuple[float, float]] = []
    folds: list[dict[str, float | int]] = []
    for held_out_n in sorted({item.n_params_b for item in observations}):
        train = [item for item in observations if item.n_params_b != held_out_n]
        test = [item for item in observations if item.n_params_b == held_out_n]
        result = fit_scaling_law(
            train,
            exponent_lower=exponent_lower,
            exponent_upper=exponent_upper,
            grid_size=grid_size,
        )
        squared_errors = []
        absolute_errors = []
        for item in test:
            prediction = result.parameters.predict(item.n_params_b, item.d_tokens_b)
            all_predictions.append((item.loss, prediction))
            squared_errors.append((item.loss - prediction) ** 2)
            absolute_errors.append(abs(item.loss - prediction))
        folds.append(
            {
                "held_out_N_params_B": held_out_n,
                "sample_count": len(test),
                "rmse": math.sqrt(statistics.fmean(squared_errors)),
                "mae": statistics.fmean(absolute_errors),
            }
        )

    squared_errors = [(actual - predicted) ** 2 for actual, predicted in all_predictions]
    absolute_errors = [abs(actual - predicted) for actual, predicted in all_predictions]
    return {
        "method": "leave-one-N-size-out",
        "fold_count": len(folds),
        "rmse": math.sqrt(statistics.fmean(squared_errors)),
        "mae": statistics.fmean(absolute_errors),
        "folds": folds,
    }


def block_bootstrap(
    observations: Sequence[Observation],
    repetitions: int,
    seed: int,
    exponent_lower: float,
    exponent_upper: float,
    grid_size: int,
) -> dict[str, object]:
    """以模型参数规模为块进行 Bootstrap，并给出五个参数的 95% 区间。"""

    if repetitions <= 0:
        return {}
    random_generator = random.Random(seed)
    grouped: dict[float, list[Observation]] = {}
    for item in observations:
        grouped.setdefault(item.n_params_b, []).append(item)
    group_keys = sorted(grouped)

    estimates: list[ScalingParameters] = []
    failed = 0
    for _ in range(repetitions):
        selected = random_generator.choices(group_keys, k=len(group_keys))
        if len(set(selected)) < 2:
            failed += 1
            continue
        sample: list[Observation] = []
        for key in selected:
            sample.extend(grouped[key])
        try:
            result = fit_scaling_law(
                sample,
                exponent_lower=exponent_lower,
                exponent_upper=exponent_upper,
                grid_size=grid_size,
                tolerance=5e-6,
            )
        except (RuntimeError, ValueError):
            failed += 1
            continue
        estimates.append(result.parameters)

    if not estimates:
        raise RuntimeError("Bootstrap 未产生任何可用拟合结果。")

    def percentile(values: Sequence[float], probability: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    intervals: dict[str, dict[str, float]] = {}
    for field_name in ScalingParameters.__dataclass_fields__:
        values = [getattr(item, field_name) for item in estimates]
        intervals[field_name] = {
            "median": percentile(values, 0.50),
            "lower_95": percentile(values, 0.025),
            "upper_95": percentile(values, 0.975),
        }
    return {
        "method": "N-size block bootstrap",
        "requested_repetitions": repetitions,
        "successful_repetitions": len(estimates),
        "failed_repetitions": failed,
        "seed": seed,
        "intervals": intervals,
    }


def save_predictions(
    path: Path, observations: Sequence[Observation], parameters: ScalingParameters
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "row_id",
                "N_params_B",
                "D_tokens_B",
                "actual_loss",
                "predicted_loss",
                "residual",
                "absolute_percentage_error",
            ),
        )
        writer.writeheader()
        for item in observations:
            predicted = parameters.predict(item.n_params_b, item.d_tokens_b)
            residual = item.loss - predicted
            writer.writerow(
                {
                    "row_id": item.row_id,
                    "N_params_B": f"{item.n_params_b:.12g}",
                    "D_tokens_B": f"{item.d_tokens_b:.12g}",
                    "actual_loss": f"{item.loss:.12g}",
                    "predicted_loss": f"{predicted:.12g}",
                    "residual": f"{residual:.12g}",
                    "absolute_percentage_error": f"{abs(residual) / item.loss:.12g}",
                }
            )


def print_report(
    input_path: Path,
    dropped: int,
    result: FitResult,
    residual_summary: Sequence[dict[str, float | int]],
    cv_result: dict[str, object] | None,
) -> None:
    parameters = result.parameters
    metrics = result.metrics
    print("=" * 78)
    print("经典标度律拟合结果")
    print("=" * 78)
    print(f"输入文件: {input_path}")
    print(f"有效样本: {metrics.sample_count}; 剔除无效样本: {dropped}")
    print(
        "模型: L(N,D) = E + A*N^(-alpha) + B*D^(-beta)\n"
        f"E     = {parameters.irreducible_loss:.10g}\n"
        f"A     = {parameters.parameter_coefficient:.10g}\n"
        f"alpha = {parameters.parameter_exponent:.10g}\n"
        f"B     = {parameters.data_coefficient:.10g}\n"
        f"beta  = {parameters.data_exponent:.10g}"
    )
    print("-" * 78)
    print(f"R²       = {metrics.r_squared:.10f}")
    print(f"调整 R²  = {metrics.adjusted_r_squared:.10f}")
    print(f"RMSE     = {metrics.rmse:.10g}")
    print(f"MAE      = {metrics.mae:.10g}")
    print(f"MAPE     = {metrics.mape:.6%}")
    print("-" * 78)
    print("按参数规模汇总的残差：")
    print(f"{'N(B)':>12} {'n':>7} {'平均残差':>14} {'RMSE':>14} {'最大绝对残差':>16}")
    for row in residual_summary:
        print(
            f"{float(row['N_params_B']):12.6g}"
            f" {int(row['sample_count']):7d}"
            f" {float(row['mean_residual']):14.6g}"
            f" {float(row['rmse']):14.6g}"
            f" {float(row['max_abs_residual']):16.6g}"
        )
    if cv_result:
        print("-" * 78)
        print(
            "留一参数规模交叉验证: "
            f"RMSE={float(cv_result['rmse']):.6g}, "
            f"MAE={float(cv_result['mae']):.6g}"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="拟合经典标度律 L=E+A*N^(-alpha)+B*D^(-beta)。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 CSV。")
    parser.add_argument("--n-column", default="N_params_B", help="参数量字段名。")
    parser.add_argument("--d-column", default="D_tokens_B", help="Token 数字段名。")
    parser.add_argument("--target", default="val_loss", help="Loss 字段名。")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="结果输出目录。"
    )
    parser.add_argument("--exponent-lower", type=float, default=0.01)
    parser.add_argument("--exponent-upper", type=float, default=1.50)
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument(
        "--cv", action="store_true", help="执行按参数规模留一交叉验证。"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="B",
        help="执行 B 次按参数规模分块的 Bootstrap；默认不执行。",
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--allow-nonphysical",
        action="store_true",
        help="允许 E<0、E>=最小 Loss 或 A/B<=0，仅用于诊断。",
    )
    return parser


def main() -> int:
    # Windows 的旧版控制台可能默认使用 GBK，无法输出 R² 等字符。
    # 明确切换为 UTF-8，同时保留不支持 reconfigure 的解释器兼容性。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    args = build_argument_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    observations, dropped = load_observations(
        input_path,
        n_column=args.n_column,
        d_column=args.d_column,
        target_column=args.target,
    )
    result = fit_scaling_law(
        observations,
        exponent_lower=args.exponent_lower,
        exponent_upper=args.exponent_upper,
        grid_size=args.grid_size,
        enforce_physical_constraints=not args.allow_nonphysical,
    )
    residual_summary = grouped_residual_summary(observations, result.parameters)

    cv_result = None
    if args.cv:
        cv_result = leave_one_size_out_cv(
            observations,
            exponent_lower=args.exponent_lower,
            exponent_upper=args.exponent_upper,
            grid_size=max(21, args.grid_size),
        )

    bootstrap_result = block_bootstrap(
        observations,
        repetitions=args.bootstrap,
        seed=args.seed,
        exponent_lower=args.exponent_lower,
        exponent_upper=args.exponent_upper,
        grid_size=max(17, min(args.grid_size, 25)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "traditional_scaling_predictions.csv"
    result_path = output_dir / "traditional_scaling_fit.json"
    save_predictions(prediction_path, observations, result.parameters)

    payload = {
        "model": "L(N,D) = E + A*N^(-alpha) + B*D^(-beta)",
        "units": {"N": "billion parameters", "D": "billion tokens"},
        "input_file": _q2_paths.repo_relative(input_path),
        "target_column": args.target,
        "dropped_rows": dropped,
        "parameters": asdict(result.parameters),
        "metrics": asdict(result.metrics),
        "residual_summary_by_N": residual_summary,
        "cross_validation": cv_result,
        "bootstrap": bootstrap_result or None,
    }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print_report(input_path, dropped, result, residual_summary, cv_result)
    print("-" * 78)
    print(f"参数与诊断: {result_path}")
    print(f"逐样本预测: {prediction_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
