#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按赛题要求完成经典标度律拟合、跨来源验证与大模型外推。

数据使用路线
------------
1. B1：经典标度律的唯一参数拟合数据；
2. B2--B5：经分簇交叉尺度归一化后，用于自适应加权结构早停与外部验证；
3. 早停选中的经典参数继续用于大模型外推；
4. B9、B10：匹配百亿参数以上模型，检验经典标度律的外推表现。

经典模型
--------
::

    L(N, D) = E + A * N^(-alpha) + B * D^(-beta)

标准标度律只含参数量 N 与训练数据量 D，不含数据质量 Q；因此本脚本只辨识这五个
参数，不拟合质量模型，也不报告任何 Q 情景。

数据质量 Q 与训练配比 p 由独立的广义标度律脚本承担。它们复用本脚本写出的 B1
基准律，但数据、参数与产物目录都相互独立：

    q2/generalized_law.py    # (E_k, A_k, B_k, eta_k, zeta_k, Psi_k) 辨识
    q2/generalized_law_figures.py   # 出图

**B6--B8 不参与任何参数估计**，仅在需要时作为事后对照。

附件 B 的实验与附件 A 相互独立，且 B7 为半合成数据，本程序不使用 B7 参数替换 B1
的真实数据参数。

默认运行：

    conda run -n GNN python q2/classic_law_analysis.py

结果写入 ``data_analysis/scaling_law_full/``。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr

from classic_law import (
    FitResult,
    Observation,
    ScalingParameters,
    fit_scaling_law,
)
from classic_law_early_stopping import (
    MODEL_PATH,
    CandidateFit,
    candidate_payload,
    cross_fitted_loss_normalization,
    derive_adaptive_weights,
    run_parameter_release_early_stopping,
    write_history_csv,
    write_predictions_csv as write_early_stopping_predictions,
    write_weight_diagnostics_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_B_DIR = PROJECT_ROOT / "data" / "real_attachments" / "B_scaling_laws"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_analysis" / "scaling_law_full"


@dataclass(frozen=True)
class Metrics:
    sample_count: int
    rmse: float
    mae: float
    mape: float
    median_ape: float
    p90_ape: float
    bias: float
    r_squared: float
    spearman_rho: float


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _float(value: str | None) -> float | None:
    try:
        parsed = float((value or "").strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_classic_observations(
    path: Path,
    row_id_column: str,
    converged_only: bool = False,
    final_checkpoint_only: bool = False,
    step_column: str = "steps",
) -> list[Observation]:
    """读取标度律观测；可按运行选择最大训练步的最终 checkpoint。"""

    indexed_rows = list(enumerate(read_csv(path), start=2))
    if converged_only:
        indexed_rows = [
            (index, row)
            for index, row in indexed_rows
            if (row.get("is_converged") or "").strip()
            in {"1", "1.0", "true", "True"}
        ]
    if final_checkpoint_only:
        latest_by_run: dict[str, tuple[int, dict[str, str]]] = {}
        latest_key: dict[str, tuple[float, float, int]] = {}
        for index, row in indexed_rows:
            run_id = (row.get(row_id_column) or str(index)).strip()
            step_value = _float(row.get(step_column))
            token_value = _float(row.get("D_tokens_B"))
            ordering = (
                step_value if step_value is not None else -math.inf,
                token_value if token_value is not None else -math.inf,
                index,
            )
            if run_id not in latest_key or ordering > latest_key[run_id]:
                latest_key[run_id] = ordering
                latest_by_run[run_id] = (index, row)
        indexed_rows = sorted(latest_by_run.values(), key=lambda item: item[0])

    observations: list[Observation] = []
    for index, row in indexed_rows:
        n_value = _float(row.get("N_params_B"))
        d_value = _float(row.get("D_tokens_B"))
        loss = _float(row.get("val_loss"))
        if (
            n_value is None
            or d_value is None
            or loss is None
            or min(n_value, d_value, loss) <= 0
        ):
            continue
        observations.append(
            Observation(
                row_id=(row.get(row_id_column) or str(index)).strip(),
                n_params_b=n_value,
                d_tokens_b=d_value,
                loss=loss,
            )
        )
    if not observations:
        raise ValueError(f"{path.name} 中没有可用的 N-D-Loss 记录。")
    return observations


def load_b3_observations(b_dir: Path) -> list[Observation]:
    observations: list[Observation] = []
    trajectory_dir = b_dir / "training_trajectories"
    files = sorted(trajectory_dir.glob("pythia_*_trajectory.csv"))
    if not files:
        raise FileNotFoundError(f"{trajectory_dir} 中没有 B3 轨迹文件。")
    for file_path in files:
        for index, row in enumerate(read_csv(file_path), start=2):
            n_value = _float(row.get("N_params_B"))
            d_value = _float(row.get("D_tokens_B"))
            loss = _float(row.get("val_loss"))
            if (
                n_value is None
                or d_value is None
                or loss is None
                or min(n_value, d_value, loss) <= 0
            ):
                continue
            observations.append(
                Observation(
                    row_id=f"{file_path.stem}:{index}",
                    n_params_b=n_value,
                    d_tokens_b=d_value,
                    loss=loss,
                )
            )
    return observations


def classic_predictions(
    observations: Sequence[Observation], parameters: ScalingParameters
) -> np.ndarray:
    return np.asarray(
        [
            parameters.predict(item.n_params_b, item.d_tokens_b)
            for item in observations
        ],
        dtype=float,
    )


def observed_losses(observations: Sequence[Observation]) -> np.ndarray:
    return np.asarray([item.loss for item in observations], dtype=float)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Metrics:
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("actual 和 predicted 必须为等长非空数组。")
    residual = actual - predicted
    absolute_percentage_error = np.abs(residual) / np.maximum(np.abs(actual), 1e-12)
    total_sum_squares = float(np.sum(np.square(actual - actual.mean())))
    residual_sum_squares = float(np.sum(np.square(residual)))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else math.nan
    )
    rho = spearmanr(actual, predicted).statistic
    return Metrics(
        sample_count=int(actual.size),
        rmse=float(np.sqrt(np.mean(np.square(residual)))),
        mae=float(np.mean(np.abs(residual))),
        mape=float(np.mean(absolute_percentage_error)),
        median_ape=float(np.median(absolute_percentage_error)),
        p90_ape=float(np.quantile(absolute_percentage_error, 0.90)),
        bias=float(np.mean(residual)),
        r_squared=float(r_squared),
        spearman_rho=float(rho),
    )


def fit_classic(
    observations: Sequence[Observation], grid_size: int
) -> FitResult:
    return fit_scaling_law(
        observations,
        exponent_lower=0.01,
        exponent_upper=1.50,
        grid_size=grid_size,
        tolerance=1e-7,
        enforce_physical_constraints=True,
    )


def candidate_to_scaling_parameters(candidate: CandidateFit) -> ScalingParameters:
    """把任一指数约束阶段转换为统一的标准标度律参数。"""

    values = candidate.parameters
    if len(values) != 5:
        raise ValueError(
            f"候选结构 {candidate.spec.name!r} 没有返回标准五参数表示。"
        )
    e_value, a_value, alpha, b_value, beta = values
    return ScalingParameters(
        irreducible_loss=float(e_value),
        parameter_coefficient=float(a_value),
        parameter_exponent=float(alpha),
        data_coefficient=float(b_value),
        data_exponent=float(beta),
    )


def affine_recalibration(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[float, float, Metrics]:
    """诊断不同数据源是否主要存在 Loss 尺度偏移，不作为主验证指标。"""

    design = np.column_stack((np.ones_like(predicted), predicted))
    intercept, slope = np.linalg.lstsq(design, actual, rcond=None)[0]
    recalibrated = intercept + slope * predicted
    return float(intercept), float(slope), calculate_metrics(actual, recalibrated)


def validate_classic_dataset(
    dataset_name: str,
    role: str,
    nature: str,
    observations: Sequence[Observation],
    b1_parameters: ScalingParameters,
    grid_size: int,
    self_fit: bool,
    note: str,
) -> dict[str, object]:
    actual = observed_losses(observations)
    b1_prediction = classic_predictions(observations, b1_parameters)
    cluster_ids = [
        (
            item.row_id.split(":", maxsplit=1)[0]
            if dataset_name == "B3"
            else item.row_id
        )
        for item in observations
    ]
    scale_normalization = cross_fitted_loss_normalization(
        actual, b1_prediction, cluster_ids=cluster_ids
    )
    intercept, slope, recalibrated_metrics = affine_recalibration(
        actual, b1_prediction
    )
    payload: dict[str, object] = {
        "dataset": dataset_name,
        "role": role,
        "nature": nature,
        "sample_count": len(observations),
        "b1_direct_transfer_metrics": asdict(
            calculate_metrics(actual, b1_prediction)
        ),
        "cross_fitted_scale_normalized_validation": scale_normalization,
        "affine_scale_diagnostic": {
            "actual_approximately": f"{intercept:.8g} + {slope:.8g} * B1_prediction",
            "intercept": intercept,
            "slope": slope,
            "metrics_after_in_sample_affine_recalibration": asdict(
                recalibrated_metrics
            ),
            "warning": "仅用于识别跨数据源 Loss 尺度差异，不是外部验证成绩。",
        },
        "note": note,
    }
    if self_fit:
        try:
            fitted = fit_classic(observations, grid_size=grid_size)
            payload["same_form_self_fit"] = {
                "parameters": asdict(fitted.parameters),
                "metrics": asdict(fitted.metrics),
            }
        except (RuntimeError, ValueError) as error:
            payload["same_form_self_fit"] = {"error": str(error)}
    return payload


def analyze_large_models(
    b_dir: Path,
    output_path: Path,
    b1_parameters: ScalingParameters,
) -> dict[str, object]:
    """联合 B9 元数据与 B10 估算 Loss，检验经典标度律在百亿参数以上的外推。

    只使用 ``L(N, D) = E + A*N^-alpha + B*D^-beta``：标准标度律不含数据质量
    维度，因此不再构造任何 Q 情景。
    """

    b9_rows = read_csv(b_dir / "supplementary_large_models.csv")
    b10_rows = read_csv(b_dir / "supplementary_large_baseline.csv")
    b9_by_name = {
        (row.get("model_name") or "").strip(): row
        for row in b9_rows
        if (row.get("model_name") or "").strip()
    }
    b10_by_name = {
        (row.get("family") or "").strip(): row
        for row in b10_rows
        if (row.get("family") or "").strip()
    }
    matched_names = sorted(set(b9_by_name).intersection(b10_by_name))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    actual_estimates: list[float] = []
    b1_predictions: list[float] = []
    written = 0
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "N_params_B",
                "D_tokens_B",
                "B10_estimated_loss",
                "B1_prediction",
                "relative_error_vs_B10_estimate",
                "publication_date",
                "organization",
                "accessibility",
                "country",
            ],
        )
        writer.writeheader()
        for name in matched_names:
            metadata = b9_by_name[name]
            baseline = b10_by_name[name]
            n_value = _float(baseline.get("N_params_B"))
            d_value = _float(baseline.get("D_tokens_B"))
            estimated_loss = _float(baseline.get("val_loss"))
            if (
                n_value is None
                or d_value is None
                or estimated_loss is None
                or min(n_value, d_value, estimated_loss) <= 0
                or n_value < 10
            ):
                continue
            classic_prediction = b1_parameters.predict(n_value, d_value)
            writer.writerow(
                {
                    "model_name": name,
                    "N_params_B": n_value,
                    "D_tokens_B": d_value,
                    "B10_estimated_loss": estimated_loss,
                    "B1_prediction": classic_prediction,
                    "relative_error_vs_B10_estimate": abs(
                        classic_prediction - estimated_loss
                    )
                    / estimated_loss,
                    "publication_date": metadata.get("publication_date", ""),
                    "organization": metadata.get("organization", ""),
                    "accessibility": metadata.get("accessibility", ""),
                    "country": metadata.get("country", ""),
                }
            )
            written += 1
            actual_estimates.append(estimated_loss)
            b1_predictions.append(classic_prediction)

    metrics = calculate_metrics(
        np.asarray(actual_estimates, dtype=float),
        np.asarray(b1_predictions, dtype=float),
    )
    return {
        "B9_row_count": len(b9_rows),
        "B10_row_count": len(b10_rows),
        "matched_name_count": len(matched_names),
        "usable_10B_plus_count": written,
        "B1_direct_transfer_against_B10_estimates": asdict(metrics),
        "limitations": [
            "B9 提供模型规模与元数据，但没有可比的实测 Loss。",
            "B10 的 val_loss 是标度律估算值，不是直接训练实验观测。",
            "B1 对 B10 的高拟合度属于内部一致性检查，不构成独立验证。",
        ],
    }


def write_markdown_report(path: Path, payload: dict[str, object]) -> None:
    """把核心数值和可信度边界写成可直接用于论文整理的 Markdown。"""

    b1 = payload["B1_main_fit"]
    validations = payload["classic_validations"]
    large = payload["large_model_extrapolation"]
    assert isinstance(b1, dict)
    assert isinstance(validations, list)
    assert isinstance(large, dict)
    bp = b1["parameters"]
    bm = b1["metrics"]
    assert isinstance(bp, dict) and isinstance(bm, dict)

    lines = [
        "# 标度律拟合与验证结果",
        "",
        "## B1 经典标度律主拟合",
        "",
        "经典模型为",
        "",
        "```text",
        "L(N,D) = E + A*N^(-alpha) + B*D^(-beta)",
        "```",
        "",
        (
            f"B1 共使用 {b1['sample_count']} 条真实轨迹记录，估计结果为 "
            f"E={bp['irreducible_loss']:.6f}，A={bp['parameter_coefficient']:.6f}，"
            f"alpha={bp['parameter_exponent']:.6f}，B={bp['data_coefficient']:.6f}，"
            f"beta={bp['data_exponent']:.6f}。拟合 R²={bm['r_squared']:.8f}，"
            f"RMSE={bm['rmse']:.6g}。"
        ),
        "",
        "## 轨迹、跨族与文献验证",
        "",
        "| 数据 | 角色 | 样本数 | 原始 RMSE | 尺度归一化 RMSE | 归一化中位相对误差 | 归一化 Spearman |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    early = payload.get("adaptive_early_stopping")
    if isinstance(early, dict) and early.get("enabled"):
        chosen = early["chosen_model"]
        weights = early["weights"]
        candidates = early["candidates"]
        assert isinstance(chosen, dict)
        assert isinstance(weights, dict)
        assert isinstance(candidates, list)
        trajectory_header_index = lines.index("## 轨迹、跨族与文献验证")
        early_lines = [
            "## 自适应权重结构早停",
            "",
            (
                f"早停选择 `{chosen['model']}`，B1 R²="
                f"{chosen['b1_metrics']['r_squared']:.8f}，"
                f"验证综合分数={chosen['validation_score']:.6f}。"
            ),
            "",
            (
                "自适应权重为 "
                + "，".join(
                    f"{name}={weights[name]:.4f}"
                    for name in ("B2", "B3", "B4", "B5")
                )
                + "。权重在候选模型比较前确定并冻结。"
            ),
            "",
            (
                "模型形式始终保持为 $L=E+A N^{-\\alpha}+B D^{-\\beta}$："
                "每个指数候选点以线性最小二乘估计 $E,A,B$，依次执行共享指数"
                "一维搜索、单方向指数释放和完整二维指数搜索。"
            ),
            "",
            "| 约束层 | 指数约束结构 | B1 R² | 验证分数 | 选模分数 | 早停 |",
            "|---:|---|---:|---:|---:|---|",
        ]
        for candidate in candidates:
            early_lines.append(
                f"| {candidate['constraint_layer']} | {candidate['model']} | "
                f"{candidate['b1_metrics']['r_squared']:.8f} | "
                f"{candidate['validation_score']:.6f} | "
                f"{candidate['selection_score']:.6f} | "
                f"{'是' if candidate['stop_triggered'] else '否'} |"
            )
        early_lines.extend(["", "## 轨迹、跨族与文献验证", ""])
        lines[trajectory_header_index : trajectory_header_index + 2] = early_lines
    for item in validations:
        direct = item["b1_direct_transfer_metrics"]
        normalization = item["cross_fitted_scale_normalized_validation"]
        metrics = normalization["normalized_metrics"]
        lines.append(
            f"| {item['dataset']} | {item['role']} | {direct['sample_count']} | "
            f"{direct['rmse']:.6f} | {metrics['rmse']:.6f} | "
            f"{metrics['median_ape']:.2%} | "
            f"{metrics['spearman_rho']:.4f} |"
        )
    lines.extend(
        [
            "",
            "B9 提供真实模型规模与元数据，但没有可比的实测 Loss；B10 的 Loss 是标度律估算值。因此 B1 对 B10 的高拟合度不能视为新的外部实验证据，B9/B10 部分应表述为大模型范围内的外推与情景分析。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def flatten_validation_rows(
    validations: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for validation in validations:
        direct = validation["b1_direct_transfer_metrics"]
        normalization = validation["cross_fitted_scale_normalized_validation"]
        normalized = normalization["normalized_metrics"]
        raw_calibrated = normalization["raw_calibrated_metrics"]
        if not all(
            isinstance(item, dict)
            for item in (direct, normalization, normalized, raw_calibrated)
        ):
            continue
        row: dict[str, object] = {
            "dataset": validation["dataset"],
            "role": validation["role"],
            "nature": validation["nature"],
            "normalization_method": normalization["method"],
            "normalization_fold_count": normalization["fold_count"],
        }
        row.update({f"raw_{key}": value for key, value in direct.items()})
        row.update(
            {f"normalized_{key}": value for key, value in normalized.items()}
        )
        row.update(
            {
                f"raw_cross_calibrated_{key}": value
                for key, value in raw_calibrated.items()
            }
        )
        row["note"] = validation["note"]
        rows.append(
            row
        )
    return rows


def write_dict_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def print_summary(payload: dict[str, object], output_dir: Path) -> None:
    b1 = payload["B1_main_fit"]
    early = payload.get("adaptive_early_stopping")
    print("=" * 88)
    print("标度律综合分析完成")
    print("=" * 88)
    if isinstance(b1, dict):
        parameters = b1["parameters"]
        metrics = b1["metrics"]
        if isinstance(parameters, dict) and isinstance(metrics, dict):
            print(
                "B1 主拟合: "
                f"E={parameters['irreducible_loss']:.6f}, "
                f"alpha={parameters['parameter_exponent']:.6f}, "
                f"beta={parameters['data_exponent']:.6f}, "
                f"R²={metrics['r_squared']:.8f}"
            )
    if isinstance(early, dict) and early.get("enabled"):
        chosen = early["chosen_model"]
        weights = early["weights"]
        print(
            f"自适应早停: {chosen['model']}, "
            + ", ".join(
                f"{name}={weights[name]:.4f}"
                for name in ("B2", "B3", "B4", "B5")
            )
        )
    for item in payload["classic_validations"]:
        direct = item["b1_direct_transfer_metrics"]
        normalization = item["cross_fitted_scale_normalized_validation"]
        metrics = normalization["normalized_metrics"]
        print(
            f"{item['dataset']}: n={direct['sample_count']}, "
            f"原始 RMSE={direct['rmse']:.6f}, "
            f"归一化 RMSE={metrics['rmse']:.6f}, "
            f"归一化中位相对误差={metrics['median_ape']:.2%}, "
            f"归一化 Spearman={metrics['spearman_rho']:.4f}"
        )
    large = payload["large_model_extrapolation"]
    if isinstance(large, dict):
        print(
            f"B9/B10: 匹配 {large['matched_name_count']} 个，"
            f"可用于 10B+ 外推 {large['usable_10B_plus_count']} 个。"
        )
    print(f"结果目录: {output_dir}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="执行 B1-B10 经典标度律拟合、跨来源验证与大模型外推。"
    )
    parser.add_argument("--b-dir", type=Path, default=DEFAULT_B_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--trajectory-validation",
        choices=("b2", "b3", "both"),
        default="both",
        help="写入主验证表的轨迹数据；自适应早停始终使用 B2--B5。",
    )
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="关闭结构早停，恢复使用未约束的 B1 五参数经典拟合。",
    )
    parser.add_argument(
        "--weight-mode",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="早停的 B2--B5 权重；默认按难度、可信度和稳定性自适应确定。",
    )
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.002)
    parser.add_argument("--early-stop-min-b1-r2", type=float, default=0.995)
    parser.add_argument("--early-stop-complexity-penalty", type=float, default=0.001)
    parser.add_argument("--validation-mape-cap", type=float, default=0.20)
    parser.add_argument("--weight-bootstrap", type=int, default=200)
    parser.add_argument("--weight-seed", type=int, default=2025)
    parser.add_argument("--weight-difficulty-power", type=float, default=0.50)
    parser.add_argument("--min-dataset-weight", type=float, default=0.05)
    parser.add_argument("--max-dataset-weight", type=float, default=0.50)

    return parser


def main() -> int:
    configure_console()
    args = build_argument_parser().parse_args()
    b_dir = args.b_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.early_stop_patience < 1:
        raise ValueError("--early-stop-patience 必须大于等于 1。")
    if args.early_stop_min_delta < 0 or args.early_stop_complexity_penalty < 0:
        raise ValueError("早停最小改善量和复杂度惩罚必须非负。")
    if not 0 < args.early_stop_min_b1_r2 <= 1:
        raise ValueError("--early-stop-min-b1-r2 必须位于 (0, 1]。")
    if args.validation_mape_cap <= 0 or args.weight_difficulty_power < 0:
        raise ValueError("验证误差截断值必须为正，难度指数必须非负。")
    if args.weight_bootstrap < 0:
        raise ValueError("--weight-bootstrap 必须非负。")
    if not 0 <= args.min_dataset_weight <= args.max_dataset_weight <= 1:
        raise ValueError("数据集权重上下限必须满足 0 <= min <= max <= 1。")

    # 1. B1 是经典标度律唯一的主拟合数据。
    b1_observations = load_classic_observations(
        b_dir / "pythia_training_log_existing.csv", row_id_column="run_id"
    )
    b1_reference_fit = fit_classic(b1_observations, grid_size=args.grid_size)

    # 早停需要固定的 B2--B5 验证集合；无论主报告展示哪条轨迹都加载它们。
    b2_observations = load_classic_observations(
        b_dir / "cerebras_training_log.csv",
        row_id_column="run_id",
        final_checkpoint_only=True,
    )
    b3_observations = load_b3_observations(b_dir)
    b4_observations = load_classic_observations(
        b_dir / "scaling_baseline.csv",
        row_id_column="family",
        converged_only=True,
    )
    b5_observations = load_classic_observations(
        b_dir / "published_scaling_data.csv",
        row_id_column="family",
        converged_only=True,
    )
    early_validation_sets: dict[str, Sequence[Observation]] = {
        "B2": b2_observations,
        "B3": b3_observations,
        "B4": b4_observations,
        "B5": b5_observations,
    }

    selected_parameters = b1_reference_fit.parameters
    selected_b1_metrics: dict[str, object] = asdict(b1_reference_fit.metrics)
    early_stopping_payload: dict[str, object] = {
        "enabled": not args.disable_early_stopping
    }
    early_output_files: dict[str, str] = {}

    if not args.disable_early_stopping:
        classic_parameter_array = np.asarray(
            [
                b1_reference_fit.parameters.irreducible_loss,
                b1_reference_fit.parameters.parameter_coefficient,
                b1_reference_fit.parameters.parameter_exponent,
                b1_reference_fit.parameters.data_coefficient,
                b1_reference_fit.parameters.data_exponent,
            ],
            dtype=float,
        )
        classic_spec = next(
            spec for spec in MODEL_PATH if spec.name == "classic"
        )
        adaptive_weights, weight_diagnostics = derive_adaptive_weights(
            validation_sets=early_validation_sets,
            classic_spec=classic_spec,
            classic_parameters=classic_parameter_array,
            raw_mape_cap=args.validation_mape_cap,
            bootstrap_count=args.weight_bootstrap,
            seed=args.weight_seed,
            difficulty_power=args.weight_difficulty_power,
            minimum_weight=args.min_dataset_weight,
            maximum_weight=args.max_dataset_weight,
        )
        weights = (
            adaptive_weights
            if args.weight_mode == "adaptive"
            else {"B2": 0.20, "B3": 0.10, "B4": 0.35, "B5": 0.35}
        )

        # 全部候选均保持 E+A*N^-alpha+B*D^-beta，仅逐层释放指数约束。
        evaluated_candidates, chosen_candidate, stop_stage = (
            run_parameter_release_early_stopping(
                b1_observations=b1_observations,
                validation_sets=early_validation_sets,
                classic_parameters=classic_parameter_array,
                weights=weights,
                raw_mape_cap=args.validation_mape_cap,
                complexity_penalty_rate=args.early_stop_complexity_penalty,
                minimum_b1_r_squared=args.early_stop_min_b1_r2,
                patience=args.early_stop_patience,
                minimum_delta=args.early_stop_min_delta,
                exponent_lower=0.01,
                exponent_upper=1.50,
                grid_size=args.grid_size,
            )
        )
        if chosen_candidate is None:
            raise RuntimeError(
                "没有早停候选模型达到 B1 拟合门槛；"
                "请降低 --early-stop-min-b1-r2。"
            )

        selected_parameters = candidate_to_scaling_parameters(chosen_candidate)
        selected_b1_metrics = chosen_candidate.b1_metrics
        history_path = output_dir / "early_stopping_history.csv"
        weights_path = output_dir / "adaptive_weights.csv"
        predictions_path = output_dir / "early_stopping_predictions.csv"
        write_history_csv(history_path, evaluated_candidates)
        write_weight_diagnostics_csv(
            weights_path, weights, weight_diagnostics
        )
        write_early_stopping_predictions(
            predictions_path,
            chosen_candidate,
            early_validation_sets,
        )
        early_output_files = {
            "early_stopping_history": str(history_path),
            "adaptive_weights": str(weights_path),
            "early_stopping_predictions": str(predictions_path),
        }
        early_stopping_payload.update(
            {
                "weight_mode": args.weight_mode,
                "weights": weights,
                "weight_diagnostics": weight_diagnostics,
                "model_path": [spec.name for spec in MODEL_PATH],
                "parameter_release_layers": [
                    ["shared_exponent"],
                    [
                        "release_parameter_exponent",
                        "release_data_exponent",
                    ],
                    ["classic"],
                ],
                "settings": {
                    "loss_scale_normalization": {
                        "reference_scale": "B1 identity scale",
                        "external_transform": "(L_source - intercept) / slope",
                        "calibration": "leave-one-trajectory-or-family-out",
                        "selection_uses_normalized_metrics": True,
                        "raw_metrics_retained": True,
                    },
                    "patience": args.early_stop_patience,
                    "minimum_delta": args.early_stop_min_delta,
                    "minimum_b1_r_squared": args.early_stop_min_b1_r2,
                    "complexity_penalty_per_extra_parameter": (
                        args.early_stop_complexity_penalty
                    ),
                    "raw_mape_cap": args.validation_mape_cap,
                    "bootstrap_count": args.weight_bootstrap,
                    "bootstrap_seed": args.weight_seed,
                    "difficulty_power": args.weight_difficulty_power,
                    "minimum_dataset_weight": args.min_dataset_weight,
                    "maximum_dataset_weight": args.max_dataset_weight,
                },
                "stop_stage": stop_stage,
                "chosen_model": candidate_payload(chosen_candidate),
                "candidates": [
                    candidate_payload(item) for item in evaluated_candidates
                ],
                "unregularized_classic_reference": {
                    "parameters": asdict(b1_reference_fit.parameters),
                    "metrics": asdict(b1_reference_fit.metrics),
                },
            }
        )
    else:
        early_stopping_payload["reason"] = "disabled by command line"

    # 2-3. B2/B3 轨迹验证，以及 B4/B5 外部验证。
    validations: list[dict[str, object]] = []
    if args.trajectory_validation in {"b2", "both"}:
        validations.append(
            validate_classic_dataset(
                dataset_name="B2",
                role="Cerebras converged-endpoint cross-family validation",
                nature="semi-synthetic",
                observations=b2_observations,
                b1_parameters=selected_parameters,
                grid_size=args.grid_size,
                self_fit=False,
                note=(
                    "每个模型规模仅保留最大训练步的最终 checkpoint；"
                    "这些端点用于检验收敛损失的跨模型族迁移。"
                ),
            )
        )
    if args.trajectory_validation in {"b3", "both"}:
        validations.append(
            validate_classic_dataset(
                dataset_name="B3",
                role="Pythia interpolated-trajectory validation",
                nature="interpolated from B1 checkpoints",
                observations=b3_observations,
                b1_parameters=selected_parameters,
                grid_size=args.grid_size,
                self_fit=False,
                note="B3 检验轨迹连续性，但不是独立模型族外验证。",
            )
        )

    validations.append(
        validate_classic_dataset(
            dataset_name="B4",
            role="cross-family convergence validation",
            nature="real curated convergence snapshots",
            observations=b4_observations,
            b1_parameters=selected_parameters,
            grid_size=args.grid_size,
            self_fit=True,
            note="跨族 Loss 可能受训练语料、分词器和验证集口径影响。",
        )
    )
    validations.append(
        validate_classic_dataset(
            dataset_name="B5",
            role="published-literature validation",
            nature="published real data",
            observations=b5_observations,
            b1_parameters=selected_parameters,
            grid_size=args.grid_size,
            self_fit=True,
            note="B5 来自多篇文献，直接误差与同形式自拟合结果应同时报告。",
        )
    )

    # 4. 经典标度律在百亿参数以上的外推。标准标度律不含 Q 维度，
    # 因此不拟合质量模型，也不构造 Q 情景。
    large_output_path = output_dir / "large_model_extrapolation.csv"
    large_model_payload = analyze_large_models(
        b_dir=b_dir,
        output_path=large_output_path,
        b1_parameters=selected_parameters,
    )


    payload: dict[str, object] = {
        "methodology": {
            "classic_law": "L=E+A*N^(-alpha)+B*D^(-beta)",
            "scope_note": (
                "本脚本只辨识经典标度律，自变量为参数量 N 与训练数据量 D。"
                "数据质量 Q 与配比 p 由独立的广义标度律脚本引入，"
                "见 q2/generalized_law.py（其辨识数据限定为 "
                "B1 + 附件 A + Q_1_to_2，B6--B8 不参与参数估计）。"
            ),
        },
        "data_credibility": {
            "B1": "real, main fit",
            "B2": "semi-synthetic; final converged checkpoint per run only",
            "B3": "interpolated from B1 checkpoints",
            "B4": "real curated cross-family snapshots",
            "B5": "real published literature data",
            "B9": "real large-model metadata without comparable observed Loss",
            "B10": "estimated Loss, not direct observation",
        },
        "B1_main_fit": {
            "sample_count": len(b1_observations),
            "parameters": asdict(selected_parameters),
            "metrics": selected_b1_metrics,
            "selection_source": (
                "adaptive structural early stopping"
                if not args.disable_early_stopping
                else "unregularized five-parameter B1 fit"
            ),
        },
        "adaptive_early_stopping": early_stopping_payload,
        "classic_validations": validations,
        "large_model_extrapolation": large_model_payload,
        "output_files": {
            "large_model_extrapolation": str(large_output_path),
            **early_output_files,
        },
    }

    report_path = output_dir / "scaling_law_report.md"
    write_markdown_report(report_path, payload)
    result_path = output_dir / "scaling_law_full_results.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_path = output_dir / "classic_validation_metrics.csv"
    write_dict_rows(validation_path, flatten_validation_rows(validations))
    payload["output_files"]["validation_metrics"] = str(validation_path)
    payload["output_files"]["full_results"] = str(result_path)
    payload["output_files"]["markdown_report"] = str(report_path)
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


    print_summary(payload, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
