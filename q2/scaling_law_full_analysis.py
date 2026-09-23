#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按赛题要求完成标度律拟合、验证、质量扩展和大模型外推。

数据使用路线
------------
1. B1：经典标度律的主要拟合数据；
2. B3（默认）：Pythia 插值训练轨迹验证。也可通过命令行改用 B2；
3. B4、B5：分别进行跨模型族和已发表文献数据验证；
4. B7（默认）：拟合含数据质量 Q 的广义标度律，并用 B8 分层检验；
5. B9、B10：匹配百亿参数以上模型，进行 Q 情景外推。

经典模型
--------
    L(N,D) = E + A*N^(-alpha) + B*D^(-beta)

质量模型
--------
    L(N,D,Q) = E_q + A_q*N^(-alpha_q)
                     + B_q*D^(-beta_q)*Q^(-gamma_q)

质量模型在 Q=1 时退化为经典形式。由于附件 A/B 的实验相互独立，且 B7
为半合成数据，本程序不使用 B7 参数替换 B1 的真实数据参数，而是从 B7
提取无量纲的有效数据质量指数

    kappa = gamma_q / beta_q,
    D_eff = D * Q^kappa.

最终在 B1 Loss 尺度上使用

    L_general(N,D,Q) = E + A*N^(-alpha) + B*(D*Q^kappa)^(-beta).

默认运行：

    conda run -n GNN python q1/scaling_law_full_analysis.py

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
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from traditional_scaling_law import (
    FitResult,
    Observation,
    ScalingParameters,
    fit_scaling_law,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_B_DIR = PROJECT_ROOT / "data" / "real_attachments" / "B_scaling_laws"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_analysis" / "scaling_law_full"


@dataclass(frozen=True)
class QualityObservation:
    experiment_id: str
    n_params_b: float
    d_tokens_b: float
    quality: float
    loss: float
    data_type: str = ""


@dataclass(frozen=True)
class QualityParameters:
    irreducible_loss: float
    parameter_coefficient: float
    parameter_exponent: float
    data_coefficient: float
    data_exponent: float
    quality_exponent: float

    @property
    def effective_data_quality_exponent(self) -> float:
        return self.quality_exponent / self.data_exponent

    def predict(
        self, n_params_b: np.ndarray, d_tokens_b: np.ndarray, quality: np.ndarray
    ) -> np.ndarray:
        return (
            self.irreducible_loss
            + self.parameter_coefficient
            * np.power(n_params_b, -self.parameter_exponent)
            + self.data_coefficient
            * np.power(d_tokens_b, -self.data_exponent)
            * np.power(quality, -self.quality_exponent)
        )


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
) -> list[Observation]:
    observations: list[Observation] = []
    for index, row in enumerate(read_csv(path), start=2):
        if converged_only and (row.get("is_converged") or "").strip() not in {
            "1",
            "1.0",
            "true",
            "True",
        }:
            continue
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


def load_quality_observations(
    path: Path, data_type: str | None = None
) -> list[QualityObservation]:
    observations: list[QualityObservation] = []
    for index, row in enumerate(read_csv(path), start=2):
        row_data_type = (row.get("data_type") or "").strip()
        if data_type is not None and row_data_type != data_type:
            continue
        n_value = _float(row.get("N_params_B"))
        d_value = _float(row.get("D_tokens_B"))
        quality = _float(row.get("Q_score"))
        loss = _float(row.get("val_loss"))
        if (
            n_value is None
            or d_value is None
            or quality is None
            or loss is None
            or min(n_value, d_value, quality, loss) <= 0
        ):
            continue
        observations.append(
            QualityObservation(
                experiment_id=(row.get("experiment_id") or str(index)).strip(),
                n_params_b=n_value,
                d_tokens_b=d_value,
                quality=quality,
                loss=loss,
                data_type=row_data_type,
            )
        )
    if not observations:
        raise ValueError(f"{path.name} 中没有符合条件的 N-D-Q-Loss 记录。")
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


def _quality_arrays(
    observations: Sequence[QualityObservation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([item.n_params_b for item in observations], dtype=float),
        np.asarray([item.d_tokens_b for item in observations], dtype=float),
        np.asarray([item.quality for item in observations], dtype=float),
        np.asarray([item.loss for item in observations], dtype=float),
    )


def fit_quality_model(
    observations: Sequence[QualityObservation],
) -> tuple[QualityParameters, Metrics, np.ndarray]:
    """多起点非线性最小二乘估计 N-D-Q 广义标度律。"""

    n_value, d_value, quality, actual = _quality_arrays(observations)
    loss_floor = max(1e-6, min(float(actual.min()) * 0.999, 5.0))

    def predict(raw_parameters: np.ndarray) -> np.ndarray:
        e_value, a_value, alpha, b_value, beta, gamma = raw_parameters
        return (
            e_value
            + a_value * np.power(n_value, -alpha)
            + b_value
            * np.power(d_value, -beta)
            * np.power(quality, -gamma)
        )

    lower = np.asarray([0.0, 1e-10, 0.001, 1e-10, 0.001, 0.001])
    upper = np.asarray([loss_floor, 50.0, 2.0, 100.0, 2.0, 2.0])
    starts = [
        np.asarray([0.70, 0.53, 0.28, 1.85, 0.08, 0.10]),
        np.asarray([1.20, 0.40, 0.34, 1.30, 0.28, 0.20]),
        np.asarray([0.25, 0.80, 0.20, 3.00, 0.15, 0.30]),
        np.asarray([min(1.50, loss_floor * 0.8), 0.30, 0.50, 2.00, 0.05, 0.05]),
    ]

    best_result = None
    best_sse = math.inf
    for start in starts:
        start = np.minimum(np.maximum(start, lower + 1e-8), upper - 1e-8)
        result = least_squares(
            lambda raw: predict(raw) - actual,
            x0=start,
            bounds=(lower, upper),
            method="trf",
            max_nfev=100000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        sse = float(np.sum(np.square(result.fun)))
        if result.success and sse < best_sse:
            best_result = result
            best_sse = sse
    if best_result is None:
        raise RuntimeError("质量模型拟合失败。")

    parameters = QualityParameters(*map(float, best_result.x))
    prediction = parameters.predict(n_value, d_value, quality)
    return parameters, calculate_metrics(actual, prediction), prediction


def leave_one_quality_level_out(
    observations: Sequence[QualityObservation],
) -> dict[str, object]:
    folds: list[dict[str, object]] = []
    all_actual: list[float] = []
    all_prediction: list[float] = []
    quality_levels = sorted({item.quality for item in observations})
    for held_out_quality in quality_levels:
        train = [item for item in observations if item.quality != held_out_quality]
        test = [item for item in observations if item.quality == held_out_quality]
        if len(train) < 20 or not test:
            continue
        parameters, _, _ = fit_quality_model(train)
        n_value, d_value, quality, actual = _quality_arrays(test)
        prediction = parameters.predict(n_value, d_value, quality)
        metrics = calculate_metrics(actual, prediction)
        all_actual.extend(actual.tolist())
        all_prediction.extend(prediction.tolist())
        folds.append(
            {
                "held_out_Q": held_out_quality,
                "sample_count": len(test),
                "metrics": asdict(metrics),
            }
        )
    overall = calculate_metrics(
        np.asarray(all_actual, dtype=float), np.asarray(all_prediction, dtype=float)
    )
    return {
        "method": "leave-one-Q-level-out",
        "fold_count": len(folds),
        "overall_metrics": asdict(overall),
        "folds": folds,
    }


def generalized_prediction(
    parameters: ScalingParameters,
    quality_kappa: float,
    n_params_b: np.ndarray,
    d_tokens_b: np.ndarray,
    quality: np.ndarray,
) -> np.ndarray:
    effective_d = d_tokens_b * np.power(quality, quality_kappa)
    return (
        parameters.irreducible_loss
        + parameters.parameter_coefficient
        * np.power(n_params_b, -parameters.parameter_exponent)
        + parameters.data_coefficient
        * np.power(effective_d, -parameters.data_exponent)
    )


def quality_direction_diagnostic(
    observations: Sequence[QualityObservation],
) -> dict[str, object]:
    """在固定 (N,D) 单元内检查 Q 与 Loss 的方向是否一致。

    题目约定 Q 越高越好，因此正常情况下 Q 与 Loss 应呈负相关。分组后再算
    相关性，避免 N、D 的主效应掩盖质量方向。
    """

    groups: dict[tuple[float, float], list[QualityObservation]] = {}
    for item in observations:
        groups.setdefault((item.n_params_b, item.d_tokens_b), []).append(item)
    correlations: list[float] = []
    for items in groups.values():
        if len(items) < 3 or len({item.quality for item in items}) < 3:
            continue
        rho = spearmanr(
            [item.quality for item in items], [item.loss for item in items]
        ).statistic
        if math.isfinite(float(rho)):
            correlations.append(float(rho))
    median_rho = float(np.median(correlations)) if correlations else math.nan
    return {
        "group_count": len(groups),
        "groups_with_correlation": len(correlations),
        "median_within_ND_spearman_Q_vs_loss": median_rho,
        "expected_sign_when_Q_is_higher_better": "negative",
        "direction_conflict": bool(math.isfinite(median_rho) and median_rho > 0),
    }


def fit_and_describe_quality(
    dataset_name: str,
    observations: Sequence[QualityObservation],
    run_cv: bool,
) -> tuple[dict[str, object], QualityParameters, np.ndarray]:
    parameters, metrics, prediction = fit_quality_model(observations)
    actual = np.asarray([item.loss for item in observations], dtype=float)
    floor_value = float(actual.min())
    floor_share = float(np.mean(np.isclose(actual, floor_value, atol=1e-12)))
    direction = quality_direction_diagnostic(observations)
    payload: dict[str, object] = {
        "dataset": dataset_name,
        "sample_count": len(observations),
        "parameters": asdict(parameters),
        "effective_data_quality_exponent_kappa": (
            parameters.effective_data_quality_exponent
        ),
        "metrics": asdict(metrics),
        "minimum_observed_loss": floor_value,
        "share_at_minimum_loss": floor_share,
        "quality_direction_diagnostic": direction,
        "boundary_diagnostic": {
            "quality_exponent_at_lower_bound": (
                parameters.quality_exponent <= 0.00101
            ),
            "irreducible_loss_near_observed_floor": (
                abs(parameters.irreducible_loss - floor_value * 0.999)
                <= max(1e-8, floor_value * 1e-5)
            ),
        },
    }
    if direction["direction_conflict"]:
        payload["warning"] = (
            "固定 N、D 后，Q 与 Loss 的方向和‘Q 越高越好’约定冲突；"
            "该分层结果只能作为数据编码/外推稳健性警告，不能用于确认质量收益。"
        )
    if run_cv:
        payload["quality_level_cross_validation"] = leave_one_quality_level_out(
            observations
        )
    return payload, parameters, prediction


def write_quality_predictions(
    path: Path,
    dataset_name: str,
    observations: Sequence[QualityObservation],
    prediction: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "dataset",
            "experiment_id",
            "data_type",
            "N_params_B",
            "D_tokens_B",
            "Q_score",
            "actual_loss",
            "predicted_loss",
            "residual",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item, predicted in zip(observations, prediction):
            writer.writerow(
                {
                    "dataset": dataset_name,
                    "experiment_id": item.experiment_id,
                    "data_type": item.data_type,
                    "N_params_B": f"{item.n_params_b:.12g}",
                    "D_tokens_B": f"{item.d_tokens_b:.12g}",
                    "Q_score": f"{item.quality:.12g}",
                    "actual_loss": f"{item.loss:.12g}",
                    "predicted_loss": f"{float(predicted):.12g}",
                    "residual": f"{item.loss - float(predicted):.12g}",
                }
            )


def parse_quality_scenarios(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        if not item.strip():
            continue
        value = float(item)
        if not 0 < value <= 1:
            raise ValueError("质量情景必须位于 (0,1]。")
        if value not in values:
            values.append(value)
    if 1.0 not in values:
        values.append(1.0)
    return sorted(values)


def analyze_large_models(
    b_dir: Path,
    output_path: Path,
    b1_parameters: ScalingParameters,
    quality_kappa: float,
    quality_scenarios: Sequence[float],
) -> dict[str, object]:
    """联合 B9 元数据和 B10 估算 Loss，进行质量情景外推。"""

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
    scenario_predictions: dict[float, list[float]] = {
        quality: [] for quality in quality_scenarios
    }
    written = 0
    scenario_fields = [f"predicted_loss_Q_{value:g}" for value in quality_scenarios]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "model_name",
            "N_params_B",
            "D_tokens_B",
            "B10_estimated_loss",
            "B1_classic_prediction_Q_1",
            "relative_error_vs_B10_estimate",
            "publication_date",
            "organization",
            "accessibility",
            "country",
        ] + scenario_fields
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
            row: dict[str, object] = {
                "model_name": name,
                "N_params_B": n_value,
                "D_tokens_B": d_value,
                "B10_estimated_loss": estimated_loss,
                "B1_classic_prediction_Q_1": classic_prediction,
                "relative_error_vs_B10_estimate": abs(
                    classic_prediction - estimated_loss
                )
                / estimated_loss,
                "publication_date": metadata.get("publication_date", ""),
                "organization": metadata.get("organization", ""),
                "accessibility": metadata.get("accessibility", ""),
                "country": metadata.get("country", ""),
            }
            for quality in quality_scenarios:
                prediction = generalized_prediction(
                    b1_parameters,
                    quality_kappa,
                    np.asarray([n_value]),
                    np.asarray([d_value]),
                    np.asarray([quality]),
                )[0]
                row[f"predicted_loss_Q_{quality:g}"] = float(prediction)
                scenario_predictions[quality].append(float(prediction))
            writer.writerow(row)
            written += 1
            actual_estimates.append(estimated_loss)
            b1_predictions.append(classic_prediction)

    metrics = calculate_metrics(
        np.asarray(actual_estimates, dtype=float),
        np.asarray(b1_predictions, dtype=float),
    )
    reference_median = float(np.median(scenario_predictions[1.0]))
    scenario_summary = {
        f"Q={quality:g}": {
            "median_predicted_loss": float(np.median(values)),
            "median_loss_difference_vs_Q_1": (
                float(np.median(values)) - reference_median
            ),
        }
        for quality, values in scenario_predictions.items()
    }
    return {
        "B9_row_count": len(b9_rows),
        "B10_row_count": len(b10_rows),
        "matched_name_count": len(matched_names),
        "usable_10B_plus_count": written,
        "B1_direct_transfer_against_B10_estimates": asdict(metrics),
        "quality_scenarios": list(quality_scenarios),
        "quality_scenario_summary": scenario_summary,
        "quality_kappa_transferred_from_quality_experiment": quality_kappa,
        "limitations": [
            "B9 提供模型规模与元数据，但没有可比的实测 Loss。",
            "B10 的 val_loss 是标度律估算值，不是直接训练实验观测。",
            "B9/B10 没有模型级 Q，Q 情景仅用于敏感性分析。",
        ],
    }


def write_markdown_report(path: Path, payload: dict[str, object]) -> None:
    """把核心数值和可信度边界写成可直接用于论文整理的 Markdown。"""

    b1 = payload["B1_main_fit"]
    validations = payload["classic_validations"]
    quality = payload["quality_main_fit"]
    b8 = payload["B8_stratified_robustness"]
    large = payload["large_model_extrapolation"]
    assert isinstance(b1, dict)
    assert isinstance(validations, list)
    assert isinstance(quality, dict)
    assert isinstance(b8, dict)
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
        "| 数据 | 角色 | 样本数 | RMSE | 中位相对误差 | Spearman |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in validations:
        metrics = item["b1_direct_transfer_metrics"]
        lines.append(
            f"| {item['dataset']} | {item['role']} | {metrics['sample_count']} | "
            f"{metrics['rmse']:.6f} | {metrics['median_ape']:.2%} | "
            f"{metrics['spearman_rho']:.4f} |"
        )
    lines.extend(
        [
            "",
            "B3 是由 B1 检查点插值得到的轨迹，只能验证轨迹连续性。B4、B5 的直接误差同时包含模型族、训练语料、分词器和验证集口径差异，因此还应结合 Spearman 排序相关及各表同形式自拟合结果判断。",
            "",
            "## 数据质量扩展",
            "",
        ]
    )
    qp = quality["parameters"]
    qm = quality["metrics"]
    qcv = quality.get("quality_level_cross_validation")
    lines.append(
        f"质量主数据为 {quality['dataset']}，拟合 R²={qm['r_squared']:.6f}，"
        f"gamma={qp['quality_exponent']:.6f}，beta_q={qp['data_exponent']:.6f}，"
        f"得到有效数据质量指数 kappa=gamma/beta_q="
        f"{quality['effective_data_quality_exponent_kappa']:.6f}。"
    )
    if isinstance(qcv, dict):
        lines.append(
            f"按 Q 水平留一验证 RMSE={qcv['overall_metrics']['rmse']:.6f}。"
        )
    lines.extend(
        [
            "",
            "转移到 B1 Loss 尺度后的广义形式为",
            "",
            "```text",
            "L(N,D,Q) = E_B1 + A_B1*N^(-alpha_B1) + B_B1*(D*Q^kappa)^(-beta_B1)",
            "```",
            "",
            "该两阶段做法使 Q=1 时严格退化为 B1 经典标度律，同时避免把半合成质量数据当成真实训练轨迹。",
            "",
            "### B8 稳健性边界",
            "",
        ]
    )
    for layer_name in ("calibrated", "extrapolated"):
        layer = b8.get(layer_name)
        if not isinstance(layer, dict) or "error" in layer:
            continue
        diagnostic = layer["quality_direction_diagnostic"]
        lines.append(
            f"- B8-{layer_name}：R²={layer['metrics']['r_squared']:.4f}，"
            f"Loss 下界占比={layer['share_at_minimum_loss']:.2%}，"
            f"固定 N、D 后 Q-Loss 的中位 Spearman="
            f"{diagnostic['median_within_ND_spearman_Q_vs_loss']:.4f}。"
        )
    lines.extend(
        [
            "",
            "B8 中固定 N、D 后 Q 与 Loss 的方向若为正，则与题目中‘质量越高越好’的定义冲突；加之 Loss=0.5 的下界堆积，B8 只适合作为编码和外推风险提示，不能反向推翻 B7 的质量效应。",
            "",
            "## B9 与 B10 的百亿参数以上外推",
            "",
            (
                f"B9 与 B10 共匹配 {large['matched_name_count']} 个模型，其中 "
                f"{large['usable_10B_plus_count']} 个可用于外推。"
            ),
            "",
            "| 质量情景 | 中位预测 Loss | 相对 Q=1 的中位 Loss 变化 |",
            "|---|---:|---:|",
        ]
    )
    for name, summary in large["quality_scenario_summary"].items():
        lines.append(
            f"| {name} | {summary['median_predicted_loss']:.6f} | "
            f"{summary['median_loss_difference_vs_Q_1']:+.6f} |"
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
        metrics = validation["b1_direct_transfer_metrics"]
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "dataset": validation["dataset"],
                "role": validation["role"],
                "nature": validation["nature"],
                **metrics,
                "note": validation["note"],
            }
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
    quality = payload["quality_main_fit"]
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
    for item in payload["classic_validations"]:
        metrics = item["b1_direct_transfer_metrics"]
        print(
            f"{item['dataset']}: n={metrics['sample_count']}, "
            f"RMSE={metrics['rmse']:.6f}, "
            f"中位相对误差={metrics['median_ape']:.2%}, "
            f"Spearman={metrics['spearman_rho']:.4f}"
        )
    if isinstance(quality, dict):
        print(
            f"质量主模型 {quality['dataset']}: "
            f"R²={quality['metrics']['r_squared']:.6f}, "
            f"kappa={quality['effective_data_quality_exponent_kappa']:.6f}"
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
        description="执行 B1-B10 标度律拟合、验证、质量扩展和大模型外推。"
    )
    parser.add_argument("--b-dir", type=Path, default=DEFAULT_B_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--trajectory-validation",
        choices=("b2", "b3", "both"),
        default="b3",
        help="B1 之外的轨迹验证数据；默认使用 B3 插值轨迹。",
    )
    parser.add_argument(
        "--quality-source",
        choices=("b6", "b7", "b8"),
        default="b7",
        help="质量模型主数据；B8 主拟合时只使用 calibrated 子集。",
    )
    parser.add_argument(
        "--quality-scenarios",
        default="0.5,0.8,1.0",
        help="B9/B10 外推所用 Q 情景，逗号分隔。",
    )
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument(
        "--skip-quality-cv",
        action="store_true",
        help="跳过按 Q 水平留一交叉验证。",
    )
    return parser


def main() -> int:
    configure_console()
    args = build_argument_parser().parse_args()
    b_dir = args.b_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. B1 是经典标度律唯一的主拟合数据。
    b1_observations = load_classic_observations(
        b_dir / "pythia_training_log_existing.csv", row_id_column="run_id"
    )
    b1_fit = fit_classic(b1_observations, grid_size=args.grid_size)

    # 2-3. B2/B3 轨迹验证，以及 B4/B5 外部验证。
    validations: list[dict[str, object]] = []
    if args.trajectory_validation in {"b2", "both"}:
        b2_observations = load_classic_observations(
            b_dir / "cerebras_training_log.csv", row_id_column="run_id"
        )
        validations.append(
            validate_classic_dataset(
                dataset_name="B2",
                role="Cerebras model-family trajectory validation",
                nature="semi-synthetic",
                observations=b2_observations,
                b1_parameters=b1_fit.parameters,
                grid_size=args.grid_size,
                self_fit=True,
                note="B2 为真实规律校准的半合成轨迹，不表述为直接实验观测。",
            )
        )
    if args.trajectory_validation in {"b3", "both"}:
        b3_observations = load_b3_observations(b_dir)
        validations.append(
            validate_classic_dataset(
                dataset_name="B3",
                role="Pythia interpolated-trajectory validation",
                nature="interpolated from B1 checkpoints",
                observations=b3_observations,
                b1_parameters=b1_fit.parameters,
                grid_size=args.grid_size,
                self_fit=False,
                note="B3 检验轨迹连续性，但不是独立模型族外验证。",
            )
        )

    b4_observations = load_classic_observations(
        b_dir / "scaling_baseline.csv",
        row_id_column="family",
        converged_only=True,
    )
    validations.append(
        validate_classic_dataset(
            dataset_name="B4",
            role="cross-family convergence validation",
            nature="real curated convergence snapshots",
            observations=b4_observations,
            b1_parameters=b1_fit.parameters,
            grid_size=args.grid_size,
            self_fit=True,
            note="跨族 Loss 可能受训练语料、分词器和验证集口径影响。",
        )
    )
    b5_observations = load_classic_observations(
        b_dir / "published_scaling_data.csv",
        row_id_column="family",
        converged_only=True,
    )
    validations.append(
        validate_classic_dataset(
            dataset_name="B5",
            role="published-literature validation",
            nature="published real data",
            observations=b5_observations,
            b1_parameters=b1_fit.parameters,
            grid_size=args.grid_size,
            self_fit=True,
            note="B5 来自多篇文献，直接误差与同形式自拟合结果应同时报告。",
        )
    )

    # 4. 质量主模型默认使用 B7；B8 按 calibrated/extrapolated 分层稳健性检验。
    quality_file_by_source = {
        "b6": "supplementary_NQ_experiment.csv",
        "b7": "supplementary_NQ_experiment_expanded.csv",
        "b8": "supplementary_NQ_experiment_large.csv",
    }
    quality_data_type = "calibrated" if args.quality_source == "b8" else None
    quality_observations = load_quality_observations(
        b_dir / quality_file_by_source[args.quality_source],
        data_type=quality_data_type,
    )
    quality_payload, quality_parameters, quality_prediction = (
        fit_and_describe_quality(
            dataset_name=args.quality_source.upper()
            + ("-calibrated" if quality_data_type else ""),
            observations=quality_observations,
            run_cv=not args.skip_quality_cv,
        )
    )
    quality_prediction_path = output_dir / "quality_model_predictions.csv"
    write_quality_predictions(
        quality_prediction_path,
        dataset_name=str(quality_payload["dataset"]),
        observations=quality_observations,
        prediction=quality_prediction,
    )

    b8_robustness: dict[str, object] = {}
    for data_type in ("calibrated", "extrapolated"):
        observations = load_quality_observations(
            b_dir / "supplementary_NQ_experiment_large.csv",
            data_type=data_type,
        )
        try:
            layer_payload, _, _ = fit_and_describe_quality(
                dataset_name=f"B8-{data_type}",
                observations=observations,
                run_cv=False,
            )
            if data_type == "extrapolated":
                existing_warning = str(layer_payload.get("warning") or "").strip()
                extrapolation_warning = (
                    "B8-extrapolated 不是直接观测，且存在 Loss 下界堆积，"
                    "参数只用于外推稳定性诊断。"
                )
                layer_payload["warning"] = " ".join(
                    item
                    for item in (existing_warning, extrapolation_warning)
                    if item
                )
            b8_robustness[data_type] = layer_payload
        except (RuntimeError, ValueError) as error:
            b8_robustness[data_type] = {"error": str(error)}

    # 5. 将 B7/B8 识别的 kappa 迁移到 B1 尺度，结合 B9/B10 做 Q 情景外推。
    quality_kappa = quality_parameters.effective_data_quality_exponent
    quality_scenarios = parse_quality_scenarios(args.quality_scenarios)
    large_output_path = output_dir / "large_model_extrapolation.csv"
    large_model_payload = analyze_large_models(
        b_dir=b_dir,
        output_path=large_output_path,
        b1_parameters=b1_fit.parameters,
        quality_kappa=quality_kappa,
        quality_scenarios=quality_scenarios,
    )

    payload: dict[str, object] = {
        "methodology": {
            "classic_law": "L=E+A*N^(-alpha)+B*D^(-beta)",
            "quality_law": (
                "L=E_q+A_q*N^(-alpha_q)+B_q*D^(-beta_q)*Q^(-gamma_q)"
            ),
            "transferred_generalized_law": (
                "L=E_B1+A_B1*N^(-alpha_B1)+"
                "B_B1*(D*Q^kappa)^(-beta_B1)"
            ),
            "kappa_definition": "gamma_q/beta_q",
            "reason_for_two_stage_fit": (
                "B1 与质量补充实验相互独立；B1 负责真实基准尺度，"
                "半合成质量实验只识别无量纲质量效率。"
            ),
        },
        "data_credibility": {
            "B1": "real, main fit",
            "B2": "semi-synthetic",
            "B3": "interpolated from B1 checkpoints",
            "B4": "real curated cross-family snapshots",
            "B5": "real published literature data",
            "B6_B7_B8": "semi-synthetic quality supplements",
            "B9": "real large-model metadata without comparable observed Loss",
            "B10": "estimated Loss, not direct observation",
        },
        "B1_main_fit": {
            "sample_count": len(b1_observations),
            "parameters": asdict(b1_fit.parameters),
            "metrics": asdict(b1_fit.metrics),
        },
        "classic_validations": validations,
        "quality_main_fit": quality_payload,
        "B8_stratified_robustness": b8_robustness,
        "transferred_generalized_parameters": {
            "B1_classic_parameters": asdict(b1_fit.parameters),
            "quality_kappa": quality_kappa,
        },
        "large_model_extrapolation": large_model_payload,
        "output_files": {
            "quality_predictions": str(quality_prediction_path),
            "large_model_extrapolation": str(large_output_path),
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
