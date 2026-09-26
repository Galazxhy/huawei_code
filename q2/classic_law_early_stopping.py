#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对标准五参数标度律执行基于参数释放路径的结构早停。

全过程保持模型形式不变：

    L(N, D) = E + A N^(-alpha) + B D^(-beta)

对给定的 ``alpha``、``beta``，始终以线性最小二乘估计 ``E``、``A``、``B``；
非线性指数则沿以下约束路径逐层释放：

    1. shared_exponent:              alpha = beta = s，一维搜索 s；
    2. release_parameter_exponent:   beta 固定为 s*，一维搜索 alpha；
    3. release_data_exponent:        alpha 固定为 s*，一维搜索 beta；
    4. classic:                      二维搜索 alpha、beta。

第二、三项是同一复杂度层的并列候选。所有网格点都只用主拟合数据选择；
多来源外部 Loss 先经分簇交叉仿射归一化映射到主拟合尺度，再用于判断是否
值得释放更多指数自由度。若验证分数连续
``patience`` 个复杂度层没有实质改善，就停止并回滚到历史最优约束结构。

默认运行：

    conda run -n GNN python q2/classic_law_early_stopping.py

结果写入 ``data_analysis/scaling_law_early_stopping/``。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.stats import spearmanr

from classic_law import Observation, fit_scaling_law


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_B_DIR = PROJECT_ROOT / "data" / "real_attachments" / "B_scaling_laws"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data_analysis" / "scaling_law_early_stopping"
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


def observed_losses(observations: Sequence[Observation]) -> np.ndarray:
    return np.asarray([item.loss for item in observations], dtype=float)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Metrics:
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("actual 和 predicted 必须为等长非空数组。")
    residual = actual - predicted
    absolute_percentage_error = np.abs(residual) / np.maximum(
        np.abs(actual), 1e-12
    )
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


def affine_recalibration(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[float, float, Metrics]:
    design = np.column_stack((np.ones_like(predicted), predicted))
    intercept, slope = np.linalg.lstsq(design, actual, rcond=None)[0]
    recalibrated = intercept + slope * predicted
    return float(intercept), float(slope), calculate_metrics(actual, recalibrated)


def cross_fitted_loss_normalization(
    actual: np.ndarray,
    predicted: np.ndarray,
    cluster_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """把外部 Loss 仿射映射到主拟合尺度，并以分簇留一避免原地校准。

    假设不同来源的观测 Loss 满足 ``L_source = a + b*L_reference``。
    每个留出簇的 ``a,b`` 仅由其余簇估计，再用逆变换
    ``(L_source-a)/b`` 得到参考尺度上的验证目标。
    """

    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("actual 和 predicted 必须为等长非空数组。")
    if cluster_ids is None:
        labels = np.asarray([str(index) for index in range(actual.size)])
    else:
        if len(cluster_ids) != actual.size:
            raise ValueError("cluster_ids 必须与观测数组等长。")
        labels = np.asarray([str(item) for item in cluster_ids])
    unique_labels = list(dict.fromkeys(labels.tolist()))

    normalized_actual = np.empty_like(actual, dtype=float)
    calibrated_raw_prediction = np.empty_like(actual, dtype=float)
    folds: list[dict[str, object]] = []

    def fit_affine(indices: np.ndarray) -> tuple[float, float]:
        x_value = predicted[indices]
        y_value = actual[indices]
        design = np.column_stack((np.ones_like(x_value), x_value))
        intercept, slope = np.linalg.lstsq(design, y_value, rcond=None)[0]
        if not math.isfinite(float(slope)) or slope <= 1e-8:
            x_scale = max(float(np.std(x_value)), 1e-12)
            slope = max(float(np.std(y_value)) / x_scale, 1e-8)
            intercept = float(np.mean(y_value) - slope * np.mean(x_value))
        return float(intercept), float(slope)

    if len(unique_labels) < 2:
        # 单簇数据无法交叉校准；保留诊断结果并显式标记为同样本校准。
        train_indices = np.arange(actual.size)
        intercept, slope = fit_affine(train_indices)
        normalized_actual[:] = (actual - intercept) / slope
        calibrated_raw_prediction[:] = intercept + slope * predicted
        folds.append(
            {
                "held_out_cluster": unique_labels[0] if unique_labels else "all",
                "calibration_sample_count": int(actual.size),
                "validation_sample_count": int(actual.size),
                "intercept": intercept,
                "slope": slope,
                "in_sample_fallback": True,
            }
        )
        method = "in-sample affine fallback"
    else:
        for label in unique_labels:
            test_mask = labels == label
            train_indices = np.flatnonzero(~test_mask)
            test_indices = np.flatnonzero(test_mask)
            intercept, slope = fit_affine(train_indices)
            normalized_actual[test_indices] = (
                actual[test_indices] - intercept
            ) / slope
            calibrated_raw_prediction[test_indices] = (
                intercept + slope * predicted[test_indices]
            )
            folds.append(
                {
                    "held_out_cluster": label,
                    "calibration_sample_count": int(train_indices.size),
                    "validation_sample_count": int(test_indices.size),
                    "intercept": intercept,
                    "slope": slope,
                    "in_sample_fallback": False,
                }
            )
        method = "leave-one-cluster-out affine loss normalization"

    normalized_metrics = calculate_metrics(normalized_actual, predicted)
    raw_calibrated_metrics = calculate_metrics(actual, calibrated_raw_prediction)
    normalized_std = max(float(np.std(normalized_actual)), 1e-12)
    return {
        "method": method,
        "fold_count": len(folds),
        "normalized_metrics": asdict(normalized_metrics),
        "normalized_rmse_over_std": normalized_metrics.rmse / normalized_std,
        "raw_calibrated_metrics": asdict(raw_calibrated_metrics),
        "folds": folds,
    }


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameter_names: tuple[str, ...]
    description: str
    predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    search_mode: str
    effective_parameter_count: int
    layer: int


@dataclass
class CandidateFit:
    spec: ModelSpec
    parameters: np.ndarray
    b1_predictions: np.ndarray
    b1_metrics: dict[str, float | int]
    validation: dict[str, dict[str, object]]
    validation_score: float
    complexity_penalty: float
    selection_score: float
    eligible: bool
    improved: bool = False
    bad_stage_count: int = 0
    stop_triggered: bool = False


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _classic(
    parameters: np.ndarray, n_value: np.ndarray, d_value: np.ndarray
) -> np.ndarray:
    e_value, a_value, alpha, b_value, beta = parameters
    return (
        e_value
        + a_value * np.power(n_value, -alpha)
        + b_value * np.power(d_value, -beta)
    )


# 四个候选始终使用同一五参数标准律，只改变指数约束与搜索维数。
MODEL_PATH = (
    ModelSpec(
        "shared_exponent",
        ("E", "A", "alpha", "B", "beta"),
        "标准标度律的共享指数约束：alpha=beta=s",
        _classic,
        "shared",
        4,
        1,
    ),
    ModelSpec(
        "release_parameter_exponent",
        ("E", "A", "alpha", "B", "beta"),
        "固定 beta=s*，单独释放参数规模指数 alpha",
        _classic,
        "release_alpha",
        5,
        2,
    ),
    ModelSpec(
        "release_data_exponent",
        ("E", "A", "alpha", "B", "beta"),
        "固定 alpha=s*，单独释放数据规模指数 beta",
        _classic,
        "release_beta",
        5,
        2,
    ),
    ModelSpec(
        "classic",
        ("E", "A", "alpha", "B", "beta"),
        "标准五参数标度律：alpha、beta 均自由",
        _classic,
        "full",
        5,
        3,
    ),
)


def arrays(
    observations: Sequence[Observation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([item.n_params_b for item in observations], dtype=float),
        np.asarray([item.d_tokens_b for item in observations], dtype=float),
        observed_losses(observations),
    )


def _linear_fit_at_exponents(
    observations: Sequence[Observation],
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """固定 ``alpha,beta``，最小二乘消去线性参数 ``E,A,B``。"""

    n_value, d_value, actual = arrays(observations)
    design = np.column_stack(
        (
            np.ones_like(actual),
            np.power(n_value, -alpha),
            np.power(d_value, -beta),
        )
    )
    # 每个指数候选仅用主拟合数据确定线性系数，验证集不进入最小二乘。
    coefficients = np.linalg.lstsq(design, actual, rcond=None)[0]
    e_value, a_value, b_value = map(float, coefficients)
    if (
        e_value < 0.0
        or a_value < 0.0
        or b_value < 0.0
        or e_value >= float(np.min(actual))
    ):
        return None
    parameters = np.asarray(
        [e_value, a_value, alpha, b_value, beta], dtype=float
    )
    prediction = _classic(parameters, n_value, d_value)
    sse = float(np.sum(np.square(actual - prediction)))
    return parameters, prediction, sse


def _search_one_exponent(
    observations: Sequence[Observation],
    mode: str,
    fixed_exponent: float | None,
    lower: float,
    upper: float,
    grid_size: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """粗网格加局部步长减半，搜索共享指数或单个已释放指数。"""

    if grid_size < 5:
        raise ValueError("grid_size 至少为 5。")

    def evaluate(value: float) -> tuple[np.ndarray, np.ndarray, float] | None:
        if mode == "shared":
            return _linear_fit_at_exponents(observations, value, value)
        if fixed_exponent is None:
            raise ValueError(f"{mode} 搜索需要固定指数。")
        if mode == "release_alpha":
            return _linear_fit_at_exponents(observations, value, fixed_exponent)
        if mode == "release_beta":
            return _linear_fit_at_exponents(observations, fixed_exponent, value)
        raise ValueError(f"未知的一维指数搜索模式：{mode}")

    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for value in np.linspace(lower, upper, grid_size):
        candidate = evaluate(float(value))
        if candidate is not None and (best is None or candidate[2] < best[2]):
            best = candidate
    if best is None:
        raise RuntimeError(f"{mode} 在给定指数范围内没有可行解。")

    step = (upper - lower) / (grid_size - 1)
    while step > tolerance:
        parameters = best[0]
        center = (
            float(parameters[2])
            if mode in {"shared", "release_alpha"}
            else float(parameters[4])
        )
        improved = False
        for value in (center - step, center + step):
            if not lower <= value <= upper:
                continue
            candidate = evaluate(value)
            if candidate is not None and candidate[2] < best[2] - 1e-18:
                best = candidate
                improved = True
        if not improved:
            step /= 2.0
    return best[0], best[1]


def fit_model(
    spec: ModelSpec,
    observations: Sequence[Observation],
    classic_parameters: np.ndarray,
    shared_exponent: float | None = None,
    exponent_lower: float = 0.01,
    exponent_upper: float = 1.50,
    grid_size: int = 31,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """按约束阶段拟合标准标度律，不改变模型函数形式。"""

    n_value, d_value, _ = arrays(observations)
    if spec.search_mode == "full":
        prediction = spec.predict(classic_parameters, n_value, d_value)
        return classic_parameters.copy(), prediction
    return _search_one_exponent(
        observations=observations,
        mode=spec.search_mode,
        fixed_exponent=shared_exponent,
        lower=exponent_lower,
        upper=exponent_upper,
        grid_size=grid_size,
        tolerance=tolerance,
    )


def safe_rho(value: float) -> float:
    return float(value) if math.isfinite(value) else -1.0


def validation_score_from_arrays(
    actual: np.ndarray,
    predicted: np.ndarray,
    raw_mape_cap: float,
    cluster_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """在交叉尺度归一化后计算跨来源验证分数。"""

    direct = calculate_metrics(actual, predicted)
    intercept, slope, shape_metrics = affine_recalibration(actual, predicted)
    target_std = max(float(np.std(actual)), 1e-12)
    shape_nrmse = shape_metrics.rmse / target_std
    normalization = cross_fitted_loss_normalization(
        actual, predicted, cluster_ids=cluster_ids
    )
    normalized_metrics = normalization["normalized_metrics"]
    assert isinstance(normalized_metrics, dict)
    scale_component = min(float(normalized_metrics["median_ape"]), raw_mape_cap)
    rank_component = 1.0 - float(
        np.clip(safe_rho(float(normalized_metrics["spearman_rho"])), -1.0, 1.0)
    )
    dataset_score = (
        scale_component
        + 0.25 * float(normalization["normalized_rmse_over_std"])
        + 0.10 * rank_component
    )
    return {
        "direct_metrics": asdict(direct),
        "affine_shape_diagnostic": {
            "intercept": intercept,
            "slope": slope,
            "metrics": asdict(shape_metrics),
            "normalized_rmse": shape_nrmse,
        },
        "cross_fitted_scale_normalization": normalization,
        "score_components": {
            "capped_normalized_median_ape": scale_component,
            "cross_fitted_normalized_nrmse": normalization[
                "normalized_rmse_over_std"
            ],
            "normalized_rank_penalty": rank_component,
        },
        "dataset_score": dataset_score,
    }


def evaluate_validation_dataset(
    dataset_name: str,
    spec: ModelSpec,
    parameters: np.ndarray,
    observations: Sequence[Observation],
    raw_mape_cap: float,
) -> dict[str, object]:
    n_value, d_value, actual = arrays(observations)
    predicted = spec.predict(parameters, n_value, d_value)
    cluster_ids = [
        validation_cluster_id(dataset_name, observation)
        for observation in observations
    ]
    result = validation_score_from_arrays(
        actual, predicted, raw_mape_cap, cluster_ids=cluster_ids
    )
    result["sample_count"] = len(observations)
    return result


def validation_cluster_id(dataset_name: str, observation: Observation) -> str:
    """给轨迹/模型族构造簇，避免把高度相关的检查点视作独立样本。"""

    if dataset_name == "B3":
        return observation.row_id.split(":", maxsplit=1)[0]
    return observation.row_id


def bounded_normalize_weights(
    raw_weights: dict[str, float], minimum: float, maximum: float
) -> dict[str, float]:
    """归一化权重，并通过水位分配满足逐数据集上下限。"""

    dataset_count = len(raw_weights)
    if dataset_count * minimum > 1.0 + 1e-12:
        raise ValueError("最小数据集权重过大，无法使权重和为 1。")
    if dataset_count * maximum < 1.0 - 1e-12:
        raise ValueError("最大数据集权重过小，无法使权重和为 1。")

    assigned: dict[str, float] = {}
    active = set(raw_weights)
    remaining = 1.0
    while active:
        raw_total = sum(max(raw_weights[name], 1e-15) for name in active)
        proposal = {
            name: remaining * max(raw_weights[name], 1e-15) / raw_total
            for name in active
        }
        too_low = [name for name, value in proposal.items() if value < minimum]
        too_high = [name for name, value in proposal.items() if value > maximum]
        if not too_low and not too_high:
            assigned.update(proposal)
            break
        for name in too_low:
            assigned[name] = minimum
            remaining -= minimum
            active.remove(name)
        for name in too_high:
            if name not in active:
                continue
            assigned[name] = maximum
            remaining -= maximum
            active.remove(name)
        if remaining < -1e-12:
            raise RuntimeError("有界权重归一化失败。")
    total = sum(assigned.values())
    # 按输入数据集顺序返回，保证 JSON、CSV 和报告可复现、易比较。
    return {name: assigned[name] / total for name in raw_weights}


def derive_adaptive_weights(
    validation_sets: dict[str, Sequence[Observation]],
    classic_spec: ModelSpec,
    classic_parameters: np.ndarray,
    raw_mape_cap: float,
    bootstrap_count: int,
    seed: int,
    difficulty_power: float,
    minimum_weight: float,
    maximum_weight: float,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]]]:
    """由外部集的先验可靠性、难度与簇重采样稳定性冻结权重。"""

    # 权重只用预先拟合的完整经典律计算一次，不随候选结构反复调整。
    reliability = {"B2": 0.55, "B3": 0.20, "B4": 1.00, "B5": 1.00}
    rng = np.random.default_rng(seed)
    diagnostics: dict[str, dict[str, float | int]] = {}
    raw_weights: dict[str, float] = {}

    for dataset_name, observations in validation_sets.items():
        n_value, d_value, actual = arrays(observations)
        predicted = classic_spec.predict(classic_parameters, n_value, d_value)
        observation_clusters = [
            validation_cluster_id(dataset_name, observation)
            for observation in observations
        ]
        reference = validation_score_from_arrays(
            actual,
            predicted,
            raw_mape_cap,
            cluster_ids=observation_clusters,
        )
        reference_score = float(reference["dataset_score"])

        clusters: dict[str, list[int]] = {}
        for index, observation in enumerate(observations):
            cluster_name = validation_cluster_id(dataset_name, observation)
            clusters.setdefault(cluster_name, []).append(index)
        cluster_names = list(clusters)

        bootstrap_scores: list[float] = []
        if bootstrap_count > 1 and len(cluster_names) > 1:
            for _ in range(bootstrap_count):
                sampled_clusters = rng.choice(
                    cluster_names, size=len(cluster_names), replace=True
                )
                sampled_indices = np.concatenate(
                    [
                        np.asarray(clusters[str(name)], dtype=int)
                        for name in sampled_clusters
                    ]
                )
                bootstrap_result = validation_score_from_arrays(
                    actual[sampled_indices],
                    predicted[sampled_indices],
                    raw_mape_cap,
                    cluster_ids=[
                        observation_clusters[int(index)]
                        for index in sampled_indices
                    ],
                )
                bootstrap_scores.append(
                    float(bootstrap_result["dataset_score"])
                )

        uncertainty = (
            float(np.std(bootstrap_scores, ddof=1))
            if len(bootstrap_scores) > 1
            else 0.0
        )
        # floor 防止 B3 分数接近零时相对不确定度发散。
        # 高误差提高难度权重，高重采样波动则降低可信度。
        stability = 1.0 / (
            1.0 + uncertainty / max(reference_score, 0.02)
        )
        difficulty = max(reference_score, 0.01) ** difficulty_power
        raw_weight = reliability[dataset_name] * difficulty * stability
        raw_weights[dataset_name] = raw_weight
        diagnostics[dataset_name] = {
            "sample_count": len(observations),
            "cluster_count": len(cluster_names),
            "reliability_prior": reliability[dataset_name],
            "reference_classic_score": reference_score,
            "bootstrap_score_std": uncertainty,
            "stability_factor": stability,
            "difficulty_factor": difficulty,
            "raw_weight": raw_weight,
        }

    weights = bounded_normalize_weights(
        raw_weights, minimum=minimum_weight, maximum=maximum_weight
    )
    for dataset_name, weight in weights.items():
        diagnostics[dataset_name]["normalized_weight"] = weight
    return weights, diagnostics


def evaluate_candidate(
    spec: ModelSpec,
    b1_observations: Sequence[Observation],
    validation_sets: dict[str, Sequence[Observation]],
    classic_parameters: np.ndarray,
    weights: dict[str, float],
    raw_mape_cap: float,
    complexity_penalty_rate: float,
    minimum_b1_r_squared: float,
    shared_exponent: float | None = None,
    exponent_lower: float = 0.01,
    exponent_upper: float = 1.50,
    grid_size: int = 31,
) -> CandidateFit:
    # 第一步只在主拟合集上拟合；外部集仅评价是否值得释放指数自由度。
    parameters, b1_prediction = fit_model(
        spec,
        b1_observations,
        classic_parameters,
        shared_exponent=shared_exponent,
        exponent_lower=exponent_lower,
        exponent_upper=exponent_upper,
        grid_size=grid_size,
    )
    b1_actual = observed_losses(b1_observations)
    b1_metrics_object = calculate_metrics(b1_actual, b1_prediction)
    b1_metrics = asdict(b1_metrics_object)
    validation = {
        dataset_name: evaluate_validation_dataset(
            dataset_name, spec, parameters, observations, raw_mape_cap
        )
        for dataset_name, observations in validation_sets.items()
    }
    validation_score = sum(
        weights[name] * float(validation[name]["dataset_score"])
        for name in validation_sets
    )
    # 每增加一个超出三参数基线的自由度都需给出可观测的验证收益。
    complexity_penalty = complexity_penalty_rate * max(
        0, spec.effective_parameter_count - 3
    )
    # 主拟合质量作为准入门槛，复杂度惩罚抑制无收益的自由度释放。
    selection_score = validation_score + complexity_penalty
    eligible = b1_metrics_object.r_squared >= minimum_b1_r_squared
    return CandidateFit(
        spec=spec,
        parameters=parameters,
        b1_predictions=b1_prediction,
        b1_metrics=b1_metrics,
        validation=validation,
        validation_score=validation_score,
        complexity_penalty=complexity_penalty,
        selection_score=selection_score,
        eligible=eligible,
    )


def run_parameter_release_early_stopping(
    b1_observations: Sequence[Observation],
    validation_sets: dict[str, Sequence[Observation]],
    classic_parameters: np.ndarray,
    weights: dict[str, float],
    raw_mape_cap: float,
    complexity_penalty_rate: float,
    minimum_b1_r_squared: float,
    patience: int,
    minimum_delta: float,
    exponent_lower: float = 0.01,
    exponent_upper: float = 1.50,
    grid_size: int = 31,
) -> tuple[list[CandidateFit], CandidateFit | None, int | None]:
    """按共享指数→单指数释放→双指数自由的层级选择结构。

    并列的 ``release_alpha/release_beta`` 先分别完成主数据拟合，再比较
    冻结权重的外部验证分数；连续 ``patience`` 层无实质改善时回滚历史最优。
    """

    specs = {spec.name: spec for spec in MODEL_PATH}
    evaluated: list[CandidateFit] = []
    best: CandidateFit | None = None
    bad_layer_count = 0
    stop_layer: int | None = None

    # 共享指数 s* 是后续两种单指数释放的共同锚点。
    shared = evaluate_candidate(
        spec=specs["shared_exponent"],
        b1_observations=b1_observations,
        validation_sets=validation_sets,
        classic_parameters=classic_parameters,
        weights=weights,
        raw_mape_cap=raw_mape_cap,
        complexity_penalty_rate=complexity_penalty_rate,
        minimum_b1_r_squared=minimum_b1_r_squared,
        exponent_lower=exponent_lower,
        exponent_upper=exponent_upper,
        grid_size=grid_size,
    )
    evaluated.append(shared)
    if shared.eligible:
        shared.improved = True
        best = shared
    shared.bad_stage_count = bad_layer_count
    anchor_exponent = float(shared.parameters[2])

    layers = (
        (
            2,
            (
                specs["release_parameter_exponent"],
                specs["release_data_exponent"],
            ),
        ),
        (3, (specs["classic"],)),
    )
    for layer, layer_specs in layers:
        # 必须完成整层比较，不能因为先看到一个候选较差就提前停止。
        layer_candidates = [
            evaluate_candidate(
                spec=spec,
                b1_observations=b1_observations,
                validation_sets=validation_sets,
                classic_parameters=classic_parameters,
                weights=weights,
                raw_mape_cap=raw_mape_cap,
                complexity_penalty_rate=complexity_penalty_rate,
                minimum_b1_r_squared=minimum_b1_r_squared,
                shared_exponent=anchor_exponent,
                exponent_lower=exponent_lower,
                exponent_upper=exponent_upper,
                grid_size=grid_size,
            )
            for spec in layer_specs
        ]
        evaluated.extend(layer_candidates)
        eligible = [candidate for candidate in layer_candidates if candidate.eligible]
        layer_best = min(eligible, key=lambda item: item.selection_score) if eligible else None

        # 改善幅度须超过 minimum_delta，否则计为一个未改进复杂度层。
        if layer_best is not None and (
            best is None
            or layer_best.selection_score < best.selection_score - minimum_delta
        ):
            best = layer_best
            layer_best.improved = True
            bad_layer_count = 0
        else:
            bad_layer_count += 1

        for candidate in layer_candidates:
            candidate.bad_stage_count = bad_layer_count
        if bad_layer_count >= patience:
            if layer_best is not None:
                layer_best.stop_triggered = True
            stop_layer = layer
            break

    return evaluated, best, stop_layer


def candidate_payload(candidate: CandidateFit) -> dict[str, object]:
    return {
        "model": candidate.spec.name,
        "description": candidate.spec.description,
        "parameter_count": candidate.spec.effective_parameter_count,
        "constraint_layer": candidate.spec.layer,
        "search_mode": candidate.spec.search_mode,
        "parameters": {
            name: float(value)
            for name, value in zip(
                candidate.spec.parameter_names, candidate.parameters
            )
        },
        "b1_metrics": candidate.b1_metrics,
        "validation": candidate.validation,
        "validation_score": candidate.validation_score,
        "complexity_penalty": candidate.complexity_penalty,
        "selection_score": candidate.selection_score,
        "eligible": candidate.eligible,
        "improved": candidate.improved,
        "bad_stage_count": candidate.bad_stage_count,
        "stop_triggered": candidate.stop_triggered,
    }


def write_history_csv(path: Path, candidates: Sequence[CandidateFit]) -> None:
    fields = [
        "stage",
        "constraint_layer",
        "search_mode",
        "model",
        "parameter_count",
        "eligible",
        "improved",
        "stop_triggered",
        "bad_stage_count",
        "b1_rmse",
        "b1_r_squared",
        "validation_score",
        "complexity_penalty",
        "selection_score",
    ]
    for dataset_name in ("B2", "B3", "B4", "B5"):
        fields.extend(
            [
                f"{dataset_name}_median_ape",
                f"{dataset_name}_r_squared",
                f"{dataset_name}_spearman_rho",
                f"{dataset_name}_affine_shape_nrmse",
                f"{dataset_name}_normalized_rmse",
                f"{dataset_name}_normalized_median_ape",
                f"{dataset_name}_normalized_r_squared",
                f"{dataset_name}_normalized_spearman_rho",
                f"{dataset_name}_normalization_fold_count",
                f"{dataset_name}_score",
            ]
        )

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage, candidate in enumerate(candidates, start=1):
            row: dict[str, object] = {
                "stage": stage,
                "constraint_layer": candidate.spec.layer,
                "search_mode": candidate.spec.search_mode,
                "model": candidate.spec.name,
                "parameter_count": candidate.spec.effective_parameter_count,
                "eligible": candidate.eligible,
                "improved": candidate.improved,
                "stop_triggered": candidate.stop_triggered,
                "bad_stage_count": candidate.bad_stage_count,
                "b1_rmse": candidate.b1_metrics["rmse"],
                "b1_r_squared": candidate.b1_metrics["r_squared"],
                "validation_score": candidate.validation_score,
                "complexity_penalty": candidate.complexity_penalty,
                "selection_score": candidate.selection_score,
            }
            for dataset_name, result in candidate.validation.items():
                direct = result["direct_metrics"]
                shape = result["affine_shape_diagnostic"]
                normalization = result["cross_fitted_scale_normalization"]
                normalized = normalization["normalized_metrics"]
                row[f"{dataset_name}_median_ape"] = direct["median_ape"]
                row[f"{dataset_name}_r_squared"] = direct["r_squared"]
                row[f"{dataset_name}_spearman_rho"] = direct["spearman_rho"]
                row[f"{dataset_name}_affine_shape_nrmse"] = shape[
                    "normalized_rmse"
                ]
                row[f"{dataset_name}_normalized_rmse"] = normalized["rmse"]
                row[f"{dataset_name}_normalized_median_ape"] = normalized[
                    "median_ape"
                ]
                row[f"{dataset_name}_normalized_r_squared"] = normalized[
                    "r_squared"
                ]
                row[f"{dataset_name}_normalized_spearman_rho"] = normalized[
                    "spearman_rho"
                ]
                row[f"{dataset_name}_normalization_fold_count"] = normalization[
                    "fold_count"
                ]
                row[f"{dataset_name}_score"] = result["dataset_score"]
            writer.writerow(row)


def write_weight_diagnostics_csv(
    path: Path,
    weights: dict[str, float],
    diagnostics: dict[str, dict[str, float | int]],
) -> None:
    fields = (
        "dataset",
        "sample_count",
        "cluster_count",
        "reliability_prior",
        "reference_classic_score",
        "bootstrap_score_std",
        "stability_factor",
        "difficulty_factor",
        "raw_weight",
        "normalized_weight",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset_name in ("B2", "B3", "B4", "B5"):
            row = {"dataset": dataset_name, **diagnostics[dataset_name]}
            row["normalized_weight"] = weights[dataset_name]
            writer.writerow(row)


def write_predictions_csv(
    path: Path,
    chosen: CandidateFit,
    validation_sets: dict[str, Sequence[Observation]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "row_id",
                "N_params_B",
                "D_tokens_B",
                "actual_loss",
                "predicted_loss",
                "residual_actual_minus_predicted",
            ),
        )
        writer.writeheader()
        for dataset_name, observations in validation_sets.items():
            n_value, d_value, actual = arrays(observations)
            predicted = chosen.spec.predict(chosen.parameters, n_value, d_value)
            for item, actual_value, predicted_value in zip(
                observations, actual, predicted
            ):
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "row_id": item.row_id,
                        "N_params_B": item.n_params_b,
                        "D_tokens_B": item.d_tokens_b,
                        "actual_loss": float(actual_value),
                        "predicted_loss": float(predicted_value),
                        "residual_actual_minus_predicted": float(
                            actual_value - predicted_value
                        ),
                    }
                )


def write_report(
    path: Path,
    chosen: CandidateFit,
    evaluated: Sequence[CandidateFit],
    stop_stage: int | None,
    weights: dict[str, float],
    weight_diagnostics: dict[str, dict[str, float | int]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# B2--B5 驱动的经典标度律早停报告",
        "",
        "## 结论",
        "",
        f"- 选中约束结构：`{chosen.spec.name}`（有效参数数 {chosen.spec.effective_parameter_count}）。",
        f"- B1 拟合：R²={chosen.b1_metrics['r_squared']:.6f}，RMSE={chosen.b1_metrics['rmse']:.6g}。",
        f"- 验证综合分数：{chosen.validation_score:.6f}；含复杂度惩罚后的选模分数：{chosen.selection_score:.6f}。",
        (
            f"- 第 {stop_stage} 个复杂度阶段触发早停，并回滚到历史最优模型。"
            if stop_stage is not None
            else "- 已评估完整个复杂度路径；最终按历史最优验证分数选模。"
        ),
        "",
        "## 策略",
        "",
        "- 所有候选模型只在 B1 上估计参数；B2、B3、B4、B5 不参与参数拟合。",
        "- B2 按 run_id 分组，仅保留最大训练步（并以最大 Token 数判定并列）的最终收敛 checkpoint。",
        "- 模型形式始终为 `L=E+A*N^(-alpha)+B*D^(-beta)`；固定指数后以线性最小二乘拟合 E、A、B。",
        "- 指数释放顺序为：共享指数一维搜索 → alpha/beta 单方向并行释放 → alpha-beta 完整二维搜索。",
        "- 不同来源的 Loss 先通过留一轨迹/模型族仿射校准映射到 B1 参考尺度，再计算用于早停的验证分数。",
        f"- B1 准入门槛：R² ≥ {args.min_b1_r2:.4f}。",
        f"- 验证分数至少改善 {args.min_delta:.4g} 才计为有效改善；耐心值为 {args.patience} 个复杂度阶段。",
        "- 单数据集分数 = 截断后的交叉归一化中位相对误差 + 0.25×交叉归一化 NRMSE + 0.10×归一化排序惩罚。",
        f"- 归一化相对误差截断上限为 {args.raw_mape_cap:.1%}，用于限制单一来源对综合分数的支配。",
        (
            "- 权重模式：自适应；权重由经典模型的验证难度、簇重采样稳定性和数据可信度先验共同确定，确定后在候选路径中冻结。"
            if args.weight_mode == "adaptive"
            else "- 权重模式：固定权重。"
        ),
        "- 最终权重："
        + "，".join(
            f"{name}={weights[name]:.4f}" for name in ("B2", "B3", "B4", "B5")
        )
        + "。",
        (
            f"- 自适应原始权重满足 `可信度先验 × 难度分数^{args.difficulty_power:g} × 稳定性`；"
            f"稳定性由 {args.weight_bootstrap} 次按轨迹/模型族分簇 Bootstrap 估计。"
            if args.weight_mode == "adaptive"
            else "- 固定模式保留原权重 B2=0.20、B3=0.10、B4=0.35、B5=0.35。"
        ),
        "",
        "## 权重诊断",
        "",
        "| 数据集 | 簇数 | 可信度先验 | 基准难度分数 | Bootstrap 标准差 | 稳定性 | 最终权重 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_name in ("B2", "B3", "B4", "B5"):
        diagnostic = weight_diagnostics[dataset_name]
        lines.append(
            "| {name} | {clusters} | {reliability:.2f} | {score:.6f} | "
            "{uncertainty:.6f} | {stability:.4f} | {weight:.4f} |".format(
                name=dataset_name,
                clusters=diagnostic["cluster_count"],
                reliability=diagnostic["reliability_prior"],
                score=diagnostic["reference_classic_score"],
                uncertainty=diagnostic["bootstrap_score_std"],
                stability=diagnostic["stability_factor"],
                weight=weights[dataset_name],
            )
        )
    lines.extend(
        [
        "",
        "## 复杂度路径",
        "",
        "| 约束层 | 候选结构 | 有效参数数 | B1 R² | 验证分数 | 选模分数 | 改善 | 早停 |",
        "|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for candidate in evaluated:
        lines.append(
            "| {stage} | {name} | {count} | {r2:.6f} | {validation:.6f} | "
            "{selection:.6f} | {improved} | {stopped} |".format(
                stage=candidate.spec.layer,
                name=candidate.spec.name,
                count=candidate.spec.effective_parameter_count,
                r2=candidate.b1_metrics["r_squared"],
                validation=candidate.validation_score,
                selection=candidate.selection_score,
                improved="是" if candidate.improved else "否",
                stopped="是" if candidate.stop_triggered else "否",
            )
        )
    lines.extend(
        [
            "",
            "## 选中结构的尺度归一化验证",
            "",
            "| 数据集 | 原始 RMSE | 归一化 RMSE | 归一化中位相对误差 | 归一化 Spearman | 校准折数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_name, result in chosen.validation.items():
        direct = result["direct_metrics"]
        normalization = result["cross_fitted_scale_normalization"]
        normalized = normalization["normalized_metrics"]
        lines.append(
            f"| {dataset_name} | {direct['rmse']:.6f} | "
            f"{normalized['rmse']:.6f} | {normalized['median_ape']:.2%} | "
            f"{normalized['spearman_rho']:.4f} | {normalization['fold_count']} |"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "B2 仅使用各运行的最终收敛 checkpoint，其直接误差仍同时包含模型族与 Loss 标尺迁移，不能单凭负 R² 判定 B1 过拟合；应结合交叉尺度归一化误差和 Spearman 排序相关。B3 是从 B1 检查点插值得到的连续轨迹，不是独立外部验证。B4、B5 更适合判断跨族趋势，但同样存在训练语料、分词器和评测口径差异。",
            "",
            "本策略按用户要求让 B5 参与早停，因此 B5 已成为模型选择数据，不能再作为完全独立的最终测试集。正式论文若需要无偏泛化估计，应另留一组文献数据，或采用按模型族分组的嵌套交叉验证。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "保持标准五参数标度律不变，沿指数参数释放路径执行结构早停。"
        )
    )
    parser.add_argument("--b-dir", type=Path, default=DEFAULT_B_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--min-b1-r2", type=float, default=0.995)
    parser.add_argument("--raw-mape-cap", type=float, default=0.20)
    parser.add_argument("--complexity-penalty", type=float, default=0.001)
    parser.add_argument(
        "--weight-mode",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="B2--B5 权重模式；默认根据验证难度和稳定性自适应确定。",
    )
    parser.add_argument("--weight-bootstrap", type=int, default=200)
    parser.add_argument("--weight-seed", type=int, default=2025)
    parser.add_argument("--difficulty-power", type=float, default=0.50)
    parser.add_argument("--min-dataset-weight", type=float, default=0.05)
    parser.add_argument("--max-dataset-weight", type=float, default=0.50)
    return parser


def main() -> int:
    configure_console()
    args = build_argument_parser().parse_args()
    if args.patience < 1:
        raise ValueError("--patience 必须大于等于 1。")
    if args.min_delta < 0 or args.complexity_penalty < 0:
        raise ValueError("--min-delta 和 --complexity-penalty 必须非负。")
    if not 0 < args.min_b1_r2 <= 1:
        raise ValueError("--min-b1-r2 必须位于 (0, 1]。")
    if args.raw_mape_cap <= 0:
        raise ValueError("--raw-mape-cap 必须为正数。")
    if args.weight_bootstrap < 0:
        raise ValueError("--weight-bootstrap 必须非负。")
    if args.difficulty_power < 0:
        raise ValueError("--difficulty-power 必须非负。")
    if not 0 <= args.min_dataset_weight <= args.max_dataset_weight <= 1:
        raise ValueError("数据集权重上下限必须满足 0 <= min <= max <= 1。")

    b_dir = args.b_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    b1_observations = load_classic_observations(
        b_dir / "pythia_training_log_existing.csv", row_id_column="run_id"
    )
    validation_sets: dict[str, Sequence[Observation]] = {
        "B2": load_classic_observations(
            b_dir / "cerebras_training_log.csv",
            row_id_column="run_id",
            final_checkpoint_only=True,
        ),
        "B3": load_b3_observations(b_dir),
        "B4": load_classic_observations(
            b_dir / "scaling_baseline.csv",
            row_id_column="family",
            converged_only=True,
        ),
        "B5": load_classic_observations(
            b_dir / "published_scaling_data.csv",
            row_id_column="family",
            converged_only=True,
        ),
    }
    classic_fit = fit_scaling_law(
        b1_observations,
        exponent_lower=0.01,
        exponent_upper=1.50,
        grid_size=args.grid_size,
        tolerance=1e-7,
        enforce_physical_constraints=True,
    )
    classic_parameters = np.asarray(
        [
            classic_fit.parameters.irreducible_loss,
            classic_fit.parameters.parameter_coefficient,
            classic_fit.parameters.parameter_exponent,
            classic_fit.parameters.data_coefficient,
            classic_fit.parameters.data_exponent,
        ],
        dtype=float,
    )

    if args.weight_mode == "adaptive":
        classic_spec = next(spec for spec in MODEL_PATH if spec.name == "classic")
        weights, weight_diagnostics = derive_adaptive_weights(
            validation_sets=validation_sets,
            classic_spec=classic_spec,
            classic_parameters=classic_parameters,
            raw_mape_cap=args.raw_mape_cap,
            bootstrap_count=args.weight_bootstrap,
            seed=args.weight_seed,
            difficulty_power=args.difficulty_power,
            minimum_weight=args.min_dataset_weight,
            maximum_weight=args.max_dataset_weight,
        )
    else:
        weights = {"B2": 0.20, "B3": 0.10, "B4": 0.35, "B5": 0.35}
        weight_diagnostics = {}
        for dataset_name, observations in validation_sets.items():
            weight_diagnostics[dataset_name] = {
                "sample_count": len(observations),
                "cluster_count": len(
                    {
                        validation_cluster_id(dataset_name, item)
                        for item in observations
                    }
                ),
                "reliability_prior": 1.0,
                "reference_classic_score": math.nan,
                "bootstrap_score_std": math.nan,
                "stability_factor": 1.0,
                "difficulty_factor": 1.0,
                "raw_weight": weights[dataset_name],
                "normalized_weight": weights[dataset_name],
            }

    evaluated, chosen, stop_stage = run_parameter_release_early_stopping(
        b1_observations=b1_observations,
        validation_sets=validation_sets,
        classic_parameters=classic_parameters,
        weights=weights,
        raw_mape_cap=args.raw_mape_cap,
        complexity_penalty_rate=args.complexity_penalty,
        minimum_b1_r_squared=args.min_b1_r2,
        patience=args.patience,
        minimum_delta=args.min_delta,
        exponent_lower=0.01,
        exponent_upper=1.50,
        grid_size=args.grid_size,
    )

    if chosen is None:
        raise RuntimeError(
            "所有候选模型均未达到 B1 拟合门槛；请降低 --min-b1-r2。"
        )

    results = {
        "method": (
            "standard scaling law with separable nonlinear least squares "
            "and exponent-release early stopping"
        ),
        "model_path": [spec.name for spec in MODEL_PATH],
        "settings": {
            "loss_scale_normalization": {
                "reference_scale": "B1 identity scale",
                "external_transform": "(L_source - intercept) / slope",
                "calibration": "leave-one-trajectory-or-family-out",
                "selection_uses_normalized_metrics": True,
                "raw_metrics_retained": True,
            },
            "weights": weights,
            "weight_mode": args.weight_mode,
            "adaptive_weight_diagnostics": weight_diagnostics,
            "adaptive_weight_settings": {
                "bootstrap_count": args.weight_bootstrap,
                "seed": args.weight_seed,
                "difficulty_power": args.difficulty_power,
                "minimum_dataset_weight": args.min_dataset_weight,
                "maximum_dataset_weight": args.max_dataset_weight,
                "reliability_priors": {
                    "B2": 0.55,
                    "B3": 0.20,
                    "B4": 1.00,
                    "B5": 1.00,
                },
            },
            "patience": args.patience,
            "minimum_delta": args.min_delta,
            "minimum_b1_r_squared": args.min_b1_r2,
            "raw_mape_cap": args.raw_mape_cap,
            "complexity_penalty_per_extra_parameter": args.complexity_penalty,
            "dataset_roles": {
                "B1": "parameter fitting only",
                "B2": "final converged checkpoint per run for cross-family selection",
                "B3": "B1-derived interpolation continuity selection; low weight",
                "B4": "cross-family model selection",
                "B5": "published-data model selection; not an independent test after use",
            },
        },
        "stop_stage": stop_stage,
        "chosen_model": candidate_payload(chosen),
        "candidates": [candidate_payload(item) for item in evaluated],
    }
    (output_dir / "early_stopping_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_history_csv(output_dir / "early_stopping_history.csv", evaluated)
    write_weight_diagnostics_csv(
        output_dir / "adaptive_weights.csv", weights, weight_diagnostics
    )
    write_predictions_csv(
        output_dir / "chosen_model_predictions.csv", chosen, validation_sets
    )
    write_report(
        output_dir / "early_stopping_report.md",
        chosen,
        evaluated,
        stop_stage,
        weights,
        weight_diagnostics,
        args,
    )

    print("结构早停完成。")
    print(f"选中模型: {chosen.spec.name}")
    print(f"B1 R^2: {chosen.b1_metrics['r_squared']:.8f}")
    print(f"验证分数: {chosen.validation_score:.8f}")
    print(f"选模分数: {chosen.selection_score:.8f}")
    print(f"早停阶段: {stop_stage if stop_stage is not None else '未提前停止'}")
    print(f"输出目录: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
