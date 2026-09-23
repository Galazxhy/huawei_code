#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 B2--B5 验证表现对经典标度律进行“结构早停”。

这里的早停不是神经网络训练中的 epoch 早停。经典标度律只有少量参数，
更合理的做法是沿着模型复杂度路径逐级拟合：

    1. compute_only:      L = E + C (N D)^(-s)
    2. shared_exponent:   L = E + A N^(-s) + B D^(-s)
    3. classic:           L = E + A N^(-alpha) + B D^(-beta)
    4. interaction:       classic + C (N D)^(-gamma)

所有候选模型只使用 B1 拟合。B2--B5 仅用于计算验证分数并决定是否继续
增加自由度。若验证分数连续 ``patience`` 个复杂度阶段没有实质改善，就停止，
并回滚到历史最优模型。

默认运行：

    conda run -n GNN python q1/scaling_law_early_stopping.py

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
from scipy.optimize import least_squares

from scaling_law_full_analysis import (
    DEFAULT_B_DIR,
    affine_recalibration,
    calculate_metrics,
    load_b3_observations,
    load_classic_observations,
    observed_losses,
)
from traditional_scaling_law import Observation, fit_scaling_law


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data_analysis" / "scaling_law_early_stopping"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameter_names: tuple[str, ...]
    description: str
    predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


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


def _compute_only(
    parameters: np.ndarray, n_value: np.ndarray, d_value: np.ndarray
) -> np.ndarray:
    e_value, coefficient, exponent = parameters
    return e_value + coefficient * np.power(n_value * d_value, -exponent)


def _shared_exponent(
    parameters: np.ndarray, n_value: np.ndarray, d_value: np.ndarray
) -> np.ndarray:
    e_value, a_value, b_value, exponent = parameters
    return (
        e_value
        + a_value * np.power(n_value, -exponent)
        + b_value * np.power(d_value, -exponent)
    )


def _classic(
    parameters: np.ndarray, n_value: np.ndarray, d_value: np.ndarray
) -> np.ndarray:
    e_value, a_value, alpha, b_value, beta = parameters
    return (
        e_value
        + a_value * np.power(n_value, -alpha)
        + b_value * np.power(d_value, -beta)
    )


def _interaction(
    parameters: np.ndarray, n_value: np.ndarray, d_value: np.ndarray
) -> np.ndarray:
    e_value, a_value, alpha, b_value, beta, c_value, gamma = parameters
    return (
        e_value
        + a_value * np.power(n_value, -alpha)
        + b_value * np.power(d_value, -beta)
        + c_value * np.power(n_value * d_value, -gamma)
    )


MODEL_PATH = (
    ModelSpec(
        "compute_only",
        ("E", "C", "s"),
        "仅使用总计算量 ND 的三参数基线",
        _compute_only,
    ),
    ModelSpec(
        "shared_exponent",
        ("E", "A", "B", "s"),
        "参数项和数据项共享指数",
        _shared_exponent,
    ),
    ModelSpec(
        "classic",
        ("E", "A", "alpha", "B", "beta"),
        "经典五参数标度律",
        _classic,
    ),
    ModelSpec(
        "classic_plus_interaction",
        ("E", "A", "alpha", "B", "beta", "C", "gamma"),
        "经典模型增加 ND 交互项；用于检测继续增复杂度是否过拟合",
        _interaction,
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


def fit_model(
    spec: ModelSpec,
    observations: Sequence[Observation],
    classic_parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_value, d_value, actual = arrays(observations)
    scale = max(float(np.std(actual)), 1e-8)
    minimum_loss = float(np.min(actual))
    loss_span = max(float(np.ptp(actual)), 0.1)

    classic_e, classic_a, classic_alpha, classic_b, classic_beta = (
        classic_parameters
    )
    shared_exponent = float((classic_alpha + classic_beta) / 2.0)

    if spec.name == "compute_only":
        starts = [
            np.asarray([0.8 * minimum_loss, loss_span, 0.20]),
            np.asarray([0.5 * minimum_loss, loss_span, 0.35]),
            np.asarray([classic_e, classic_a + classic_b, shared_exponent / 2]),
        ]
        lower = np.asarray([0.0, 0.0, 0.005])
        upper = np.asarray([minimum_loss * 0.9999, 100.0, 2.0])
    elif spec.name == "shared_exponent":
        starts = [
            np.asarray(
                [classic_e, classic_a, classic_b, shared_exponent]
            ),
            np.asarray([0.8 * minimum_loss, 0.5, 1.0, 0.25]),
            np.asarray([0.5 * minimum_loss, 1.0, 1.0, 0.5]),
        ]
        lower = np.asarray([0.0, 0.0, 0.0, 0.005])
        upper = np.asarray([minimum_loss * 0.9999, 100.0, 100.0, 2.0])
    elif spec.name == "classic":
        # 使用可分离非线性最小二乘得到的稳定 B1 解，不重复数值优化。
        prediction = spec.predict(classic_parameters, n_value, d_value)
        return classic_parameters.copy(), prediction
    else:
        starts = [
            np.asarray(
                [
                    classic_e,
                    classic_a,
                    classic_alpha,
                    classic_b,
                    classic_beta,
                    interaction_coefficient,
                    interaction_exponent,
                ]
            )
            for interaction_coefficient in (1e-4, 0.02, 0.10)
            for interaction_exponent in (0.10, 0.30, 0.60)
        ]
        lower = np.asarray([0.0, 0.0, 0.005, 0.0, 0.005, 0.0, 0.005])
        upper = np.asarray(
            [minimum_loss * 0.9999, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0]
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (spec.predict(parameters, n_value, d_value) - actual) / scale

    best_parameters: np.ndarray | None = None
    best_cost = math.inf
    for start in starts:
        clipped_start = np.minimum(np.maximum(start, lower + 1e-10), upper - 1e-10)
        result = least_squares(
            residual,
            clipped_start,
            bounds=(lower, upper),
            method="trf",
            max_nfev=5000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        cost = float(np.sum(np.square(residual(result.x))))
        if result.success and cost < best_cost:
            best_cost = cost
            best_parameters = result.x

    if best_parameters is None:
        raise RuntimeError(f"{spec.name} 拟合失败。")
    return best_parameters, spec.predict(best_parameters, n_value, d_value)


def safe_rho(value: float) -> float:
    return float(value) if math.isfinite(value) else -1.0


def validation_score_from_arrays(
    actual: np.ndarray,
    predicted: np.ndarray,
    raw_mape_cap: float,
) -> dict[str, object]:
    """计算跨数据源稳健分数；既保留尺度误差，也强调曲线形状。"""

    direct = calculate_metrics(actual, predicted)
    intercept, slope, shape_metrics = affine_recalibration(actual, predicted)
    target_std = max(float(np.std(actual)), 1e-12)
    shape_nrmse = shape_metrics.rmse / target_std
    scale_component = min(direct.median_ape, raw_mape_cap)
    rank_component = 1.0 - float(
        np.clip(safe_rho(direct.spearman_rho), -1.0, 1.0)
    )
    dataset_score = (
        scale_component + 0.25 * shape_nrmse + 0.10 * rank_component
    )
    return {
        "direct_metrics": asdict(direct),
        "affine_shape_diagnostic": {
            "intercept": intercept,
            "slope": slope,
            "metrics": asdict(shape_metrics),
            "normalized_rmse": shape_nrmse,
        },
        "score_components": {
            "capped_median_ape": scale_component,
            "affine_shape_nrmse": shape_nrmse,
            "rank_penalty": rank_component,
        },
        "dataset_score": dataset_score,
    }


def evaluate_validation_dataset(
    spec: ModelSpec,
    parameters: np.ndarray,
    observations: Sequence[Observation],
    raw_mape_cap: float,
) -> dict[str, object]:
    n_value, d_value, actual = arrays(observations)
    predicted = spec.predict(parameters, n_value, d_value)
    result = validation_score_from_arrays(actual, predicted, raw_mape_cap)
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
    """由基准经典模型的难度和簇重采样稳定性一次性确定权重。"""

    reliability = {"B2": 0.55, "B3": 0.20, "B4": 1.00, "B5": 1.00}
    rng = np.random.default_rng(seed)
    diagnostics: dict[str, dict[str, float | int]] = {}
    raw_weights: dict[str, float] = {}

    for dataset_name, observations in validation_sets.items():
        n_value, d_value, actual = arrays(observations)
        predicted = classic_spec.predict(classic_parameters, n_value, d_value)
        reference = validation_score_from_arrays(
            actual, predicted, raw_mape_cap
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
) -> CandidateFit:
    parameters, b1_prediction = fit_model(
        spec, b1_observations, classic_parameters
    )
    b1_actual = observed_losses(b1_observations)
    b1_metrics_object = calculate_metrics(b1_actual, b1_prediction)
    b1_metrics = asdict(b1_metrics_object)
    validation = {
        dataset_name: evaluate_validation_dataset(
            spec, parameters, observations, raw_mape_cap
        )
        for dataset_name, observations in validation_sets.items()
    }
    validation_score = sum(
        weights[name] * float(validation[name]["dataset_score"])
        for name in validation_sets
    )
    # 每增加一个超出三参数基线的自由度都需给出可观测的验证收益。
    complexity_penalty = complexity_penalty_rate * max(
        0, len(spec.parameter_names) - 3
    )
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


def update_early_stopping_state(
    candidate: CandidateFit,
    best: CandidateFit | None,
    bad_stage_count: int,
    stage: int,
    patience: int,
    minimum_delta: float,
) -> tuple[CandidateFit | None, int, int | None]:
    """处理一个已完成的复杂度阶段，返回更新后的早停状态。"""

    if not candidate.eligible:
        candidate.bad_stage_count = bad_stage_count
        return best, bad_stage_count, None

    if best is None or candidate.selection_score < best.selection_score - minimum_delta:
        best = candidate
        candidate.improved = True
        bad_stage_count = 0
    else:
        bad_stage_count += 1
    candidate.bad_stage_count = bad_stage_count

    if bad_stage_count >= patience:
        candidate.stop_triggered = True
        return best, bad_stage_count, stage
    return best, bad_stage_count, None


def candidate_payload(candidate: CandidateFit) -> dict[str, object]:
    return {
        "model": candidate.spec.name,
        "description": candidate.spec.description,
        "parameter_count": len(candidate.spec.parameter_names),
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
                f"{dataset_name}_score",
            ]
        )

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage, candidate in enumerate(candidates, start=1):
            row: dict[str, object] = {
                "stage": stage,
                "model": candidate.spec.name,
                "parameter_count": len(candidate.spec.parameter_names),
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
                row[f"{dataset_name}_median_ape"] = direct["median_ape"]
                row[f"{dataset_name}_r_squared"] = direct["r_squared"]
                row[f"{dataset_name}_spearman_rho"] = direct["spearman_rho"]
                row[f"{dataset_name}_affine_shape_nrmse"] = shape[
                    "normalized_rmse"
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
        f"- 选中模型：`{chosen.spec.name}`（{len(chosen.spec.parameter_names)} 个参数）。",
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
        f"- B1 准入门槛：R² ≥ {args.min_b1_r2:.4f}。",
        f"- 验证分数至少改善 {args.min_delta:.4g} 才计为有效改善；耐心值为 {args.patience} 个复杂度阶段。",
        "- 单数据集分数 = 截断后的中位相对误差 + 0.25×仿射校准后 NRMSE + 0.10×排序惩罚。",
        f"- 直接误差截断上限为 {args.raw_mape_cap:.1%}，用于减弱不同 Loss 口径造成的整体偏移。",
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
        "| 阶段 | 模型 | 参数数 | B1 R² | 验证分数 | 选模分数 | 改善 | 早停 |",
        "|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for stage, candidate in enumerate(evaluated, start=1):
        lines.append(
            "| {stage} | {name} | {count} | {r2:.6f} | {validation:.6f} | "
            "{selection:.6f} | {improved} | {stopped} |".format(
                stage=stage,
                name=candidate.spec.name,
                count=len(candidate.spec.parameter_names),
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
            "## 解释限制",
            "",
            "B2 的直接误差同时包含模型族与 Loss 标尺迁移，不能单凭 B2 的负 R² 判定 B1 过拟合；应结合仿射校准后的曲线形状误差和 Spearman 排序相关。B3 是从 B1 检查点插值得到的连续轨迹，不是独立外部验证。B4、B5 更适合判断跨族趋势，但同样存在训练语料、分词器和评测口径差异。",
            "",
            "本策略按用户要求让 B5 参与早停，因此 B5 已成为模型选择数据，不能再作为完全独立的最终测试集。正式论文若需要无偏泛化估计，应另留一组文献数据，或采用按模型族分组的嵌套交叉验证。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只用 B1 拟合，并基于 B2--B5 验证表现执行结构早停。"
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
            b_dir / "cerebras_training_log.csv", row_id_column="run_id"
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

    evaluated: list[CandidateFit] = []
    chosen: CandidateFit | None = None
    bad_stage_count = 0
    stop_stage: int | None = None
    for stage, spec in enumerate(MODEL_PATH, start=1):
        candidate = evaluate_candidate(
            spec=spec,
            b1_observations=b1_observations,
            validation_sets=validation_sets,
            classic_parameters=classic_parameters,
            weights=weights,
            raw_mape_cap=args.raw_mape_cap,
            complexity_penalty_rate=args.complexity_penalty,
            minimum_b1_r_squared=args.min_b1_r2,
        )
        evaluated.append(candidate)
        chosen, bad_stage_count, stop_stage = update_early_stopping_state(
            candidate=candidate,
            best=chosen,
            bad_stage_count=bad_stage_count,
            stage=stage,
            patience=args.patience,
            minimum_delta=args.min_delta,
        )
        if stop_stage is not None:
            break

    if chosen is None:
        raise RuntimeError(
            "所有候选模型均未达到 B1 拟合门槛；请降低 --min-b1-r2。"
        )

    results = {
        "method": "B1-only fitting with B2-B5 structural early stopping",
        "model_path": [spec.name for spec in MODEL_PATH],
        "settings": {
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
                "B2": "semi-synthetic cross-family trajectory selection",
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
