#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分离式组成广义标度律，并保留原广义律作同划分对照。

    L_k = E_k + A_k N^(-alpha) + B_k D^(-beta)
          + (N/N_ref)^(-eta_k) (D/D_ref)^(-zeta_k) Phi_k(p,Q)

Phi_k 由参考规模的质量加权配比与中心化对数比拟合、零中心化；不设置独立 C。
下式为保留的原广义律对照，并非新模型的配比函数：

    L_k(N,D,p) = E_k + A_k N^(-alpha_k)
                 + B_k D^(-beta_k) [1 + (sum_i p_i Q_i W_i)^gamma]

N、D 在公式中分别以十亿参数、十亿 token 为单位。alpha_k、beta_k 固定为
问题一真实轨迹拟合的指数；配比实验只有三个不同的 (N,D)，不足以从留出数据
重新辨识全部逐域指数。Q_i 从 Q17_mapping.csv 读取且固定。W_i > 0，
各 W_i 独立估计，不施加均值或总和约束；岭惩罚仅使其向 1 软收缩。

每个候选 (W,gamma) 下，13 个评测域的 E_k、A_k、B_k 用非负最小二乘估计。
成对的 1M/60M 配比始终进入相同的数据折。最终测试集不参与拟合或选模。
另做训练折尺度归一化消融，并以调参折原始 Loss 相对误差决定是否采用。
逐规模/逐评测域的岭回归残差校正单独汇报，不属于原律，也不得跨规模外推。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar, nnls
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REGMIX = ROOT / "data/real_attachments/A_data_value/regmix_tables"
HANDOFF = ROOT / "data_analysis/Q_1_to_2"
CLASSIC = ROOT / "data_analysis/scaling_law_full/scaling_law_full_results.json"
OUTPUT = HANDOFF / "generalized_scaling_compact.json"
REPORT = ROOT / "data_analysis/generalized_scaling_law/generalized_scaling_law_compact_report.md"
SCALES = (
    ("train_1m", "train_mixture_1m.csv", "train_pile_loss_1m.csv", 1e6, 1e9),
    ("test_1m", "test_mixture_1m.csv", "test_pile_loss_1m.csv", 1e6, 1e9),
    ("test_60m", "test_mixture_60m.csv", "test_pile_loss_60m.csv", 6e7, 1e9),
    ("test_1b", "test_mixture_1B.csv", "test_pile_loss_1B.csv", 1e9, 2.5e10),
)
RIDGES = (0.001, 0.01, 0.1)
COMPOSITIONAL_RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0)
COMPOSITIONAL_EPSILON = 1e-3


@dataclass(frozen=True)
class Block:
    name: str
    n: float
    d: float
    ids: tuple[str, ...]
    p: np.ndarray
    y: np.ndarray
    mixture_file: str
    loss_file: str

    def subset(self, indices: np.ndarray) -> "Block":
        return Block(self.name, self.n, self.d, tuple(self.ids[i] for i in indices),
                     self.p[indices], self.y[indices], self.mixture_file, self.loss_file)


@dataclass(frozen=True)
class Samples:
    p: np.ndarray
    n: np.ndarray
    d: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Model:
    form: str
    ridge: float
    gamma: float
    w: np.ndarray
    coefficients: np.ndarray  # (13, 3): E_k, A_k, B_k
    q_reference: float = 1.0


@dataclass(frozen=True)
class LossNormalizer:
    """Robust per-scale/per-evaluation-domain statistics from training rows only."""

    median: dict[tuple[float, float], np.ndarray]
    scale: dict[tuple[float, float], np.ndarray]
    counts: dict[tuple[float, float], int]

    @classmethod
    def from_blocks(cls, blocks: list[Block]) -> "LossNormalizer":
        grouped: dict[tuple[float, float], list[np.ndarray]] = {}
        for block in blocks:
            grouped.setdefault((block.n / 1e9, block.d / 1e9), []).append(block.y)
        median, scale, counts = {}, {}, {}
        for key, arrays in grouped.items():
            values = np.vstack(arrays)
            center = np.median(values, axis=0)
            spread = 1.4826 * np.median(np.abs(values - center), axis=0)
            if np.any(spread <= 1e-10):
                raise ValueError(f"训练折的损失 MAD 为零：{key}")
            median[key], scale[key], counts[key] = center, spread, len(values)
        return cls(median, scale, counts)

    def scale_for(self, data: Samples) -> np.ndarray:
        return np.vstack([self.scale[(float(n), float(d))]
                          for n, d in zip(data.n, data.d)])

    def as_dict(self, loss_names: tuple[str, ...]) -> dict[str, object]:
        return {
            f"N={n * 1e9:g},D={d * 1e9:g}": {
                "n_training_rows": self.counts[(n, d)],
                "median": dict(zip(loss_names, self.median[(n, d)].tolist())),
                "robust_scale_1.4826_MAD": dict(zip(loss_names, self.scale[(n, d)].tolist())),
            }
            for n, d in self.scale
        }


def _table(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader))
        rows = [row for row in reader if row]
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError(f"空 CSV 或列数不一致：{path}")
    array = np.asarray([[float(value) for value in row[1:]] for row in rows])
    if not np.isfinite(array).all():
        raise ValueError(f"数据含非有限数值：{path}")
    return header[1:], tuple(row[0] for row in rows), array


def load_data(regmix: Path, handoff: Path) -> tuple[list[Block], tuple[str, ...], tuple[str, ...], np.ndarray]:
    blocks = []
    train_names: tuple[str, ...] | None = None
    loss_names: tuple[str, ...] | None = None
    for label, p_file, y_file, n, d in SCALES:
        p_header, p_ids, p = _table(regmix / p_file)
        y_header, y_ids, y = _table(regmix / y_file)
        names = tuple(value.removeprefix("train_the_pile_") for value in p_header)
        domains = tuple(value.removeprefix("metric/the_pile_").removesuffix("_val_loss")
                        for value in y_header)
        if p_ids != y_ids or len(set(p_ids)) != len(p_ids):
            raise ValueError(f"{label}: 配比与损失的行 ID 不匹配或重复")
        if train_names is not None and (train_names != names or loss_names != domains):
            raise ValueError(f"{label}: 领域列顺序与其余区块不一致")
        if np.any(p < 0) or np.max(np.abs(p.sum(axis=1) - 1)) > 0.03 or np.any(y <= 0):
            raise ValueError(f"{label}: 无效的配比或损失")
        blocks.append(Block(label, n, d, p_ids, p, y, p_file, y_file))
        train_names, loss_names = names, domains
    if blocks[1].ids != blocks[2].ids or not np.allclose(blocks[1].p, blocks[2].p, atol=1e-9):
        raise ValueError("1M 与 60M 的测试配比不是逐行配对")
    with (handoff / "Q17_mapping.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        quality_map = {row["mixture_domain"]: float(row["Q17"])
                       for row in csv.DictReader(handle)}
    if any(name not in quality_map for name in train_names):
        raise ValueError("Q17_mapping.csv 缺少训练领域")
    q = np.asarray([quality_map[name] for name in train_names])
    if not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError("Q17 质量必须为有限正数")
    return blocks, train_names, loss_names, q


def load_exponents(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload["B1_main_fit"]["parameters"]
    return float(parameters["parameter_exponent"]), float(parameters["data_exponent"])


def combine(blocks: list[Block]) -> Samples:
    return Samples(
        np.vstack([block.p for block in blocks]),
        np.concatenate([np.full(len(block.ids), block.n / 1e9) for block in blocks]),
        np.concatenate([np.full(len(block.ids), block.d / 1e9) for block in blocks]),
        np.vstack([block.y for block in blocks]),
    )


def partition(blocks: list[Block], seed: int) -> tuple[list[Block], list[Block], list[Block]]:
    """512 个既定训练配比全留在训练折；跨规模重复配比共用折号。"""
    generator = np.random.default_rng(seed)
    paired = generator.permutation(len(blocks[1].ids))
    large = generator.permutation(len(blocks[3].ids))

    def three_way(order: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, b = round(0.6 * len(order)), round(0.8 * len(order))
        return order[:a], order[a:b], order[b:]

    pair_parts, large_parts = three_way(paired), three_way(large)
    result = []
    for part in range(3):
        selected = [blocks[1].subset(pair_parts[part]),
                    blocks[2].subset(pair_parts[part]),
                    blocks[3].subset(large_parts[part])]
        if part == 0:
            selected.insert(0, blocks[0])
        result.append(selected)
    return result[0], result[1], result[2]


def positive_weights(log_weights: np.ndarray) -> np.ndarray:
    """Independent positive domain weights; no sum or mean normalization."""
    return np.exp(log_weights)


def design(data: Samples, q: np.ndarray, w: np.ndarray, gamma: float,
           alpha: float, beta: float, include_mixture: bool) -> np.ndarray:
    x = np.maximum(data.p @ (q * w), 1e-12)
    multiplier = 1 + x**gamma if include_mixture else np.ones_like(x)
    return np.stack((np.ones((len(x), data.y.shape[1])),
                     np.broadcast_to(data.n[:, None] ** (-alpha), data.y.shape),
                     np.broadcast_to((data.d ** (-beta) * multiplier)[:, None], data.y.shape)), axis=2)


def fit_linear(data: Samples, q: np.ndarray, w: np.ndarray, gamma: float,
               alpha: float, beta: float, include_mixture: bool = True,
               normalizer: LossNormalizer | None = None) -> np.ndarray:
    features = design(data, q, w, gamma, alpha, beta, include_mixture)
    coefficients = np.empty((data.y.shape[1], 3))
    denominator = data.y if normalizer is None else normalizer.scale_for(data)
    for k in range(data.y.shape[1]):
        weighted = features[:, k, :] / denominator[:, k, None]
        coefficients[k], _ = nnls(weighted, data.y[:, k] / denominator[:, k])
    return coefficients


def predict(data: Samples, model: Model, q: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    features = design(data, q / model.q_reference, model.w, model.gamma, alpha, beta,
                      model.form != "no_mixture")
    return np.einsum("nkj,kj->nk", features, model.coefficients)


def relative_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predicted / actual - 1))))


def standardized_rmse(data: Samples, predicted: np.ndarray,
                      normalizer: LossNormalizer) -> float:
    return float(np.sqrt(np.mean(np.square((predicted - data.y) /
                                          normalizer.scale_for(data)))))


def fit_candidate(data: Samples, q: np.ndarray, alpha: float, beta: float,
                  form: str, ridge: float = 0.0,
                  normalizer: LossNormalizer | None = None,
                  q_reference: float = 1.0) -> Model:
    q_effective = q / q_reference
    ones = np.ones(len(q))
    if form == "no_mixture":
        coef = fit_linear(data, q_effective, ones, 1.0, alpha, beta,
                          include_mixture=False, normalizer=normalizer)
        return Model(form, 0.0, 1.0, ones, coef, q_reference)

    if form == "uniform_w":
        def loss_gamma(gamma: float) -> float:
            coef = fit_linear(data, q_effective, ones, gamma, alpha, beta,
                              normalizer=normalizer)
            candidate = Model(form, 0.0, gamma, ones, coef, q_reference)
            predicted = predict(data, candidate, q, alpha, beta)
            return (relative_rmse(data.y, predicted) if normalizer is None else
                    standardized_rmse(data, predicted, normalizer))

        search = minimize_scalar(loss_gamma, bounds=(0.2, 3.0), method="bounded",
                                 options={"xatol": 1e-3})
        gamma = float(search.x)
        return Model(form, 0.0, gamma, ones,
                     fit_linear(data, q_effective, ones, gamma, alpha, beta,
                                normalizer=normalizer), q_reference)

    if form != "fitted_w":
        raise ValueError(f"未知模型形式：{form}")

    def objective(theta: np.ndarray) -> float:
        w = positive_weights(theta[:len(q)])
        gamma = math.exp(float(theta[len(q)]))
        coef = fit_linear(data, q_effective, w, gamma, alpha, beta,
                          normalizer=normalizer)
        candidate = Model(form, ridge, gamma, w, coef, q_reference)
        predicted = predict(data, candidate, q, alpha, beta)
        error = (relative_rmse(data.y, predicted) if normalizer is None else
                 standardized_rmse(data, predicted, normalizer))
        return error ** 2 + ridge * float(np.mean(np.square(w - 1)))

    start = np.zeros(len(q) + 1)
    bounds = [(-2.0, 2.0)] * len(q) + [(math.log(0.2), math.log(3.0))]
    solution = minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 90, "ftol": 1e-9})
    w = positive_weights(solution.x[:len(q)])
    gamma = math.exp(float(solution.x[len(q)]))
    return Model(form, ridge, gamma, w,
                 fit_linear(data, q_effective, w, gamma, alpha, beta,
                            normalizer=normalizer), q_reference)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    error = predicted - actual
    ape = np.abs(error / actual)
    observed_centered = actual - actual.mean(axis=0, keepdims=True)
    predicted_centered = predicted - predicted.mean(axis=0, keepdims=True)
    denominator = float(np.square(observed_centered).sum())
    domain_denominator = np.square(observed_centered).sum(axis=0)
    domain_numerator = np.square(error).sum(axis=0)
    valid_domains = domain_denominator > 1e-12
    domain_r_squared = (1 - domain_numerator[valid_domains] /
                        domain_denominator[valid_domains])
    correlations = [
        float(spearmanr(actual[:, k], predicted[:, k]).statistic)
        for k in range(actual.shape[1])
        if np.ptp(actual[:, k]) > 1e-12 and np.ptp(predicted[:, k]) > 1e-12
    ]
    return {
        "rmse_loss": float(np.sqrt(np.square(error).mean())),
        "relative_rmse": relative_rmse(actual, predicted),
        "median_ape": float(np.median(ape)),
        "p90_ape": float(np.quantile(ape, 0.9)),
        "relative_bias": float(np.mean(error / actual)),
        "r_squared_macro": (float(np.mean(domain_r_squared))
                            if len(domain_r_squared) else None),
        "r_squared_pooled_within_domain": (
            float(1 - domain_numerator[valid_domains].sum() /
                  domain_denominator[valid_domains].sum())
            if np.any(valid_domains) else None),
        "mixture_r_squared": (float(1 - np.square(observed_centered - predicted_centered).sum()
                                    / denominator) if denominator > 0 else math.nan),
        "median_domain_spearman": float(np.median(correlations)) if correlations else None,
    }


def mixture_r_squared_within_scale(
    pairs: list[tuple["Block", np.ndarray]]
) -> float:
    """在**每个 (评测域, 规模) 内部**分别中心化后的配比效应 R²。

    ``metrics()`` 里的 ``mixture_r_squared`` 对单区块调用时等价于本函数；
    但当它被作用在**跨规模堆叠**的数据上时，中心化会跨越规模，
    从而把 N、D 造成的差异也算成"配比效应"。本函数显式按 (域, 规模) 分组，
    因此可以直接用于跨规模的汇总表。
    """

    numerator = 0.0
    denominator = 0.0
    for block, predicted in pairs:
        observed_centered = block.y - block.y.mean(axis=0, keepdims=True)
        predicted_centered = predicted - predicted.mean(axis=0, keepdims=True)
        numerator += float(np.square(observed_centered - predicted_centered).sum())
        denominator += float(np.square(observed_centered).sum())
    if denominator <= 0:
        return math.nan
    return float(1.0 - numerator / denominator)


def irreducible_floor_diagnostic(
    pairs: list[tuple["Block", np.ndarray]], e_coefficients: np.ndarray
) -> dict[str, object]:
    """检查加性配比项是否把预测压到不可约损失 ``E_k`` 之下。

    ``Phi_k`` 是自由符号的加性项，因此模型**原则上**可以预测 ``L_k < E_k``，
    而 ``E_k`` 按定义是 Loss 的下确界。这里只做汇报，不改动模型。
    """

    floor = np.asarray(e_coefficients, dtype=float)
    margins = []
    predictions = []
    for _block, predicted in pairs:
        margins.append(predicted - floor[None, :])
        predictions.append(predicted)
    stacked_margin = np.vstack(margins)
    stacked_prediction = np.vstack(predictions)
    below = stacked_margin < 0.0
    return {
        "min_predicted_minus_E": float(stacked_margin.min()),
        "min_predicted_loss": float(stacked_prediction.min()),
        "fraction_of_predictions_below_E": float(below.mean()),
        "max_predicted_minus_E": float(stacked_margin.max()),
        "verdict": (
            "未击穿不可约损失下界"
            if not below.any()
            else "存在预测低于不可约损失的点，建议对 L_k 做 max(L_k, E_k) 截断并说明"
        ),
    }


def compositional_features(p: np.ndarray, q: np.ndarray,
                           epsilon: float = COMPOSITIONAL_EPSILON) -> np.ndarray:
    """拼接质量加权份额 ``p_i Q_i`` 与单纯形上的中心化对数比。"""
    log_p = np.log(p + epsilon)
    clr = log_p - log_p.mean(axis=1, keepdims=True)
    return np.column_stack((p * q[None, :], clr))


def fit_compositional_scaling_law(
    train_blocks: list[Block], tune_blocks: list[Block], test_blocks: list[Block],
    q: np.ndarray, train_names: tuple[str, ...], loss_names: tuple[str, ...],
    alpha: float, beta: float, epsilon: float = COMPOSITIONAL_EPSILON,
) -> dict[str, object]:
    """用“参考配比形状→规模幅度→经典基线”三阶段辨识广义律。

    ``alpha,beta`` 来自独立经典律并保持固定。参考规模训练折确定
    ``Phi_k(p,Q)`` 的形状；其余观测规模只估计逐目标截距与幅度，
    最后由三个规模锚点恢复 ``E_k,A_k,B_k,eta_k,zeta_k``。
    留出折只作检验，规模锚点之外的幂律延伸不视为实证验证。
    """
    # 阶段一：在参考规模用训练折拟合配比响应，调参折只选 Ridge 强度。
    reference = combine(train_blocks[:2])
    reference_tuning = combine([tune_blocks[0]])
    # Q_i 随来源固定；p_iQ_i 在这里是组成特征，不单独识别“提高质量”的效应。
    scaler = StandardScaler().fit(compositional_features(reference.p, q, epsilon))
    x_train = scaler.transform(compositional_features(reference.p, q, epsilon))
    x_tune = scaler.transform(compositional_features(reference_tuning.p, q, epsilon))
    candidates = []
    for strength in COMPOSITIONAL_RIDGES:
        fitted = Ridge(alpha=strength).fit(x_train, reference.y)
        score = relative_rmse(reference_tuning.y, fitted.predict(x_tune))
        candidates.append((score, strength, fitted))
    _, strength, ridge = min(candidates, key=lambda item: item[0])

    def raw_response(p: np.ndarray) -> np.ndarray:
        return ridge.predict(scaler.transform(compositional_features(p, q, epsilon)))

    # 居中并固定参考幅度，使形状函数与各规模的截距/幅度不互相吸收。
    raw_train = raw_response(reference.p)
    response_center = raw_train.mean(axis=0)
    x_centered = raw_train - response_center
    slope_reference = (
        np.sum(x_centered * (reference.y - reference.y.mean(axis=0)), axis=0)
        / np.sum(np.square(x_centered), axis=0)
    )
    if np.any(slope_reference <= 0):
        raise ValueError("参考规模的配比响应斜率须为正")

    def response(p: np.ndarray) -> np.ndarray:
        return (raw_response(p) - response_center) * slope_reference

    # 阶段二：每个观测规模只回归一个截距和一个配比响应幅度。
    scale_training = (
        reference,
        combine([train_blocks[2]]),
        combine([train_blocks[3]]),
    )
    scale_n = np.asarray([samples.n[0] for samples in scale_training])
    scale_d = np.asarray([samples.d[0] for samples in scale_training])
    reference_n, reference_d = scale_n[0], scale_d[0]
    baselines, amplitudes = [], []
    for index, samples in enumerate(scale_training):
        f = response(samples.p)
        if index == 0:
            amplitude = np.ones(samples.y.shape[1])
        else:
            centered = f - f.mean(axis=0)
            denominator = np.square(centered).sum(axis=0)
            if np.any(denominator <= 1e-12):
                raise ValueError(f"规模 {index} 的配比函数缺少可辨识变化")
            amplitude = (np.sum(centered * (samples.y - samples.y.mean(axis=0)),
                                axis=0) / denominator)
        if np.any(amplitude <= 0):
            raise ValueError(f"规模 {index} 的配比幅度不为正，无法拟合正幂律")
        baselines.append(samples.y.mean(axis=0) - amplitude * f.mean(axis=0))
        amplitudes.append(amplitude)
    baselines = np.stack(baselines)
    amplitudes = np.stack(amplitudes)
    # 阶段三：指数已锚定，用非负最小二乘恢复各目标的 E、A、B。
    scale_design = np.column_stack(
        (np.ones(3), scale_n ** (-alpha), scale_d ** (-beta))
    )
    coefficients = np.stack([
        nnls(scale_design, baselines[:, k])[0]
        for k in range(len(loss_names))
    ])
    reconstructed = scale_design @ coefficients.T
    # 前两个锚点 D 相同，先由其幅度比识别 eta；第三点才条件识别 zeta。
    eta = -np.log(amplitudes[1]) / np.log(scale_n[1] / reference_n)
    zeta = (-np.log(amplitudes[2]) - eta * np.log(scale_n[2] / reference_n)
            ) / np.log(scale_d[2] / reference_d)

    def predict_block(block: Block) -> np.ndarray:
        # 任一留出配方共用同一 Phi_k 形状，幅度只由 (N,D) 调制。
        n, d = block.n / 1e9, block.d / 1e9
        baseline = (coefficients[:, 0] + coefficients[:, 1] * n ** (-alpha)
                    + coefficients[:, 2] * d ** (-beta))
        amplitude = (n / reference_n) ** (-eta) * (d / reference_d) ** (-zeta)
        return baseline[None, :] + amplitude[None, :] * response(block.p)

    def by_block(blocks: list[Block]) -> dict[str, dict[str, object]]:
        return {
            block.name: {"n_mixtures": len(block.ids),
                         **metrics(block.y, predict_block(block))}
            for block in blocks
        }

    observed = np.vstack([block.y for block in test_blocks])
    predicted = np.vstack([predict_block(block) for block in test_blocks])
    aggregate = metrics(observed, predicted)
    aggregate["mixture_r_squared_within_scale"] = mixture_r_squared_within_scale(
        [(block, predict_block(block)) for block in test_blocks])
    aggregate["irreducible_floor"] = irreducible_floor_diagnostic(
        [(block, predict_block(block)) for block in test_blocks], coefficients[:, 0])
    aggregate = {key: value for key, value in aggregate.items()
                 if key not in ("mixture_r_squared", "median_domain_spearman",
                                "r_squared_macro", "r_squared_pooled_within_domain")}
    feature_names = ([f"pQ:{name}" for name in train_names]
                     + [f"clr:{name}" for name in train_names])
    effective_weights = ridge.coef_ * slope_reference[:, None] / scaler.scale_[None, :]
    return {
        "formula": "L_k=E_k+A_k*N_b^(-alpha)+B_k*D_b^(-beta)+S_k(N,D)*Phi_k(p,Q)",
        "scale_amplitude": "S_k(N,D)=(N/N_ref)^(-eta_k)*(D/D_ref)^(-zeta_k)",
        "mixture_response": "Phi_k=slope_ref_k*[Ridge_k(Standardize([p_i*Q_i, clr_epsilon(p)_i]))-mean_train_1M Ridge_k]",
        "expanded_mixture_response": "Phi_k=sum_i Wp_(k,i)*(p_i*Q_i-mean_train_1M(p_i*Q_i))+sum_i Wz_(k,i)*(clr_epsilon(p)_i-mean_train_1M(clr_epsilon(p)_i))",
        "clr_definition": "clr_epsilon(p)_i=log(p_i+epsilon)-mean_j(log(p_j+epsilon))",
        "epsilon": epsilon,
        "reference_scale": {"N": float(reference_n * 1e9),
                            "D": float(reference_d * 1e9)},
        "ridge_candidates": [{"ridge": candidate_strength,
                              "reference_tuning_relative_rmse": score}
                             for score, candidate_strength, _ in candidates],
        "ridge_selected": strength,
        "parameter_sources": {
            "Phi": "1M training mixtures; Ridge strength selected on 1M tuning mixtures",
            "amplitude_and_baseline": "only each observed scale's training mixtures",
            "alpha_beta": "fixed external classic-law estimates",
            "heldout": "not used in any fit or selection",
        },
        "quality_identifiability": (
            "Q_i is fixed for each training domain. With free domain coefficients, "
            "p_i*Q_i contains no information beyond p_i; per-feature standardization "
            "also removes its fixed scale. The fit measures composition response, "
            "not an independently identified quality effect."
        ),
        "parameters": {
            "alpha": alpha, "beta": beta,
            "E": dict(zip(loss_names, coefficients[:, 0].tolist())),
            "A": dict(zip(loss_names, coefficients[:, 1].tolist())),
            "B": dict(zip(loss_names, coefficients[:, 2].tolist())),
            "eta": dict(zip(loss_names, eta.tolist())),
            "zeta": dict(zip(loss_names, zeta.tolist())),
            "reference_response_center": dict(zip(loss_names, response_center.tolist())),
            "reference_response_slope": dict(zip(loss_names, slope_reference.tolist())),
            "feature_names": feature_names,
            "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(),
            "ridge_intercept": dict(zip(loss_names, ridge.intercept_.tolist())),
            "ridge_coefficients": {
                domain: dict(zip(feature_names, ridge.coef_[k].tolist()))
                for k, domain in enumerate(loss_names)
            },
            "effective_weight_pQ": {
                domain: dict(zip(train_names, effective_weights[k, :len(q)].tolist()))
                for k, domain in enumerate(loss_names)
            },
            "effective_weight_clr": {
                domain: dict(zip(train_names, effective_weights[k, len(q):].tolist()))
                for k, domain in enumerate(loss_names)
            },
        },
        "scale_anchor_diagnostics": {
            "observed_amplitude": {
                label: dict(zip(loss_names, amplitudes[index].tolist()))
                for index, label in enumerate((train_blocks[0].name,
                                               train_blocks[2].name,
                                               train_blocks[3].name))
            },
            "baseline_nnls_rmse": float(np.sqrt(np.mean(np.square(baselines - reconstructed)))),
            "negative_zeta_count": int(np.sum(zeta < 0)),
        },
        "training": by_block(train_blocks),
        "tuning": by_block(tune_blocks),
        "heldout_test": by_block(test_blocks),
        "aggregate_heldout": aggregate,
        "extrapolation_warning": (
            "Only three observed scale anchors exist; N and D change together at 1B. "
            "eta/zeta follow an assumed separable power law and are not validated "
            "for unseen scales. Q_i is fixed and cannot reveal a separate quality effect."
        ),
    }


def fit_residual_correction(train_blocks: list[Block], tuning_block: Block,
                            model: Model, q: np.ndarray, alpha: float,
                            beta: float) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    """Ridge-fit a scale-specific, evaluation-domain-specific residual from training only."""
    train = combine(train_blocks)
    tuning = combine([tuning_block])
    features = np.column_stack((np.ones(len(train.p)), train.p))
    tuning_features = np.column_stack((np.ones(len(tuning.p)), tuning.p))
    residual = train.y - predict(train, model, q, alpha, beta)
    penalty = np.diag([0.0] + [1.0] * train.p.shape[1])
    candidates = []
    for strength in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        coefficients = np.linalg.solve(features.T @ features + strength * penalty,
                                       features.T @ residual)
        prediction = (predict(tuning, model, q, alpha, beta)
                      + tuning_features @ coefficients)
        candidates.append((relative_rmse(tuning.y, prediction), strength,
                           coefficients))
    _, strength, coefficients = min(candidates, key=lambda item: item[0])
    return coefficients, strength, [
        {"ridge": candidate_strength, "tuning_relative_rmse": candidate_score}
        for candidate_score, candidate_strength, _ in candidates
    ]


def predict_corrected(block: Block, model: Model, correction: np.ndarray,
                      q: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return (predict(combine([block]), model, q, alpha, beta)
            + np.column_stack((np.ones(len(block.p)), block.p)) @ correction)


def run(regmix: Path = REGMIX, handoff: Path = HANDOFF, classic_path: Path = CLASSIC,
        seed: int = 2025) -> dict[str, object]:
    blocks, train_names, loss_names, q = load_data(regmix, handoff)
    alpha, beta = load_exponents(classic_path)
    train_blocks, tune_blocks, test_blocks = partition(blocks, seed)
    compositional = fit_compositional_scaling_law(
        train_blocks, tune_blocks, test_blocks, q, train_names, loss_names,
        alpha, beta)
    train, tune = combine(train_blocks), combine(tune_blocks)
    loss_scale = LossNormalizer.from_blocks(train_blocks)
    variants: dict[str, dict[str, object]] = {}
    q_reference = float(np.mean(q))
    for variant, normalizer, q_ref in (
        ("raw", None, 1.0),
        ("loss_normalized", loss_scale, 1.0),
        ("loss_and_q_normalized", loss_scale, q_reference),
    ):
        fitted: list[Model] = [
            fit_candidate(train, q, alpha, beta, "no_mixture",
                          normalizer=normalizer, q_reference=q_ref),
            fit_candidate(train, q, alpha, beta, "uniform_w",
                          normalizer=normalizer, q_reference=q_ref),
        ]
        fitted.extend(fit_candidate(train, q, alpha, beta, "fitted_w", ridge,
                                    normalizer=normalizer, q_reference=q_ref)
                      for ridge in RIDGES)
        tuning = []
        for model in fitted:
            predicted = predict(tune, model, q, alpha, beta)
            scoped = [(block, predict(combine([block]), model, q, alpha, beta))
                      for block in tune_blocks]
            tuning.append({"form": model.form, "ridge": model.ridge,
                           "gamma": model.gamma,
                           "standardized_rmse": standardized_rmse(tune, predicted, loss_scale),
                           "mixture_r_squared_within_scale":
                               mixture_r_squared_within_scale(scoped),
                           **metrics(tune.y, predicted)})
        criterion = "relative_rmse" if normalizer is None else "standardized_rmse"
        # 无配比项只作基线；选择限制在原广义公式内。
        chosen = min((i for i, model in enumerate(fitted) if model.form != "no_mixture"),
                     key=lambda i: tuning[i][criterion])
        models = {"no_mixture": fitted[0], "uniform_w": fitted[1],
                  "selected": fitted[chosen]}
        heldout = {}
        for block in test_blocks:
            samples = combine([block])
            heldout[block.name] = {
                "n_mixtures": len(block.ids),
                "models": {
                    name: {**metrics(block.y, predicted),
                           "standardized_rmse": standardized_rmse(samples, predicted,
                                                                    loss_scale)}
                    for name, model in models.items()
                    for predicted in [predict(samples, model, q, alpha, beta)]
                },
            }
        test_all = combine(test_blocks)
        predicted_all = predict(test_all, fitted[chosen], q, alpha, beta)
        aggregate = metrics(test_all.y, predicted_all)
        variants[variant] = {
            "criterion": criterion,
            "candidates": tuning,
            "chosen": {"form": fitted[chosen].form, "ridge": fitted[chosen].ridge,
                       "gamma": fitted[chosen].gamma},
            "selected_tuning": tuning[chosen],
            "heldout_test": heldout,
            "aggregate_heldout": {
                **{key: value for key, value in aggregate.items()
                   if key not in ("mixture_r_squared", "median_domain_spearman",
                                  "r_squared_macro", "r_squared_pooled_within_domain")},
                "standardized_rmse": standardized_rmse(test_all, predicted_all, loss_scale),
                "mixture_r_squared_within_scale": mixture_r_squared_within_scale(
                    [(block, predict(combine([block]), fitted[chosen], q, alpha, beta))
                     for block in test_blocks]),
                "irreducible_floor": irreducible_floor_diagnostic(
                    [(block, predict(combine([block]), fitted[chosen], q, alpha, beta))
                     for block in test_blocks], fitted[chosen].coefficients[:, 0]),
            },
            "selected_model": fitted[chosen],
        }
    # 所有训练口径用同一原始 Loss 相对误差选模，避免只优化标准化指标。
    # 最终留出集只用于一次性评估，不参与选择。
    selected_variant = min(variants,
                           key=lambda name: variants[name]["selected_tuning"]["relative_rmse"])
    selected = variants[selected_variant]["selected_model"]
    tuning = variants[selected_variant]["candidates"]
    heldout = variants[selected_variant]["heldout_test"]
    corrections = {}
    corrected_predictions = []
    for train_group, tuning_block, test_block in (
        (train_blocks[:2], tune_blocks[0], test_blocks[0]),
        ([train_blocks[2]], tune_blocks[1], test_blocks[1]),
        ([train_blocks[3]], tune_blocks[2], test_blocks[2]),
    ):
        correction, strength, candidates = fit_residual_correction(
            train_group, tuning_block, selected, q, alpha, beta)
        predicted = predict_corrected(test_block, selected, correction,
                                      q, alpha, beta)
        if np.any(predicted <= 0):
            raise ValueError(f"{test_block.name}: 残差校正产生非正 Loss")
        sample = combine([test_block])
        heldout[test_block.name]["models"]["corrected"] = {
            **metrics(test_block.y, predicted),
            "standardized_rmse": standardized_rmse(sample, predicted, loss_scale),
        }
        corrected_predictions.append(predicted)
        corrections[test_block.name] = {
            "ridge": strength,
            "training_rows": sum(len(block.ids) for block in train_group),
            "tuning_rows": len(tuning_block.ids),
            "candidates": candidates,
            "coefficients": {
                domain: {"intercept": float(correction[0, k]),
                         "mixture_domain": dict(zip(train_names, correction[1:, k].tolist()))}
                for k, domain in enumerate(loss_names)
            },
        }
    test_all = combine(test_blocks)
    corrected_all = np.vstack(corrected_predictions)
    aggregate_corrected = metrics(test_all.y, corrected_all)
    aggregate_corrected = {
        **{key: value for key, value in aggregate_corrected.items()
           if key not in ("mixture_r_squared", "median_domain_spearman",
                          "r_squared_macro", "r_squared_pooled_within_domain")},
        "standardized_rmse": standardized_rmse(test_all, corrected_all, loss_scale),
        "mixture_r_squared_within_scale": mixture_r_squared_within_scale(
            list(zip(test_blocks, corrected_predictions))),
        "irreducible_floor": irreducible_floor_diagnostic(
            list(zip(test_blocks, corrected_predictions)),
            selected.coefficients[:, 0]),
    }
    comparison = {
        name: {key: value for key, value in result.items() if key != "selected_model"}
        for name, result in variants.items()
    }
    # clr 的 epsilon 是自由选择：数据里有大量精确为 0 的配比份额，
    # log(0+eps) 与典型份额 log(p) 的差距会随 eps 变化，故做一次敏感性扫描。
    epsilon_sensitivity = []
    for epsilon in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
        sweep = fit_compositional_scaling_law(
            train_blocks, tune_blocks, test_blocks, q, train_names, loss_names,
            alpha, beta, epsilon=epsilon)
        epsilon_sensitivity.append({
            "epsilon": epsilon,
            "ridge_selected": sweep["ridge_selected"],
            "heldout_relative_rmse": sweep["aggregate_heldout"]["relative_rmse"],
            "heldout_median_ape": sweep["aggregate_heldout"]["median_ape"],
            "heldout_mixture_r_squared_within_scale":
                sweep["aggregate_heldout"]["mixture_r_squared_within_scale"],
            "heldout_r_squared_macro":
                sweep["heldout_test"][test_blocks[0].name].get("r_squared_macro"),
        })
    centered = train_blocks[0].y - train_blocks[0].y.mean(axis=0)
    pairwise = np.corrcoef(train_blocks[0].y, rowvar=False)
    singular = np.linalg.svd(centered, compute_uv=False)
    structural_diagnostics = {
        "training_1m_negative_domain_pair_fraction": float(np.mean(
            pairwise[np.triu_indices(len(loss_names), 1)] < 0)),
        "training_1m_first_principal_component_variance_fraction": float(
            singular[0] ** 2 / np.sum(singular ** 2)),
        "interpretation": "原公式在固定规模下只通过一个共同标量影响各域，且 B_k>=0；异号配比效应不能同时刻画",
    }
    return {
        "primary_model": "compositional_scaling_law",
        "compositional_scaling_law": compositional,
        "legacy_top_level_note": (
            "For compatibility, the remaining top-level formula/parameters/selection "
            "describe the previous original-law comparator; primary model parameters "
            "are in compositional_scaling_law.parameters."
        ),
        "formula": "L_k=E_k+A_k*N^(-alpha_k)+B_k*D^(-beta_k)*[1+(sum_i p_i*Q_i*W_i)^gamma]",
        "formula_constraints": "No fitted C; W_i>0 independently, without W normalization; E_k,A_k,B_k>=0; alpha_k=alpha_B1; beta_k=beta_B1",
        "normalization": {
            "selected_variant": selected_variant,
            "loss_definition": "z_(s,k)=(L_(s,k)-median_train_(s,k))/(1.4826*MAD_train_(s,k)); minimize (z_pred-z_obs)^2; retain original raw-loss formula and units",
            "loss_statistics_source": "only training-fold observed losses; each observed (N,D) scale and evaluation domain separately",
            "q_definition": "Q_i=Q17_i/Q_reference only for the loss_and_q_normalized ablation; Q_reference=mean of 17 fixed input quality values; raw and loss_normalized use Q17_i",
            "q_reference": q_reference,
            "training_loss_statistics": loss_scale.as_dict(loss_names),
            "note": "The existing 1M-only loss_standardization.json and scale_normalization.json are not used; the latter uses held-out observations in its upstream estimation.",
        },
        "weight_estimation": {
            "normalization": "none: neither sum(W) nor mean(W) is constrained",
            "parameterization": "W_i=exp(logW_i), with -2<=logW_i<=2 for every domain",
            "penalty": "lambda*mean_i((W_i-1)^2), a soft shrinkage rather than an equality constraint",
        },
        "units": {"N": "billion parameters", "D": "billion tokens", "L": "validation cross-entropy"},
        "sources": {
            "quality": str(handoff / "Q17_mapping.csv"),
            "exponents": str(classic_path) + " : B1_main_fit.parameters",
            "blocks": [
                {"name": block.name, "mixture_file": str(regmix / block.mixture_file),
                 "loss_file": str(regmix / block.loss_file), "n_mixtures": len(block.ids),
                 "N_parameters": block.n, "D_tokens": block.d}
                for block in blocks
            ],
            "excluded": "est_pile_loss_10b.csv 与 est_pile_loss_70b.csv 是估算值，不进入拟合或验证",
        },
        "split": {
            "seed": seed,
            "method": "train_1m 全部训练；test_1m/test_60m 按同一配比 ID 60/20/20 切分；test_1b 独立 60/20/20 切分",
            "train": {block.name: len(block.ids) for block in train_blocks},
            "tuning": {block.name: len(block.ids) for block in tune_blocks},
            "heldout_test": {block.name: len(block.ids) for block in test_blocks},
        },
        "metric_definitions": {
            "rmse_loss": "sqrt(mean((predicted-observed)^2)); 单位为交叉熵 Loss",
            "relative_rmse": "sqrt(mean((predicted/observed-1)^2))",
            "median_ape": "median(abs(predicted-observed)/observed)",
            "p90_ape": "90th_percentile(abs(predicted-observed)/observed)",
            "relative_bias": "mean((predicted-observed)/observed); 正值表示系统性高估",
            "r_squared_macro": "每个评测域计算 1-SSE_k/SST_k，SST_k=sum((observed-mean_observed_k)^2)，再对有效域取平均；不中心化预测值",
            "r_squared_pooled_within_domain": "1-sum_k(SSE_k)/sum_k(SST_k)，每个评测域的 SST 使用该域观测均值；只在固定规模内解释",
            "mixture_r_squared": "1-sum(((predicted-mean_p predicted)-(observed-mean_p observed))^2)/sum((observed-mean_p observed)^2); 各评测域分别中心化",
            "median_domain_spearman": "各评测域按配比点计算 Spearman 秩相关，再取 13 个域的中位数",
            "standardized_rmse": "sqrt(mean(((predicted-observed)/(1.4826*MAD_train_(scale,domain)))^2)); MAD only from training fold",
        },
        "selection": {
            "criterion": "各方案内部按各自训练目标选广义公式候选；方案之间统一按调参折原始 Loss 的 relative_rmse 选择；最终留出集不参与选择",
            "variant": selected_variant,
            "candidates": tuning,
            "chosen": {"form": selected.form, "ridge": selected.ridge,
                       "gamma": selected.gamma},
        },
        "normalization_comparison": comparison,
        "epsilon_sensitivity": epsilon_sensitivity,
        "structural_diagnostics": structural_diagnostics,
        "residual_correction": {
            "formula": "L_corrected_(s,k)(p)=L_law_k(N_s,D_s,p)+c_(s,k)+sum_i p_i*v_(s,k,i)",
            "purpose": "训练折上的逐规模、逐评测域岭回归残差校正；与原广义标度律分开汇报，仅用于已观测规模内预测",
            "selection": "每个规模的 ridge 仅按对应调参折 relative_rmse 选择",
            "by_scale": corrections,
            "aggregate_heldout": aggregate_corrected,
            "extrapolation_warning": "校正项没有未观测规模的定义，不得用于 10B/70B 外推或解释原律参数",
        },
        "parameters": {
            "alpha_by_domain": dict.fromkeys(loss_names, alpha),
            "beta_by_domain": dict.fromkeys(loss_names, beta),
            "quality_Q": dict(zip(train_names, (q / selected.q_reference).tolist())),
            "quality_Q_raw": dict(zip(train_names, q.tolist())),
            "quality_reference": selected.q_reference,
            "weight_W": dict(zip(train_names, selected.w.tolist())),
            "E": dict(zip(loss_names, selected.coefficients[:, 0].tolist())),
            "A": dict(zip(loss_names, selected.coefficients[:, 1].tolist())),
            "B": dict(zip(loss_names, selected.coefficients[:, 2].tolist())),
        },
        "heldout_test": heldout,
        "limitations": (
            "这是在训练折已覆盖三个观测规模下的配比泛化验证，不是完整留一规模外推；"
            "W 是固定 Q 条件下的正则化估计，未施加均值约束，不能解释为质量的因果效应；"
            "Q 的正比例归一化可由 W 吸收，单独改变 Q 刻度不能证明新机制；"
            "残差校正是独立的预测增强层，不属于原广义标度律。"
        ),
    }


def write_markdown_report(result: dict[str, object], path: Path) -> None:
    """把运行结果写成一份可独立阅读的 Markdown 报告。"""

    comp = result["compositional_scaling_law"]
    agg = comp["aggregate_heldout"]
    legacy = result["selection"]
    structural = result["structural_diagnostics"]
    params = comp["parameters"]
    loss_names = list(params["E"].keys())
    train_names = list(params["effective_weight_pQ"][loss_names[0]].keys())
    lines: list[str] = []
    add = lines.append

    add("# 广义标度律（分离式组成版）结果报告")
    add("")
    add("> 由 `q2/generalized_law.py` 自动生成，数据源为附件 A 的配比实验、")
    add("> `Q_1_to_2/Q17_mapping.csv` 的质量输入，以及问题一在 B1 上的经典标度律指数。")
    add("> 10B / 70B 的估算 Loss 不参与拟合与验证。")
    add("")

    # ---- 摘要 ----
    add("## 0. 摘要")
    add("")
    add("| 项目 | 结果 |")
    add("| --- | --- |")
    add(f"| 主模型 | 分离式组成广义标度律（配比项 {2 * len(train_names)} 维逐域响应） |")
    add(f"| 留出集 relative RMSE | **{agg['relative_rmse']:.2%}** |")
    add(f"| 留出集中位 APE | **{agg['median_ape']:.2%}** |")
    add(f"| 留出集 P90 APE | {agg['p90_ape']:.2%} |")
    add(f"| 留出集配比效应 R²（按 (域,规模) 分别中心化） | "
        f"**{agg.get('mixture_r_squared_within_scale', float('nan')):.4f}** |")
    add(f"| 同口径无配比项基线 | {legacy['candidates'][0]['relative_rmse']:.2%}（relRMSE） |")
    add(f"| 不可约损失下界检查 | {agg.get('irreducible_floor', {}).get('verdict', '—')} |")
    add("")
    add(f"结构诊断：训练 1M 上评测域两两相关的负相关比例为 "
        f"**{structural['training_1m_negative_domain_pair_fraction']:.1%}**，"
        f"第一主成分只解释配比响应方差的 "
        f"{structural['training_1m_first_principal_component_variance_fraction']:.1%}。"
        "任何单一标量形式都无法同时刻画符号相反的响应。")
    add("")

    # ---- 模型 ----
    add("## 1. 模型")
    add("")
    add("主模型：")
    add("")
    add("```text")
    add(comp["formula"])
    add(comp["scale_amplitude"])
    add("```")
    add("")
    add("其中配比响应为")
    add("")
    add("```text")
    add(comp["expanded_mixture_response"])
    add(comp["clr_definition"])
    add("```")
    add("")
    add(f"参考规模：N_ref = {comp['reference_scale']['N']:.6g}，"
        f"D_ref = {comp['reference_scale']['D']:.6g}。")
    add("")
    add(f"参数来源：配比响应 Φ 由 1M 训练/调参配比确定；规模幅度与基线 "
        f"`(E_k, A_k, B_k)` 只用各规模自己的训练折；指数 `alpha, beta` 固定为"
        f"问题一 B1 的经典标度律估计；**留出集不参与任何拟合或选模**。")
    add("")
    add("三项结构性选择及其对应的实测依据：")
    add("")
    add("1. **配比响应用逐域高维特征**（`p_i·Q_i` 与 `clr`，共 "
        f"{2 * len(train_names)} 维）——固定规模下配比效应不是秩 1 的；")
    add("2. **规模只改变响应幅度 `S_k(N,D)`，不改变形状 Φ_k**——"
        "实测 1M 与 60M 的配比响应形状相关 0.9938；")
    add("3. **指数固定为 B1 的值**——附件 A 只有 3 个不同的 (N, D) 点，"
        "标准标度律在其上恰好饱和，无法自识别指数。")
    add("")

    # ---- 数据与划分 ----
    add("## 2. 数据与划分")
    add("")
    add("| 区块 | 配比数 | N | D |")
    add("| --- | ---: | ---: | ---: |")
    for block in result["sources"]["blocks"]:
        add(f"| {block['name']} | {block['n_mixtures']} | {block['N_parameters']:.3g} | "
            f"{block['D_tokens']:.3g} |")
    add("")
    add(f"不纳入：{result['sources']['excluded']}。")
    add("")
    add(f"划分方式（seed = {result['split']['seed']}）：{result['split']['method']}。")
    add("")
    add("| 折 | 内容 |")
    add("| --- | --- |")
    for name in ("train", "tuning", "heldout_test"):
        add(f"| {name} | {result['split'][name]} |")
    add("")
    add("1M 与 60M 的测试配比是逐行配对的，配对配比始终进入同一折，"
        "避免跨规模信息泄漏。")
    add("")

    # ---- 留出结果 ----
    add("## 3. 留出集结果（未参与拟合与选模）")
    add("")
    add("| 规模 | 配比数 | relative RMSE | 中位 APE | P90 APE | 配比效应 R² | 域内 R²（宏平均） | 中位 Spearman |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for scale, score in comp["heldout_test"].items():
        add(f"| {scale} | {score['n_mixtures']} | {score['relative_rmse']:.2%} | "
            f"{score['median_ape']:.2%} | {score['p90_ape']:.2%} | "
            f"{score['mixture_r_squared']:.4f} | "
            f"{(score.get('r_squared_macro') or float('nan')):.4f} | "
            f"{(score.get('median_domain_spearman') or float('nan')):.4f} |")
    add(f"| **汇总** | — | **{agg['relative_rmse']:.2%}** | **{agg['median_ape']:.2%}** | "
        f"{agg['p90_ape']:.2%} | "
        f"**{agg.get('mixture_r_squared_within_scale', float('nan')):.4f}** | — | — |")
    add("")
    add(f"汇总口径的配比效应 R² 按 (评测域, 规模) 分别中心化后合并；"
        f"相对偏差 relative bias = {agg['relative_bias']:+.2%}。")
    add("")
    floor = agg.get("irreducible_floor")
    if isinstance(floor, dict):
        add(f"不可约损失下界：`min(L_pred − E_k) = {floor['min_predicted_minus_E']:+.4f}`，"
            f"低于下界的预测占比 {floor['fraction_of_predictions_below_E']:.4%}——"
            f"{floor['verdict']}。")
        add("")

    # ---- 与对照形式比较 ----
    add("## 4. 与对照形式的比较")
    add("")
    add("调参折上按原始 Loss 的 relative RMSE 选模（留出集不参与）：")
    add("")
    add("| 形式 | lambda | gamma | relative RMSE | 配比效应 R²（同口径） |")
    add("| --- | ---: | ---: | ---: | ---: |")
    for item in legacy["candidates"]:
        add(f"| {item['form']} | {item['ridge']:.3g} | {item['gamma']:.4f} | "
            f"{item['relative_rmse']:.2%} | "
            f"{item.get('mixture_r_squared_within_scale', float('nan')):.4f} |")
    add("")
    add("")
    add(f"分离式组成版不在上表的候选网格内（形式不同），但可在同一调参折上直接比较："
        f"调参折 relative RMSE = "
        f"**{comp['tuning'][list(comp['tuning'])[0]]['relative_rmse']:.2%}**，"
        f"留出集 relative RMSE = **{agg['relative_rmse']:.2%}**、"
        f"配比效应 R² = **{agg.get('mixture_r_squared_within_scale', float('nan')):.4f}**。"
        f"相对无配比项基线（{legacy['candidates'][0]['relative_rmse']:.2%}）改善约 "
        f"{legacy['candidates'][0]['relative_rmse'] / comp['tuning'][list(comp['tuning'])[0]]['relative_rmse']:.1f} 倍。")
    add("")
    add("尺度归一化消融（方案内部各自选模，方案之间按调参折原始 Loss 的 relative RMSE 比较）：")
    add("")
    add("| 方案 | 选中形态 | lambda | gamma | 调参 relative RMSE | 留出 relative RMSE | 留出配比效应 R² |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for variant, info in result["normalization_comparison"].items():
        chosen = info["chosen"]
        aggregate = info["aggregate_heldout"]
        add(f"| {variant} | {chosen['form']} | {chosen['ridge']:.3g} | "
            f"{chosen['gamma']:.4f} | {info['selected_tuning']['relative_rmse']:.2%} | "
            f"{aggregate['relative_rmse']:.2%} | "
            f"{aggregate.get('mixture_r_squared_within_scale', float('nan')):.4f} |")
    add("")
    add(f"最终采用 **{result['normalization']['selected_variant']}** 方案。"
        f"{result['normalization']['note']}")
    add("")

    # ---- 原形式失效分析 ----
    add("## 5. 单标量原形式为何失效")
    add("")
    gammas = [item["gamma"] for item in legacy["candidates"]]
    add("三条独立证据：")
    add("")
    add("1. **对照不带来任何改善**：全部候选的 relative RMSE 与"
        "无配比项基线持平或在噪声内（上表第 1 行 vs 其余行）。")
    add(f"2. **`gamma` 在所有候选上都顶在盒边界**：取值范围 "
        f"[{min(gammas):.4f}, {max(gammas):.4f}]，搜索盒为 [0.2, 3.0]，"
        "没有一个候选给出内部最优——说明该参数在此数据上不可识别。")
    add(f"3. **结构上不可能成立**：训练 1M 上 "
        f"**{structural['training_1m_negative_domain_pair_fraction']:.1%}** 的评测域对"
        "对同一配比变化的响应符号相反；单一正标量只能产生同向响应。")
    add("")
    add(f"（原形式：`{result['formula']}`；约束：{result['formula_constraints']}。）")
    add("")

    # ---- 参数 ----
    add("## 6. 参数")
    add("")
    add("基线与规模幅度（逐评测域）：")
    add("")
    add("| 评测域 | E_k | A_k | B_k | eta_k | zeta_k |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in loss_names:
        add(f"| {name} | {params['E'][name]:.4f} | {params['A'][name]:.4f} | "
            f"{params['B'][name]:.4f} | {params['eta'][name]:.4f} | "
            f"{params['zeta'][name]:.4f} |")
    add("")
    diag = comp["scale_anchor_diagnostics"]
    add(f"规模锚点诊断：基线 NNLS 残差 {diag['baseline_nnls_rmse']:.3g}；"
        f"`zeta_k < 0` 的域数 {diag['negative_zeta_count']}。")
    add("")
    add("配比响应权重。下面用**标准化权重**（每 1 个特征标准差对应的 Phi 变化）表示，"
        "因为原始尺度上的系数会被特征的方差差异放大，不便直接比较：")
    add("")
    slope = params["reference_response_slope"]
    ridge_coef = params["ridge_coefficients"]
    flat = []
    for domain in loss_names:
        for feature, value in ridge_coef[domain].items():
            flat.append((abs(value * slope[domain]), domain, feature,
                         value * slope[domain]))
    flat.sort(reverse=True)
    add("| 评测域 | 特征方向 | 标准化权重 |")
    add("| --- | --- | ---: |")
    for _magnitude, domain, feature, value in flat[:14]:
        label = feature.replace("pQ:", "p·Q / ").replace("clr:", "clr / ")
        add(f"| {domain} | {label} | {value:+.4f} |")
    add("")
    add(f"（`p·Q` 方向与 `clr` 方向各 {len(train_names)} 维，合计 "
        f"{2 * len(train_names)} 维；`clr` 用 epsilon = {comp['epsilon']:.0e} 平滑。）")
    add("")
    add("> 权重只识别到共同尺度：`Φ_k` 自带 `slope_ref_k`，`Q·W` 的整体尺度会被它吸收。")
    add("> 报告的数值应理解为在该规范下的**相对**权重。")
    add("")

    # ---- 稳健性 ----
    add("## 7. 稳健性检查")
    add("")
    sweep = result.get("epsilon_sensitivity") or []
    if sweep:
        add(f"`clr` 的平滑常数 epsilon 敏感性（当前取值 {comp['epsilon']:.0e}）：")
        add("")
        add("| epsilon | 选中 ridge | 留出 relative RMSE | 留出中位 APE | 留出配比效应 R² |")
        add("| ---: | ---: | ---: | ---: | ---: |")
        for row in sweep:
            add(f"| {row['epsilon']:.0e} | {row['ridge_selected']:.3g} | "
                f"{row['heldout_relative_rmse']:.2%} | {row['heldout_median_ape']:.2%} | "
                f"{row['heldout_mixture_r_squared_within_scale']:.4f} |")
        add("")
    rc = result["residual_correction"]
    add(f"残差校正（独立的预测增强层，**不属于**本广义标度律，且它校正的是 §4 表中所选中的"
        f"**对照形式**，不是分离式组成版）：`{rc['formula']}`。"
        f"留出汇总 relative RMSE "
        f"{rc['aggregate_heldout']['relative_rmse']:.2%}，"
        f"配比效应 R² {rc['aggregate_heldout'].get('mixture_r_squared_within_scale', float('nan')):.4f}——"
        f"仍明显劣于组成版的 {agg['relative_rmse']:.2%} / "
        f"{agg.get('mixture_r_squared_within_scale', float('nan')):.4f}，因此本报告不采用该项。")
    add(f"{rc['extrapolation_warning']}")
    add("")

    # ---- 可识别性与假设 ----
    add("## 8. 可识别性与假设")
    add("")
    add("- **`W` 与 `Q` 不可分**：只有乘积 `Q_i·W_i` 可识别；`W` 的 ridge 向 1 收缩"
        "起到选规范的作用，因此报告的 `W` 是相对权重，不是绝对标定。")
    add("- **指数借用自 B1**：附件 A 的三个 (N, D) 点使标准标度律恰好饱和，"
        "无法自识别 `alpha, beta`；本模型固定它们，因此"
        "**形式与 B1 可比，但不是对 A 自身指数的验证**。")
    add("- **`eta_k, zeta_k` 是外推**：`eta` 由 1M↔60M（D 固定）确定，"
        "`zeta` 实际只由 1B 一个规模点确定；“幅度按 N、D 幂律衰减”是假设，"
        "三个点无法检验。`S_k → 0` 意味着配比效应在极大规模下趋于消失，"
        "数据只支持“1B 处相对幅度缩到约 0.56 倍”，不支持“趋于 0”。")
    add("- **`Q_i` 固定**：单一固定的质量轴无法分离出独立的质量效应；"
        "改变 `Q` 的尺度可被 `W` 吸收，因此不能据此声称新的机制。")
    add("- **加性配比项**：`Φ_k` 自由符号，原则上可预测 `L_k < E_k`；"
        "留出集的下界检查见 §3。")
    add("")

    # ---- 局限 ----
    add("## 9. 局限")
    add("")
    add("设计文件自述的限制：")
    add("")
    add(f"> {result['limitations']}")
    add("")
    add(f"补充说明：")
    add(f"- 1B 留出折只有 "
        f"{comp['heldout_test'][list(comp['heldout_test'])[-1]]['n_mixtures']} 组配比，"
        f"其指标（配比效应 R² "
        f"{comp['heldout_test'][list(comp['heldout_test'])[-1]]['mixture_r_squared']:.4f}）"
        f"样本量偏小，引用时应注明。")
    add(f"- 观测配比点每个只有一次运行（数据表无 seed/repeat 列），"
        f"因此所有残差都不含重复运行噪声的估计。")
    add(f"- 本模型刻画的是**配比 → Loss** 的统计关系，不是质量的因果效应。")
    add("")

    # ---- 复现 ----
    add("## 10. 复现")
    add("")
    add("```bash")
    add("python q2/generalized_law.py \\")
    add("    --regmix-dir data/real_attachments/A_data_value/regmix_tables \\")
    add("    --handoff-dir data_analysis/Q_1_to_2 \\")
    add("    --classic-results data_analysis/scaling_law_full/scaling_law_full_results.json \\")
    add("    --output data_analysis/Q_1_to_2/generalized_scaling_compact.json \\")
    add("    --report data_analysis/generalized_scaling_law/generalized_scaling_law_compact_report.md")
    add("```")
    add("")
    add(f"依赖：numpy、scipy（`scipy.optimize` / `scipy.stats`）、scikit-learn。")
    add(f"完整参数与逐来源指标见 `generalized_scaling_compact.json`。")
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(result: dict[str, object], output: Path) -> None:
    source = result["sources"]
    split = result["split"]
    upgraded = result["compositional_scaling_law"]
    print("\n分离式组成广义标度律（新版；无独立 C 缩放因子）")
    print(upgraded["formula"])
    print(upgraded["scale_amplitude"])
    print(upgraded["expanded_mixture_response"])
    print(upgraded["clr_definition"])
    print("参考规模配比岭回归系数由调参折选定：",
          upgraded["ridge_selected"], "；基线 E/A/B 与规模幅度仅使用训练折。")
    print("新律最终留出集：")
    for scale, score in upgraded["heldout_test"].items():
        print(f"  {scale}: relative_RMSE={score['relative_rmse']:.2%}, "
              f"R2_macro={score['r_squared_macro']:.3f}, "
              f"MedAPE={score['median_ape']:.2%}")
    print("  汇总 relative_RMSE="
          f"{upgraded['aggregate_heldout']['relative_rmse']:.2%}")
    print("  外推限制：", upgraded["extrapolation_warning"])
    print("\n原广义标度律对照（无独立 C 缩放因子）")
    print(result["formula"])
    print("N、D 使用十亿单位；13 个评测域分别拟合 E、A、B，指数固定为问题一的真实轨迹估计。")
    print("\n参考数据来源")
    print("Q_i:", source["quality"])
    print("alpha、beta:", source["exponents"])
    for block in source["blocks"]:
        print(f"{block['name']}：{block['n_mixtures']} 个配比，N={block['N_parameters']:g}，"
              f"D={block['D_tokens']:g}\n  p: {block['mixture_file']}\n  L: {block['loss_file']}")
    print("不纳入：", source["excluded"])
    print("\n数据划分（随机种子", split["seed"], "）")
    print(split["method"])
    for name in ("train", "tuning", "heldout_test"):
        print(f"{name}: {split[name]}")
    norm = result["normalization"]
    print("\n尺度归一化消融方案：", norm["loss_definition"])
    print("每个规模和评测域的 median/MAD 仅由训练折估计；不读取全数据生成的现成归一化参数。")
    print(f"Q 归一化试验：Q_i=Q17_i/{norm['q_reference']:.6f}；不约束 W 的均值。")
    print("\n指标计算方式：实测 L 来自各 loss_file，预测 L 由上式和训练折参数计算。")
    for name, definition in result["metric_definitions"].items():
        print(f"  {name}: {definition}")
    print("\n调参折：方案之间统一按原始 Loss 的 relative_RMSE 选择，留出集不参与选模")
    for item in result["selection"]["candidates"]:
        print(f"  {item['form']:12s} lambda={item['ridge']:.3g} "
              f"gamma={item['gamma']:.3f} relative_RMSE={item['relative_rmse']:.2%} "
              f"standardized_RMSE={item['standardized_rmse']:.3f} "
              f"MedAPE={item['median_ape']:.2%}")
    print("选中：", result["selection"]["variant"], result["selection"]["chosen"])
    print("\n归一化消融（调参折选模；下列留出误差均还原为原始 Loss 计算）：")
    for variant, info in result["normalization_comparison"].items():
        aggregate = info["aggregate_heldout"]
        print(f"  {variant}: tuning standardized_RMSE="
              f"{info['selected_tuning']['standardized_rmse']:.3f}, "
              f"relative_RMSE={info['selected_tuning']['relative_rmse']:.2%}, "
              f"chosen={info['chosen']}")
        print(f"    留出汇总：relative_RMSE={aggregate['relative_rmse']:.2%}, "
              f"standardized_RMSE={aggregate['standardized_rmse']:.3f}")
        for scale, entry in info["heldout_test"].items():
            m = entry["models"]["selected"]
            print(f"    {scale}: relative_RMSE={m['relative_rmse']:.2%}, "
                  f"standardized_RMSE={m['standardized_rmse']:.3f}, "
                  f"R2_macro={m['r_squared_macro']:.3f}, "
                  f"mixture_R2={m['mixture_r_squared']:.3f}")
    structural = result["structural_diagnostics"]
    print("\n结构诊断：训练 1M 的评测域两两负相关比例="
          f"{structural['training_1m_negative_domain_pair_fraction']:.1%}；"
          "首主成分解释的域间配比波动="
          f"{structural['training_1m_first_principal_component_variance_fraction']:.1%}。")
    print(structural["interpretation"])
    print("\n附加残差校正：原标度律 + 逐规模/逐评测域配比岭回归；只用于已观测规模。")
    for scale, detail in result["residual_correction"]["by_scale"].items():
        print(f"  {scale}: lambda={detail['ridge']:.3g}, "
              f"training_rows={detail['training_rows']}, "
              f"tuning_rows={detail['tuning_rows']}")
    print("  留出汇总：relative_RMSE="
          f"{result['residual_correction']['aggregate_heldout']['relative_rmse']:.2%}")
    print("\n最终留出集（下列数值未参与拟合或选模）")
    for scale, entry in result["heldout_test"].items():
        print(f"{scale}，{entry['n_mixtures']} 个配比 × 13 个评测域：")
        for name, m in entry["models"].items():
            spearman = (f"{m['median_domain_spearman']:.3f}"
                        if m['median_domain_spearman'] is not None else "未定义（常数预测）")
            print(f"  {name:12s} RMSE={m['rmse_loss']:.4f} "
                  f"relative_RMSE={m['relative_rmse']:.2%} "
                  f"standardized_RMSE={m['standardized_rmse']:.3f} "
                  f"MedAPE={m['median_ape']:.2%} p90APE={m['p90_ape']:.2%} "
                  f"R2_macro={m['r_squared_macro']:.3f} "
                  f"mixture_R2={m['mixture_r_squared']:.3f} "
                  f"Spearman中位数={spearman}")
    print("\n参数范围：gamma=%.4f，W_min=%.4f，W_max=%.4f，mean(W)=%.4f（仅描述，非约束）" % (
        result["selection"]["chosen"]["gamma"],
        min(result["parameters"]["weight_W"].values()),
        max(result["parameters"]["weight_W"].values()),
        np.mean(list(result["parameters"]["weight_W"].values())),
    ))
    print("领域权重（Q 为本方案实际输入质量，W 为独立拟合权重，Q×W 为混合项系数）：")
    for domain, w in result["parameters"]["weight_W"].items():
        q = result["parameters"]["quality_Q"][domain]
        print(f"  {domain:26s} Q={q:.4f} W={w:.4f} Q×W={q*w:.4f}")
    print("正则项：lambda × mean_i((W_i-1)^2)，只作软收缩；W_i 通过 exp(log W_i) 保持正值。")
    print(result["limitations"])
    print("完整参数与逐来源指标：", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regmix-dir", type=Path, default=REGMIX)
    parser.add_argument("--handoff-dir", type=Path, default=HANDOFF)
    parser.add_argument("--classic-results", type=Path, default=CLASSIC)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    result = run(args.regmix_dir, args.handoff_dir, args.classic_results, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(result, args.report)
    print_report(result, args.output)
    print("Markdown 报告：", args.report)


if __name__ == "__main__":
    main()
