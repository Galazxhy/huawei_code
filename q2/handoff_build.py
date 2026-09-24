#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 `data_analysis/Q2_to_Q3/`：把第二问的广义标度律与相关数据交付给第三问。

产出
----
* `交付文档.md`                     —— 交付说明（人读）
* `Q2_to_Q3_inputs.json`            —— 接口入口（机读）
* `generalized_scaling_law.json`     —— 广义标度律的形式、参数、适用区间
* `generalized_scaling_law_parameters.csv` —— 逐评测域的 E/A/B/eta/zeta
* `mixture_response.json`            —— 配比响应 Phi_k(p,Q) 的完整可复算参数
* `quality_mixture_inputs.csv`       —— 17 个训练域的 Q17 与配比使用情况
* `context_length_feasibility.csv`   —— 由 C7 解析出的 L 可行取值与临界值
* `quality_cost_functions.json`      —— 附录 B 的三类 g(Q) 及由此得到的质量成本
* `elasticity_and_substitution.json` —— 弹性、替代条件、域间替代/互补
* `generalized_law_evaluator.py`     —— 可直接 import 的 L(N,D,p,Q) 求值器

用法
----
    python q2/handoff_build.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPACT = ROOT / "data_analysis/Q_1_to_2/generalized_scaling_compact.json"
Q17_MAPPING = ROOT / "data_analysis/Q_1_to_2/Q17_mapping.csv"
TRAIN_DOMAIN_NAMES = ROOT / "data_analysis/Q_1_to_2/train_domain_names.csv"
LOSS_DOMAIN_NAMES = ROOT / "data_analysis/Q_1_to_2/loss_domain_names.csv"
C7 = ROOT / "data/real_attachments/C_efficiency_evolution/model_architecture_metadata.csv"
B6 = ROOT / "data/real_attachments/B_scaling_laws/supplementary_NQ_experiment.csv"
B7 = ROOT / "data/real_attachments/B_scaling_laws/supplementary_NQ_experiment_expanded.csv"
OUT = ROOT / "data_analysis/Q2_to_Q3"

#: 题面给的长文本注意力系数：C_attn = eta * N * D * L_ctx
ATTENTION_ETA = 2e-4
#: 题面给的三档预算（FLOPs）
BUDGETS = (1e19, 1e22, 1e24)
#: 附录 B 的质量成本函数（C_Q = D * [g(Q) - g(Q0)]_+）
QUALITY_COST_FORMS = {
    "exponential": {"g": "gamma * exp(lambda * Q)", "gamma": 1e7, "lambda": 6.0},
    "power": {"g": "gamma * Q ** lambda", "gamma": 5e9, "lambda": 4.0},
    "log_asymptotic": {"g": "gamma * log(1 + lambda * Q)", "gamma": 2e9, "lambda": 10.0},
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_names(path: Path, column: str) -> list[str]:
    return [row[column] for row in read_csv_rows(path)]


# --------------------------------------------------------------------------- #
# 1. 广义标度律求值器（与 compact 模型逐位一致）
# --------------------------------------------------------------------------- #
class GeneralizedLaw:
    """复现 `generalized_law.py` 的分离式组成广义标度律。"""

    def __init__(self, payload: dict) -> None:
        comp = payload["compositional_scaling_law"]
        self.raw = comp
        self.loss_domains = list(comp["parameters"]["E"].keys())
        params = comp["parameters"]
        self.alpha = float(params["alpha"])
        self.beta = float(params["beta"])
        self.epsilon = float(comp["epsilon"])
        self.feature_names = list(params["feature_names"])
        self.feature_mean = np.asarray(params["feature_mean"], dtype=float)
        self.feature_scale = np.asarray(params["feature_scale"], dtype=float)
        self.reference_slope = np.asarray(
            [params["reference_response_slope"][d] for d in self.loss_domains]
        )
        self.response_center = np.asarray(
            [params["reference_response_center"][d] for d in self.loss_domains]
        )
        self.ridge_intercept = np.asarray(
            [params["ridge_intercept"][d] for d in self.loss_domains]
        )
        self.ridge_coefficients = np.asarray(
            [
                [params["ridge_coefficients"][d][f] for f in self.feature_names]
                for d in self.loss_domains
            ]
        )  # (13, 34)
        self.e_coefficients = np.asarray(
            [params["E"][d] for d in self.loss_domains]
        )
        self.a_coefficients = np.asarray(
            [params["A"][d] for d in self.loss_domains]
        )
        self.b_coefficients = np.asarray(
            [params["B"][d] for d in self.loss_domains]
        )
        self.eta = np.asarray([params["eta"][d] for d in self.loss_domains])
        self.zeta = np.asarray([params["zeta"][d] for d in self.loss_domains])
        self.reference_n = float(comp["reference_scale"]["N"]) / 1e9
        self.reference_d = float(comp["reference_scale"]["D"]) / 1e9
        # 训练域顺序：feature_names 的前半段是 "pQ:<train domain>"
        self.train_domains = [
            name.split(":", 1)[1]
            for name in self.feature_names
            if name.startswith("pQ:")
        ]

    # -- 配比响应 ---------------------------------------------------------- #
    def _features(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        log_p = np.log(p + self.epsilon)
        clr = log_p - log_p.mean(axis=1, keepdims=True)
        return np.column_stack((p * q[None, :], clr))

    def response(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """``Phi_k(p, Q)``，(n, 17) 配比 × (17,) 质量 -> (n, 13)。"""

        standardized = (self._features(p, q) - self.feature_mean) / self.feature_scale
        raw = self.ridge_intercept[None, :] + standardized @ self.ridge_coefficients.T
        return (raw - self.response_center[None, :]) * self.reference_slope[None, :]

    def amplitude(self, n_params: float, d_tokens: float) -> np.ndarray:
        """``S_k(N, D)``；``n_params`` 与 ``d_tokens`` 以十亿为单位。"""

        return (n_params / self.reference_n) ** (-self.eta) * (
            d_tokens / self.reference_d
        ) ** (-self.zeta)

    def predict(
        self,
        n_params: float,
        d_tokens: float,
        p: np.ndarray,
        q: np.ndarray,
        quality_scale: float = 1.0,
    ) -> np.ndarray:
        """返回 13 个评测域的预测 Loss。

        ``n_params`` / ``d_tokens`` 以十亿为单位；``p`` 为 (n, 17) 且每行和为 1；
        ``quality_scale`` 把所有 ``Q_i`` 同比例缩放（t 倍），用于 Q3 的质量情景。
        """

        p = np.atleast_2d(np.asarray(p, dtype=float))
        q = np.asarray(q, dtype=float) * float(quality_scale)
        base = (
            self.e_coefficients[None, :]
            + self.a_coefficients[None, :] * n_params ** (-self.alpha)
            + self.b_coefficients[None, :] * d_tokens ** (-self.beta)
        )
        amplitude = self.amplitude(n_params, d_tokens)[None, :]
        return base + amplitude * self.response(p, q)


# --------------------------------------------------------------------------- #
# 2. B6/B7：题面附录 A 要求的质量相关数据
# --------------------------------------------------------------------------- #
def fit_quality_law(path: Path) -> dict[str, float]:
    """在 B6/B7 上拟合 ``L = E + A*N^-alpha + B*(D*Q)^-beta``（网格 + 内层闭式解）。"""

    rows = read_csv_rows(path)
    n = np.array([float(r["N_params_B"]) for r in rows])
    d = np.array([float(r["D_tokens_B"]) for r in rows])
    q = np.array([float(r["Q_score"]) for r in rows])
    loss = np.array([float(r["val_loss"]) for r in rows])
    best = None
    for alpha in np.linspace(0.05, 0.80, 301):
        for beta in np.linspace(0.02, 0.60, 291):
            design = np.column_stack(
                [np.ones_like(n), n ** -alpha, (d * q) ** -beta]
            )
            coef, *_ = np.linalg.lstsq(design, loss, rcond=None)
            rss = float(np.square(loss - design @ coef).sum())
            if best is None or rss < best[0]:
                best = (rss, alpha, beta, coef)
    assert best is not None
    rss, alpha, beta, coef = best
    return {
        "E": float(coef[0]),
        "A": float(coef[1]),
        "alpha": float(alpha),
        "B": float(coef[2]),
        "beta": float(beta),
        "rmse": float(np.sqrt(rss / loss.size)),
        "r_squared": float(1 - rss / np.square(loss - loss.mean()).sum()),
        "sample_count": int(loss.size),
        "N_unit": "billion parameters",
        "D_unit": "billion tokens",
    }


def quality_elasticity(fit: dict[str, float], n: float, d: float, q: float) -> dict:
    """``ln L = ln(E + A N^-alpha + B (DQ)^-beta)`` 对 ``ln Q`` 的弹性。"""

    term = fit["B"] * (d * q) ** (-fit["beta"])
    total = fit["E"] + fit["A"] * n ** (-fit["alpha"]) + term
    elasticity = -fit["beta"] * term / total
    return {
        "N_billion_params": n,
        "D_billion_tokens": d,
        "Q": q,
        "predicted_loss": float(total),
        "quality_share_of_loss": float(term / total),
        "elasticity_dlnL_dlnQ": float(elasticity),
        "d_loss_per_unit_Q": float(elasticity * total / q),
    }


# --------------------------------------------------------------------------- #
# 3. C7：上下文长度可行取值
# --------------------------------------------------------------------------- #
def context_length_grid() -> list[dict[str, object]]:
    rows = read_csv_rows(C7)
    lengths: dict[int, list[str]] = {}
    for row in rows:
        raw = (row.get("max_position_embeddings") or "").strip()
        if not raw or raw.upper() == "NA":
            continue
        lengths.setdefault(int(float(raw)), []).append(row["model_name"])
    critical = 6.0 / ATTENTION_ETA
    records = []
    for length in sorted(lengths):
        records.append(
            {
                "L_ctx": length,
                "model_count": len(lengths[length]),
                "example_models": lengths[length][:4],
                "attention_over_training_ratio": length * ATTENTION_ETA / 6.0,
                "attention_share_of_budget": length * ATTENTION_ETA
                / (6.0 + length * ATTENTION_ETA),
                "exceeds_critical": bool(length > critical),
                "note": (
                    "注意力开销超过基础训练开销"
                    if length > critical
                    else "基础训练开销仍占主导"
                ),
            }
        )
    return records, critical


# --------------------------------------------------------------------------- #
# 4. 质量成本函数
# --------------------------------------------------------------------------- #
def g_value(form: str, q: float) -> float:
    spec = QUALITY_COST_FORMS[form]
    if form == "exponential":
        return spec["gamma"] * math.exp(spec["lambda"] * q)
    if form == "power":
        return spec["gamma"] * q ** spec["lambda"]
    return spec["gamma"] * math.log(1.0 + spec["lambda"] * q)


def quality_cost_table(q0: float, grid: tuple[float, ...]) -> dict[str, object]:
    table = {}
    for form in QUALITY_COST_FORMS:
        base = g_value(form, q0)
        table[form] = {
            "expression": QUALITY_COST_FORMS[form]["g"],
            "parameters": {
                "gamma": QUALITY_COST_FORMS[form]["gamma"],
                "lambda": QUALITY_COST_FORMS[form]["lambda"],
            },
            "g_at_Q0": base,
            "per_token_cost": {
                f"Q={q:.2f}": max(g_value(form, q) - base, 0.0) for q in grid
            },
        }
    return table


# --------------------------------------------------------------------------- #
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads(COMPACT.read_text(encoding="utf-8"))
    law = GeneralizedLaw(payload)

    train_domains = law.train_domains
    loss_domains = law.loss_domains
    quality = np.asarray(
        [
            float(
                next(
                    row["Q17"]
                    for row in read_csv_rows(Q17_MAPPING)
                    if row["mixture_domain"] == name
                )
            )
            for name in train_domains
        ],
        dtype=float,
    )
    q_reference = float(quality.mean())

    # ---- 自检：复现 compact 报告的留出指标 --------------------------------- #
    blocks_rows = [
        # N、D 以十亿为单位：1e6 参数 = 1e-3 B，1e9 token = 1.0 B
        ("train_1m", 1e-3, 1.0, "train_mixture_1m.csv", "train_pile_loss_1m.csv"),
        ("test_1m", 1e-3, 1.0, "test_mixture_1m.csv", "test_pile_loss_1m.csv"),
        ("test_60m", 0.06, 1.0, "test_mixture_60m.csv", "test_pile_loss_60m.csv"),
        ("test_1b", 1.0, 25.0, "test_mixture_1B.csv", "test_pile_loss_1B.csv"),
    ]
    regmix = ROOT / "data/real_attachments/A_data_value/regmix_tables"
    verification = []
    recorded: list[np.ndarray] = []
    for label, n_b, d_b, p_file, l_file in blocks_rows:
        with (regmix / p_file).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            p = np.asarray([[float(v) for v in row[1:]] for row in reader if row])
        with (regmix / l_file).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            y = np.asarray([[float(v) for v in row[1:]] for row in reader if row])
        recorded.append(p)
        predicted = law.predict(n_b, d_b, p, quality)
        error = predicted - y
        verification.append(
            {
                "block": label,
                "n_mixtures": int(y.shape[0]),
                "relative_rmse": float(np.sqrt(np.mean(np.square(error / y)))),
                "median_ape": float(np.median(np.abs(error / y))),
            }
        )

    # 逐域拟合优度：在**单一规模内**（test_60m）按评测域计算 1 - SSE/SST，
    # 避免把规模间差异算成"逐域解释力"。
    with (regmix / "test_mixture_60m.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.reader(handle)
        next(reader)
        p_60m = np.asarray([[float(v) for v in row[1:]] for row in reader if row])
    with (regmix / "test_pile_loss_60m.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.reader(handle)
        next(reader)
        y_60m = np.asarray([[float(v) for v in row[1:]] for row in reader if row])
    predicted_60m = law.predict(0.06, 1.0, p_60m, quality)
    per_domain_r2 = [
        1.0
        - np.square(predicted_60m[:, k] - y_60m[:, k]).sum()
        / np.square(y_60m[:, k] - y_60m[:, k].mean()).sum()
        for k in range(len(loss_domains))
    ]

    # ---- 配比可行域 ------------------------------------------------------- #
    # 附件 A 只在有限个配比上测过 Loss；模型的配比响应由 34 维特征上的 Ridge 给出，
    # 在数据覆盖之外没有约束。等权配比 1/17 的熵是 ln 17 = 2.833，而记录到的配比
    # 最大熵只有 2.381，等权点落在**数据之外**：在该点上律给出的 Loss 比该规模实测
    # 均值低 24%~45%，因此不能作为参考点，更不能当作第三问的可行配比。
    recorded_matrix = np.vstack(recorded)
    entropy = -(recorded_matrix * np.log(np.clip(recorded_matrix, 1e-12, None))).sum(axis=1)
    entropy_order = np.argsort(entropy)
    support = {
        "definition": "H(p) = -sum_i p_i log p_i，nats；按 17 维配比直接计算，不涉及模型",
        "recorded_entropy_min": float(entropy.min()),
        "recorded_entropy_max": float(entropy.max()),
        "uniform_mixture_entropy": float(np.log(len(train_domains))),
        "recorded_max_share_per_domain": {
            name: float(recorded_matrix[:, j].max())
            for j, name in enumerate(train_domains)
        },
        "n_recorded_mixtures": int(recorded_matrix.shape[0]),
        "warning": (
            "等权配比 1/17 的熵为 ln 17 = 2.833，超出记录到的最大熵 "
            f"{entropy.max():.3f}；实测该点上律的预测比同规模均值低 24%~45%。"
            "第三问的配比变量必须落在记录到的可行域内，否则目标函数是外推值。"
        ),
    }

    def percentile_mixture(percentile: float) -> np.ndarray:
        """记录到的配比中，熵最接近给定分位数的那个（真实观测点，非构造点）。"""

        index = entropy_order[
            int(np.clip(round(percentile * (len(entropy_order) - 1)), 0, len(entropy_order) - 1))
        ]
        return recorded_matrix[index]

    # ---- 弹性与替代条件 --------------------------------------------------- #
    ref_n, ref_d = 1.0, 25.0  # 1B 参数 / 25B token，与 1B 观测锚点一致
    p_reference = percentile_mixture(0.5)[None, :]
    q_fixed = quality

    def aggregate(n_b: float, d_b: float, quality_scale: float = 1.0) -> float:
        return float(
            law.predict(n_b, d_b, p_reference, q_fixed, quality_scale).mean()
        )

    h = 1e-4

    def elasticities(n_b: float, d_b: float) -> dict[str, float]:
        """对数弹性；以相对扰动做中心差分，``d ln f / d ln x``。"""

        base = aggregate(n_b, d_b)
        eps_n_value = (
            aggregate(n_b * (1 + h), d_b) - aggregate(n_b * (1 - h), d_b)
        ) / (2 * h) / base
        eps_d_value = (
            aggregate(n_b, d_b * (1 + h)) - aggregate(n_b, d_b * (1 - h))
        ) / (2 * h) / base
        eps_q_value = (
            aggregate(n_b, d_b, 1 + h) - aggregate(n_b, d_b, 1 - h)
        ) / (2 * h) / base
        return {
            "N_billion_params": n_b,
            "D_billion_tokens": d_b,
            "predicted_mean_loss": float(base),
            "dlnL_dlnN": float(eps_n_value),
            "dlnL_dlnD": float(eps_d_value),
            "dlnL_dlnQ_scale": float(eps_q_value),
        }

    reference_points = [elasticities(1.0, 25.0), elasticities(0.06, 1.0),
                        elasticities(10.0, 100.0)]
    base_loss = reference_points[0]["predicted_mean_loss"]

    b7_fit = fit_quality_law(B7)
    b6_fit = fit_quality_law(B6)
    quality_points = [
        quality_elasticity(b7_fit, 1.0, 25.0, q) for q in (0.3, 0.5, 0.7, 0.9, 1.0)
    ]

    # 替代条件：令 dL 相等的 dN/dQ（用 B7 的律，因为它显式含 Q）
    substitution = []
    for q in (0.3, 0.5, 0.7, 0.9):
        term_d = b7_fit["B"] * b7_fit["beta"] * (25.0 * q) ** (-b7_fit["beta"]) / q
        term_n = b7_fit["A"] * b7_fit["alpha"] * 1.0 ** (-b7_fit["alpha"])
        # dL = -A*alpha*N^(-alpha-1) dN - B*beta*D^-beta*Q^(-beta-1) dQ = 0
        # => dN/dQ = term_d / term_n  (N 以十亿计)
        ratio = term_d / term_n
        substitution.append(
            {
                "Q": q,
                "dN_dQ_billion_params_per_unit_Q": float(ratio),
                "relative_substitution_dlnN_dlnQ": float(ratio * q / 1.0),
                "N_gain_for_quality_plus_0.1_percent": float(ratio * 0.1 / 1.0 * 100),
            }
        )

    # 域间替代/互补：配比响应矩阵的列相关
    design = None
    weights = law.raw["parameters"].get("effective_weight_pQ")
    if weights:
        design = np.asarray(
            [
                [weights[domain][trainer] for trainer in train_domains]
                for domain in loss_domains
            ],
            dtype=float,
        )

    reference_entropy = float(
        -(p_reference[0] * np.log(np.clip(p_reference[0], 1e-12, None))).sum()
    )
    sensitivity = []
    for percentile in (0.1, 0.5, 0.9):
        mixture = percentile_mixture(percentile)
        h_value = float(-(mixture * np.log(np.clip(mixture, 1e-12, None))).sum())
        saved = p_reference.copy()
        p_reference[:] = mixture
        point = elasticities(1.0, 25.0)
        p_reference[:] = saved
        sensitivity.append(
            {
                "entropy_percentile": percentile,
                "mixture_entropy": h_value,
                "mixture": dict(zip(train_domains, mixture.tolist())),
                **point,
            }
        )

    elasticity_payload = {
        "reference_point": {
            "mixture_entropy": reference_entropy,
            "mixture": dict(zip(train_domains, p_reference[0].tolist())),
            "source": (
                "附件 A 记录到的配比中，熵位于中位数的那个真实配比"
                f"（H = {reference_entropy:.3f} nats）；不是 1/17 等权配比"
                "（H = ln 17 超出数据可行域，见 mixture_support）"
            ),
            "quality_scale": 1.0,
            "note": "共 3 个参考点，见 elasticity_from_q2_law.points",
        },
        "mixture_support": support,
        "reference_mixture_sensitivity": {
            "at": "N=1B, D=25B",
            "points": sensitivity,
            "note": (
                "弹性随参考配比变化；三行分别是记录到的配比中熵处于 10/50/90 分位的"
                "真实配比。第三问若改变配比，应先确认仍在 mixture_support 内。"
            ),
        },
        "elasticity_from_q2_law": {
            "points": reference_points,
            "note": (
                "N、D 的弹性直接来自 Q2 的广义标度律。"
                "Q 的弹性是把全部 Q_i 同比例缩放后数值微分得到的，"
                "属于**外推**：附件 A 中 Q_i 固定，模型无法识别独立的质量效应，"
                "因此第三问应以 B6/B7 的质量弹性为准（见下条）。"
            ),
        },
        "quality_law_from_B6_B7": {
            "form": "L = E + A*N^(-alpha) + B*(D*Q)^(-beta)",
            "B7": b7_fit,
            "B6": b6_fit,
            "elasticity_at_reference_points": quality_points,
            "note": (
                "题面附录 A 要求质量相关部分使用 B6（或 B7、B8）。"
                "B7 的完整全因子设计 450 点，方向正确，是推荐来源；"
                "B8 的 Q 方向与 B6/B7 相反且有硬下界截断，不得用于标定。"
            ),
        },
        "substitution_condition": {
            "derivation": (
                "在 L = E + A*N^(-alpha) + B*(D*Q)^(-beta) 下令 dL=0，得 "
                "dN/dQ = B*beta*(D*Q)^(-beta) / (A*alpha*Q*N^(-alpha-1))。"
            ),
            "points": substitution,
        },
    }
    if design is not None:
        design = np.asarray(design, dtype=float)
        correlation = np.corrcoef(design.T) if design.shape[1] > 1 else np.eye(1)
        pairs = []
        for i in range(len(train_domains)):
            for j in range(i + 1, len(train_domains)):
                pairs.append((float(correlation[i, j]), train_domains[i], train_domains[j]))
        pairs.sort()
        elasticity_payload["domain_interaction"] = {
            "source": "compact 模型 effective_weight_pQ（13 评测域 × 17 训练域）的列相关",
            "most_complementary_pairs": [
                {"domains": [a, b], "correlation": c} for c, a, b in pairs[:5]
            ],
            "most_substitutable_pairs": [
                {"domains": [a, b], "correlation": c} for c, a, b in pairs[-5:][::-1]
            ],
            "caveat": (
                "这是模型内部权重的相关，不是因果替代关系；"
                "只用于提示哪些域的配比在模型中起相似作用。"
            ),
        }

    # ---- C7 / 成本函数 ---------------------------------------------------- #
    lengths, critical = context_length_grid()
    quality_cost = quality_cost_table(q0=q_reference, grid=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0))

    # ---- 写出：接口入口 --------------------------------------------------- #
    inputs = {
        "handoff": "第二问 → 第三问",
        "deliverables": {
            "law": "generalized_scaling_law.json",
            "law_parameters": "generalized_scaling_law_parameters.csv",
            "mixture_response": "mixture_response.json",
            "quality_and_mixture": "quality_mixture_inputs.csv",
            "context_length": "context_length_feasibility.csv",
            "quality_cost": "quality_cost_functions.json",
            "elasticity": "elasticity_and_substitution.json",
            "evaluator": "generalized_law_evaluator.py",
        },
        "model": payload["compositional_scaling_law"]["formula"],
        "model_scale_amplitude": payload["compositional_scaling_law"]["scale_amplitude"],
        "loss_domains": loss_domains,
        "train_domains": train_domains,
        "quality_vector_Q17": dict(zip(train_domains, quality.tolist())),
        "quality_reference_mean": q_reference,
        "units": {
            "N": "billion parameters (N_b)",
            "D": "billion tokens (D_b)",
            "L": "validation cross-entropy (nats)",
            "p": "17-dim simplex",
        },
        "q3_external_inputs": {
            "attention_eta": ATTENTION_ETA,
            "attention_cost": "C_attn = eta * N * D * L_ctx",
            "training_cost": "C_train = 6 * N * D",
            "critical_context_length": critical,
            "budgets_FLOPs": list(BUDGETS),
            "quality_cost_forms": list(QUALITY_COST_FORMS),
            "source": "题面附录 B（成本函数参数）与 C7（max_position_embeddings）",
        },
        "mixture_support": support,
        "verification": verification,
        "scope_note": (
            "本目录只交付第二问的模型与参数，以及第三问需要的外生输入整理；"
            "不包含第三问的优化模型、也不包含任何最优解。"
        ),
    }

    (OUT / "Q2_to_Q3_inputs.json").write_text(
        json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 广义标度律本体 --------------------------------------------------- #
    law_payload = {
        "formula": payload["compositional_scaling_law"]["formula"],
        "scale_amplitude": payload["compositional_scaling_law"]["scale_amplitude"],
        "mixture_response_expanded": payload["compositional_scaling_law"][
            "expanded_mixture_response"
        ],
        "clr_definition": payload["compositional_scaling_law"]["clr_definition"],
        "epsilon": law.epsilon,
        "reference_scale": {
            "N": law.reference_n * 1e9,
            "D": law.reference_d * 1e9,
        },
        "exponents_fixed_from_B1": {"alpha": law.alpha, "beta": law.beta},
        "per_domain_parameters": {
            name: {
                "E_k": float(law.e_coefficients[k]),
                "A_k": float(law.a_coefficients[k]),
                "B_k": float(law.b_coefficients[k]),
                "eta_k": float(law.eta[k]),
                "zeta_k": float(law.zeta[k]),
            }
            for k, name in enumerate(loss_domains)
        },
        "identifiability": {
            "quality_elasticity_not_identifiable_from_A": (
                "附件 A 中每个训练域的 Q_i 固定，模型只能识别 p_i·Q_i 的组合；"
                "独立的质量效应必须取自 B6/B7（见 elasticity_and_substitution.json）。"
            ),
            "weights_only_up_to_common_scale": (
                "配比响应带 slope_ref_k，Q·W 的整体尺度被它吸收；"
                "报告的权重只有相对意义。"
            ),
            "exponents_borrowed_from_B1": (
                "附件 A 只有 3 个不同的 (N, D) 点，标准标度律在其上恰好饱和，"
                "无法自识别指数；alpha、beta 固定为 B1 的识别值。"
            ),
            "eta_zeta_are_extrapolation": (
                "eta 由 1M↔60M 确定，zeta 实际只由 1B 一个规模点确定；"
                "幅度幂律衰减是假设，三个点无法检验。"
            ),
        },
        "per_domain_r_squared_within_scale_test_60m": {
            name: float(per_domain_r2[k]) for k, name in enumerate(loss_domains)
        },
        "fit_quality_heldout": payload["compositional_scaling_law"]["heldout_test"],
        "fit_quality_aggregate": {
            k: v
            for k, v in payload["compositional_scaling_law"]["aggregate_heldout"].items()
            if k != "irreducible_floor"
        },
        "self_check_reproduction": verification,
        "caveat": payload["compositional_scaling_law"]["extrapolation_warning"],
    }
    (OUT / "generalized_scaling_law.json").write_text(
        json.dumps(law_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (OUT / "generalized_scaling_law_parameters.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "loss_domain",
                "E_k",
                "A_k",
                "B_k",
                "eta_k",
                "zeta_k",
                "r_squared_within_scale_test_60m",
            ]
        )
        for k, name in enumerate(loss_domains):
            writer.writerow(
                [
                    name,
                    f"{law.e_coefficients[k]:.10g}",
                    f"{law.a_coefficients[k]:.10g}",
                    f"{law.b_coefficients[k]:.10g}",
                    f"{law.eta[k]:.10g}",
                    f"{law.zeta[k]:.10g}",
                    f"{per_domain_r2[k]:.6g}",
                ]
            )

    # ---- 配比响应（可复算） ----------------------------------------------- #
    mixture_payload = {
        "purpose": "Φ_k(p,Q) 的完整可复算参数；给定 p 与 Q 即可还原 13 个评测域的配比响应",
        "definition": payload["compositional_scaling_law"]["expanded_mixture_response"],
        "clr_definition": payload["compositional_scaling_law"]["clr_definition"],
        "epsilon": law.epsilon,
        "feature_names": law.feature_names,
        "feature_mean": law.feature_mean.tolist(),
        "feature_scale": law.feature_scale.tolist(),
        "ridge_intercept": dict(zip(loss_domains, law.ridge_intercept.tolist())),
        "ridge_coefficients": {
            name: dict(zip(law.feature_names, law.ridge_coefficients[k].tolist()))
            for k, name in enumerate(loss_domains)
        },
        "reference_response_center": dict(
            zip(loss_domains, law.response_center.tolist())
        ),
        "reference_response_slope": dict(
            zip(loss_domains, law.reference_slope.tolist())
        ),
        "usage": (
            "标准化 X = (features - feature_mean) / feature_scale；"
            "raw = ridge_intercept + X @ ridge_coefficients^T；"
            "Phi = (raw - reference_response_center) * reference_response_slope"
        ),
        "important": (
            "Φ 已在参考规模的训练配比上零中心化，因此它的**绝对值没有意义**，"
            "只有相对于参考配比的变化有意义。"
        ),
    }
    (OUT / "mixture_response.json").write_text(
        json.dumps(mixture_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 质量与配比输入 --------------------------------------------------- #
    shares = np.zeros(len(train_domains))
    total = 0
    for _label, _n, _d, p_file, _l in blocks_rows:
        with (regmix / p_file).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            p = np.asarray([[float(v) for v in row[1:]] for row in reader if row])
        shares += p.mean(axis=0) * p.shape[0]
        total += p.shape[0]
    shares /= total
    with (OUT / "quality_mixture_inputs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "train_domain",
                "Q17",
                "mean_mixture_share_in_A",
                "min_share_in_A",
                "max_share_in_A",
            ]
        )
        all_p = []
        for _label, _n, _d, p_file, _l in blocks_rows:
            with (regmix / p_file).open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                next(reader)
                all_p.append(
                    np.asarray([[float(v) for v in row[1:]] for row in reader if row])
                )
        stacked = np.vstack(all_p)
        for k, name in enumerate(train_domains):
            writer.writerow(
                [
                    name,
                    f"{quality[k]:.8g}",
                    f"{shares[k]:.8g}",
                    f"{stacked[:, k].min():.8g}",
                    f"{stacked[:, k].max():.8g}",
                ]
            )

    # ---- C7 上下文长度 ---------------------------------------------------- #
    with (OUT / "context_length_feasibility.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "L_ctx",
                "model_count_in_C7",
                "example_models",
                "C_attn_over_C_train",
                "attention_share_of_budget",
                "exceeds_critical",
                "note",
            ]
        )
        for record in lengths:
            writer.writerow(
                [
                    record["L_ctx"],
                    record["model_count"],
                    "; ".join(record["example_models"]),
                    f"{record['attention_over_training_ratio']:.6g}",
                    f"{record['attention_share_of_budget']:.6g}",
                    record["exceeds_critical"],
                    record["note"],
                ]
            )

    (OUT / "quality_cost_functions.json").write_text(
        json.dumps(
            {
                "definition": "C_Q = D * [g(Q) - g(Q0)]_+",
                "baseline_quality_Q0": {
                    "value": q_reference,
                    "source": (
                        "附件 A 的 17 域质量均值（Q17_mapping.csv 的 Q17 均值）；"
                        "第三问如改用其他 Q0 口径须在论文中说明并做敏感性分析"
                    ),
                },
                "forms": quality_cost,
                "cross_form_comparison": {
                    "per_token_cost_at_Q_1.0": {
                        form: quality_cost[form]["per_token_cost"]["Q=1.00"]
                        for form in quality_cost
                    },
                    "note": (
                        "三种形式在 Q→1 时的每 token 增量成本相差数个量级，"
                        "因此第三问必须报告成本函数选择对最优解的影响。"
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (OUT / "elasticity_and_substitution.json").write_text(
        json.dumps(elasticity_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(inputs, ensure_ascii=False, indent=2)[:1200])
    print("\n自检（用交付的求值器复现 compact 结果）：")
    for row in verification:
        print(
            f"  {row['block']:9s} n={row['n_mixtures']:4d} "
            f"relRMSE={row['relative_rmse']:.4f} 中位APE={row['median_ape']:.4f}"
        )


if __name__ == "__main__":
    main()
