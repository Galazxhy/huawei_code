"""问题三统一模型合同与只读求值适配器。

本模块只负责把问题二交付的配比响应、B7质量响应和题面成本合同接起来。
它不执行第三问优化，也不把外推点伪装成观测验证。

外部单位合同
------------
``n_params`` 为参数个数，``d_tokens`` 为 token 个数，成本为 FLOPs；
``p`` 为17维单纯形配比，``Q`` 为全局质量治理水平。

质量语义
--------
Q1的 ``q_i`` 是固定的领域质量向量，用于问题二的配比响应；这里的 ``Q``
表示在固定配比质量基础上通过统一治理动作达到的全局质量水平，主分析范围
为 ``Q0 <= Q <= 1``。二者不能在未声明的情况下互相替代。

质量桥接
--------
以问题二的13域配比Loss为基线，使用B7质量律在相同(N,D)处的质量变化：

    L3_ratio = L_mix * (F_B7(N,D,Q) / F_B7(N,D,Q0)) ** kappa
    L3_add = L_mix + kappa * (F_B7(N,D,Q) - F_B7(N,D,Q0))

主情景 ``kappa=1``；它是跨数据源的建模假设，必须与其它桥接情景做敏感性
分析。Q=Q0时严格回到问题二的配比预测。

配比可信域
----------
优化配比必须属于A4-A5训练配方的凸包。凸包检验由线性规划完成；仅有
``p_i >= p_min`` 不视为支持域约束。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
HANDOFF_Q1 = ROOT / "00_交接输入/Q1_to_Q2"
HANDOFF_Q2 = ROOT / "00_交接输入/Q2_to_Q3"
ATTACHMENTS = ROOT / "00_交接输入/支持与复核数据"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_q2_evaluator():
    path = HANDOFF_Q2 / "generalized_law_evaluator.py"
    spec = importlib.util.spec_from_file_location("q2_handoff_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Q2 evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GeneralizedLawEvaluator


@dataclass(frozen=True)
class ContractConfig:
    eta_attention: float = 2e-4
    quality_bridge_kappa: float = 1.0
    quality_bridge_mode: str = "ratio"
    q_lower_is_q0: bool = True
    support_tolerance: float = 2e-8
    probability_tolerance: float = 2e-8
    loss_floor: float = 1e-12
    n_bounds: tuple[float, float] = (1e6, 1e11)
    d_bounds: tuple[float, float] = (1e9, 1e13)
    allow_scale_extrapolation: bool = True
    q0_override: float | None = None
    quality_cost_scale: float = 1.0


class Q3ModelContract:
    """问题三的可审计统一求值器。"""

    def __init__(self, config: ContractConfig | None = None) -> None:
        self.config = config or ContractConfig()
        self.inputs = json.loads((HANDOFF_Q2 / "Q2_to_Q3_inputs.json").read_text(encoding="utf-8"))
        self.quality_cost_payload = json.loads(
            (HANDOFF_Q2 / "quality_cost_functions.json").read_text(encoding="utf-8")
        )
        self.elasticity_payload = json.loads(
            (HANDOFF_Q2 / "elasticity_and_substitution.json").read_text(encoding="utf-8")
        )
        self.train_domains = list(self.inputs["train_domains"])
        self.loss_domains = list(self.inputs["loss_domains"])
        self.q_i = np.asarray(
            [self.inputs["quality_vector_Q17"][d] for d in self.train_domains], dtype=float
        )
        self.q0 = float(self.inputs["quality_reference_mean"])
        self.q2 = _load_q2_evaluator().from_directory(HANDOFF_Q2)
        self._validate_handoff()
        if self.config.q0_override is not None:
            self.q0 = float(self.config.q0_override)
        if not 0 < self.q0 <= 1:
            raise ValueError("Q0 must be in (0, 1]")
        if not np.isfinite(self.config.quality_cost_scale) or self.config.quality_cost_scale <= 0:
            raise ValueError("quality_cost_scale must be positive and finite")
        if self.config.quality_bridge_mode not in ("ratio", "additive"):
            raise ValueError("invalid bridge mode")
        if not np.isfinite(self.config.quality_bridge_kappa) or self.config.quality_bridge_kappa < 0:
            raise ValueError("kappa must be nonnegative and finite; zero is the null-effect control")
        for bounds in (self.config.n_bounds, self.config.d_bounds):
            if len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
                raise ValueError("invalid search bounds")
        self.b7 = self.elasticity_payload["quality_law_from_B6_B7"]["B7"]
        self.context_table = pd.read_csv(HANDOFF_Q2 / "context_length_feasibility.csv")
        self.contexts = tuple(int(x) for x in self.context_table["L_ctx"].tolist())
        c7 = pd.read_csv(ATTACHMENTS / "C_efficiency_evolution/model_architecture_metadata.csv")
        observed_contexts = sorted(c7["max_position_embeddings"].dropna().astype(int).unique())
        if list(self.contexts) != observed_contexts:
            raise ValueError("handoff contexts disagree with original C7")
        self.critical_context = 6.0 / self.config.eta_attention
        self.observed_scale_anchors = ((1e6, 1e9), (6e7, 1e9), (1e9, 25e9))
        self.support_points = self._load_support_points()
        self.support_matrix = self.support_points
        self._residual_scale = self._calibrate_residual_scale()

    def _validate_handoff(self) -> None:
        q1 = pd.read_csv(HANDOFF_Q1 / "Q17_mapping.csv")
        domains = pd.read_csv(HANDOFF_Q1 / "train_domain_names.csv")["domain"].tolist()
        losses = pd.read_csv(HANDOFF_Q1 / "loss_domain_names.csv")["loss_domain"].tolist()
        if domains != self.train_domains or losses != self.loss_domains:
            raise ValueError("Q1/Q2 domain order mismatch")
        if q1["mixture_domain"].duplicated().any() or set(q1["mixture_domain"]) != set(domains):
            raise ValueError("Q17 domain mapping mismatch")
        values = q1.set_index("mixture_domain").loc[domains, "Q17"].to_numpy(float)
        if not np.allclose(values, self.q_i, atol=1e-12, rtol=0):
            raise ValueError("Q1/Q2 quality mismatch")
        if self.q2.train_domains != domains or self.q2.loss_domains != losses:
            raise ValueError("evaluator domain order mismatch")
        if not np.allclose(self.q2.quality, values, atol=1e-12, rtol=0):
            raise ValueError("evaluator Q17 mismatch")
        if not np.isclose(values.mean(), self.q0, atol=1e-12, rtol=0):
            raise ValueError("Q0 must agree with declared equal-domain baseline")

    def _load_support_points(self) -> np.ndarray:
        files = [
            ("train_mixture_1m.csv", "train"),
        ]
        frames = []
        for filename, kind in files:
            path = ATTACHMENTS / "A_data_value/regmix_tables" / filename
            frame = pd.read_csv(path)
            cols = [f"train_the_pile_{d}" for d in self.train_domains]
            values = frame[cols].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"invalid mixture values in {path}")
            row_sum = values.sum(axis=1, keepdims=True)
            if np.any(row_sum <= 0):
                raise ValueError(f"nonpositive mixture row in {path}")
            frames.append(values / row_sum)
        return np.vstack(frames)

    def _calibrate_residual_scale(self) -> dict[str, float]:
        """Use supplied test tables for an empirical residual scale only.

        This is not a formal predictive interval: the handoff does not contain
        fit covariance, test IDs or repeated seeds.
        """
        blocks = [
            ("test_1m", "test_mixture_1m.csv", "test_pile_loss_1m.csv", 1e6, 1e9),
            ("test_60m", "test_mixture_60m.csv", "test_pile_loss_60m.csv", 6e7, 1e9),
            ("test_1b", "test_mixture_1B.csv", "test_pile_loss_1B.csv", 1e9, 25e9),
        ]
        result: dict[str, float] = {}
        base = ATTACHMENTS / "A_data_value/regmix_tables"
        for name, mix_file, loss_file, n, d in blocks:
            mix = pd.read_csv(base / mix_file, index_col=0)
            loss = pd.read_csv(base / loss_file, index_col=0)
            if not mix.index.is_unique or not loss.index.is_unique or set(mix.index) != set(loss.index):
                raise ValueError(f"mixture/Loss experiment ID mismatch: {name}")
            loss = loss.loc[mix.index]
            p = mix[[f"train_the_pile_{x}" for x in self.train_domains]].to_numpy(dtype=float)
            p /= p.sum(axis=1, keepdims=True)
            y = loss[[f"metric/the_pile_{x}_val_loss" for x in self.loss_domains]].to_numpy(dtype=float)
            if not np.isfinite(p).all() or not np.isfinite(y).all() or (p < 0).any() or (y <= 0).any():
                raise ValueError(f"invalid test data: {name}")
            pred = np.asarray([self.q2.predict(n, d, row) for row in p])
            result[name] = float(np.sqrt(np.mean((pred - y) ** 2)))
        return result

    def _validate_units(self, n_params: float, d_tokens: float) -> None:
        if not np.isfinite(n_params) or not np.isfinite(d_tokens):
            raise ValueError("N and D must be finite")
        if n_params <= 0 or d_tokens <= 0:
            raise ValueError("N and D must be positive absolute counts")
        if n_params < 1e4 or d_tokens < 1e6:
            raise ValueError(
                "N and D appear to be in billion units; pass absolute parameter/token counts"
            )
        for name, value, bounds in (("N", n_params, self.config.n_bounds),
                                    ("D", d_tokens, self.config.d_bounds)):
            if value < bounds[0] * (1 - 1e-12) or value > bounds[1] * (1 + 1e-12):
                raise ValueError(f"{name} outside declared scenario bounds {bounds}")

    def _validate_q(self, Q: float) -> None:
        if not np.isfinite(Q) or Q <= 0 or Q > 1:
            raise ValueError("Q must lie in (0, 1]")
        if self.config.q_lower_is_q0 and Q < self.q0 - self.config.probability_tolerance:
            raise ValueError(f"Q={Q} is below the baseline Q0={self.q0}")

    def _validate_context(self, L_ctx: int) -> None:
        if int(L_ctx) != L_ctx or int(L_ctx) not in self.contexts:
            raise ValueError(f"L_ctx must be one of the C7 values {self.contexts}")

    def _validate_probability(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(-1)
        if p.size != len(self.train_domains):
            raise ValueError(f"p must have {len(self.train_domains)} entries")
        if not np.isfinite(p).all() or (p < -self.config.probability_tolerance).any():
            raise ValueError("p must be finite and nonnegative")
        if abs(float(p.sum()) - 1.0) > self.config.probability_tolerance:
            raise ValueError("p must sum to one")
        return np.maximum(p, 0.0) / p.sum()

    def support_certificate(self, p: np.ndarray) -> dict[str, Any]:
        """Return an exact convex-hull membership certificate for p."""
        p = self._validate_probability(p)
        A_eq = np.vstack([self.support_matrix.T, np.ones(self.support_matrix.shape[0])])
        b_eq = np.r_[p, 1.0]
        fit = linprog(
            c=np.zeros(self.support_matrix.shape[0]),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0.0, None),
            method="highs",
        )
        residual = None
        if fit.success and fit.x is not None:
            residual = float(np.max(np.abs(self.support_matrix.T @ fit.x - p)))
        return {
            "inside_convex_hull": bool(fit.success and (residual or 0.0) <= self.config.support_tolerance),
            "lp_success": bool(fit.success),
            "max_reconstruction_error": residual,
            "support_points": int(self.support_matrix.shape[0]),
            "message": str(fit.message),
        }

    def _require_support(self, p: np.ndarray) -> np.ndarray:
        p = self._validate_probability(p)
        certificate = self.support_certificate(p)
        if not certificate["inside_convex_hull"]:
            raise ValueError(
                "p is outside the convex hull of A4-A5 training mixtures; "
                "optimization must remain in the declared mixture support region"
            )
        return p

    def _g(self, Q: float, form: str) -> float:
        item = self.quality_cost_payload["forms"][form]
        gamma = float(item["parameters"]["gamma"]) * self.config.quality_cost_scale
        lam = float(item["parameters"]["lambda"])
        if form == "exponential":
            return gamma * float(np.exp(lam * Q))
        if form == "power":
            return gamma * Q**lam
        if form == "log_asymptotic":
            return gamma * float(np.log1p(lam * Q))
        raise ValueError(f"unknown quality cost form: {form}")

    def g_prime(self, Q: float, form: str) -> float:
        params = self.quality_cost_payload["forms"][form]["parameters"]
        gamma = float(params["gamma"]) * self.config.quality_cost_scale
        lam = float(params["lambda"])
        if form == "exponential":
            return gamma * lam * float(np.exp(lam * Q))
        if form == "power":
            return gamma * lam * Q ** (lam - 1)
        if form == "log_asymptotic":
            return gamma * lam / (1 + lam * Q)
        raise ValueError("invalid cost form")

    def costs(self, n_params: float, d_tokens: float, Q: float, L_ctx: int,
              quality_cost_form: str = "exponential") -> dict[str, float]:
        self._validate_units(n_params, d_tokens)
        self._validate_q(Q)
        self._validate_context(L_ctx)
        c_train = 6.0 * n_params * d_tokens
        c_attn = self.config.eta_attention * n_params * d_tokens * int(L_ctx)
        c_quality = d_tokens * max(self._g(Q, quality_cost_form) - self._g(self.q0, quality_cost_form), 0.0)
        return {
            "C_train_flops": c_train,
            "C_attn_flops": c_attn,
            "C_quality_flops": c_quality,
            "C_total_flops": c_train + c_attn + c_quality,
            "attention_to_training_ratio": c_attn / c_train,
            "quality_cost_form": quality_cost_form,
            "quality_cost_scale": self.config.quality_cost_scale,
        }

    def _quality_loss(self, n_params: float, d_tokens: float, Q: float) -> float:
        n_b = n_params / 1e9
        d_b = d_tokens / 1e9
        return float(
            self.b7["E"]
            + self.b7["A"] * n_b ** (-self.b7["alpha"])
            + self.b7["B"] * (d_b * Q) ** (-self.b7["beta"])
        )

    def predict_losses(self, n_params: float, d_tokens: float, p: np.ndarray,
                       Q: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Shared numerical kernel; public evaluate applies all guards."""
        base_loss = np.asarray(self.q2.predict(n_params, d_tokens, p), dtype=float)
        q_loss = self._quality_loss(n_params, d_tokens, Q)
        q0_loss = self._quality_loss(n_params, d_tokens, self.q0)
        ratio = q_loss / q0_loss
        if self.config.quality_bridge_mode == "ratio":
            final_loss = base_loss * ratio ** self.config.quality_bridge_kappa
        elif self.config.quality_bridge_mode == "additive":
            final_loss = base_loss + self.config.quality_bridge_kappa * (q_loss - q0_loss)
        else:
            raise ValueError(f"unknown quality bridge mode: {self.config.quality_bridge_mode}")
        return final_loss, base_loss, float(ratio)

    def evaluate(self, n_params: float, d_tokens: float, p: np.ndarray, Q: float,
                 L_ctx: int, quality_cost_form: str = "exponential",
                 budget_flops: float | None = None) -> dict[str, Any]:
        self._validate_units(n_params, d_tokens)
        self._validate_q(Q)
        self._validate_context(L_ctx)
        p = self._require_support(p)
        final_loss, base_loss, ratio = self.predict_losses(n_params, d_tokens, p, Q)
        if not np.isfinite(final_loss).all() or (final_loss <= self.config.loss_floor).any():
            raise ValueError("unified predictor returned nonpositive or nonfinite Loss")
        scale_supported = any(
            np.isclose(n_params, n0, rtol=0.0, atol=1e-6 * n0)
            and np.isclose(d_tokens, d0, rtol=0.0, atol=1e-6 * d0)
            for n0, d0 in self.observed_scale_anchors
        )
        if not scale_supported and not self.config.allow_scale_extrapolation:
            raise ValueError("scale extrapolation disabled for this run")
        costs = self.costs(n_params, d_tokens, Q, L_ctx, quality_cost_form)
        budget_margin = None
        if budget_flops is not None:
            if not np.isfinite(budget_flops) or budget_flops <= 0:
                raise ValueError("budget_flops must be a positive finite FLOPs budget")
            budget_margin = float(budget_flops - costs["C_total_flops"])
            if budget_margin < -max(1.0, 1e-10 * budget_flops):
                raise ValueError(
                    f"budget exceeded: total={costs['C_total_flops']:.6g}, "
                    f"budget={budget_flops:.6g}"
                )
        scale_distances = np.linalg.norm(
            np.log10(np.array([n_params, d_tokens]) / np.asarray(self.observed_scale_anchors)), axis=1)
        nearest_scale = int(np.argmin(scale_distances))
        return {
            "N_params": float(n_params),
            "D_tokens": float(d_tokens),
            "Q": float(Q),
            "Q0": self.q0,
            "p": p.tolist(),
            "L_ctx": int(L_ctx),
            "loss_by_domain": dict(zip(self.loss_domains, final_loss.tolist())),
            "mean_loss": float(final_loss.mean()),
            "base_mean_loss_q2": float(base_loss.mean()),
            "quality_ratio": float(ratio),
            "quality_bridge_kappa": self.config.quality_bridge_kappa,
            "quality_bridge_mode": self.config.quality_bridge_mode,
            "support_certificate": self.support_certificate(p),
            "scale_supported_by_A_mixture": scale_supported,
            "observed_scale_anchors": [list(x) for x in self.observed_scale_anchors],
            "external_scale_warning": not scale_supported,
            "nearest_observed_scale": list(self.observed_scale_anchors[nearest_scale]),
            "nearest_scale_log10_distance": float(scale_distances[nearest_scale]),
            "nearest_training_mixture_l1": float(np.abs(self.support_points - p).sum(axis=1).min()),
            "evidence_levels": {"mixture_geometry": "A4 training hull certificate",
                                "scale_response": "observed scale" if scale_supported else "conditional extrapolation",
                                "quality_response": "null effect control" if self.config.quality_bridge_kappa == 0 else
                                                    "uncalibrated transfer from B7 semi-synthetic quality law",
                                "quality_cost": "problem-specified cost on assumed common Q scale"},
            "scenario_N_bounds": list(self.config.n_bounds),
            "scenario_D_bounds": list(self.config.d_bounds),
            "budget_flops": None if budget_flops is None else float(budget_flops),
            "budget_margin_flops": budget_margin,
            "empirical_residual_scale_only": True,
            "residual_scale_by_observed_block": self._residual_scale,
            "costs": costs,
        }

    def source_manifest(self) -> dict[str, Any]:
        files = [
            HANDOFF_Q1 / "Q17_mapping.csv",
            HANDOFF_Q1 / "train_domain_names.csv",
            HANDOFF_Q1 / "loss_domain_names.csv",
            HANDOFF_Q2 / "Q2_to_Q3_inputs.json",
            HANDOFF_Q2 / "generalized_scaling_law.json",
            HANDOFF_Q2 / "mixture_response.json",
            HANDOFF_Q2 / "elasticity_and_substitution.json",
            HANDOFF_Q2 / "quality_cost_functions.json",
            HANDOFF_Q2 / "context_length_feasibility.csv",
            HANDOFF_Q2 / "generalized_law_evaluator.py",
        ]
        files += [
            ATTACHMENTS / "C_efficiency_evolution/model_architecture_metadata.csv",
            ATTACHMENTS / "B_scaling_laws/supplementary_NQ_experiment_expanded.csv",
        ]
        for filename in [
            "train_mixture_1m.csv",
            "test_mixture_1m.csv",
            "test_mixture_60m.csv",
            "test_mixture_1B.csv",
            "test_pile_loss_1m.csv",
            "test_pile_loss_60m.csv",
            "test_pile_loss_1B.csv",
        ]:
            files.append(ATTACHMENTS / "A_data_value/regmix_tables" / filename)
        files.append(ATTACHMENTS / "A_data_value/regmix_tables/train_pile_loss_1m.csv")
        files.append(ATTACHMENTS / "来源清单.json")
        files += sorted((ROOT / "02_代码与实验").glob("*.py"))
        files.append(ROOT / "01_项目管理/问题三统一执行方案.md")
        return {str(p.relative_to(ROOT.parent)): sha256(p) for p in files}


if __name__ == "__main__":
    model = Q3ModelContract()
    p = model.support_points.mean(axis=0)
    output = model.evaluate(1e9, 25e9, p, model.q0, 2048)
    print(json.dumps({
        "mean_loss": output["mean_loss"],
        "costs": output["costs"],
        "support": output["support_certificate"],
        "external_scale_warning": output["external_scale_warning"],
    }, ensure_ascii=False, indent=2))
