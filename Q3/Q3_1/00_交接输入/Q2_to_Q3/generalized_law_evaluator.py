#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二问交付给第三问的广义标度律求值器（自包含，只依赖 numpy）。

模型
----
::

    L_k(N, D, p, Q) = E_k + A_k * N_b^(-alpha) + B_k * D_b^(-beta)
                      + S_k(N, D) * Phi_k(p, Q)

    S_k(N, D)  = (N_b / N_ref)^(-eta_k) * (D_b / D_ref)^(-zeta_k)
    Phi_k(p,Q) = slope_k * [ Ridge_k( (X - mean) / scale ) - center_k ]
    X          = [ p_i * Q_i  (17 维) , clr_eps(p)_i (17 维) ]
    clr_eps(p)_i = log(p_i + eps) - (1/17) * sum_j log(p_j + eps)

**单位**：``N_b`` 与 ``D_b`` 以**十亿**为单位（1e9 参数、1e9 token）；
``L`` 为验证集交叉熵（nats）；``p`` 为 17 维单纯形上的配比。

用法
----
::

    from generalized_law_evaluator import GeneralizedLawEvaluator

    law = GeneralizedLawEvaluator.from_directory(".")   # 本目录
    p = np.full(17, 1 / 17)
    losses = law.predict(n_params=1e9, d_tokens=2.5e10, p=p)   # 13 个评测域的 Loss
    print(losses.mean())                                       # 标量目标可取均值

注意
----
* 传入的 ``n_params`` / ``d_tokens`` 是**绝对个数**（不是十亿），内部会除以 1e9。
* ``Phi_k`` 已在参考规模的训练配比上零中心化，**绝对值没有意义**，
  只有相对于参考配比的变化有意义。
* 本模型对全局质量缩放的响应是**外推**：附件 A 中每个域的 Q_i 固定，
  模型无法识别独立的质量效应。第三问的质量弹性请以 B6/B7 为准
  （见 ``elasticity_and_substitution.json``）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["GeneralizedLawEvaluator"]


class GeneralizedLawEvaluator:
    """分离式组成广义标度律的求值器。"""

    def __init__(self, payload: dict, mixture_payload: dict, quality: np.ndarray,
                 train_domains: list[str]) -> None:
        self.loss_domains: list[str] = list(mixture_payload["ridge_intercept"].keys())
        self.train_domains: list[str] = list(train_domains)
        self.quality = np.asarray(quality, dtype=float)

        self.epsilon = float(mixture_payload["epsilon"])
        self.feature_names: list[str] = list(mixture_payload["feature_names"])
        self.feature_mean = np.asarray(mixture_payload["feature_mean"], dtype=float)
        self.feature_scale = np.asarray(mixture_payload["feature_scale"], dtype=float)
        self.ridge_intercept = np.asarray(
            [mixture_payload["ridge_intercept"][d] for d in self.loss_domains]
        )
        self.ridge_coefficients = np.asarray(
            [
                [mixture_payload["ridge_coefficients"][d][f] for f in self.feature_names]
                for d in self.loss_domains
            ]
        )
        self.response_center = np.asarray(
            [mixture_payload["reference_response_center"][d] for d in self.loss_domains]
        )
        self.response_slope = np.asarray(
            [mixture_payload["reference_response_slope"][d] for d in self.loss_domains]
        )

        domain = payload["per_domain_parameters"]
        self.e_coefficients = np.asarray([domain[d]["E_k"] for d in self.loss_domains])
        self.a_coefficients = np.asarray([domain[d]["A_k"] for d in self.loss_domains])
        self.b_coefficients = np.asarray([domain[d]["B_k"] for d in self.loss_domains])
        self.eta = np.asarray([domain[d]["eta_k"] for d in self.loss_domains])
        self.zeta = np.asarray([domain[d]["zeta_k"] for d in self.loss_domains])
        self.alpha = float(payload["exponents_fixed_from_B1"]["alpha"])
        self.beta = float(payload["exponents_fixed_from_B1"]["beta"])
        self.reference_n = float(payload["reference_scale"]["N"]) / 1e9
        self.reference_d = float(payload["reference_scale"]["D"]) / 1e9

    # ------------------------------------------------------------------ #
    @classmethod
    def from_directory(cls, directory: str | Path) -> "GeneralizedLawEvaluator":
        """从交付目录读取参数并构造求值器。"""

        directory = Path(directory)
        payload = json.loads(
            (directory / "generalized_scaling_law.json").read_text(encoding="utf-8")
        )
        mixture = json.loads(
            (directory / "mixture_response.json").read_text(encoding="utf-8")
        )
        inputs = json.loads(
            (directory / "Q2_to_Q3_inputs.json").read_text(encoding="utf-8")
        )
        train_domains = list(inputs["train_domains"])
        quality = np.asarray(
            [inputs["quality_vector_Q17"][name] for name in train_domains], dtype=float
        )
        return cls(payload, mixture, quality, train_domains)

    # ------------------------------------------------------------------ #
    def _features(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        log_p = np.log(p + self.epsilon)
        clr = log_p - log_p.mean(axis=1, keepdims=True)
        return np.column_stack((p * q[None, :], clr))

    def mixture_response(self, p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
        """``Phi_k(p, Q)``，返回 (n, 13)。"""

        p = np.atleast_2d(np.asarray(p, dtype=float))
        q = self.quality if q is None else np.asarray(q, dtype=float)
        standardized = (self._features(p, q) - self.feature_mean) / self.feature_scale
        raw = self.ridge_intercept[None, :] + standardized @ self.ridge_coefficients.T
        return (raw - self.response_center[None, :]) * self.response_slope[None, :]

    def scale_amplitude(self, n_params: float, d_tokens: float) -> np.ndarray:
        """``S_k(N, D)``，返回 (13,)。``n_params``/``d_tokens`` 为绝对个数。"""

        n_b = np.asarray(n_params, dtype=float) / 1e9
        d_b = np.asarray(d_tokens, dtype=float) / 1e9
        return (n_b / self.reference_n) ** (-self.eta) * (
            d_b / self.reference_d
        ) ** (-self.zeta)

    def predict(
        self,
        n_params: float,
        d_tokens: float,
        p: np.ndarray,
        quality_scale: float = 1.0,
        q: np.ndarray | None = None,
    ) -> np.ndarray:
        """返回 13 个评测域的预测 Loss；``quality_scale`` 把所有 Q_i 同比例缩放。"""

        n_b = float(n_params) / 1e9
        d_b = float(d_tokens) / 1e9
        q_effective = (self.quality if q is None else np.asarray(q, dtype=float))
        q_effective = q_effective * float(quality_scale)
        base = (
            self.e_coefficients
            + self.a_coefficients * n_b ** (-self.alpha)
            + self.b_coefficients * d_b ** (-self.beta)
        )
        amplitude = self.scale_amplitude(n_params, d_tokens)
        return base + amplitude * self.mixture_response(p, q_effective)[0]

    def predict_scalar(
        self,
        n_params: float,
        d_tokens: float,
        p: np.ndarray,
        quality_scale: float = 1.0,
        q: np.ndarray | None = None,
    ) -> float:
        """13 个评测域 Loss 的算术平均，便于作为优化目标。"""

        return float(
            self.predict(n_params, d_tokens, p, quality_scale, q).mean()
        )

    def elasticities(
        self, n_params: float, d_tokens: float, p: np.ndarray, step: float = 1e-4
    ) -> dict[str, float]:
        """在给定点上的对数弹性（中心差分）。"""

        base = self.predict_scalar(n_params, d_tokens, p)
        d_n = (
            self.predict_scalar(n_params * (1 + step), d_tokens, p)
            - self.predict_scalar(n_params * (1 - step), d_tokens, p)
        ) / (2 * step) / base
        d_d = (
            self.predict_scalar(n_params, d_tokens * (1 + step), p)
            - self.predict_scalar(n_params, d_tokens * (1 - step), p)
        ) / (2 * step) / base
        d_q = (
            self.predict_scalar(n_params, d_tokens, p, quality_scale=1 + step)
            - self.predict_scalar(n_params, d_tokens, p, quality_scale=1 - step)
        ) / (2 * step) / base
        return {
            "predicted_mean_loss": base,
            "dlnL_dlnN": float(d_n),
            "dlnL_dlnD": float(d_d),
            "dlnL_dlnQ_scale": float(d_q),
        }


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    law = GeneralizedLawEvaluator.from_directory(here)
    equal = np.full(len(law.train_domains), 1.0 / len(law.train_domains))
    print("评测域:", law.loss_domains)
    print("训练域:", law.train_domains)
    for n, d in ((1e9, 2.5e10), (6e7, 1e9), (1e10, 1e11)):
        print(
            f"N={n:.3g}, D={d:.3g}: 13 域均值 Loss = "
            f"{law.predict_scalar(n, d, equal):.4f}"
        )
    print("弹性 @(1B, 25B):", law.elasticities(1e9, 2.5e10, equal))
