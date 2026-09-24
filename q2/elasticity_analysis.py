#!/usr/bin/env python3
"""Marginal loss benefit and elasticities for quality and mixture allocation.

Quality is analysed with the separately fitted quality scaling law. Mixture
allocation is analysed with the compositional generalized law, using feasible
directions on the simplex. The two laws have different loss calibrations and
their derivatives must not be added or divided to infer a substitution rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data_analysis" / "Q2_to_Q3"
OUTPUT = ROOT / "data_analysis" / "quality_mixture_elasticity"
QUALITY_DESIGN = (ROOT / "data" / "real_attachments" / "B_scaling_laws"
                  / "supplementary_NQ_experiment_expanded.csv")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HANDOFF))

import generalized_law as compact  # noqa: E402
from generalized_law_evaluator import GeneralizedLawEvaluator  # noqa: E402

ANCHORS = (
    ("1M / 1B tokens", "test_1m"),
    ("60M / 1B tokens", "test_60m"),
    ("1B / 25B tokens", "test_1b"),
)
MIN_ACTIVE_SHARE = 0.05
MAX_ACTIVE_SHARE = 0.95
SHARE_STEP = 0.01
SHARE_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def quality_point(fit: dict, n_b: float, d_b: float, q: float) -> dict:
    """Loss, marginal gain, elasticity, and finite +0.01 quality gain."""

    quality_term = fit["B"] * (d_b * q) ** (-fit["beta"])
    base = fit["E"] + fit["A"] * n_b ** (-fit["alpha"])
    loss = base + quality_term
    derivative = -fit["beta"] * quality_term / q
    return {
        "predicted_loss_nats": loss,
        "marginal_benefit_nats_per_unit_Q": -derivative,
        "approx_benefit_nats_for_plus_0_01_Q": -0.01 * derivative,
        "exact_benefit_nats_for_plus_0_01_Q": (
            loss - (base + fit["B"] * (d_b * (q + 0.01)) ** (-fit["beta"])))
            if q + 0.01 <= 1.0 else None,
        "loss_elasticity_wrt_Q": derivative * q / loss,
    }


def quality_analysis(fits: dict, reference_q: float) -> dict:
    """Analytic derivatives of E+A*N^-alpha+B*(D*Q)^-beta."""

    out = {}
    for fit_name in ("B6", "B7"):
        fit = fits[fit_name]
        points = []
        for q in (0.5, reference_q, 0.7, 0.9):
            n_b, d_b = 1.0, 25.0
            points.append({"Q": q, **quality_point(fit, n_b, d_b, q)})
        out[fit_name] = {
            "source_fit": {k: fit[k] for k in ("sample_count", "rmse", "r_squared")},
            "N_billion_parameters": 1.0,
            "D_billion_tokens": 25.0,
            "points": points,
        }
    return out


def quality_factorial_grid(fits: dict) -> tuple[list[dict], dict]:
    """Evaluate fitted laws on the observed 9 x 5 x 10 quality design."""

    with QUALITY_DESIGN.open("r", encoding="utf-8-sig", newline="") as handle:
        design = list(csv.DictReader(handle))
    axes = {
        "N_billion_parameters": sorted({float(r["N_params_B"]) for r in design}),
        "D_billion_tokens": sorted({float(r["D_tokens_B"]) for r in design}),
        "Q": sorted({float(r["Q_score"]) for r in design}),
    }
    expected = np.prod([len(values) for values in axes.values()])
    combinations = {(float(r["N_params_B"]), float(r["D_tokens_B"]),
                     float(r["Q_score"])) for r in design}
    if len(design) != expected or len(combinations) != expected:
        raise ValueError("The quality experiment is not a complete unique factorial grid")
    rows = []
    for raw in design:
        n = float(raw["N_params_B"])
        d = float(raw["D_tokens_B"])
        q = float(raw["Q_score"])
        primary = quality_point(fits["B7"], n, d, q)
        comparison = quality_point(fits["B6"], n, d, q)
        rows.append({
            "N_billion_parameters": n, "D_billion_tokens": d, "Q": q,
            "observed_loss_nats": float(raw["val_loss"]),
            **{f"B7_{key}": value for key, value in primary.items()},
            "B6_marginal_benefit_nats_per_unit_Q": comparison[
                "marginal_benefit_nats_per_unit_Q"],
            "B6_loss_elasticity_wrt_Q": comparison["loss_elasticity_wrt_Q"],
        })
    observed = np.array([r["observed_loss_nats"] for r in rows])
    predicted = np.array([r["B7_predicted_loss_nats"] for r in rows])
    rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
    if not np.isclose(rmse, fits["B7"]["rmse"], rtol=0.05):
        raise ValueError(f"B7 grid RMSE {rmse:.6f} differs from stored fit")
    grid_info = {
        "source": str(QUALITY_DESIGN.relative_to(ROOT)).replace("\\", "/"),
        "axes": axes, "n_points": len(rows), "reproduced_B7_rmse": rmse,
        "marginal_benefit_range": [
            min(r["B7_marginal_benefit_nats_per_unit_Q"] for r in rows),
            max(r["B7_marginal_benefit_nats_per_unit_Q"] for r in rows),
        ],
        "elasticity_range": [
            min(r["B7_loss_elasticity_wrt_Q"] for r in rows),
            max(r["B7_loss_elasticity_wrt_Q"] for r in rows),
        ],
    }
    return rows, grid_info


def mixture_gradient(law: GeneralizedLawEvaluator, p: np.ndarray,
                     n: float, d: float) -> np.ndarray:
    """Return dL/dp in ambient coordinates, shape (samples, eval, sources).

    Only differences along a simplex tangent direction are interpreted below.
    """

    m = len(law.train_domains)
    effective = law.ridge_coefficients / law.feature_scale[None, :]
    effective *= law.response_slope[:, None]
    weighted = effective[:, :m] * law.quality[None, :]
    clr_weights = effective[:, m:]
    clr_centered = clr_weights - clr_weights.mean(axis=1, keepdims=True)
    phi_gradient = (weighted[None, :, :]
                    + clr_centered[None, :, :] / (p[:, None, :] + law.epsilon))
    return phi_gradient * law.scale_amplitude(n, d)[None, :, None]


def proportional_upweight(law: GeneralizedLawEvaluator, p: np.ndarray,
                          n: float, d: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Increase source i, reduce every other source proportionally.

    The returned derivative is dL/dp_i along that feasible direction.
    Linearized benefit for +1 percentage point is -0.01 * derivative. The dimensionless
    elasticity is p_i/L * derivative (a 1% *relative* increase in p_i).
    """

    gradient = mixture_gradient(law, p, n, d)
    weighted_sum = np.einsum("sdi,si->sd", gradient, p)
    other_mean = ((weighted_sum[:, :, None] - p[:, None, :] * gradient)
                  / np.maximum(1 - p[:, None, :], 1e-12))
    derivative = gradient - other_mean
    loss = law.predict(n, d, p, warn_outside_support=False)
    benefit_1pp = -0.01 * derivative
    elasticity = derivative * p[:, None, :] / loss[:, :, None]
    return benefit_1pp, elasticity, loss


def check_derivative(law: GeneralizedLawEvaluator, p: np.ndarray,
                     n: float, d: float, analytic_benefit: np.ndarray) -> None:
    """Fail if an analytic share derivative disagrees with a finite difference."""

    active = np.argwhere((p >= 0.05) & (p <= 0.8))
    if not len(active):
        raise ValueError("No interior mixture found for derivative check")
    row, source = (int(v) for v in active[0])
    share = p[row].copy()
    step = 1e-6
    plus, minus = share.copy(), share.copy()
    plus *= 1 - step / (1 - share[source])
    minus *= 1 + step / (1 - share[source])
    plus[source] = share[source] + step
    minus[source] = share[source] - step
    numeric = (law.predict(n, d, plus, warn_outside_support=False)
               - law.predict(n, d, minus, warn_outside_support=False)) / (2 * step)
    expected = -analytic_benefit[row, :, source] / 0.01
    if not np.allclose(numeric, expected, rtol=2e-4, atol=2e-5):
        raise ValueError("Mixture marginal benefit failed finite-difference check")


def percentile(values: np.ndarray, q: float) -> float | None:
    return float(np.quantile(values, q)) if values.size else None


def finite_share_gain(law: GeneralizedLawEvaluator, p: np.ndarray,
                      n: float, d: float, base_loss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact predicted loss gain for a feasible +1pp proportional upweight."""

    support = law.support()
    maximum = np.asarray([support["recorded_max_share_per_domain"][name]
                          for name in law.train_domains])
    gains = np.full((len(p), base_loss.shape[1], p.shape[1]), np.nan)
    valid = np.zeros((len(p), p.shape[1]), dtype=bool)
    for source in range(p.shape[1]):
        candidate = p * (1 - SHARE_STEP / np.maximum(1 - p[:, [source]], 1e-12))
        candidate[:, source] = p[:, source] + SHARE_STEP
        allowed = (p[:, source] <= MAX_ACTIVE_SHARE) & \
                  (candidate[:, source] <= maximum[source] + 1e-9) & \
                  ~law.outside_support(candidate)
        if np.any(allowed):
            predicted = law.predict(n, d, candidate[allowed],
                                    warn_outside_support=False)
            gains[allowed, :, source] = base_loss[allowed] - predicted
        valid[:, source] = allowed
    return gains, valid


def formal_global_quality_elasticity(law: GeneralizedLawEvaluator, p: np.ndarray,
                                     n: float, d: float, loss: np.ndarray) -> np.ndarray:
    """Fitted-law response to scaling every fixed Q_i; NOT identified by data."""

    m = len(law.train_domains)
    effective_pq = (law.ridge_coefficients[:, :m]
                    / law.feature_scale[None, :m]) * law.response_slope[:, None]
    derivative = ((p * law.quality[None, :]) @ effective_pq.T
                  * law.scale_amplitude(n, d)[None, :])
    return derivative.mean(axis=1) / loss.mean(axis=1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse(output_dir: Path) -> dict:
    law = GeneralizedLawEvaluator.from_directory(HANDOFF)
    blocks, train_names, loss_names, q = compact.load_data(compact.REGMIX,
                                                            compact.HANDOFF)
    if list(train_names) != law.train_domains or list(loss_names) != law.loss_domains:
        raise ValueError("Domain ordering differs between fitted law and raw mixtures")
    if not np.allclose(q, law.quality):
        raise ValueError("Quality scores differ between fitted law and raw mixtures")
    result = json.loads((HANDOFF / "elasticity_and_substitution.json").read_text(
        encoding="utf-8"))
    q0 = float(np.mean(q))
    quality_fits = result["quality_law_from_B6_B7"]
    quality = quality_analysis(quality_fits, q0)
    quality_grid_rows, quality_grid_info = quality_factorial_grid(quality_fits)

    by_name = {block.name: block for block in blocks}
    summary_rows, domain_rows, p_grid_rows = [], [], []
    anchor_info, formal_quality = {}, {}
    for anchor, block_name in ANCHORS:
        block = by_name[block_name]
        # The released CSV shares are rounded to three decimals. Enforce the
        # simplex before differentiating; this changes each share by <0.2%.
        p = block.p / block.p.sum(axis=1, keepdims=True)
        benefit, elasticity, loss = proportional_upweight(law, p, block.n, block.d)
        check_derivative(law, p, block.n, block.d, benefit)
        anchor_info[anchor] = {"n_recorded_mixtures": len(p),
                               "N_parameters": block.n, "D_tokens": block.d,
                               "mean_predicted_loss_nats": float(loss.mean())}
        formal = formal_global_quality_elasticity(law, p, block.n, block.d, loss)
        formal_quality[anchor] = {
            "minimum": float(formal.min()),
            "median": float(np.median(formal)),
            "maximum": float(formal.max()),
            "fraction_positive": float(np.mean(formal > 0)),
            "interpretation": "Algebraic derivative of the fitted model under "
                              "a global Q scale change; Q never varied "
                              "independently in the mixture data.",
        }
        finite_gain, finite_valid = finite_share_gain(law, p, block.n, block.d,
                                                      loss)
        mean_benefit = benefit.mean(axis=1)
        mean_finite_gain = finite_gain.mean(axis=1)
        mean_elasticity = ((-100 * mean_benefit) * p
                           / loss.mean(axis=1)[:, None])
        # Above uses derivative = -benefit/0.01; the factor 100 restores it.
        for source, name in enumerate(train_names):
            mask = (p[:, source] >= MIN_ACTIVE_SHARE) & \
                   (p[:, source] <= MAX_ACTIVE_SHARE)
            finite_mask = mask & finite_valid[:, source]
            n_active = int(mask.sum())
            domain_medians = np.median(benefit[mask, :, source], axis=0) \
                if n_active else np.full(len(loss_names), np.nan)
            summary_rows.append({
                "anchor": anchor, "source": name, "quality_Q_i": q[source],
                "n_active": n_active, "n_total": len(p),
                "median_active_share": percentile(p[mask, source], 0.5),
                "median_benefit_nats_per_plus_1pp": percentile(
                    mean_benefit[mask, source], 0.5),
                "n_feasible_plus_1pp": int(finite_mask.sum()),
                "median_exact_gain_nats_for_plus_1pp": percentile(
                    mean_finite_gain[finite_mask, source], 0.5),
                "p10_exact_gain_nats_for_plus_1pp": percentile(
                    mean_finite_gain[finite_mask, source], 0.1),
                "p90_exact_gain_nats_for_plus_1pp": percentile(
                    mean_finite_gain[finite_mask, source], 0.9),
                "p10_benefit_nats_per_plus_1pp": percentile(
                    mean_benefit[mask, source], 0.1),
                "p90_benefit_nats_per_plus_1pp": percentile(
                    mean_benefit[mask, source], 0.9),
                "median_loss_elasticity_relative_share": percentile(
                    mean_elasticity[mask, source], 0.5),
                "p10_loss_elasticity_relative_share": percentile(
                    mean_elasticity[mask, source], 0.1),
                "p90_loss_elasticity_relative_share": percentile(
                    mean_elasticity[mask, source], 0.9),
                "evaluation_domains_with_median_benefit": int(
                    np.sum(domain_medians > 0)) if n_active else None,
            })
            if n_active:
                active_indices = np.flatnonzero(mask)
                ordered = active_indices[np.argsort(p[active_indices, source])]
                for quantile in SHARE_QUANTILES:
                    row = int(ordered[round(quantile * (n_active - 1))])
                    p_grid_rows.append({
                        "anchor": anchor, "source": name,
                        "share_quantile": quantile, "mixture_id": block.ids[row],
                        "n_active": n_active, "p_i": p[row, source],
                        "mixture_entropy": float(law.entropy(p[row])[0]),
                        "mean_predicted_loss_nats": float(loss[row].mean()),
                        "mean_marginal_benefit_nats_per_plus_1pp":
                            float(mean_benefit[row, source]),
                        "mean_loss_elasticity_relative_share":
                            float(mean_elasticity[row, source]),
                        "plus_1pp_within_coarse_support": bool(
                            finite_valid[row, source]),
                        "mean_exact_gain_nats_for_plus_1pp": (
                            float(mean_finite_gain[row, source])
                            if finite_valid[row, source] else None),
                    })
            for k, domain in enumerate(loss_names):
                domain_rows.append({
                    "anchor": anchor, "source": name, "evaluation_domain": domain,
                    "n_active": n_active,
                    "median_benefit_nats_per_plus_1pp": percentile(
                        benefit[mask, k, source], 0.5),
                    "median_exact_gain_nats_for_plus_1pp": percentile(
                        finite_gain[finite_mask, k, source], 0.5),
                    "median_loss_elasticity_relative_share": percentile(
                        elasticity[mask, k, source], 0.5),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "quality_factorial_grid.csv", quality_grid_rows)
    write_csv(output_dir / "mixture_source_summary.csv", summary_rows)
    write_csv(output_dir / "mixture_domain_detail.csv", domain_rows)
    write_csv(output_dir / "mixture_share_quantile_grid.csv", p_grid_rows)
    payload = {
        "definitions": {
            "quality": "B6/B7: -dL/dQ, loss benefit in nats per unit Q; "
                       "elasticity=(Q/L)*dL/dQ.",
            "mixture": "At fixed N,D,Q, raise p_i and reduce all other shares "
                       "proportionally. Marginal benefit per +1pp is the "
                       "linearized -0.01*dL/dp_i; the separate exact +1pp "
                       "gain is computed only if coarse empirical support is "
                       "respected. Elasticity=(p_i/L)*dL/dp_i, for a 1% "
                       "relative share change.",
            "active_filter": f"{MIN_ACTIVE_SHARE} <= p_i <= {MAX_ACTIVE_SHARE}",
            "interpretation": "Positive benefit reduces loss. Negative elasticity "
                              "means a relative share increase reduces loss.",
        },
        "quality_reference_Q_mean_17_sources": q0,
        "quality": quality,
        "quality_factorial_grid": quality_grid_info,
        "mixture_share_quantile_grid": {
            "quantiles": list(SHARE_QUANTILES), "n_rows": len(p_grid_rows),
            "construction": "For each source and scale anchor, choose actual "
                            "recorded mixtures nearest the 10th, 25th, 50th, "
                            "75th, and 90th percentiles of active source share. "
                            "Other shares vary; these are not isolated p_i sweeps.",
        },
        "mixture_anchor_info": anchor_info,
        "formal_unidentified_quality_response_from_mixture_law": formal_quality,
        "limitations": [
            "Each Q_i is fixed per source in the mixture data. The generalized "
            "law's formal dL/dQ_i is not an identified quality effect.",
            "B6/B7 and the generalized mixture law have distinct response "
            "calibrations; their marginal benefits cannot be combined into a "
            "quality-versus-mixture substitution rate.",
            "Mixture benefits are local model sensitivities, not causal "
            "effects or held-out derivative observations. Rare active sources "
            "are excluded below 5% share; inspect n_active and n_feasible_plus_1pp.",
            "N and D change together between 60M and 1B anchors, so their "
            "separate scale effects are not identified by that comparison.",
        ],
        "source_files": [
            "data_analysis/Q2_to_Q3/elasticity_and_substitution.json",
            "data_analysis/Q2_to_Q3/generalized_scaling_law.json",
            "data_analysis/Q2_to_Q3/mixture_response.json",
            "data_analysis/Q_1_to_2/Q17_mapping.csv",
            "data/real_attachments/B_scaling_laws/supplementary_NQ_experiment_expanded.csv",
            "data/real_attachments/A_data_value/regmix_tables",
        ],
    }
    (output_dir / "allocation_elasticity.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "quality_elasticity.json").write_text(
        json.dumps({"reference_Q": q0, "models": quality}, indent=2),
        encoding="utf-8")
    b7 = quality["B7"]["points"][1]
    b6 = quality["B6"]["points"][1]
    lines = [
        "# 质量与配比的边际效益和弹性",
        "",
        "## 定义与数据口径",
        "",
        "质量采用独立拟合的质量标度律："
        "\\(L=E+AN^{-\\alpha}+B(DQ)^{-\\beta}\\)。"
        "其边际收益为 \\( -\\partial L/\\partial Q "
        "=\\beta B(DQ)^{-\\beta}/Q\\)，"
        "损失弹性为 \\((Q/L)\\partial L/\\partial Q\\)。"
        "参数、token 均以十亿为单位。",
        "",
        "配比采用已拟合的组成型广义标度律。在固定 \\(N,D,Q\\) 下，"
        "把来源 \\(i\\) 的份额提高，其他来源按原份额比例缩减；"
        "这是满足 \\(\\sum_i p_i=1\\) 的方向导数。"
        "表中的边际收益为导数乘 0.01 的线性近似，另给出完整提高 1 个百分点的"
        "模型预测差。弹性是来源份额相对提高 1% 时 Loss 的相对变化，"
        "不是份额提高 1 个百分点。正的收益和负的弹性均表示 Loss 降低。",
        "",
        "其中 \\(w^{pQ},w^{clr}\\) 是模型在标准化逆变换后的有效系数。"
        "记 \\(g_{ki}=\\partial L_k/\\partial p_i\\) 为形式偏导，"
        "\\(\\bar w_k^{clr}=17^{-1}\\sum_j w_{kj}^{clr}\\)，则"
        "\\(g_{ki}=S_k[Q_iw_{ki}^{pQ}+(w_{ki}^{clr}-\\bar w_k^{clr})/"
        "(p_i+\\varepsilon)]\\)。可行方向的导数为"
        "\\(m_{ki}=g_{ki}-\\sum_{j\\ne i}p_jg_{kj}/(1-p_i)\\)；"
        "边际收益为 \\(-0.01m_{ki}\\)，弹性为 \\(p_i/L_k\\cdot m_{ki}\\)。",
        "",
        "仅统计初始来源份额在 5%–95% 的真实记录配比；有限变动还需满足"
        "记录到的熵区间与该来源最大份额。这样的支持域筛选仍只是粗检验，"
        "不能证明扰动后的联合配比分布有充分数据覆盖。",
        "",
        "## 质量 Q：独立质量实验",
        "",
        f"在原始全因子实验的 {quality_grid_info['n_points']} 个已观测设计点上"
        "（9 个 N × 5 个 D × 10 个 Q）逐点计算解析导数，"
        f"重算拟合 RMSE 为 {quality_grid_info['reproduced_B7_rmse']:.5f} nats。"
        "该步骤是固定已拟合参数后的网格评估，不是重新搜索最优拟合参数。",
        "",
        "下表固定 N=1B、D=50B token，横向比较不同 Q；完整的 N–D–Q 网格见 CSV。",
        "",
        "| Q | 边际收益 (nats/单位 Q) | 损失弹性 | +0.01 Q 精确收益 (nats) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for q_value in (0.1, 0.3, 0.5, 0.7, 0.9):
        row = next(r for r in quality_grid_rows
                   if np.isclose(r["N_billion_parameters"], 1.0)
                   and np.isclose(r["D_billion_tokens"], 50.0)
                   and np.isclose(r["Q"], q_value))
        lines.append(
            f"| {q_value:.1f} | "
            f"{row['B7_marginal_benefit_nats_per_unit_Q']:.4f} | "
            f"{row['B7_loss_elasticity_wrt_Q']:+.4f} | "
            f"{row['B7_exact_benefit_nats_for_plus_0_01_Q']:.5f} |"
        )
    lines += [
        "",
        f"在 N=1B、D=25B、Q={q0:.3f} 处，B7 的边际收益为 "
        f"{b7['marginal_benefit_nats_per_unit_Q']:.4f} nats/单位 Q；"
        f"Q 增加 0.01 的精确预测收益为 "
        f"{b7['exact_benefit_nats_for_plus_0_01_Q']:.5f} nats，"
        f"弹性为 {b7['loss_elasticity_wrt_Q']:.4f}。"
        f"B6 对应弹性为 {b6['loss_elasticity_wrt_Q']:.4f}，"
        "两组独立质量实验在该参考点给出接近的局部斜率。"
        "这里的 Q=0.622 是 17 个来源质量分数的均值，仅为共同参考点；"
        "其与独立质量实验的 Q 标定一致性仍须另行检验。",
        "",
        "## 配比 p：组成型模型的局部敏感度",
        "",
        f"除全部来源和评测域的分布汇总外，另构造 {len(p_grid_rows)} 个"
        "来源份额分位点评估：每个规模、每个有足够份额的来源，"
        "从真实记录中选取该份额的 10/25/50/75/90 分位附近配比。"
        "因此它展示不同实际配比背景下的局部响应，而非只改变一个来源的纯粹实验。",
        "",
        "下表仅列每个规模锚点中至少有 10 条有效配比、且至少有 10 条"
        "可行 +1pp 扰动的来源。名次表示模型局部响应，不是数据来源的因果质量排序。",
        "",
        "| 规模锚点 | 来源 | 有效/可行条数 | +1pp 精确收益 (nats) | 损失弹性 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for anchor, _ in ANCHORS:
        eligible = [row for row in summary_rows if row["anchor"] == anchor
                    and row["n_active"] >= 10
                    and row["n_feasible_plus_1pp"] >= 10]
        ranked = sorted(eligible,
                        key=lambda row: row["median_exact_gain_nats_for_plus_1pp"],
                        reverse=True)
        for row in ranked[:3]:
            lines.append(
                f"| {anchor} | {row['source']} | {row['n_active']}/"
                f"{row['n_feasible_plus_1pp']} | "
                f"{row['median_exact_gain_nats_for_plus_1pp']:+.4f} | "
                f"{row['median_loss_elasticity_relative_share']:+.4f} |"
            )
    lines += [
        "",
        "完整的 17 来源×13 评测域结果、每项支持度及配比分位点结果见 CSV。"
        "某些来源在不同评测域的边际收益符号不同，不能把各域响应压缩成"
        "一个普适的来源权重。",
        "",
        "## 解释边界",
        "",
        "- 组成型数据里的每个 Q_i 对来源固定。因此广义律的形式导数 "
        "\\(\\partial L/\\partial Q_i\\) 不可解释为已经识别的独立质量干预；"
        "质量弹性取自单独变化 Q 的质量实验。"
        f"在 1M 锚点的记录配比上，广义律形式上的全局 Q 弹性范围为 "
        f"{formal_quality['1M / 1B tokens']['minimum']:+.3f} 至 "
        f"{formal_quality['1M / 1B tokens']['maximum']:+.3f}，"
        "仅作不可辨识性诊断。",
        "- 质量律与配比律的 Loss 标定不同，不能将两张表的绝对边际收益直接"
        "相比或相除，推导所谓质量—配比最优替代率。",
        "- 配比弹性是拟合模型的局部导数，并无直接观测的导数真值；"
        "尤其接近零份额时对 clr 平滑参数敏感，故报告中排除低于 5% 的来源份额。",
        "- 1B 锚点只有 64 条记录配比；60M 到 1B 同时改变了参数量和 token 数，"
        "不能仅凭此比较分别识别两者的作用。",
        "",
        "## 图与复现",
        "",
        "`figures/quality_curves` 展示质量边际收益和弹性沿 Q 的变化；"
        "`figures/quality_heatmap` 展示固定 Q 时 N–D 网格上的弹性；"
        "`figures/mixture_gain_intervals` 展示三个规模的来源收益分布；"
        "`figures/mixture_quantile_slices` 展示真实来源份额分位点上的局部弹性；"
        "`figures/representative_mixtures` 用饼图展示三组真实配比的组成，"
        "饼图本身不表示边际收益；"
        "`figures/representative_mixtures_loss_cylinders` 将相同三组配比的"
        "扇区作为顶面、将整组配比的预测 Loss 映射为共同零点的圆柱高度，"
        "并标注相对低熵配比的预测及实测收益。"
        "圆柱扇区不能解读为各来源独立产生的收益。每图同时提供 PNG 和 SVG。",
        "",
        "运行 `python q2/elasticity_analysis.py` 更新数据与报告，"
        "再运行 `python q2/elasticity_figures.py` 更新图像。",
        "",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines),
                                                   encoding="utf-8")
    print(f"Wrote results to {output_dir}")
    print("Quality at N=1B, D=25B tokens (separate quality law):")
    for name in ("B7", "B6"):
        point = quality[name]["points"][1]
        print(f"  {name} Q={q0:.3f}: benefit per +0.01 Q "
              f"{point['approx_benefit_nats_for_plus_0_01_Q']:.5f} nats, "
              f"elasticity {point['loss_elasticity_wrt_Q']:.4f}")
    print("Mixture: +1 pp source share, other shares reduced proportionally; "
          "positive benefit lowers loss")
    for anchor, _ in ANCHORS:
        eligible = [r for r in summary_rows if r["anchor"] == anchor
                    and r["n_active"] >= 10
                    and r["n_feasible_plus_1pp"] >= 10]
        ranked = sorted(eligible,
                        key=lambda r: r["median_exact_gain_nats_for_plus_1pp"],
                        reverse=True)
        print(f"  {anchor}: {len(eligible)} sources with >=10 active and feasible mixtures")
        for row in ranked[:3]:
            print(f"    {row['source']:<20} exact +1pp gain "
                  f"{row['median_exact_gain_nats_for_plus_1pp']:+.4f} nats, "
                  f"elasticity {row['median_loss_elasticity_relative_share']:+.4f}, "
                  f"n={row['n_feasible_plus_1pp']}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    analyse(args.output_dir)


if __name__ == "__main__":
    main()
