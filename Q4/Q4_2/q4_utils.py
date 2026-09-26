"""Shared helpers for the Q4 capability-frontier state model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CDATA = ROOT.parents[1] / "data" / "real_attachments" / "C_efficiency_evolution"
Q3_RESULT = ROOT.parents[1] / "results" / "Q3" / "result_q3_cost_allocation.csv"

ANALYSIS_END = pd.Timestamp("2025-03-31 23:59:59.999999")
LEVEL_A_C = 2.15e22
LEVEL_B_C = 1e24
#: The three budgets the problem specifies as anchor points.  Problem 3 scans a
#: wider grid; ``Q3/Q3_3/write_q4_anchors.py`` reads this constant so the two
#: sides cannot drift apart.
Q3_ANCHOR_BUDGETS = (1e19, 1e22, 1e24)
BENCH = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
MAIN_TYPES = ["💬 chat models (RLHF, DPO, IFT, ...)", "🔶 fine-tuned on domain-specific datasets"]
PRETRAIN_TYPES = ["🟢 pretrained", "🟩 continuously pretrained"]
FAMILY_PATTERN = re.compile(
    r"(llama|qwen|mistral|mixtral|gemma|phi|deepseek|yi|falcon|pythia|olmo|command-r|rwkv|mamba|internlm|baichuan|glm|bloom|opt)",
    flags=re.I,
)


def model_family(name: str) -> str:
    value = str(name).lower()
    match = FAMILY_PATTERN.search(value)
    if match:
        return match.group(1).lower()
    return value.split("/", 1)[0]


def load_q3_anchors(cost_form: str = "exponential") -> list[tuple[float, float]]:
    q3 = pd.read_csv(Q3_RESULT)
    keep = q3[
        q3["scenario"].eq("main")
        & q3["context"].astype(float).eq(4096)
        & q3["cost_form"].eq(cost_form)
        & q3["mode"].eq("joint_reduced_hull")
        & q3["bridge"].eq("ratio")
        & q3["kappa"].astype(float).eq(1.0)
    ].copy()
    keep["budget"] = pd.to_numeric(keep["budget"])
    keep["loss"] = pd.to_numeric(keep["loss"])
    keep = keep.sort_values("budget")
    if keep.shape[0] != len(Q3_ANCHOR_BUDGETS) or not np.allclose(keep["budget"], Q3_ANCHOR_BUDGETS):
        raise ValueError(
            "Q3 main anchors are incomplete or use unexpected budgets; "
            f"expected {list(Q3_ANCHOR_BUDGETS)} (scenario=main, context=4096, "
            f"cost_form={cost_form}, mode=joint_reduced_hull, bridge=ratio, kappa=1.0)"
        )
    return list(zip(keep["budget"].astype(float), keep["loss"].astype(float)))


Q3_ANCHORS = load_q3_anchors()


def evidence_level(compute: float) -> str:
    if compute <= LEVEL_A_C:
        return "A_supported"
    if compute <= LEVEL_B_C:
        return "B_shape_extrapolation"
    return "C_beyond_Q3_support"


def log_interp_lstar(
    compute: float | np.ndarray,
    anchors: list[tuple[float, float]] | None = None,
    allow_boundary_extrapolation: bool = False,
) -> float | np.ndarray:
    """Log-log interpolation; optional continuation uses the nearest segment slope."""
    anchors = anchors or Q3_ANCHORS
    xs = np.log10([x for x, _ in anchors])
    ys = np.log10([y for _, y in anchors])
    values = np.asarray(compute, dtype=float)
    x = np.atleast_1d(np.log10(values))
    y = np.interp(x, xs, ys)
    if allow_boundary_extrapolation:
        left = x < xs[0]
        right = x > xs[-1]
        y[left] = ys[0] + (x[left] - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        y[right] = ys[-1] + (x[right] - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    else:
        y[(x < xs[0]) | (x > xs[-1])] = np.nan
    result = 10**y
    return float(result[0]) if np.ndim(compute) == 0 else result


@dataclass
class BridgeFit:
    knots: np.ndarray
    coef: np.ndarray
    penalty: float
    sigma_high: float
    sigma_medium: float
    sigma_high_raw: float
    sigma_medium_raw: float
    variance_constraint_active: bool
    iterations: int

    @property
    def min_loss(self) -> float:
        return float(self.knots[0])

    def _design(self, loss: np.ndarray, medium: np.ndarray) -> np.ndarray:
        hinge = np.maximum(self.knots[None, :] - loss[:, None], 0.0)
        return np.column_stack([np.ones(len(loss)), medium.astype(float), hinge])

    def predict(
        self,
        loss: float | np.ndarray,
        medium: bool | np.ndarray = False,
        tail_fraction: float = 1.0,
    ) -> np.ndarray:
        values = np.atleast_1d(loss).astype(float)
        medium_values = np.broadcast_to(np.asarray(medium, dtype=bool), values.shape)
        full = self._design(values, medium_values) @ self.coef
        below = values < self.min_loss
        if tail_fraction < 1.0 and below.any():
            boundary = self._design(np.full(below.sum(), self.min_loss), medium_values[below]) @ self.coef
            full[below] = boundary + tail_fraction * (full[below] - boundary)
        return np.clip(full, 0.0, 100.0)


def fit_bridge(
    c6: pd.DataFrame,
    penalty: float = 1.0,
    knots: np.ndarray | None = None,
    max_iter: int = 30,
    tol: float = 1e-7,
) -> BridgeFit:
    """Heteroskedastic monotone hinge spline with a comparability-level intercept.

    The shared shape identifies score contrasts. The Medium intercept absorbs the
    systematic level mismatch between same-validation-set and cross-source losses.
    """
    data = c6.dropna(subset=["Val_Loss", "LB_Average", "Loss_Comparability"]).copy()
    x = data["Val_Loss"].to_numpy(float)
    y = data["LB_Average"].to_numpy(float)
    high = data["Loss_Comparability"].str.contains("High", na=False).to_numpy()
    medium = ~high
    if knots is None:
        knots = np.unique(np.quantile(x, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    knots = np.asarray(knots, dtype=float)
    hinge = np.maximum(knots[None, :] - x[:, None], 0.0)
    design = np.column_stack([np.ones(len(x)), medium.astype(float), hinge])
    n_coef = design.shape[1]
    bounds_low = np.r_[-np.inf, -np.inf, np.zeros(len(knots))]
    bounds_high = np.full(n_coef, np.inf)
    # First differences regularize slope changes while preserving beta >= 0.
    penalty_matrix = np.zeros((max(len(knots) - 1, 1), n_coef))
    if len(knots) > 1:
        for j in range(len(knots) - 1):
            penalty_matrix[j, 2 + j] = 1.0
            penalty_matrix[j, 2 + j + 1] = -1.0
    weights = np.ones(len(data), dtype=float)
    old = np.full(len(data), np.nan)
    sigma_h_raw = sigma_m_raw = sigma_h = sigma_m = 1.0
    for iteration in range(1, max_iter + 1):
        root_w = np.sqrt(weights)
        aug_x = design * root_w[:, None]
        aug_y = y * root_w
        if penalty > 0 and len(knots) > 1:
            aug_x = np.vstack([aug_x, np.sqrt(penalty) * penalty_matrix])
            aug_y = np.r_[aug_y, np.zeros(penalty_matrix.shape[0])]
        coef = lsq_linear(aug_x, aug_y, bounds=(bounds_low, bounds_high), lsmr_tol="auto").x
        pred = design @ coef
        resid = y - pred
        sigma_all = max(float(np.std(resid, ddof=1)), 1e-3)
        sigma_h_raw = max(float(np.std(resid[high], ddof=1)), 1e-3) if high.sum() >= 2 else sigma_all
        sigma_m_raw = max(float(np.std(resid[~high], ddof=1)), 1e-3) if (~high).sum() >= 2 else sigma_all
        # Small High strata otherwise collapse to near-zero variance and dominate FGLS.
        # Five pooled prior degrees of freedom give an empirical-Bayes variance floor.
        prior_df = 5.0
        sigma_h = float(np.sqrt(((max(high.sum() - 1, 0)) * sigma_h_raw**2 + prior_df * sigma_all**2) / (max(high.sum() - 1, 0) + prior_df)))
        sigma_m_shrunk = float(np.sqrt(((max((~high).sum() - 1, 0)) * sigma_m_raw**2 + prior_df * sigma_all**2) / (max((~high).sum() - 1, 0) + prior_df)))
        sigma_m = max(sigma_m_shrunk, sigma_h)
        weights = np.where(high, 1.0 / sigma_h**2, 1.0 / sigma_m**2)
        weights /= weights.mean()
        if np.all(np.isfinite(old)) and np.max(np.abs(pred - old)) < tol:
            break
        old = pred
    return BridgeFit(
        knots=knots,
        coef=coef,
        penalty=penalty,
        sigma_high=sigma_h,
        sigma_medium=sigma_m,
        sigma_high_raw=sigma_h_raw,
        sigma_medium_raw=sigma_m_raw,
        variance_constraint_active=sigma_m_raw < sigma_h_raw,
        iterations=iteration,
    )


def select_bridge_penalty(c6: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Select spline roughness by leave-one-model-family-out prediction."""
    data = c6.copy()
    data["family"] = data["Model"].str.split("/").str[0]
    global_knots = np.unique(np.quantile(data["Val_Loss"], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    rows = []
    for penalty in [0.01, 0.1, 1.0, 10.0, 100.0]:
        squared = []
        absolute = []
        medium_data = data[~data["Loss_Comparability"].str.contains("High", na=False)]
        for family, test in medium_data.groupby("family"):
            train = data.drop(index=test.index)
            fit = fit_bridge(train, penalty=penalty, knots=global_knots)
            medium = ~test["Loss_Comparability"].str.contains("High", na=False).to_numpy()
            error = test["LB_Average"].to_numpy() - fit.predict(test["Val_Loss"], medium=medium)
            squared.extend(error**2)
            absolute.extend(np.abs(error))
        high_idx = data.index[data["Loss_Comparability"].str.contains("High", na=False)]
        for idx in high_idx:
            train = data.drop(index=idx)
            test = data.loc[[idx]]
            fit = fit_bridge(train, penalty=penalty, knots=global_knots)
            error = test["LB_Average"].to_numpy() - fit.predict(test["Val_Loss"], medium=False)
            squared.extend(error**2)
            absolute.extend(np.abs(error))
        rows.append(
            {
                "penalty": penalty,
                "n": len(squared),
                "rmse": float(np.sqrt(np.mean(squared))),
                "mae": float(np.mean(absolute)),
            }
        )
    scores = pd.DataFrame(rows)
    selected = float(scores.sort_values(["rmse", "penalty"]).iloc[0]["penalty"])
    return selected, scores


def month_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.period_range(start.to_period("M"), end.to_period("M"), freq="M").end_time


def rolling_quantile(
    frame: pd.DataFrame,
    date_col: str,
    value_col: str,
    months: int,
    q: float,
    min_n: int,
    end: pd.Timestamp = ANALYSIS_END,
) -> pd.DataFrame:
    df = frame[[date_col, value_col]].dropna().copy()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df["date"].notna() & df["date"].le(end)]
    rows: list[dict] = []
    for window_end in month_grid(df["date"].min(), end):
        window_start = window_end - pd.DateOffset(months=months)
        vals = df.loc[(df["date"] > window_start) & (df["date"] <= window_end), value_col]
        if len(vals) >= min_n:
            rows.append(
                {"date": window_end, "n": len(vals), f"q{round(q * 100)}": float(np.quantile(vals, q))}
            )
    return pd.DataFrame(rows)


def robust_slope(y: np.ndarray, x: np.ndarray | None = None) -> float:
    """Theil median pairwise slope without an additional dependency."""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float)
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(len(y) - 1)
        for j in range(i + 1, len(y))
        if x[j] != x[i] and np.isfinite(y[i]) and np.isfinite(y[j])
    ]
    return float(np.median(slopes)) if slopes else np.nan


def load_frontier(name: str = "F_frontier_main.csv") -> pd.DataFrame:
    return pd.read_csv(OUT / name, parse_dates=["date"])


def load_compute_frontier() -> pd.DataFrame:
    return pd.read_csv(OUT / "C_frontier.csv", parse_dates=["date"])
