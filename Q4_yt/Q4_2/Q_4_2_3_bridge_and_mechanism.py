"""Fit the stratified Loss-Benchmark bridge and historical state decomposition."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from q4_utils import (
    CDATA,
    OUT,
    evidence_level,
    fit_bridge,
    load_q3_anchors,
    load_compute_frontier,
    load_frontier,
    log_interp_lstar,
    select_bridge_penalty,
)


def bridge_cv_splits(c6: pd.DataFrame):
    data = c6.copy()
    data["family"] = data["Model"].str.split("/").str[0]
    medium = ~data["Loss_Comparability"].str.contains("High", na=False)
    for family, test in data[medium].groupby("family"):
        yield f"Medium_family:{family}", data.drop(index=test.index), test
    for idx in data.index[~medium]:
        yield f"High_LOO:{idx}", data.drop(index=idx), data.loc[[idx]]


def linear_prediction(train: pd.DataFrame, test: pd.DataFrame, high_only: bool = False) -> np.ndarray:
    fit_data = train[train["Loss_Comparability"].str.contains("High", na=False)] if high_only else train
    slope, intercept = np.polyfit(fit_data["Val_Loss"], fit_data["LB_Average"], 1)
    return np.clip(intercept + slope * test["Val_Loss"].to_numpy(float), 0.0, 100.0)


def compare_bridge_models(c6: pd.DataFrame, penalty: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    for split, train, test in bridge_cv_splits(c6):
        medium_test = ~test["Loss_Comparability"].str.contains("High", na=False).to_numpy()
        main = fit_bridge(train, penalty=penalty)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0.0, y_max=100.0)
        iso.fit(train["Val_Loss"], train["LB_Average"])
        model_predictions = {
            "pooled_linear": linear_prediction(train, test),
            "high_only_linear": linear_prediction(train, test, high_only=True),
            "uncalibrated_isotonic": iso.predict(test["Val_Loss"]),
            "source_calibrated_monotone_spline": main.predict(test["Val_Loss"], medium=medium_test),
        }
        for model, pred in model_predictions.items():
            for position, (idx, row) in enumerate(test.iterrows()):
                predictions.append(
                    {
                        "split": split,
                        "row_index": idx,
                        "model_name": row["Model"],
                        "comparability": row["Loss_Comparability"],
                        "bridge": model,
                        "actual": float(row["LB_Average"]),
                        "prediction": float(pred[position]),
                    }
                )
    pred = pd.DataFrame(predictions)
    pred["error"] = pred["prediction"] - pred["actual"]
    metrics = (
        pred.groupby("bridge")
        .agg(n=("error", "size"), rmse=("error", lambda x: np.sqrt(np.mean(x**2))), mae=("error", lambda x: np.mean(np.abs(x))))
        .reset_index()
        .sort_values("rmse")
    )
    return pred, metrics


def pythia_leave_one_sensitivity(
    c6: pd.DataFrame, penalty: float, compute0: float, compute1: float, delta_f: float
) -> pd.DataFrame:
    high_indices = c6.index[c6["Loss_Comparability"].str.contains("High", na=False)]
    anchors = load_q3_anchors("exponential")
    losses = log_interp_lstar(
        np.array([compute0, compute1]), anchors=anchors, allow_boundary_extrapolation=True
    )
    rows = []
    for idx in [None, *high_indices.tolist()]:
        train = c6 if idx is None else c6.drop(index=idx)
        fit = fit_bridge(train, penalty=penalty)
        b = fit.predict(losses, tail_fraction=1.0)
        scale_gain = max(float(b[1] - b[0]), 0.0)
        rows.append(
            {
                "omitted": "none" if idx is None else str(c6.loc[idx, "Model"]),
                "tail_slope_score_per_loss": float(-fit.coef[2:].sum()),
                "scale_gain_upper": scale_gain,
                "scale_share_upper": scale_gain / delta_f if delta_f > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    c6 = pd.read_csv(CDATA / "loss_benchmark_bridge_expanded.csv")
    penalty, penalty_cv = select_bridge_penalty(c6)
    penalty_cv.to_csv(OUT / "bridge_penalty_cv.csv", index=False, encoding="utf-8-sig")
    bridge = fit_bridge(c6, penalty=penalty)
    bridge_cv_predictions, bridge_comparison = compare_bridge_models(c6, penalty)
    bridge_cv_predictions.to_csv(OUT / "bridge_model_cv_predictions.csv", index=False, encoding="utf-8-sig")
    bridge_comparison.to_csv(OUT / "bridge_model_comparison.csv", index=False, encoding="utf-8-sig")
    fitted = c6.copy()
    fitted_medium = ~fitted["Loss_Comparability"].str.contains("High", na=False).to_numpy()
    fitted["H"] = bridge.predict(fitted["Val_Loss"], medium=fitted_medium)
    fitted["resid"] = fitted["LB_Average"] - fitted["H"]
    fitted.to_csv(OUT / "C6_bridge_fit.csv", index=False, encoding="utf-8-sig")

    grid = np.linspace(c6["Val_Loss"].min(), c6["Val_Loss"].max(), 300)
    pd.DataFrame({"loss": grid, "benchmark": bridge.predict(grid)}).to_csv(
        OUT / "C6_bridge_curve.csv", index=False, encoding="utf-8-sig"
    )

    frontier = load_frontier()
    compute = load_compute_frontier().dropna(subset=["C_q90_reported"])
    merged = pd.merge_asof(
        frontier.sort_values("date"),
        compute.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=62),
    ).dropna(subset=["C_q90_reported"])
    merged["L_star"] = log_interp_lstar(
        merged["C_q90_reported"].to_numpy(), allow_boundary_extrapolation=True
    )
    # The absolute source intercept cancels in contrasts. Outside C6 support,
    # saturation and full tail continuation define the partial-identification set.
    merged["B"] = bridge.predict(merged["L_star"], tail_fraction=0.0)
    merged["B_tail"] = bridge.predict(merged["L_star"], tail_fraction=1.0)
    merged["evidence"] = merged["C_q90_reported"].map(evidence_level)
    t0 = merged.iloc[0]
    merged["delta_F"] = merged["q90"] - t0["q90"]
    merged["delta_B"] = merged["B"] - t0["B"]
    merged["T"] = merged["delta_F"] - merged["delta_B"]
    merged["delta_B_tail"] = merged["B_tail"] - t0["B_tail"]
    merged["T_tail"] = merged["delta_F"] - merged["delta_B_tail"]
    merged.to_csv(OUT / "mechanism_series.csv", index=False, encoding="utf-8-sig")

    last = merged.iloc[-1]
    delta_f = float(last["delta_F"])
    delta_scale = float(last["delta_B"])
    if abs(delta_scale) < 1e-10:
        delta_scale = 0.0
    delta_tech = float(last["T"])
    delta_scale_tail = float(last["delta_B_tail"])
    delta_tech_tail = float(last["T_tail"])
    support_valid = bool(
        merged["C_q90_reported"].le(1e24).all()
        and merged["L_star"].between(c6["Val_Loss"].min(), c6["Val_Loss"].max()).all()
    )
    valid_shares = support_valid and delta_f > 0 and delta_scale >= 0 and delta_tech >= 0
    high = c6["Loss_Comparability"].str.contains("High", na=False)
    pythia_sensitivity = pythia_leave_one_sensitivity(
        c6, penalty, float(t0["C_q90_reported"]), float(last["C_q90_reported"]), delta_f
    )
    pythia_sensitivity.to_csv(OUT / "bridge_pythia_leave_one.csv", index=False, encoding="utf-8-sig")

    # Bootstrap the upper endpoint of the partially identified scale share.
    rng = np.random.default_rng(20260924)
    capability = pd.read_csv(OUT / "capability_sample_main.csv", parse_dates=["Submission Date"])
    compute_sample = pd.read_csv(OUT / "compute_sample.csv", parse_dates=["date"])
    anchor_sets = [load_q3_anchors(name) for name in ["exponential", "power", "log_asymptotic"]]
    contribution_draws = []
    for draw in range(1000):
        boot_c6 = pd.concat(
            [group.iloc[rng.integers(0, len(group), len(group))] for _, group in c6.groupby(high)],
            ignore_index=True,
        )
        boot_bridge = fit_bridge(boot_c6, penalty=penalty)
        anchors = anchor_sets[int(rng.integers(0, len(anchor_sets)))]
        endpoint_values = []
        endpoint_computes = []
        for endpoint in [pd.Timestamp(t0["date"]), pd.Timestamp(last["date"])]:
            scores = capability[
                (capability["Submission Date"] > endpoint - pd.DateOffset(months=3))
                & (capability["Submission Date"] <= endpoint)
            ]["Y"].dropna().to_numpy(float)
            computes = compute_sample[
                (compute_sample["date"] > endpoint - pd.DateOffset(months=3))
                & (compute_sample["date"] <= endpoint)
                & compute_sample["compute_source"].eq("reported")
                & compute_sample["Confidence"].isin(["Confident", "Likely"])
            ]["reported_compute"].dropna().to_numpy(float)
            endpoint_values.append(float(np.quantile(rng.choice(scores, len(scores), replace=True), 0.90)))
            endpoint_computes.append(
                float(10 ** np.quantile(np.log10(rng.choice(computes, len(computes), replace=True)), 0.90))
            )
        losses = log_interp_lstar(
            np.asarray(endpoint_computes), anchors=anchors, allow_boundary_extrapolation=True
        )
        full_b = boot_bridge.predict(losses, tail_fraction=1.0)
        draw_df = endpoint_values[1] - endpoint_values[0]
        upper_scale = max(float(full_b[1] - full_b[0]), 0.0)
        contribution_draws.append(
            {
                "draw": draw,
                "delta_F": draw_df,
                "scale_gain_lower": 0.0,
                "scale_gain_upper": upper_scale,
                "scale_share_upper": upper_scale / draw_df if draw_df > 0 else np.nan,
            }
        )
    contribution_draws = pd.DataFrame(contribution_draws)
    contribution_draws.to_csv(OUT / "contribution_identified_set_bootstrap.csv", index=False, encoding="utf-8-sig")
    share_quantiles = contribution_draws["scale_share_upper"].dropna().quantile([0.025, 0.5, 0.975])
    summary = {
        "bridge_model": "heteroskedastic monotone hinge spline with Medium-level intercept",
        "bridge_penalty": penalty,
        "bridge_medium_intercept": float(bridge.coef[1]),
        "bridge_cv_comparison": bridge_comparison.to_dict(orient="records"),
        "pythia_leave_one_scale_share_upper_range": [
            float(pythia_sensitivity["scale_share_upper"].min()),
            float(pythia_sensitivity["scale_share_upper"].max()),
        ],
        "bridge_n_high": int(high.sum()),
        "bridge_n_medium": int((~high).sum()),
        "bridge_spearman_all": float(spearmanr(c6["Val_Loss"], c6["LB_Average"]).statistic),
        "bridge_sigma_high": bridge.sigma_high,
        "bridge_sigma_medium": bridge.sigma_medium,
        "bridge_sigma_high_raw": bridge.sigma_high_raw,
        "bridge_sigma_medium_raw": bridge.sigma_medium_raw,
        "variance_constraint_active": bridge.variance_constraint_active,
        "t0": str(t0["date"]),
        "t1": str(last["date"]),
        "F0": float(t0["q90"]),
        "F1": float(last["q90"]),
        "C0": float(t0["C_q90_reported"]),
        "C1": float(last["C_q90_reported"]),
        "B0": float(t0["B"]),
        "B1": float(last["B"]),
        "delta_F": delta_f,
        "delta_F_scale": delta_scale,
        "delta_F_tech": delta_tech,
        "delta_F_scale_tail_sensitivity": delta_scale_tail,
        "delta_F_tech_tail_sensitivity": delta_tech_tail,
        "scale_share_sensitivity_range": [
            float(min(delta_scale, delta_scale_tail) / delta_f),
            float(max(delta_scale, delta_scale_tail) / delta_f),
        ] if delta_f > 0 and min(delta_scale, delta_scale_tail) >= 0 else None,
        "scale_share_upper_bootstrap": {
            "q2_5": float(share_quantiles.loc[0.025]),
            "median": float(share_quantiles.loc[0.5]),
            "q97_5": float(share_quantiles.loc[0.975]),
            "draws": 1000,
            "probability_below_10pct": float((contribution_draws["scale_share_upper"] < 0.10).mean()),
            "probability_below_25pct": float((contribution_draws["scale_share_upper"] < 0.25).mean()),
            "probability_below_50pct": float((contribution_draws["scale_share_upper"] < 0.50).mean()),
        },
        "shares_reportable": valid_shares,
        "scale_share": float(delta_scale / delta_f) if valid_shares else None,
        "tech_share": float(delta_tech / delta_f) if valid_shares else None,
        "evidence_t0": t0["evidence"],
        "evidence_t1": last["evidence"],
        "aligned_points": int(len(merged)),
        "warning": "Point contribution shares are not identified: compute exceeds Q3 support and L* is below C6 support. The tail result is a labeled sensitivity bound.",
    }
    (OUT / "decomposition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
