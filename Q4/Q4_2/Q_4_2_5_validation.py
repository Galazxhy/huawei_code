"""Sensitivity, short-horizon backtesting, and matched-sample cross-checks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from q4_utils import CDATA, OUT, ROOT, fit_bridge, robust_slope, rolling_quantile, select_bridge_penalty


def frontier_sensitivity() -> pd.DataFrame:
    sample = pd.read_csv(OUT / "capability_sample_main.csv", parse_dates=["Submission Date"])
    rows = []
    for months in (2, 3, 4):
        for quantile in (0.90, 0.95):
            f = rolling_quantile(sample, "Submission Date", "Y", months, quantile, min_n=5)
            if f.empty:
                continue
            col = f"q{round(quantile * 100)}"
            rows.append(
                {
                    "sample": "main",
                    "months": months,
                    "quantile": quantile,
                    "points": len(f),
                    "first": f[col].iloc[0],
                    "last": f[col].iloc[-1],
                    "net_change": f[col].iloc[-1] - f[col].iloc[0],
                    "monthly_slope": robust_slope(f[col].to_numpy()),
                }
            )
    for name in ("F_frontier_strict_license.csv", "F_frontier_pretrained.csv"):
        f = pd.read_csv(OUT / name)
        rows.append(
            {
                "sample": name.replace("F_frontier_", "").replace(".csv", ""),
                "months": 3,
                "quantile": 0.90,
                "points": len(f),
                "first": f["q90"].iloc[0],
                "last": f["q90"].iloc[-1],
                "net_change": f["q90"].iloc[-1] - f["q90"].iloc[0],
                "monthly_slope": robust_slope(f["q90"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def bridge_family_cv() -> pd.DataFrame:
    c6 = pd.read_csv(CDATA / "loss_benchmark_bridge_expanded.csv")
    penalty, _ = select_bridge_penalty(c6)
    c6["family"] = c6["Model"].str.split("/").str[0]
    rows = []
    medium_data = c6[~c6["Loss_Comparability"].str.contains("High", na=False)]
    for family, test in medium_data.groupby("family"):
        train = c6.drop(index=test.index)
        model = fit_bridge(train, penalty=penalty)
        medium = ~test["Loss_Comparability"].str.contains("High", na=False).to_numpy()
        pred = model.predict(test["Val_Loss"], medium=medium)
        rows.append(
            {
                "held_out_family": family,
                "n": len(test),
                "rmse": float(np.sqrt(np.mean((test["LB_Average"].to_numpy() - pred) ** 2))),
                "mae": float(np.mean(np.abs(test["LB_Average"].to_numpy() - pred))),
                "scheme": "leave_one_medium_family_out",
            }
        )
    high_idx = c6.index[c6["Loss_Comparability"].str.contains("High", na=False)]
    for idx in high_idx:
        train = c6.drop(index=idx)
        test = c6.loc[[idx]]
        model = fit_bridge(train, penalty=penalty)
        pred = model.predict(test["Val_Loss"], medium=False)
        error = float(test["LB_Average"].iloc[0] - pred[0])
        rows.append(
            {
                "held_out_family": f"High_LOO_{idx}",
                "n": 1,
                "rmse": abs(error),
                "mae": abs(error),
                "scheme": "leave_one_high_observation_out",
            }
        )
    return pd.DataFrame(rows)


def rolling_origin_backtest() -> pd.DataFrame:
    series = pd.read_csv(OUT / "mechanism_series.csv", parse_dates=["date"])
    rows = []
    for origin in range(4, len(series) - 1):
        train = series.iloc[: origin + 1]
        drift = robust_slope(train["T"].to_numpy())
        logc_rate = max(robust_slope(train["logC_q90_reported"].tail(5).to_numpy()), 0.0)
        b_slope = robust_slope(train["B"].tail(5).to_numpy())
        direct_slope = robust_slope(train["q90"].to_numpy())
        one_step_change = np.diff(train["q90"].to_numpy())
        methods = {
            "unified_state_model": drift + b_slope,
            "last_value": 0.0,
            "direct_theil_sen": direct_slope,
        }
        for horizon in (1, 2, 3):
            target = origin + horizon
            if target >= len(series):
                continue
            actual = float(series["q90"].iloc[target])
            for method, monthly_change in methods.items():
                pred = float(train["q90"].iloc[-1] + horizon * monthly_change)
                innovation = one_step_change - monthly_change
                sd = max(float(np.std(innovation, ddof=1)), 0.5)
                half = 1.96 * sd * np.sqrt(horizon)
                rows.append(
                    {
                        "origin": train["date"].iloc[-1],
                        "horizon_months": horizon,
                        "model": method,
                        "prediction": pred,
                        "actual": actual,
                        "error": pred - actual,
                        "lower_95": pred - half,
                        "upper_95": pred + half,
                        "interval_width": 2 * half,
                        "covered_95": pred - half <= actual <= pred + half,
                        "compute_rate_diagnostic": logc_rate,
                    }
                )
    return pd.DataFrame(rows)


def matched_crosscheck() -> dict:
    match = pd.read_csv(ROOT.parent / "Q4_1" / "results" / "C1_C4_entity_match.csv")
    score = pd.read_csv(CDATA / "leaderboard_cleaned.csv")
    score["Y"] = score[["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]].mean(axis=1)
    reliable = match[match["match_level"].isin(["high", "medium"])].merge(
        score[["Model", "Y", "Submission Date", "Type"]], left_on="model", right_on="Model", how="left"
    )
    reliable["c4_compute"] = pd.to_numeric(reliable["c4_compute"], errors="coerce")
    usable = reliable.dropna(subset=["c4_compute", "Y"])
    rho = spearmanr(np.log10(usable["c4_compute"]), usable["Y"]) if len(usable) >= 3 else None
    usable = usable.rename(columns={"model": "c1_model", "Model": "leaderboard_model"})
    usable.to_csv(OUT / "matched_sample_crosscheck.csv", index=False, encoding="utf-8-sig")
    return {
        "reliable_matches": int(len(reliable)),
        "with_compute_and_score": int(len(usable)),
        "spearman_log_compute_score": float(rho.statistic) if rho else None,
        "pvalue": float(rho.pvalue) if rho else None,
        "role": "directional cross-check only; fuzzy entity matches do not identify causal effects",
    }


def main() -> None:
    sensitivity = frontier_sensitivity()
    sensitivity.to_csv(OUT / "frontier_sensitivity.csv", index=False, encoding="utf-8-sig")
    bridge_cv = bridge_family_cv()
    bridge_cv.to_csv(OUT / "bridge_family_cv.csv", index=False, encoding="utf-8-sig")
    backtest = rolling_origin_backtest()
    backtest.to_csv(OUT / "rolling_origin_backtest.csv", index=False, encoding="utf-8-sig")
    comparison = (
        backtest.groupby(["model", "horizon_months"])
        .agg(
            n=("error", "size"),
            mae=("error", lambda x: np.mean(np.abs(x))),
            coverage_95=("covered_95", "mean"),
            mean_interval_width=("interval_width", "mean"),
        )
        .reset_index()
    )
    overall = (
        backtest.groupby("model")
        .agg(
            n=("error", "size"),
            mae=("error", lambda x: np.mean(np.abs(x))),
            coverage_95=("covered_95", "mean"),
            mean_interval_width=("interval_width", "mean"),
        )
        .reset_index()
    )
    overall["horizon_months"] = "overall"
    comparison = pd.concat([comparison, overall], ignore_index=True)
    comparison.to_csv(OUT / "forecast_baseline_comparison.csv", index=False, encoding="utf-8-sig")
    matched = matched_crosscheck()
    main = sensitivity[(sensitivity["sample"] == "main") & (sensitivity["months"] == 3) & (sensitivity["quantile"] == 0.90)].iloc[0]
    summary = {
        "main_frontier_net_change": float(main["net_change"]),
        "sensitivity_same_positive_direction": bool((sensitivity["net_change"] > 0).mean() >= 0.75),
        "sensitivity_positive_fraction": float((sensitivity["net_change"] > 0).mean()),
        "bridge_family_cv_weighted_rmse": float(np.sqrt(np.average(bridge_cv["rmse"] ** 2, weights=bridge_cv["n"]))),
        "backtest_predictions": int(len(backtest)),
        "forecast_baseline_comparison": comparison.to_dict(orient="records"),
        "matched_crosscheck": matched,
    }
    (OUT / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
