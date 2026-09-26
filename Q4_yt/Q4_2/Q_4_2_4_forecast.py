"""Forecast the 12-month frontier and an exploratory 24-month extension."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from q4_utils import (
    CDATA,
    OUT,
    evidence_level,
    fit_bridge,
    load_q3_anchors,
    log_interp_lstar,
    model_family,
    robust_slope,
)


SEED = 20260924
N_DRAWS = 2000


def extrapolated_lstar(compute: float, anchors: list[tuple[float, float]], slope_fraction: float) -> float:
    if compute <= anchors[-1][0]:
        return float(log_interp_lstar(compute, anchors=anchors))
    x1, y1 = np.log10(anchors[-2])
    x2, y2 = np.log10(anchors[-1])
    slope = (y2 - y1) / (x2 - x1)
    return float(10 ** (y2 + slope_fraction * slope * (np.log10(compute) - x2)))


def bridge_with_tail(bridge, loss: float, tail_fraction: float) -> float:
    return float(bridge.predict(loss, medium=False, tail_fraction=tail_fraction)[0])


def c3_long_run_check() -> dict:
    c3 = pd.read_csv(CDATA / "leaderboard_extended_timeseries.csv")
    hist = c3[c3["Source"].eq("Historical (papers/reports)")].copy()
    hist["Average"] = pd.to_numeric(hist["Average"], errors="coerce")
    yearly = hist.groupby("Year").agg(n=("Average", "size"), frontier_q90=("Average", lambda x: x.quantile(0.9))).reset_index()
    yearly.to_csv(OUT / "C3_historical_frontier.csv", index=False, encoding="utf-8-sig")
    return {
        "records": int(len(hist)),
        "years": int(len(yearly)),
        "q90_slope_points_per_year": robust_slope(yearly["frontier_q90"].to_numpy(), yearly["Year"].to_numpy()),
        "role": "low-density order-of-magnitude check only; benchmark coverage changes by year",
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    mechanism = pd.read_csv(OUT / "mechanism_series.csv", parse_dates=["date"])
    c6 = pd.read_csv(CDATA / "loss_benchmark_bridge_expanded.csv")
    decomposition = json.loads((OUT / "decomposition_summary.json").read_text(encoding="utf-8"))
    bridge_penalty = float(decomposition["bridge_penalty"])
    bridge = fit_bridge(c6, penalty=bridge_penalty)
    q3_cost_forms = [load_q3_anchors(name) for name in ["exponential", "power", "log_asymptotic"]]
    compute_summary = json.loads((OUT / "compute_frontier_summary.json").read_text(encoding="utf-8"))
    sample = pd.read_csv(OUT / "capability_sample_main.csv", parse_dates=["Submission Date"])
    compute_sample = pd.read_csv(OUT / "compute_sample.csv", parse_dates=["date"])

    last = mechanism.iloc[-1]
    f_last = float(last["q90"])
    c_last = float(last["C_q90_reported"])
    recent = max(float(compute_summary["slope_recent_log10_per_month"]), 0.0)
    early = max(float(compute_summary["slope_early_log10_per_month"]), 0.0)
    slow_rate = min(recent, early)
    compute_paths = {
        "compute_platform": 0.0,
        "no_slowdown_counterfactual": max(recent, early),
    }
    t_increments = np.diff(mechanism["T"].to_numpy(float))
    tech_drift = robust_slope(mechanism["T"].to_numpy())
    tech_noise = max(float(np.std(t_increments - tech_drift, ddof=1)), 0.2)
    anchors_main = q3_cost_forms[0]
    b_last = bridge_with_tail(bridge, extrapolated_lstar(c_last, anchors_main, 1.0), 0.5)

    technology_paths = {"current_tech_drift": tech_drift, "half_tech_drift": 0.5 * tech_drift}
    rows = []
    for compute_path, rate in compute_paths.items():
        for technology_path, drift in technology_paths.items():
            for horizon in (12, 24):
                scenario = f"{compute_path}+{technology_path}"
                c_future = c_last * 10 ** (rate * horizon)
                l_future = extrapolated_lstar(c_future, anchors_main, 1.0)
                b_future = bridge_with_tail(bridge, l_future, 0.5)
                point = float(np.clip(f_last + (b_future - b_last) + drift * horizon, 0, 100))
                is_primary = compute_path == "compute_platform" and technology_path == "current_tech_drift" and horizon == 12
                rows.append(
                    {
                        "scenario": scenario,
                        "compute_path": compute_path,
                        "technology_path": technology_path,
                        "horizon_months": horizon,
                        "forecast": point,
                        "compute": c_future,
                        "loss_star": l_future,
                        "scale_gain": b_future - b_last,
                        "tech_gain": drift * horizon,
                        "evidence": evidence_level(c_future),
                        "status": "primary" if is_primary else "exploratory",
                    }
                )

    # Conditional end-to-end simulation. Beyond 1e24, the terminal Q3 slope is varied
    # from saturation (0) to full continuation (1) rather than treated as known.
    ability_window = sample[
        (sample["Submission Date"] > last["date"] - pd.DateOffset(months=3))
        & (sample["Submission Date"] <= last["date"])
    ].dropna(subset=["Y"]).copy()
    ability_window["model_family"] = ability_window["Model"].map(model_family)
    ability_window["month"] = ability_window["Submission Date"].dt.to_period("M")
    ability_blocks = [
        group["Y"].to_numpy(float)
        for _, group in ability_window.groupby(["month", "model_family"])
    ]
    compute_window = compute_sample[
        (compute_sample["date"] > last["date"] - pd.DateOffset(months=3))
        & (compute_sample["date"] <= last["date"])
        & compute_sample["compute_source"].eq("reported")
        & compute_sample["Confidence"].isin(["Confident", "Likely"])
    ]["reported_compute"].dropna().to_numpy(float)
    frontier_log = mechanism["logC_q90_reported"].dropna().to_numpy(float)
    high = c6["Loss_Comparability"].str.contains("High", na=False)
    groups = [c6[high], c6[~high]]
    draws = []
    oat_draws = []
    fixed_sources = [
        "ability_window",
        "model_family",
        "compute_window",
        "Q3_form",
        "bridge_sample",
        "tail_slope",
        "technology_drift",
    ]

    def evaluate(
        horizon: int,
        f0: float,
        c0: float,
        rate: float,
        bridge_fit,
        anchors,
        terminal_fraction: float,
        tail_fraction: float,
        drift: float,
        innovation: float,
    ) -> tuple[float, float, float, float]:
        start_l = extrapolated_lstar(c0, anchors, terminal_fraction)
        start_b = bridge_with_tail(bridge_fit, start_l, tail_fraction)
        future_c = c0 * 10 ** (rate * horizon)
        future_l = extrapolated_lstar(future_c, anchors, terminal_fraction)
        future_b = bridge_with_tail(bridge_fit, future_l, tail_fraction)
        scale_gain = future_b - start_b
        tech_gain = drift * horizon + innovation
        forecast = float(np.clip(f0 + scale_gain + tech_gain, 0, 100))
        return forecast, scale_gain, tech_gain, future_c

    for draw in range(N_DRAWS):
        chosen_blocks = rng.integers(0, len(ability_blocks), len(ability_blocks))
        sampled_selected = [
            rng.choice(ability_blocks[idx], len(ability_blocks[idx]), replace=True)
            for idx in chosen_blocks
        ]
        sampled_all = [rng.choice(block, len(block), replace=True) for block in ability_blocks]
        f0 = float(np.quantile(np.concatenate(sampled_selected), 0.90))
        f0_fixed_ability = float(
            np.quantile(np.concatenate([ability_blocks[idx] for idx in chosen_blocks]), 0.90)
        )
        f0_fixed_family = float(np.quantile(np.concatenate(sampled_all), 0.90))
        c0 = float(
            10 ** np.quantile(np.log10(rng.choice(compute_window, len(compute_window), replace=True)), 0.90)
        )
        boot_c6 = pd.concat(
            [g.iloc[rng.integers(0, len(g), len(g))] for g in groups], ignore_index=True
        )
        boot_bridge = fit_bridge(boot_c6, penalty=bridge_penalty)
        anchors = q3_cost_forms[int(rng.integers(0, len(q3_cost_forms)))]
        terminal_fraction = float(rng.uniform(0.0, 1.0))
        tail_fraction = float(rng.uniform(0.0, 1.0))
        block = rng.choice(np.diff(frontier_log[-7:]), 6, replace=True)
        sampled_rate = max(float(np.median(block)), 0.0)
        rate = min(sampled_rate, early)
        sampled_drift = float(rng.normal(tech_drift, max(tech_noise / np.sqrt(len(mechanism)), 0.1)))
        for horizon in (12, 24):
            rw_noise = float(rng.normal(0.0, tech_noise * np.sqrt(horizon)))
            value, scale_gain, tech_gain, future_c = evaluate(
                horizon,
                f0,
                c0,
                rate,
                boot_bridge,
                anchors,
                terminal_fraction,
                tail_fraction,
                sampled_drift,
                rw_noise,
            )
            draws.append(
                {
                    "draw": draw,
                    "horizon_months": horizon,
                    "forecast": value,
                    "start_frontier": f0,
                    "scale_gain": scale_gain,
                    "tech_gain": tech_gain,
                    "future_compute": future_c,
                    "terminal_slope_fraction": terminal_fraction,
                }
            )
            if horizon == 12:
                oat_draws.append({"draw": draw, "fixed_source": "none", "forecast": value})
                variants = {
                    "ability_window": (f0_fixed_ability, c0, rate, boot_bridge, anchors, terminal_fraction, tail_fraction, sampled_drift, rw_noise),
                    "model_family": (f0_fixed_family, c0, rate, boot_bridge, anchors, terminal_fraction, tail_fraction, sampled_drift, rw_noise),
                    "compute_window": (f0, c_last, slow_rate, boot_bridge, anchors, terminal_fraction, tail_fraction, sampled_drift, rw_noise),
                    "Q3_form": (f0, c0, rate, boot_bridge, anchors_main, terminal_fraction, tail_fraction, sampled_drift, rw_noise),
                    "bridge_sample": (f0, c0, rate, bridge, anchors, terminal_fraction, tail_fraction, sampled_drift, rw_noise),
                    "tail_slope": (f0, c0, rate, boot_bridge, anchors, 0.5, 0.5, sampled_drift, rw_noise),
                    "technology_drift": (f0, c0, rate, boot_bridge, anchors, terminal_fraction, tail_fraction, tech_drift, 0.0),
                }
                for source in fixed_sources:
                    fixed_value, _, _, _ = evaluate(12, *variants[source])
                    oat_draws.append({"draw": draw, "fixed_source": source, "forecast": fixed_value})
    draws = pd.DataFrame(draws)
    draws.to_csv(OUT / "forecast_draws.csv", index=False, encoding="utf-8-sig")
    oat_draws = pd.DataFrame(oat_draws)
    oat_draws.to_csv(OUT / "uncertainty_oat_draws.csv", index=False, encoding="utf-8-sig")
    full_variance = float(oat_draws.loc[oat_draws["fixed_source"].eq("none"), "forecast"].var(ddof=1))
    source_labels = {
        "ability_window": "能力窗口抽样",
        "model_family": "模型族依赖",
        "compute_window": "算力前沿",
        "Q3_form": "Q3成本形式",
        "bridge_sample": "桥接样本",
        "tail_slope": "支撑域外斜率",
        "technology_drift": "技术漂移与创新",
    }
    oat_rows = []
    for source in fixed_sources:
        fixed_variance = float(
            oat_draws.loc[oat_draws["fixed_source"].eq(source), "forecast"].var(ddof=1)
        )
        oat_rows.append(
            {
                "source": source,
                "source_cn": source_labels[source],
                "full_variance": full_variance,
                "fixed_variance": fixed_variance,
                "variance_reduction_rate": 1.0 - fixed_variance / full_variance,
                "full_sd": np.sqrt(full_variance),
                "fixed_sd": np.sqrt(fixed_variance),
            }
        )
    oat = pd.DataFrame(oat_rows).sort_values("variance_reduction_rate", ascending=False)
    oat.to_csv(OUT / "uncertainty_oat_variance_reduction.csv", index=False, encoding="utf-8-sig")
    intervals = draws.groupby("horizon_months")["forecast"].quantile([0.025, 0.5, 0.975]).unstack()
    intervals.columns = ["lower_2_5", "median", "upper_97_5"]
    intervals = intervals.reset_index()
    intervals.to_csv(OUT / "forecast_intervals.csv", index=False, encoding="utf-8-sig")
    forecasts = pd.DataFrame(rows)
    forecasts = forecasts.merge(intervals, on="horizon_months", how="left")
    interval_cols = ["lower_2_5", "median", "upper_97_5"]
    primary_scenario = "compute_platform+current_tech_drift"
    forecasts.loc[forecasts["scenario"] != primary_scenario, interval_cols] = np.nan
    forecasts.to_csv(OUT / "forecast_scenarios.csv", index=False, encoding="utf-8-sig")
    forecasts.to_csv(OUT / "forecast_scenario_matrix.csv", index=False, encoding="utf-8-sig")

    variance_rows = []
    for horizon, group in draws.groupby("horizon_months"):
        components = group[["start_frontier", "scale_gain", "tech_gain"]].var(ddof=1)
        shares = components / components.sum()
        for name in components.index:
            variance_rows.append(
                {"horizon_months": horizon, "component": name, "variance_proxy": components[name], "share": shares[name]}
            )
    pd.DataFrame(variance_rows).to_csv(OUT / "uncertainty_decomposition.csv", index=False, encoding="utf-8-sig")
    uncertainty_sources = pd.DataFrame(
        [
            ["ability_window", "resample models in terminal three-month window", "starting frontier"],
            ["model_family", "month-by-family block bootstrap in capability audit", "dependence sensitivity"],
            ["compute_window", "resample trusted reported-compute models", "starting compute frontier"],
            ["Q3_form", "switch exponential, power, log-asymptotic anchors", "mechanism scenario"],
            ["bridge_sample", "stratified High/Medium bootstrap", "Loss-to-score mapping"],
            ["tail_slope", "uniform saturation-to-full-continuation fraction", "support-boundary uncertainty"],
            ["technology_drift", "drift estimation plus random-walk innovation", "future non-scale progress"],
        ],
        columns=["source", "propagation", "affects"],
    )
    uncertainty_sources.to_csv(OUT / "uncertainty_sources.csv", index=False, encoding="utf-8-sig")

    c3 = c3_long_run_check()
    summary = {
        "simulation_draws": N_DRAWS,
        "seed": SEED,
        "tech_drift_points_per_month": tech_drift,
        "tech_innovation_sd": tech_noise,
        "compute_rate_recent_log10_per_month": recent,
        "compute_rate_early_log10_per_month": early,
        "compute_rate_slowdown_log10_per_month": slow_rate,
        "primary_12_month": intervals[intervals["horizon_months"] == 12].iloc[0].to_dict(),
        "exploratory_24_month": intervals[intervals["horizon_months"] == 24].iloc[0].to_dict(),
        "interval_interpretation": "95% conditional simulation interval, not a frequentist confidence interval",
        "scenario_matrix_12_month": forecasts[forecasts["horizon_months"] == 12][
            ["compute_path", "technology_path", "forecast", "scale_gain", "tech_gain"]
        ].to_dict(orient="records"),
        "uncertainty_oat_variance_reduction": oat[
            ["source", "source_cn", "variance_reduction_rate", "fixed_variance"]
        ].to_dict(orient="records"),
        "uncertainty_oat_interpretation": "one-at-a-time variance reduction; non-additive and not a Sobol index",
        "c3_long_run_check": c3,
    }
    (OUT / "forecast_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
