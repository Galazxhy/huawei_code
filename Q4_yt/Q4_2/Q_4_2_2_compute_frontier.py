"""Construct leakage-free open-weight language-model compute frontiers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from q4_utils import ANALYSIS_END, CDATA, OUT, month_grid, robust_slope


def load_compute_sample() -> pd.DataFrame:
    c4 = pd.read_csv(CDATA / "epoch_all_ai_models.csv", low_memory=False)
    c4["date"] = pd.to_datetime(c4["Publication date"], errors="coerce")
    language = c4["Domain"].fillna("").str.contains("Language", case=False)
    lm_task = c4["Task"].fillna("").str.contains("language modeling|question answering", case=False)
    openw = c4[
        c4["Open model weights?"].eq("Yes")
        & language
        & lm_task
        & c4["date"].notna()
        & c4["date"].le(ANALYSIS_END)
    ].copy()
    openw["reported_compute"] = pd.to_numeric(openw["Training compute (FLOP)"], errors="coerce")
    params = pd.to_numeric(openw["Parameters"], errors="coerce")
    tokens = pd.to_numeric(openw["Training dataset size (total)"], errors="coerce")
    openw["estimated_compute"] = 6.0 * params * tokens
    openw["compute"] = openw["reported_compute"].fillna(openw["estimated_compute"])
    openw["compute_source"] = np.select(
        [openw["reported_compute"].notna(), openw["reported_compute"].isna() & openw["estimated_compute"].notna()],
        ["reported", "6ND-estimated"],
        default="missing",
    )
    openw["trusted_confidence"] = openw["Confidence"].isin(["Confident", "Likely"])
    return openw


def build_frontier(sample: pd.DataFrame, value: str, suffix: str, min_n: int = 4) -> pd.DataFrame:
    rows = []
    valid = sample.dropna(subset=[value]).copy()
    valid["log_compute"] = np.log10(valid[value].astype(float))
    for end in month_grid(valid["date"].min(), ANALYSIS_END):
        start = end - pd.DateOffset(months=3)
        vals = valid.loc[(valid["date"] > start) & (valid["date"] <= end), "log_compute"]
        if len(vals) < min_n:
            continue
        row = {"date": end, f"n_{suffix}": len(vals)}
        for q in (0.50, 0.75, 0.90):
            row[f"logC_q{round(q * 100)}_{suffix}"] = float(np.quantile(vals, q))
            row[f"C_q{round(q * 100)}_{suffix}"] = float(10 ** row[f"logC_q{round(q * 100)}_{suffix}"])
        rows.append(row)
    return pd.DataFrame(rows)


def period_slope(series: pd.DataFrame, col: str, start: str, end: str) -> float:
    block = series[series["date"].between(pd.Timestamp(start), pd.Timestamp(end))].dropna(subset=[col])
    return robust_slope(block[col].to_numpy()) if len(block) >= 4 else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = load_compute_sample()
    reported = sample[sample["compute_source"].eq("reported") & sample["trusted_confidence"]]
    all_sources = sample[sample["compute_source"].isin(["reported", "6ND-estimated"])]
    main = build_frontier(reported, "reported_compute", "reported")
    sensitivity = build_frontier(all_sources, "compute", "all")
    frontier = pd.merge(main, sensitivity, on="date", how="outer").sort_values("date")
    frontier.to_csv(OUT / "C_frontier.csv", index=False, encoding="utf-8-sig")
    sample[["Model", "date", "Confidence", "reported_compute", "estimated_compute", "compute", "compute_source"]].to_csv(
        OUT / "compute_sample.csv", index=False, encoding="utf-8-sig"
    )

    # Equal-length-ish regimes around the June 2024 jump expose the subsequent plateau.
    early = period_slope(frontier, "logC_q90_reported", "2023-06-01", "2024-06-30")
    recent = period_slope(frontier, "logC_q90_reported", "2024-07-01", "2025-03-31")
    last = frontier.dropna(subset=["C_q90_reported"]).iloc[-1]
    summary = {
        "analysis_end": str(ANALYSIS_END),
        "filter": "open weights; Language domain; language modeling or QA task",
        "eligible_models": int(len(sample)),
        "reported_compute": int(sample["reported_compute"].notna().sum()),
        "reported_trusted_main": int(len(reported)),
        "estimated_only": int((sample["compute_source"] == "6ND-estimated").sum()),
        "missing_compute": int((sample["compute_source"] == "missing").sum()),
        "frontier_points": int(frontier["C_q90_reported"].notna().sum()),
        "slope_recent_log10_per_month": float(recent),
        "slope_early_log10_per_month": float(early),
        "slowdown_observed": bool(recent < early),
        "last_frontier_q90_flops": float(last["C_q90_reported"]),
        "last_frontier_evidence": "within_Q3" if last["C_q90_reported"] <= 1e24 else "beyond_Q3",
    }
    (OUT / "compute_frontier_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
