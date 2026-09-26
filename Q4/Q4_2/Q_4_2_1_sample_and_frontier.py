"""Freeze the open-weight samples and construct capability frontiers."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from q4_utils import (
    ANALYSIS_END,
    BENCH,
    CDATA,
    MAIN_TYPES,
    OUT,
    PRETRAIN_TYPES,
    ROOT,
    model_family,
    rolling_quantile,
)


PREP = ROOT.parent / "Q4_1" / "results"
RESEARCH_LICENSE = re.compile(
    r"apache|mit|bsd|cc-by|cc0|openrail|bigscience|llama|gemma|qwen|deepseek|research",
    flags=re.I,
)


def capability_sample() -> pd.DataFrame:
    c1 = pd.read_csv(CDATA / "leaderboard_cleaned.csv")
    c2 = pd.read_csv(CDATA / "leaderboard_enhanced.csv")
    meta = c2[["Model", "Epoch_AI_Open_Weights", "Epoch_AI_Publication_Date"]].drop_duplicates("Model")
    merged = c1.merge(meta, on="Model", how="left", validate="many_to_one")
    merged["Submission Date"] = pd.to_datetime(merged["Submission Date"], errors="coerce")
    merged["complete_six"] = merged[BENCH].notna().all(axis=1)
    merged["Y"] = merged[BENCH].mean(axis=1)
    merged["strict_research_license"] = merged["Hub License"].fillna("").str.contains(RESEARCH_LICENSE)
    merged["analysis_eligible"] = (
        merged["Epoch_AI_Open_Weights"].eq("Yes")
        & merged["complete_six"]
        & merged["Submission Date"].le(ANALYSIS_END)
    )
    return merged


def make_frontier(sample: pd.DataFrame, types: list[str], q: float = 0.90, months: int = 3) -> pd.DataFrame:
    subset = sample[sample["analysis_eligible"] & sample["Type"].isin(types)].copy()
    return rolling_quantile(subset, "Submission Date", "Y", months=months, q=q, min_n=5)


def validate_capability_metric(sample: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    x = sample[BENCH].to_numpy(float)
    k = x.shape[1]
    alpha = k / (k - 1) * (1.0 - x.var(axis=0, ddof=1).sum() / x.sum(axis=1).var(ddof=1))
    z = StandardScaler().fit_transform(x)
    pca = PCA().fit(z)
    pc1 = pca.transform(z)[:, 0]
    if np.corrcoef(sample["Y"], pc1)[0, 1] < 0:
        pc1 = -pc1
    z_mean = z.mean(axis=1)
    enriched = sample.copy()
    enriched["Y_equal"] = enriched["Y"]
    enriched["Y_zmean"] = z_mean
    enriched["Y_pc1"] = pc1
    correlation = np.corrcoef(enriched[["Y_equal", "Y_zmean", "Y_pc1"]].to_numpy(), rowvar=False)
    frontier_rows = []
    for metric in ["Y_equal", "Y_zmean", "Y_pc1"]:
        f = rolling_quantile(enriched, "Submission Date", metric, months=3, q=0.90, min_n=5)
        col = "q90"
        f["metric"] = metric
        f["net_change"] = f[col].iloc[-1] - f[col].iloc[0]
        frontier_rows.append(f.rename(columns={col: "frontier"}))
    alternative = pd.concat(frontier_rows, ignore_index=True)
    summary = {
        "cronbach_alpha_raw_scores": float(alpha),
        "pca_standardized_pc1_explained_ratio": float(pca.explained_variance_ratio_[0]),
        "corr_equal_pc1": float(correlation[0, 2]),
        "corr_equal_zmean": float(correlation[0, 1]),
        "corr_zmean_pc1": float(correlation[1, 2]),
        "all_three_frontiers_same_direction": bool(
            alternative.groupby("metric")["net_change"].first().gt(0).all()
        ),
        "frontier_net_change_by_metric": {
            key: float(value)
            for key, value in alternative.groupby("metric")["net_change"].first().items()
        },
    }
    return summary, alternative


def family_sensitivity(sample: pd.DataFrame, draws: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    work = sample.copy()
    work["model_family"] = work["Model"].map(model_family)
    work["month"] = work["Submission Date"].dt.to_period("M")
    balanced = work.loc[work.groupby(["month", "model_family"])["Y"].idxmax()].copy()
    frontier = rolling_quantile(balanced, "Submission Date", "Y", months=3, q=0.90, min_n=5)
    frontier.to_csv(OUT / "F_frontier_family_balanced.csv", index=False, encoding="utf-8-sig")

    endpoints = [pd.Timestamp("2024-06-30 23:59:59.999999"), ANALYSIS_END]
    endpoint_blocks = []
    for endpoint in endpoints:
        window = work[
            (work["Submission Date"] > endpoint - pd.DateOffset(months=3))
            & (work["Submission Date"] <= endpoint)
        ]
        endpoint_blocks.append([group["Y"].to_numpy(float) for _, group in window.groupby(["month", "model_family"])])
    rng = np.random.default_rng(20260925)
    rows = []
    for draw in range(draws):
        values = []
        for blocks in endpoint_blocks:
            chosen = rng.integers(0, len(blocks), len(blocks))
            values.append(np.concatenate([blocks[i] for i in chosen]))
        f0, f1 = [float(np.quantile(value, 0.90)) for value in values]
        rows.append({"draw": draw, "F0": f0, "F1": f1, "net_change": f1 - f0})
    bootstrap = pd.DataFrame(rows)
    summary = {
        "family_rule": "known base-architecture regex; fallback to Hugging Face organization",
        "families": int(work["model_family"].nunique()),
        "family_month_cells": int(work.groupby(["month", "model_family"]).ngroups),
        "balanced_models": int(len(balanced)),
        "balanced_frontier_net_change": float(frontier["q90"].iloc[-1] - frontier["q90"].iloc[0]),
        "block_bootstrap_draws": draws,
        "block_bootstrap_positive_probability": float((bootstrap["net_change"] > 0).mean()),
        "block_bootstrap_net_change_quantiles": {
            str(q): float(bootstrap["net_change"].quantile(q)) for q in [0.025, 0.5, 0.975]
        },
    }
    return frontier, bootstrap, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    merged = capability_sample()
    main_sample = merged[merged["analysis_eligible"] & merged["Type"].isin(MAIN_TYPES)].copy()
    pretrained = merged[merged["analysis_eligible"] & merged["Type"].isin(PRETRAIN_TYPES)].copy()
    strict = main_sample[main_sample["strict_research_license"]].copy()

    metric_summary, alternative_frontiers = validate_capability_metric(main_sample)
    alternative_frontiers.to_csv(OUT / "ability_metric_alternative_frontiers.csv", index=False, encoding="utf-8-sig")
    (OUT / "ability_metric_validation.json").write_text(
        json.dumps(metric_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _, family_bootstrap, family_summary = family_sensitivity(main_sample)
    family_bootstrap.to_csv(OUT / "family_block_bootstrap.csv", index=False, encoding="utf-8-sig")
    (OUT / "family_sensitivity_summary.json").write_text(
        json.dumps(family_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    f_main = make_frontier(merged, MAIN_TYPES)
    f_pretrain = make_frontier(merged, PRETRAIN_TYPES)
    f_strict = rolling_quantile(strict, "Submission Date", "Y", months=3, q=0.90, min_n=5)
    f_main.to_csv(OUT / "F_frontier_main.csv", index=False, encoding="utf-8-sig")
    f_pretrain.to_csv(OUT / "F_frontier_pretrained.csv", index=False, encoding="utf-8-sig")
    f_strict.to_csv(OUT / "F_frontier_strict_license.csv", index=False, encoding="utf-8-sig")
    type_comparison = pd.DataFrame(
        [
            {
                "model_type": "chat_finetuned",
                "sample_n": len(main_sample),
                "frontier_points": len(f_main),
                "first_frontier": f_main["q90"].iloc[0],
                "last_frontier": f_main["q90"].iloc[-1],
                "net_change": f_main["q90"].iloc[-1] - f_main["q90"].iloc[0],
            },
            {
                "model_type": "pretrained",
                "sample_n": len(pretrained),
                "frontier_points": len(f_pretrain),
                "first_frontier": f_pretrain["q90"].iloc[0],
                "last_frontier": f_pretrain["q90"].iloc[-1],
                "net_change": f_pretrain["q90"].iloc[-1] - f_pretrain["q90"].iloc[0],
            },
        ]
    )
    type_comparison.to_csv(OUT / "model_type_frontier_comparison.csv", index=False, encoding="utf-8-sig")
    main_sample[["Model", "Submission Date", "Hub License", "Type", *BENCH, "Y"]].to_csv(
        OUT / "capability_sample_main.csv", index=False, encoding="utf-8-sig"
    )

    # C8 is an independent task-level aggregation check; it is not mixed into the model-level sample.
    c8 = pd.read_csv(PREP / "C8_per_task_scores.csv")
    c8["date"] = pd.to_datetime(c8["timestamp"], format="%Y-%m-%dT%H-%M-%S", errors="coerce")
    c8_col = {"MATH Lvl 5": "MATH_Lvl5", "MMLU-PRO": "MMLU_PRO"}
    task_rows = []
    for task in BENCH:
        col = c8_col.get(task, task)
        task_data = c8[["date", col]].rename(columns={col: "score"})
        f = rolling_quantile(task_data, "date", "score", months=3, q=0.90, min_n=20)
        f = f.rename(columns={"q90": "frontier"})
        f["task"] = task
        task_rows.append(f[["date", "task", "frontier", "n"]])
    task_frontier = pd.concat(task_rows, ignore_index=True)
    task_frontier.to_csv(OUT / "C8_task_frontier.csv", index=False, encoding="utf-8-sig")
    pivot = task_frontier.pivot(index="date", columns="task", values="frontier").sort_index()
    monthly = (pivot.diff() > 0).sum(axis=1)
    breadth = pd.DataFrame(
        {
            "date": pivot.index,
            "tasks_increasing": monthly,
            "tasks_available": pivot.notna().sum(axis=1),
            "breadth": monthly / pivot.notna().sum(axis=1),
        }
    )
    breadth.to_csv(OUT / "C8_breadth.csv", index=False, encoding="utf-8-sig")
    net_change = (pivot.iloc[-1] - pivot.iloc[0]).rename("net_change").reset_index()
    net_change["improved"] = net_change["net_change"] > 0
    net_change.to_csv(OUT / "C8_task_net_change.csv", index=False, encoding="utf-8-sig")
    score_columns = ["IFEval", "BBH", "MATH_Lvl5", "GPQA", "MUSR", "MMLU_PRO"]
    c8_audit = {
        "rows": int(len(c8)),
        "unique_models": int(c8["model"].nunique()),
        "duplicate_model_rows": int(c8.duplicated("model", keep=False).sum()),
        "duplicate_record_files": int(c8.duplicated("record_file", keep=False).sum()),
        "zero_counts": {col: int(c8[col].eq(0).sum()) for col in score_columns},
        "zero_rates": {col: float(c8[col].eq(0).mean()) for col in score_columns},
        "missing_counts": {col: int(c8[col].isna().sum()) for col in score_columns},
        "timestamp_min": str(c8["date"].min()),
        "timestamp_max": str(c8["date"].max()),
        "audit_interpretation": "Exact zeros are retained as valid task scores; duplicate model/file counts diagnose snapshot duplication.",
    }
    (OUT / "C8_quality_audit.json").write_text(
        json.dumps(c8_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "analysis_end": str(ANALYSIS_END),
        "main_sample_n": int(len(main_sample)),
        "strict_license_n": int(len(strict)),
        "pretrained_sample_n": int(len(pretrained)),
        "open_weights_yes_all_dates": int(merged["Epoch_AI_Open_Weights"].eq("Yes").sum()),
        "main_date_range": [str(main_sample["Submission Date"].min()), str(main_sample["Submission Date"].max())],
        "frontier_main_points": int(len(f_main)),
        "frontier_pretrained_points": int(len(f_pretrain)),
        "c8_complete_models": int(c8["six_dim_complete"].sum()),
        "c8_tasks_net_improved": int(net_change["improved"].sum()),
        "c8_monthly_breadth_median": float(breadth["breadth"].iloc[1:].median()),
        "ability_metric_validation": metric_summary,
        "family_sensitivity": family_summary,
    }
    (OUT / "sample_frontier_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
