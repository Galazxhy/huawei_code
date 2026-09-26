"""Analyze conflict sources at indicator, relation and domain levels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUT = Path(__file__).resolve().parent / "results"


def load_flag(dataset: str) -> tuple[np.ndarray, list[str]]:
    with np.load(OUT / f"{dataset}_anomaly.npz", allow_pickle=True) as archive:
        return archive["flag"], [str(x) for x in archive["names"]]


def main() -> None:
    flag_a1, names = load_flag("A1")
    with np.load(OUT / "A1_anomaly.npz", allow_pickle=True) as archive:
        r_a1 = archive["r"]
    _, _ = load_flag("A2")
    _, _ = load_flag("A3")

    # Indicator conflict incidence R_j.
    indicator_rows = []
    for j, name in enumerate(names):
        indicator_rows.append({
            "indicator": name,
            "A1_conflict_rate": float(flag_a1[:, j].mean()),
            "A1_mean_r": float(r_a1[:, j].mean()),
            "A1_mean_r_when_flagged": float(r_a1[flag_a1[:, j], j].mean()) if flag_a1[:, j].any() else 0.0,
        })
    indicator = pd.DataFrame(indicator_rows).sort_values("A1_mean_r_when_flagged", ascending=False)
    indicator.to_csv(OUT / "A1_conflict_indicator_rates.csv", index=False, encoding="utf-8-sig")

    # Indicator severity for extension stability comparison.
    severity_rows = []
    for dataset in ("A1", "A2", "A3"):
        with np.load(OUT / f"{dataset}_anomaly.npz", allow_pickle=True) as archive:
            r_mat = archive["r"]
            flag_mat = archive["flag"]
        for j, name in enumerate(names):
            severity_rows.append({
                "dataset": dataset, "indicator": name,
                "conflict_rate": float(flag_mat[:, j].mean()),
                "mean_r": float(r_mat[:, j].mean()),
                "mean_r_when_flagged": float(r_mat[flag_mat[:, j], j].mean()) if flag_mat[:, j].any() else 0.0,
            })
    severity = pd.DataFrame(severity_rows)
    severity.to_csv(OUT / "conflict_indicator_severity.csv", index=False, encoding="utf-8-sig")

    # Relation-level co-occurrence.
    co_occurrence = (flag_a1[:, :, None] & flag_a1[:, None, :]).mean(axis=0)
    edge_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            edge_rows.append({"source": names[i], "target": names[j], "joint_conflict_rate": float(co_occurrence[i, j])})
    edges = pd.DataFrame(edge_rows)
    edges.to_csv(OUT / "A1_conflict_pair_cooccurrence.csv", index=False, encoding="utf-8-sig")

    # Domain conflict levels.
    domain_rows = []
    for dataset in ("A1", "A2", "A3"):
        scores = pd.read_csv(OUT / f"{dataset}_conflict_scores.csv")
        high = scores.C_conflict.quantile(0.99)
        for domain, group in scores.groupby("domain", sort=True):
            domain_rows.append({
                "dataset": dataset, "domain": domain,
                "mean_C": float(group.C_conflict.mean()),
                "median_C": float(group.C_conflict.median()),
                "p99_C": float(group.C_conflict.quantile(0.99)),
                "high_conflict_rate": float((group.C_conflict > high).mean()),
                "n": len(group),
            })
    domain_table = pd.DataFrame(domain_rows)
    domain_table.to_csv(OUT / "conflict_domain_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "indicator_conflict_rate_min": float(indicator.A1_conflict_rate.min()),
        "indicator_conflict_rate_max": float(indicator.A1_conflict_rate.max()),
        "top_indicators": indicator.head(8).indicator.tolist(),
        "bottom_indicators": indicator.tail(8).indicator.tolist(),
        "top_conflict_pairs": edges.sort_values("joint_conflict_rate", ascending=False).head(10).to_dict("records"),
    }
    (OUT / "conflict_cause_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ind = indicator.head(18).iloc[::-1]
    sns.barplot(data=ind, x="A1_mean_r_when_flagged", y="indicator", color="#c06d4b", ax=axes[0])
    axes[0].set_title("A1 indicator conflict severity (mean r when flagged)")
    axes[0].set_xlabel("mean standardized anomaly r")
    domain_plot = domain_table[domain_table.dataset == "A1"].sort_values("mean_C")
    sns.barplot(data=domain_plot, x="mean_C", y="domain", color="#216e77", ax=axes[1])
    axes[1].set_title("A1 domain mean conflict C")
    axes[1].set_xlabel("mean C")
    fig.tight_layout()
    fig.savefig(OUT / "fig_Q1_2_indicator_domain_conflict.png", dpi=180)
    plt.close(fig)

    # Pairwise co-occurrence heatmap for the full 22x22 matrix.
    ordered = indicator.indicator.tolist()
    matrix = pd.DataFrame(co_occurrence, index=names, columns=names).loc[ordered, ordered]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(matrix, cmap="rocket_r", square=True, ax=ax, cbar_kws={"label": "joint conflict rate"})
    ax.set_title("Pairwise joint conflict rate")
    fig.tight_layout()
    fig.savefig(OUT / "fig_Q1_2_conflict_pair_heatmap.png", dpi=180)
    plt.close(fig)

    print(indicator.head(12).to_string(index=False))
    print(domain_table[domain_table.dataset == "A1"].to_string(index=False))


if __name__ == "__main__":
    main()
