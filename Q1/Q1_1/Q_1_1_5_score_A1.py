"""Fit nonnegative within-cluster weights; report A1 quality and uncertainty."""

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
SEED = 20260923


def derive_weights(group: list[str], correlations: pd.DataFrame) -> np.ndarray:
    if len(group) == 1:
        return np.ones(1)
    corr = correlations.loc[group, group].to_numpy(dtype=float)
    # Nonnegative Perron eigenvector of absolute correlation avoids negative
    # coefficients, which would invert a normalized higher-is-better signal.
    _, vectors = np.linalg.eigh(np.abs(corr))
    weights = np.abs(vectors[:, -1])
    return weights / weights.sum()


def score_clusters(frame: pd.DataFrame, group_weights: dict[int, dict[str, float]]) -> pd.DataFrame:
    components = {}
    for group, weights in group_weights.items():
        components[f"H_{group}"] = frame[list(weights)].to_numpy(dtype=float) @ np.array(list(weights.values()))
    result = pd.DataFrame(components)
    result["Q0"] = result.mean(axis=1)
    return result


def main() -> None:
    frame = pd.read_csv(OUT / "A1_directional_signals.csv")
    groups = pd.read_csv(OUT / "A1_indicator_clusters.csv")
    corr = pd.read_csv(OUT / "A1_balanced_spearman.csv", index_col=0)
    group_weights = {}
    rows = []
    for cluster, slice_ in groups.groupby("cluster", sort=True):
        names = slice_.indicator.tolist()
        weights = derive_weights(names, corr)
        group_weights[int(cluster)] = dict(zip(names, map(float, weights)))
        rows.extend({"cluster": int(cluster), "indicator": name, "within_cluster_weight": float(weight),
                     "effective_weight_in_Q0": float(weight / groups.cluster.nunique())}
                    for name, weight in zip(names, weights))
    pd.DataFrame(rows).to_csv(OUT / "A1_model_indicator_weights.csv", index=False, encoding="utf-8-sig")
    (OUT / "A1_frozen_hierarchy.json").write_text(
        json.dumps({"group_weights": group_weights, "cluster_count": len(group_weights),
                    "weight_rule": "nonnegative leading eigenvector of absolute domain-balanced Spearman matrix within cluster",
                    "between_cluster_rule": "equal weights, one share per cluster",
                    "quality_name": "Q0, before the separately modeled conflict-resolution task"}, indent=2),
        encoding="utf-8",
    )

    scores = score_clusters(frame, group_weights)
    scored = pd.concat([frame[["id", "domain"]], scores], axis=1)
    if len(scored) != 51230 or scored.Q0.isna().any() or not scored.Q0.between(0, 1).all():
        raise ValueError("A1 scoring coverage or 0-1 bound failed")
    scored.to_csv(OUT / "A1_sample_Q0.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(SEED)
    summary = []
    for domain, group in scored.groupby("domain", sort=True):
        values = group.Q0.to_numpy()
        n = len(values)
        # Chunked bootstrap prevents retaining a large B x n matrix.
        boot = [np.mean(values[rng.integers(0, n, n)]) for _ in range(500)]
        summary.append({"dataset": "A1", "domain": domain, "n": n, "mean_Q0": float(values.mean()),
                        "median_Q0": float(np.median(values)), "ci95_low": float(np.quantile(boot, .025)),
                        "ci95_high": float(np.quantile(boot, .975)), "std_Q0": float(values.std(ddof=1))})
    report = pd.DataFrame(summary).sort_values("mean_Q0", ascending=False)
    report.to_csv(OUT / "A1_domain_Q0.csv", index=False, encoding="utf-8-sig")
    overall = {"A1_micro_Q0": float(scored.Q0.mean()), "A1_macro_Q0": float(report.mean_Q0.mean()),
               "A1_rows": len(scored), "clusters": len(group_weights), "domain_bootstrap_replicates": 500}
    (OUT / "A1_overall_Q0.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ordered = report.iloc[::-1]
    ax.errorbar(ordered.mean_Q0, ordered.domain, xerr=[ordered.mean_Q0 - ordered.ci95_low,
                                                         ordered.ci95_high - ordered.mean_Q0],
                fmt="o", color="#216e77", capsize=4, markersize=8)
    ax.set(xlim=(max(0, report.mean_Q0.min() - .08), min(1, report.mean_Q0.max() + .08)),
           xlabel="Mean Q0 (95% bootstrap CI)", ylabel="A1 domain", title="Initial quality score across seven domains")
    fig.tight_layout()
    fig.savefig(OUT / "fig_A1_domain_Q0.png", dpi=200)
    plt.close(fig)
    print(report.to_string(index=False))
    print("Overall:", overall)


if __name__ == "__main__":
    main()
