"""Sensitivity checks for the frozen hierarchical quality model.

This script perturbs explicitly assumed choices (middle-suitability band,
middle-vs-positive direction, and cluster count K) and reports how strongly the
resulting initial quality score Q0 changes. It intentionally keeps the frozen
hierarchy fixed for direction/band perturbations, so those numbers quantify the
effect of the stated assumptions rather than refitting a new model for each
perturbation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from Q_1_1_3_freeze_scale import DIRECTIONS, percentile
from Q_1_1_5_score_A1 import derive_weights, score_clusters


OUT = Path(__file__).resolve().parent / "results"
SEED = 20260923

FINAL_FIELDS = [name for name in DIRECTIONS if not name.startswith("qurater_")] + ["qurater"]
QURATER_COLUMNS = [name for name in DIRECTIONS if name.startswith("qurater_")]


def suitability(p: np.ndarray, direction: str, band: float) -> np.ndarray:
    if direction == "positive":
        return p
    if direction == "negative":
        return 1 - p
    return np.clip(np.minimum(p / band, (1 - p) / band), 0, 1)


def transform_frame(frame, references, medians, domain_medians, directions, band):
    features = {}
    for field, (direction, _) in directions.items():
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float, copy=True)
        invalid = ~np.isfinite(values)
        if domain_medians is not None and "domain" in frame:
            for domain, domain_median in domain_medians[field].items():
                mask = invalid & (frame["domain"].to_numpy() == domain)
                values[mask] = domain_median
        values[~np.isfinite(values)] = medians[field]
        features[field] = suitability(percentile(values, references[field]), direction, band)
    result = pd.DataFrame({name: features[name] for name in FINAL_FIELDS if name != "qurater"})
    result["qurater"] = np.mean([features[name] for name in QURATER_COLUMNS], axis=0)
    return result[FINAL_FIELDS]


def compare_to_baseline(signals, group_weights, frame, baseline):
    scored = score_clusters(signals, group_weights)
    scored["id"] = frame["id"].to_numpy()
    scored["domain"] = frame["domain"].to_numpy()
    merged = scored.merge(baseline[["id", "Q0"]], on="id", suffixes=("", "_base"))
    sample_spearman = float(spearmanr(merged["Q0"], merged["Q0_base"]).statistic)
    domain_rank_spearman = float(spearmanr(
        merged.groupby("domain")["Q0"].mean(),
        merged.groupby("domain")["Q0_base"].mean(),
    ).statistic)
    top_n = math.ceil(len(merged) * 0.10)
    top_new = set(np.argpartition(merged["Q0"].to_numpy(), -top_n)[-top_n:])
    top_base = set(np.argpartition(merged["Q0_base"].to_numpy(), -top_n)[-top_n:])
    top10_overlap = len(top_new & top_base) / top_n
    return sample_spearman, domain_rank_spearman, top10_overlap


def main() -> None:
    frame = pd.read_csv(OUT / "A1_scalarized_signals.csv")
    baseline = pd.read_csv(OUT / "A1_sample_Q0.csv", usecols=["id", "domain", "Q0"])
    params = json.loads((OUT / "A1_frozen_scale.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((OUT / "A1_frozen_hierarchy.json").read_text(encoding="utf-8"))
    base_group_weights = {int(k): v for k, v in hierarchy["group_weights"].items()}
    with np.load(OUT / "A1_frozen_ecdf.npz") as archive:
        references = {field: [archive[f"{field}__{i}"].copy() for i in range(7)] for field in DIRECTIONS}

    base_signals = pd.read_csv(OUT / "A1_directional_signals.csv")[FINAL_FIELDS]
    rows = []

    def record(variant, description, signals, group_weights):
        sample_sp, domain_sp, top10 = compare_to_baseline(signals, group_weights, frame, baseline)
        rows.append({
            "variant": variant,
            "description": description,
            "sample_spearman_to_baseline": round(sample_sp, 6),
            "domain_rank_spearman_to_baseline": round(domain_sp, 6),
            "top10_overlap_with_baseline": round(top10, 6),
        })

    # Direction/band perturbations with the baseline K=8 hierarchy fixed.
    record("baseline", "baseline K=8, band=0.20, middle directions", base_signals, base_group_weights)

    alt_directions = dict(DIRECTIONS)
    alt_directions["qurater_required_expertise"] = ("positive", "expertise treated as higher-is-better")
    record("expertise_positive", "QuRating expertise middle -> positive", transform_frame(
        frame, references, params["median_fallback_global"], params["median_fallback_by_domain"],
        alt_directions, 0.20), base_group_weights)

    middle_fields = [name for name, (direction, _) in DIRECTIONS.items() if direction == "middle"]
    all_positive = dict(DIRECTIONS)
    for name in middle_fields:
        all_positive[name] = ("positive", "middle suitability replaced by higher-is-better")
    record("middle_to_positive", "all middle directions -> positive", transform_frame(
        frame, references, params["median_fallback_global"], params["median_fallback_by_domain"],
        all_positive, 0.20), base_group_weights)

    for band in (0.10, 0.30):
        record(f"band_{band:.2f}", f"middle suitability band width {band:.2f}",
               transform_frame(frame, references, params["median_fallback_global"],
                               params["median_fallback_by_domain"], dict(DIRECTIONS), band),
               base_group_weights)

    # Cluster-count perturbations. Refit only the cluster partition and weights,
    # keeping the standardized signals identical to the baseline model.
    corr = pd.read_csv(OUT / "A1_balanced_spearman.csv", index_col=0)
    corr = corr.loc[FINAL_FIELDS, FINAL_FIELDS]
    distance = np.clip(1 - np.abs(corr.to_numpy(dtype=float)), 0, 1)
    np.fill_diagonal(distance, 0)
    hierarchy_linkage = linkage(squareform(distance, checks=False), method="average")
    for k in (7, 8, 9):
        labels = fcluster(hierarchy_linkage, k, criterion="maxclust")
        cluster_groups: dict[int, list[str]] = {}
        for name, label in zip(FINAL_FIELDS, labels):
            cluster_groups.setdefault(int(label), []).append(name)
        group_weights = {}
        for cluster, names in cluster_groups.items():
            weights = derive_weights(names, corr)
            group_weights[cluster] = dict(zip(names, map(float, weights)))
        record(f"K_{k}", f"cluster count K={k}", base_signals, group_weights)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "A1_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = summary.sort_values("domain_rank_spearman_to_baseline")
    sns.barplot(data=plot_data, x="domain_rank_spearman_to_baseline", y="variant", color="#216e77", ax=ax)
    ax.set(xlim=(0, 1), xlabel="Domain-rank Spearman vs baseline Q0",
           title="Sensitivity of seven-domain quality ranking to modeling choices")
    for i, value in enumerate(plot_data.domain_rank_spearman_to_baseline):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_A1_sensitivity_ranks.png", dpi=200)
    plt.close(fig)

    print(summary.to_string(index=False))
    print("Sensitivity outputs written to", OUT)


if __name__ == "__main__":
    main()
