"""Analyze balanced rank correlations, domain effects and stable signal groups."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import silhouette_score


OUT = Path(__file__).resolve().parent / "results"
SEED = 20260923


def weighted_correlation(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    center = np.average(data, axis=0, weights=weights)
    centered = data - center
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    deviations = np.sqrt(np.diag(covariance))
    denominator = np.outer(deviations, deviations)
    return np.clip(np.divide(covariance, denominator, out=np.eye(data.shape[1]), where=denominator > 0), -1, 1)


def cluster_from_ranks(ranks: np.ndarray, weights: np.ndarray):
    correlation = weighted_correlation(ranks, weights)
    distance = np.clip(1 - np.abs(correlation), 0, 1)
    np.fill_diagonal(distance, 0)
    hierarchy = linkage(squareform(distance, checks=False), method="average")
    return correlation, distance, hierarchy


def main() -> None:
    frame = pd.read_csv(OUT / "A1_directional_signals.csv")
    names = list(frame.columns[2:])
    domain = frame["domain"].to_numpy()
    groups = {d: np.flatnonzero(domain == d) for d in np.unique(domain)}
    weights = np.array([1 / (len(groups[d]) * len(groups)) for d in domain])
    x = frame[names].to_numpy(dtype=float)
    ranks = np.column_stack([rankdata(x[:, i], method="average") for i in range(x.shape[1])])
    correlation, distance, hierarchy = cluster_from_ranks(ranks, weights)

    rng = np.random.default_rng(SEED)
    bootstrap_ranks: list[np.ndarray] = []
    bootstrap_weights: list[np.ndarray] = []
    for _ in range(120):
        idx = np.concatenate([rng.choice(indices, size=min(1200, len(indices)), replace=True) for indices in groups.values()])
        br = np.column_stack([rankdata(x[idx, j]) for j in range(len(names))])
        bw = np.concatenate([np.full(min(1200, len(indices)), 1 / (len(groups) * min(1200, len(indices)))) for indices in groups.values()])
        bootstrap_ranks.append(br)
        bootstrap_weights.append(bw)

    candidate_rows = []
    for k in range(3, 11):
        labels = fcluster(hierarchy, k, criterion="maxclust")
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(names):
            continue
        silhouette = float(silhouette_score(distance, labels, metric="precomputed"))
        agreements = []
        for br, bw in zip(bootstrap_ranks, bootstrap_weights):
            _, _, bh = cluster_from_ranks(br, bw)
            b_labels = fcluster(bh, k, criterion="maxclust")
            pairs = [((labels[i] == labels[j]) == (b_labels[i] == b_labels[j]))
                     for i in range(len(names)) for j in range(i + 1, len(names))]
            agreements.append(np.mean(pairs))
        candidate_rows.append({"k": k, "silhouette": silhouette, "bootstrap_pair_agreement_mean": float(np.mean(agreements)),
                               "bootstrap_pair_agreement_p05": float(np.quantile(agreements, .05))})
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(OUT / "A1_cluster_k_selection.csv", index=False, encoding="utf-8-sig")
    # Balanced rule: favor k with both separation and reproducible assignments.
    selected = candidates.loc[(candidates.silhouette + candidates.bootstrap_pair_agreement_mean).idxmax()]
    chosen_k = int(selected.k)
    labels = fcluster(hierarchy, chosen_k, criterion="maxclust")
    clusters = pd.DataFrame({"indicator": names, "cluster": labels})
    clusters.to_csv(OUT / "A1_indicator_clusters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(correlation, index=names, columns=names).to_csv(OUT / "A1_balanced_spearman.csv", encoding="utf-8-sig")

    variability = []
    for j, name in enumerate(names):
        means = np.array([x[indices, j].mean() for indices in groups.values()])
        within = np.mean([x[indices, j].var() for indices in groups.values()])
        between = means.var()
        variability.append({"indicator": name, "between_domain_variance": between,
                            "within_domain_variance": within, "domain_variance_share_eta": between / (between + within) if between + within else 0})
    pd.DataFrame(variability).to_csv(OUT / "A1_domain_variance_shares.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    counts = frame.domain.value_counts().sort_values()
    sns.barplot(x=counts.values, y=counts.index, color="#216e77", ax=ax)
    ax.set(xlabel="A1 records", ylabel="Domain", title="A1 sample size by domain")
    for i, count in enumerate(counts.values):
        ax.text(count + counts.max() * .01, i, f"{count:,}", va="center", fontsize=9)
    ax.set_xlim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(OUT / "fig_A1_domain_counts.png", dpi=200)
    plt.close(fig)

    order = leaves_list(hierarchy)
    ordered_names = [names[i] for i in order]
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(pd.DataFrame(correlation, index=names, columns=names).loc[ordered_names, ordered_names],
                cmap="vlag", center=0, vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"label": "Domain-balanced Spearman"})
    ax.set_title(f"A1 signal correlations (hierarchical order, K={chosen_k})")
    fig.tight_layout()
    fig.savefig(OUT / "fig_A1_clustered_correlations.png", dpi=180)
    plt.close(fig)

    var_frame = pd.DataFrame(variability).sort_values("domain_variance_share_eta")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=var_frame, x="domain_variance_share_eta", y="indicator", color="#c06d4b", ax=ax)
    ax.set(xlim=(0, 1), xlabel="Share of balanced variance attributable to domain", ylabel="Signal",
           title="Domain effects in normalized A1 signals")
    fig.tight_layout()
    fig.savefig(OUT / "fig_A1_domain_variance.png", dpi=180)
    plt.close(fig)

    result = {"selected_k": chosen_k, "selection_score": float(selected.silhouette + selected.bootstrap_pair_agreement_mean),
              "bootstrap_replicates": len(bootstrap_ranks), "bootstrap_sample_per_domain": 1200,
              "cluster_sizes": {str(k): int(v) for k, v in clusters.cluster.value_counts().sort_index().items()},
              "limitation": "absolute rank correlation can group oppositely directed indicators; inspect cluster semantics and comparison model"}
    (OUT / "A1_structure_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Selected K:", chosen_k, "cluster sizes:", result["cluster_sizes"])
    print(clusters.to_string(index=False))


if __name__ == "__main__":
    main()
