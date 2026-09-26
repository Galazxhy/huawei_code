"""Compare DBHQI with equal-weight, CRITIC and global-PCA baselines."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA


OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    signals = pd.read_csv(OUT / "A1_directional_signals.csv")
    dbhqi = pd.read_csv(OUT / "A1_sample_Q0.csv", usecols=["id", "domain", "Q0"])
    cols = list(signals.columns[2:])
    x = signals[cols].to_numpy(dtype=float)
    equal = x.mean(axis=1)

    std = x.std(axis=0, ddof=1)
    corr = np.corrcoef(x, rowvar=False)
    critic = std * np.sum(1 - corr, axis=1)
    critic = critic / critic.sum()
    critic_score = x @ critic

    pca = PCA(n_components=1, random_state=20260923)
    pca_score = pca.fit_transform(x).ravel()
    if spearmanr(pca_score, equal).statistic < 0:
        pca_score *= -1
    pca_score = (pca_score - pca_score.min()) / (pca_score.max() - pca_score.min())

    comparison = dbhqi.copy()
    comparison["equal_weight"] = equal
    comparison["critic"] = critic_score
    comparison["global_pca"] = pca_score
    comparison.to_csv(OUT / "A1_baseline_scores.csv", index=False, encoding="utf-8-sig")

    methods = ["Q0", "equal_weight", "critic", "global_pca"]
    stats = []
    for left in methods:
        for right in methods:
            stats.append({"method_1": left, "method_2": right,
                          "spearman": float(spearmanr(comparison[left], comparison[right]).statistic)})
    pd.DataFrame(stats).to_csv(OUT / "A1_baseline_rank_correlations.csv", index=False, encoding="utf-8-sig")
    top_n = int(np.ceil(len(comparison) * .10))
    top_sets = {m: set(np.argpartition(comparison[m].to_numpy(), -top_n)[-top_n:]) for m in methods}
    overlap_rows = []
    for method in methods[1:]:
        overlap_rows.append({"method": method, "top10_overlap_with_Q0": len(top_sets["Q0"] & top_sets[method]) / top_n,
                             "domain_rank_spearman_with_Q0": float(spearmanr(
                                 comparison.groupby("domain")["Q0"].mean(),
                                 comparison.groupby("domain")[method].mean()).statistic)})
    pd.DataFrame(overlap_rows).to_csv(OUT / "A1_baseline_robustness.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"indicator": cols, "critic_weight": critic,
                  "global_pca_loading": pca.components_[0]}).to_csv(OUT / "A1_baseline_weights.csv", index=False, encoding="utf-8-sig")

    matrix = comparison[methods].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="crest", vmin=0, vmax=1, square=True, ax=ax)
    ax.set_title("Rank agreement across initial quality models")
    fig.tight_layout()
    fig.savefig(OUT / "fig_A1_model_rank_agreement.png", dpi=200)
    plt.close(fig)
    summary = {"global_pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
               "top10_count": top_n, "note": "Baselines are robustness comparators, not replacements for semantic validation."}
    (OUT / "A1_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(matrix.to_string())
    print(pd.DataFrame(overlap_rows).to_string(index=False))


if __name__ == "__main__":
    main()
