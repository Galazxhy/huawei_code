"""Compare the graph-conflict measure with simple dispersion baselines."""

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
from scipy.stats import spearmanr


OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    signals = pd.read_csv(OUT / "A1_directional_signals.csv")
    names = list(signals.columns[2:])
    x = signals[names].to_numpy(dtype=float)
    c_var = x.var(axis=1, ddof=1)
    c_max = x.max(axis=1) - x.min(axis=1)
    graph = pd.read_csv(OUT / "A1_conflict_scores.csv")["C_conflict"].to_numpy()

    rows = []
    for label, c in (("variance", c_var), ("max_pair_gap", c_max)):
        rows.append({
            "baseline": label,
            "spearman_with_graph_C": float(spearmanr(graph, c).statistic),
            "top1pct_overlap_with_graph": float(len(set(np.argpartition(graph, -int(len(graph)*0.01))[-int(len(graph)*0.01):]) &
                                                   set(np.argpartition(c, -int(len(graph)*0.01))[-int(len(graph)*0.01):])) / int(len(graph)*0.01)),
        })
    baseline = pd.DataFrame(rows)
    baseline.to_csv(OUT / "conflict_baseline_comparison.csv", index=False, encoding="utf-8-sig")
    (OUT / "conflict_baseline_summary.json").write_text(json.dumps(baseline.to_dict("records"), indent=2) + "\n", encoding="utf-8")

    final = pd.read_csv(OUT / "A1_final_quality.csv")
    # Visualize Q0 vs Q*, emphasizing the most conflicted samples.
    sample = final
    if len(final) > 20000:
        sample = final.sample(20000, random_state=20260923)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(sample.Q0, sample.Q_star, c=sample.C_conflict, cmap="viridis", s=10, alpha=0.5, vmin=0, vmax=np.quantile(sample.C_conflict, 0.995))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Q^(0) before conflict resolution", ylabel="Q* after conflict resolution",
           title="Conflict-adjusted quality vs base quality")
    fig.colorbar(sc, label="Conflict C")
    fig.tight_layout()
    fig.savefig(OUT / "fig_Q1_2_Q0_vs_Qstar.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.scatterplot(x=graph, y=c_var, s=6, alpha=0.3, ax=axes[0])
    axes[0].set(xlabel="Graph conflict C", ylabel="Signal variance", title="C vs variance")
    sns.scatterplot(x=graph, y=c_max, s=6, alpha=0.3, ax=axes[1])
    axes[1].set(xlabel="Graph conflict C", ylabel="Max pairwise gap", title="C vs max gap")
    fig.tight_layout()
    fig.savefig(OUT / "fig_Q1_2_baseline_conflict.png", dpi=180)
    plt.close(fig)

    print(baseline.to_string(index=False))
    print("Q* vs Q0 correlation:", float(np.corrcoef(final.Q0, final.Q_star)[0, 1]))
    high = final.C_conflict >= final.C_conflict.quantile(0.99)
    print("High-conflict median |Q*-Q0|:", float(np.median(np.abs(final.loc[high, "Q_star"] - final.loc[high, "Q0"]))))
    print("Low-conflict median |Q*-Q0|:", float(np.median(np.abs(final.loc[~high, "Q_star"] - final.loc[~high, "Q0"]))))


if __name__ == "__main__":
    main()
