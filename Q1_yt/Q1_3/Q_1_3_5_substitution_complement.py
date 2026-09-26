"""Compute 17x17 substitution effects and complementarity network."""

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


def main() -> None:
    domains = pd.read_csv(OUT / "train_domain_names.csv")["domain"].tolist()
    loss_domains = pd.read_csv(OUT / "loss_domain_names.csv")["loss_domain"].tolist()
    B = pd.read_csv(OUT / "B_multitask_sparse_scheffe.csv", index_col=0)
    beta = B.loc[domains].to_numpy(dtype=float)          # 17 x 13
    gamma = {frozenset(name.split(":")): B.loc[name].to_numpy(dtype=float)
             for name in B.index if ":" in name}          # pair -> 13

    P_train = pd.read_csv(OUT / "P_train_1m.csv").to_numpy(dtype=float)
    p_ref = P_train.mean(axis=0)

    # derivative dz_k / dp_i at p_ref.
    deriv = np.zeros((17, 13), dtype=float)
    for i in range(17):
        d = beta[i].copy()
        for j in range(17):
            if i == j:
                continue
            d += gamma[frozenset({domains[i], domains[j]})] * p_ref[j]
        deriv[i] = d

    substitution = np.zeros((17, 17), dtype=float)
    for i in range(17):
        for j in range(17):
            substitution[i, j] = float((deriv[i] - deriv[j]).mean())
    sub_df = pd.DataFrame(substitution, index=domains, columns=domains)
    sub_df.to_csv(OUT / "substitution_matrix.csv", encoding="utf-8-sig")

    complement = np.zeros((17, 17), dtype=float)
    for i in range(17):
        for j in range(i + 1, 17):
            c = -float(gamma[frozenset({domains[i], domains[j]})].mean())
            complement[i, j] = complement[j, i] = c
    comp_df = pd.DataFrame(complement, index=domains, columns=domains)
    comp_df.to_csv(OUT / "complementarity_matrix.csv", encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sub_df, cmap="vlag", center=0, square=True, ax=ax,
                cbar_kws={"label": "S(i<-j): dG when moving share j->i (lower=better)"})
    ax.set_title("Domain substitution effect matrix at mean mixture")
    fig.tight_layout()
    fig.savefig(OUT / "fig_Q1_3_substitution_matrix.png", dpi=180)
    plt.close(fig)

    # Complementarity network: top |C_ij| edges.
    fig, ax = plt.subplots(figsize=(11, 9))
    angle = np.linspace(0, 2 * np.pi, 17, endpoint=False)
    pos = np.column_stack([np.cos(angle), np.sin(angle)])
    threshold = np.quantile(np.abs(complement[np.triu_indices(17, 1)]), 0.75)
    for i in range(17):
        for j in range(i + 1, 17):
            if abs(complement[i, j]) >= threshold:
                color = "#216e77" if complement[i, j] > 0 else "#c06d4b"
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color=color, alpha=0.7, linewidth=0.5 + 2.5 * abs(complement[i, j]) / max(abs(complement).max(), 1e-9))
    ax.scatter(pos[:, 0], pos[:, 1], s=240, color="white", edgecolor="#333")
    for d, (x0, y0) in zip(domains, pos):
        ax.text(x0 * 1.12, y0 * 1.12, d.replace("_", " "), ha="center", va="center", fontsize=7)
    ax.axis("off")
    ax.set(xlim=(-1.45, 1.45), ylim=(-1.45, 1.45), title="Domain complementarity network (top quartile |C|)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_Q1_3_complementarity_network.png", dpi=180)
    plt.close(fig)

    summary = {
        "reference_mixture": p_ref.tolist(),
        "best_substitution_pairs": [
            {"from": domains[int(k % 17)], "to": domains[int(k // 17)], "S": float(substitution[int(k // 17), int(k % 17)])}
            for k in np.argsort(substitution, axis=None)[:10]
        ],
        "most_complementary_pairs": sorted(
            [{"pair": f"{domains[i]}:{domains[j]}", "C": float(complement[i, j])}
             for i in range(17) for j in range(i + 1, 17)],
            key=lambda x: -x["C"],
        )[:10],
        "most_redundant_pairs": sorted(
            [{"pair": f"{domains[i]}:{domains[j]}", "C": float(complement[i, j])}
             for i in range(17) for j in range(i + 1, 17)],
            key=lambda x: x["C"],
        )[:10],
    }
    (OUT / "substitution_complement_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("best substitution S(i<-j) rows (smaller = better):")
    print(sub_df.stack().sort_values().head(10).to_string())
    print("complementary top pairs:")
    print(comp_df.where(np.triu(np.ones(comp_df.shape), 1).astype(bool)).stack().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
