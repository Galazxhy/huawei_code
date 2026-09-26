"""Learn the domain-balanced quality dependency graph and test edge stability."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


OUT = Path(__file__).resolve().parent / "results"
SEED = 20260923

RHO0 = 0.10
PI0 = 0.95
BOOTSTRAP_REPLICATES = 300


def weighted_correlation(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    center = np.average(data, axis=0, weights=weights)
    centered = data - center
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    deviations = np.sqrt(np.diag(covariance))
    denominator = np.outer(deviations, deviations)
    return np.clip(np.divide(covariance, denominator, out=np.eye(data.shape[1]), where=denominator > 0), -1, 1)


def main() -> None:
    frame = pd.read_csv(OUT / "A1_directional_signals.csv")
    names = list(frame.columns[2:])
    domain = frame["domain"].to_numpy()
    groups = {d: np.flatnonzero(domain == d) for d in np.unique(domain)}
    weights = np.array([1 / (len(groups[d]) * len(groups)) for d in domain])
    x = frame[names].to_numpy(dtype=float)
    ranks = np.column_stack([rankdata(x[:, j]) for j in range(x.shape[1])])
    full_corr = weighted_correlation(ranks, weights)

    rng = np.random.default_rng(SEED)
    boot_corr = np.zeros((BOOTSTRAP_REPLICATES, len(names), len(names)), dtype=float)
    for b in range(BOOTSTRAP_REPLICATES):
        idx = np.concatenate([rng.choice(indices, size=min(1200, len(indices)), replace=True)
                              for indices in groups.values()])
        br = np.column_stack([rankdata(x[idx, j]) for j in range(len(names))])
        boot_corr[b] = weighted_correlation(br, np.ones(len(idx)) / len(idx))

    presence = np.abs(boot_corr) >= RHO0
    stability = presence.mean(axis=0)
    sign_agreement = np.zeros((len(names), len(names)), dtype=float)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                sign_agreement[i, j] = 1.0
                continue
            sign_agreement[i, j] = float(np.mean(np.sign(boot_corr[:, i, j]) == np.sign(full_corr[i, j])))

    edge_mask = (np.abs(full_corr) >= RHO0) & (stability >= PI0)
    np.fill_diagonal(edge_mask, False)

    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rows.append({
                "source": names[i],
                "target": names[j],
                "rho": round(float(full_corr[i, j]), 6),
                "bootstrap_presence": round(float(stability[i, j]), 6),
                "bootstrap_sign_agreement": round(float(sign_agreement[i, j]), 6),
                "edge": bool(edge_mask[i, j]),
            })
    edges = pd.DataFrame(rows)
    edges.to_csv(OUT / "A1_quality_dependency_edges.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(full_corr, index=names, columns=names).to_csv(
        OUT / "A1_quality_dependency_correlation.csv", encoding="utf-8-sig")

    adjacency = pd.DataFrame(edge_mask, index=names, columns=names).astype(int)
    adjacency.to_csv(OUT / "A1_quality_dependency_adjacency.csv", encoding="utf-8-sig")

    degree = adjacency.sum(axis=1)
    summary = {
        "rho0": RHO0, "pi0": PI0, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "n_nodes": len(names), "n_edges": int(edge_mask.sum() / 2),
        "isolated_nodes": degree[degree == 0].index.tolist(),
        "degree_min": int(degree.min()), "degree_max": int(degree.max()),
        "degree_median": float(degree.median()),
    }
    (OUT / "A1_relation_graph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 9))
    angle = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    pos = np.column_stack([np.cos(angle), np.sin(angle)])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if edge_mask[i, j]:
                color = "#216e77" if full_corr[i, j] > 0 else "#c06d4b"
                width = 0.5 + 3.0 * min(1.0, abs(full_corr[i, j]))
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color=color, linewidth=width, alpha=0.7, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=210, color="white", edgecolor="#333", zorder=2)
    for name, (x0, y0) in zip(names, pos):
        label = name.replace("rps_doc_", "").replace("rps_lines_", "").replace("modernbert_", "mb_")
        ax.text(x0 * 1.12, y0 * 1.12, label, ha="center", va="center", fontsize=7)
    ax.set(xlim=(-1.45, 1.45), ylim=(-1.45, 1.45), title=f"Quality dependency graph (|rho|>={RHO0}, stability>={PI0})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_Q1_2_relation_graph.png", dpi=180)
    plt.close(fig)

    print("Relation graph:", summary)
    print("Degree per indicator:")
    print(degree.to_string())


if __name__ == "__main__":
    main()
