"""Sensitivity of conflict detection and Q* to alpha and lambda."""

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


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
Q1_1 = ROOT.parent / "Q1_1"

PRIMARY_ALPHA = 0.025
PRIMARY_LAMBDA = 0.5


def cluster_quality(r, flag, tau_by_jd, domains, domain_arr, weights, hierarchy, h_cols, alpha, lambda_):
    names = weights.indicator.tolist()
    clusters = sorted(weights.cluster.unique().tolist())
    cluster_of = dict(zip(weights.indicator, weights.cluster))
    within_weight = dict(zip(weights.indicator, weights.within_cluster_weight))
    reliability = np.ones_like(r)
    for j, name in enumerate(names):
        for domain in domains:
            mask = domain_arr == domain
            if not mask.any():
                continue
            tau = tau_by_jd[(name, domain)]
            excess = np.maximum(r[mask, j] - tau, 0.0)
            reliability[mask, j] = np.where(flag[mask, j], np.exp(-lambda_ * excess), 1.0)
    cluster_rel = np.empty((len(domain_arr), len(clusters)))
    for ci, cluster in enumerate(clusters):
        members = weights.loc[weights.cluster == cluster, "indicator"].tolist()
        idx = [names.index(m) for m in members]
        w = np.array([within_weight[m] for m in members])
        cluster_rel[:, ci] = reliability[:, idx] @ w
    U = cluster_rel.mean(axis=1)
    adaptive = (cluster_rel / len(clusters)) / np.maximum(U, 1e-9)[:, None]
    h = hierarchy[h_cols].to_numpy(dtype=float)
    q_star = (adaptive * h).sum(axis=1)
    n_conflict = flag.sum(axis=1)
    A = n_conflict / len(names)
    B = np.where(n_conflict > 0, (flag * r).sum(axis=1) / np.maximum(n_conflict, 1), 0.0)
    C = np.sqrt(A * B)
    return C, U, q_star


def main() -> None:
    weights = pd.read_csv(Q1_1 / "results" / "A1_model_indicator_weights.csv")
    hierarchy = pd.read_csv(Q1_1 / "results" / "A1_sample_Q0.csv")
    clusters = sorted(weights.cluster.unique().tolist())
    h_cols = [f"H_{c}" for c in clusters]
    names = weights.indicator.tolist()
    domains = sorted(hierarchy.domain.unique().tolist())

    scores = pd.read_csv(OUT / "A1_conflict_scores.csv")
    with np.load(OUT / "A1_anomaly.npz", allow_pickle=True) as archive:
        r = archive["r"]
    domain_arr = scores.domain.to_numpy()
    base_C = scores.C_conflict.to_numpy()
    base_Q = pd.read_csv(OUT / "A1_final_quality.csv").Q_star.to_numpy()

    rows = []

    def evaluate(alpha, lambda_):
        flag = np.zeros_like(r, dtype=bool)
        tau_by_jd = {}
        for j, name in enumerate(names):
            for domain in domains:
                mask = domain_arr == domain
                rj = r[mask, j]
                sorted_r = np.sort(rj)
                tau = float(np.quantile(sorted_r, 1 - alpha))
                tau_by_jd[(name, domain)] = max(tau, 1e-6)
                p_conf = 1 - np.searchsorted(sorted_r, rj, side="left") / len(sorted_r)
                flag[mask, j] = p_conf < alpha
        C, U, q_star = cluster_quality(r, flag, tau_by_jd, domains, domain_arr, weights, hierarchy, h_cols, alpha, lambda_)
        top_n = int(len(C) * 0.01)
        top_base = set(np.argpartition(base_C, -top_n)[-top_n:])
        top_new = set(np.argpartition(C, -top_n)[-top_n:])
        domain_rank = spearmanr(
            pd.Series(C).groupby(domain_arr).mean(),
            pd.Series(base_C).groupby(domain_arr).mean(),
        ).statistic
        rows.append({
            "alpha": alpha, "lambda": lambda_,
            "spearman_C_to_primary": float(spearmanr(base_C, C).statistic),
            "top1pct_C_overlap": len(top_base & top_new) / top_n,
            "domain_C_rank_spearman": float(domain_rank),
            "spearman_Qstar_to_primary": float(spearmanr(base_Q, q_star).statistic),
            "mean_U": float(U.mean()),
            "flagged_indicator_rate": float(flag.mean()),
        })

    for alpha in (0.01, 0.025, 0.05, 0.10):
        evaluate(alpha, PRIMARY_LAMBDA)
    for lambda_ in (0.3, 0.5, 1.0):
        evaluate(PRIMARY_ALPHA, lambda_)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "conflict_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    alpha_rows = summary[summary["lambda"] == PRIMARY_LAMBDA]
    sns.lineplot(data=alpha_rows, x="alpha", y="spearman_Qstar_to_primary", marker="o", ax=axes[0], label="Q* rank agreement")
    sns.lineplot(data=alpha_rows, x="alpha", y="top1pct_C_overlap", marker="s", ax=axes[0], label="top1% C overlap")
    axes[0].set(xlabel="alpha", ylabel="agreement with primary")
    lambda_rows = summary[summary.alpha == PRIMARY_ALPHA]
    sns.lineplot(data=lambda_rows, x="lambda", y="spearman_Qstar_to_primary", marker="o", ax=axes[1], label="Q* rank agreement")
    sns.lineplot(data=lambda_rows, x="lambda", y="top1pct_C_overlap", marker="s", ax=axes[1], label="top1% C overlap")
    axes[1].set(xlabel="lambda", ylabel="agreement with primary")
    fig.tight_layout()
    fig.savefig(OUT / "fig_Q1_2_sensitivity.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
