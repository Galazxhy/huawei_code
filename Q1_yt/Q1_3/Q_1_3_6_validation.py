"""Validate the mixture-response model on same-scale, cross-scale and extrapolated sets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


OUT = Path(__file__).resolve().parent / "results"
SEED = 20260923


def hellinger(P: np.ndarray, P_train: np.ndarray) -> np.ndarray:
    roots = np.sqrt(P)
    root_train = np.sqrt(P_train)
    return np.min(np.sqrt(0.5 * ((roots[:, None, :] - root_train[None, :, :]) ** 2).sum(axis=2)), axis=1)


def pairwise_concordance(a: np.ndarray, b: np.ndarray, n_pairs: int = 8000) -> float:
    n = len(a)
    rng = np.random.default_rng(SEED)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    valid = i != j
    i, j = i[valid], j[valid]
    return float(np.mean(np.sign(a[i] - a[j]) == np.sign(b[i] - b[j])))


def main() -> None:
    domains = pd.read_csv(OUT / "train_domain_names.csv")["domain"].tolist()
    loss_domains = pd.read_csv(OUT / "loss_domain_names.csv")["loss_domain"].tolist()
    B = pd.read_csv(OUT / "B_multitask_sparse_scheffe.csv", index_col=0).to_numpy(dtype=float)  # 153 x 13
    X_train = pd.read_csv(OUT / "X2_train_1m.csv").to_numpy(dtype=float)
    Z_train = pd.read_csv(OUT / "Z_train.csv").to_numpy(dtype=float)

    def predict(key: str) -> np.ndarray:
        X = pd.read_csv(OUT / f"X2_{key}.csv").to_numpy(dtype=float)
        return X @ B

    # Same-scale validation (1M).
    pred_1m = predict("test_1m")
    obs_1m = pd.read_csv(OUT / "Z_test_1m.csv").to_numpy(dtype=float)
    per_task = []
    for k, loss_name in enumerate(loss_domains):
        p, o = pred_1m[:, k], obs_1m[:, k]
        ss_res = np.sum((o - p) ** 2)
        ss_tot = np.sum((o - o.mean()) ** 2)
        per_task.append({"loss_domain": loss_name, "rmse": float(np.sqrt(np.mean((o - p) ** 2))),
                         "mae": float(np.mean(np.abs(o - p))), "r2": float(1 - ss_res / max(ss_tot, 1e-9)),
                         "pearson": float(np.corrcoef(o, p)[0, 1])})
    per_task_df = pd.DataFrame(per_task)
    per_task_df.to_csv(OUT / "validation_1m_per_task.csv", index=False, encoding="utf-8-sig")
    macro_rmse = float(per_task_df.rmse.mean())
    macro_r2 = float(per_task_df.r2.mean())

    G_pred_1m = pred_1m.mean(axis=1)
    G_obs_1m = obs_1m.mean(axis=1)
    pred_vs_obs = pd.DataFrame({"pred_G": G_pred_1m, "obs_G": G_obs_1m})
    pred_vs_obs.to_csv(OUT / "G_pred_obs_1m.csv", index=False, encoding="utf-8-sig")

    # Cross-scale and extrapolation rank/concordance.
    rows = []
    for key, kind in (("test_60m", "60M test"), ("test_1b", "1B test"),
                      ("est_10b", "10B extrapolated"), ("est_70b", "70B extrapolated")):
        pred = predict(key)
        obs = pd.read_csv(OUT / f"Z_{key}.csv").to_numpy(dtype=float)
        g_pred, g_obs = pred.mean(axis=1), obs.mean(axis=1)
        rows.append({
            "scale": kind, "n": len(g_pred),
            "spearman_rank": float(spearmanr(g_pred, g_obs).statistic),
            "pairwise_concordance": pairwise_concordance(g_pred, g_obs),
        })
    scale_table = pd.DataFrame(rows)
    scale_table.to_csv(OUT / "cross_scale_validation.csv", index=False, encoding="utf-8-sig")

    # Composition-space distance vs same-scale error.
    P_train = pd.read_csv(OUT / "P_train_1m.csv").to_numpy(dtype=float)
    P_test_1m = pd.read_csv(OUT / "P_test_1m.csv").to_numpy(dtype=float)
    d_train = hellinger(P_test_1m, P_train)
    error_1m = np.abs(G_pred_1m - G_obs_1m)
    dist_df = pd.DataFrame({"hellinger_to_train": d_train, "abs_error_G": error_1m})
    dist_df.to_csv(OUT / "error_vs_distance_1m.csv", index=False, encoding="utf-8-sig")

    summary = {
        "test_1m_macro_rmse": macro_rmse,
        "test_1m_macro_r2": macro_r2,
        "cross_scale": rows,
        "hellinger_test_1m": {"median": float(np.median(d_train)), "p90": float(np.quantile(d_train, 0.90))},
    }
    (OUT / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=G_obs_1m, y=G_pred_1m, s=22, ax=axes[0])
    lim = [min(G_obs_1m.min(), G_pred_1m.min()), max(G_obs_1m.max(), G_pred_1m.max())]
    axes[0].plot(lim, lim, color="gray", linestyle="--")
    axes[0].set(xlabel="observed G (standardized)", ylabel="predicted G", title="1M validation: macro G")
    sns.scatterplot(x=d_train, y=error_1m, s=22, ax=axes[1])
    axes[1].set(xlabel="Hellinger distance to nearest train mixture", ylabel="|pred G - obs G|",
                title="Error vs composition-space distance")
    fig.tight_layout()
    fig.savefig(OUT / "fig_Q1_3_validation.png", dpi=180)
    plt.close(fig)

    print(per_task_df.to_string(index=False))
    print("macro rmse", macro_rmse, "macro r2", macro_r2)
    print(scale_table.to_string(index=False))


if __name__ == "__main__":
    main()
