"""Evidence checks: domain support, interpretable pairwise effects, cross-scale slopes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


OUT = Path(__file__).resolve().parent / "results"
DELTA = 0.01


def scheffe(P: np.ndarray, domains: list[str]) -> np.ndarray:
    parts = [P]
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            parts.append((P[:, i] * P[:, j])[:, None])
    return np.hstack(parts)


def main() -> None:
    domains = pd.read_csv(OUT / "train_domain_names.csv")["domain"].tolist()
    loss_domains = pd.read_csv(OUT / "loss_domain_names.csv")["loss_domain"].tolist()
    B = pd.read_csv(OUT / "B_multitask_sparse_scheffe.csv", index_col=0).to_numpy(dtype=float)
    P_train = pd.read_csv(OUT / "P_train_1m.csv").to_numpy(dtype=float)
    p_ref = P_train.mean(axis=0)

    # 1. Domain coverage/support.
    support_rows = []
    for i, d in enumerate(domains):
        support_rows.append({"domain": d, "mean_share": float(P_train[:, i].mean()),
                             "nonzero_share": float((P_train[:, i] > 1e-4).mean()),
                             "max_share": float(P_train[:, i].max())})
    support = pd.DataFrame(support_rows)
    support.to_csv(OUT / "mixture_domain_support.csv", index=False, encoding="utf-8-sig")

    # 2. Interpretable pairwise joint effect at reference (actual standardized G change).
    G_ref = float((scheffe(p_ref[None, :], domains) @ B.mean(axis=1)).item())
    joint = np.zeros((len(domains), len(domains)))
    for i in range(len(domains)):
        for j in range(len(domains)):
            p = p_ref.copy()
            p[i] += DELTA
            p[j] += DELTA
            others = [k for k in range(len(domains)) if k not in (i, j)]
            total_other = p[others].sum()
            if total_other > 1e-9:
                p[others] *= (1 - 2 * DELTA) / total_other
            joint[i, j] = float((scheffe(p[None, :], domains) @ B.mean(axis=1)).item() - G_ref)
    joint_df = pd.DataFrame(joint, index=domains, columns=domains)
    joint_df.to_csv(OUT / "pairwise_joint_effect_at_reference.csv", encoding="utf-8-sig")

    # 3. Per-domain cross-scale rank and scale slopes.
    rows = []
    for key, kind in (("test_60m", "60M test"), ("test_1b", "1B test"),
                      ("est_10b", "10B extrapolated"), ("est_70b", "70B extrapolated")):
        pred = pd.read_csv(OUT / f"X2_{key}.csv").to_numpy(dtype=float) @ B
        obs = pd.read_csv(OUT / f"Z_{key}.csv").to_numpy(dtype=float)
        for k, loss_name in enumerate(loss_domains):
            p_k, o_k = pred[:, k], obs[:, k]
            rho = float(spearmanr(p_k, o_k).statistic)
            b = float(np.polyfit(p_k, o_k, 1)[0])
            rows.append({"scale": kind, "loss_domain": loss_name,
                         "spearman_rank": rho, "scale_slope_b": b})
        g_pred, g_obs = pred.mean(axis=1), obs.mean(axis=1)
        rows.append({"scale": kind, "loss_domain": "__MACRO_G__",
                     "spearman_rank": float(spearmanr(g_pred, g_obs).statistic),
                     "scale_slope_b": float(np.polyfit(g_pred, g_obs, 1)[0])})
    cross_domain = pd.DataFrame(rows)
    cross_domain.to_csv(OUT / "cross_scale_per_domain.csv", index=False, encoding="utf-8-sig")

    print("Domain support:")
    print(support.to_string(index=False))
    print("Cross-scale macro slopes:")
    print(cross_domain[cross_domain.loss_domain == "__MACRO_G__"].to_string(index=False))


if __name__ == "__main__":
    main()
