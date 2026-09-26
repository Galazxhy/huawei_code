"""Convert conflict into indicator reliability and sample-adaptive Q*."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
Q1_1 = ROOT.parent / "Q1_1"
LAMBDA = 0.5


def main() -> None:
    weights = pd.read_csv(Q1_1 / "results" / "A1_model_indicator_weights.csv")
    clusters = sorted(weights.cluster.unique().tolist())
    cluster_of = dict(zip(weights.indicator, weights.cluster))
    within_weight = dict(zip(weights.indicator, weights.within_cluster_weight))
    scale_json = json.loads((OUT / "conflict_scale.json").read_text(encoding="utf-8"))
    scales = scale_json["scales"]
    domains = scale_json["domains"]

    domain_summary = []
    for dataset, h_file in (("A1", "A1_sample_Q0.csv"), ("A2", "A2_sample_Q0.csv"), ("A3", "A3_sample_Q0.csv")):
        hierarchy = pd.read_csv(Q1_1 / "results" / h_file)
        scores = pd.read_csv(OUT / f"{dataset}_conflict_scores.csv")
        with np.load(OUT / f"{dataset}_anomaly.npz", allow_pickle=True) as archive:
            r = archive["r"]
            flag = archive["flag"]
            np_names = [str(x) for x in archive["names"]]
        names = np_names
        if set(names) != set(weights.indicator.tolist()):
            raise ValueError(f"Indicator set mismatch for {dataset}")

        reliability = np.ones_like(r, dtype=float)
        for j, name in enumerate(names):
            for domain in domains:
                mask = scores.domain.to_numpy() == domain
                if not mask.any():
                    continue
                tau = scales[name][domain]["r_threshold_alpha"]
                excess = np.maximum(r[mask, j] - tau, 0.0)
                reliability[mask, j] = np.where(flag[mask, j], np.exp(-LAMBDA * excess), 1.0)

        cluster_rel = np.empty((len(scores), len(clusters)), dtype=float)
        for ci, cluster in enumerate(clusters):
            members = weights.loc[weights.cluster == cluster, "indicator"].tolist()
            idx = [names.index(m) for m in members]
            w = np.array([within_weight[m] for m in members])
            cluster_rel[:, ci] = reliability[:, idx] @ w
        U = cluster_rel.mean(axis=1)
        adaptive = (cluster_rel / len(clusters)) / np.maximum(U, 1e-9)[:, None]

        h_cols = [f"H_{c}" for c in clusters]
        h = hierarchy[h_cols].to_numpy(dtype=float)
        q_star = (adaptive * h).sum(axis=1)

        final = scores[["id", "domain", "n_conflict", "A_coverage", "B_intensity", "C_conflict"]].copy()
        final["U_reliability"] = U
        final["Q0"] = hierarchy["Q0"].to_numpy()
        final["Q_star"] = q_star
        final.to_csv(OUT / f"{dataset}_final_quality.csv", index=False, encoding="utf-8-sig")

        for domain, group in final.groupby("domain", sort=True):
            domain_summary.append({
                "dataset": dataset, "domain": domain, "n": len(group),
                "mean_Q0": float(group.Q0.mean()),
                "reliability_weighted_Qstar": float((group.U_reliability * group.Q_star).sum() / group.U_reliability.sum()),
                "mean_Qstar": float(group.Q_star.mean()),
                "mean_U": float(group.U_reliability.mean()),
                "mean_C": float(group.C_conflict.mean()),
            })
        print(f"{dataset}: Q* and U written; rows={len(final):,}; "
              f"U mean={float(U.mean()):.4f}, corr(Q0,Q*)={float(np.corrcoef(final.Q0, q_star)[0,1]):.4f}")

    domain_table = pd.DataFrame(domain_summary)
    domain_table["delta_Qstar_minus_Q0"] = domain_table.reliability_weighted_Qstar - domain_table.mean_Q0
    domain_table.to_csv(OUT / "Qstar_domain_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "reliability_summary.json").write_text(json.dumps({
        "lambda": LAMBDA, "clusters": clusters,
        "rule": "indicator reliability -> within-cluster weighted reliability -> adaptive cluster weight",
        "U_definition": "mean cluster reliability = sum_k (1/K) R_ik",
        "Qstar_definition": "sum_k v_ik* H_ik, where v_ik* = (1/K)R_ik/U_i",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
