"""Two final checks: model-gap uncertainty and interpretability eligibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.model_selection import RepeatedKFold


OUT = Path(__file__).resolve().parent / "results"
SEED = 20260923


def macro_r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y_true - Y_true.mean(axis=0)) ** 2, axis=0)
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-9)))


def clr(P: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    P_pos = np.clip(P, eps, None)
    P_pos = P_pos / P_pos.sum(axis=1, keepdims=True)
    logp = np.log(P_pos)
    return logp - logp.mean(axis=1, keepdims=True)


def main() -> None:
    domains = pd.read_csv(OUT / "train_domain_names.csv")["domain"].tolist()
    X2 = pd.read_csv(OUT / "X2_train_1m.csv").to_numpy(dtype=float)
    Y = pd.read_csv(OUT / "Z_train.csv").to_numpy(dtype=float)
    P = pd.read_csv(OUT / "P_train_1m.csv").to_numpy(dtype=float)
    Xc = clr(P)
    info = json.loads((OUT / "scheffe_fit_summary.json").read_text(encoding="utf-8"))
    alpha = float(info["best_multitask"]["alpha"])
    l1_ratio = float(info["best_multitask"]["l1_ratio"])
    ridge_alpha = float(info["best_ridge"]["alpha"])

    # 1. Repeated CV model gap.
    rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    gap_rows = []
    for tr, va in rkf.split(X2):
        r = Ridge(alpha=ridge_alpha, random_state=SEED)
        r.fit(Xc[tr], Y[tr])
        r2_ilr = macro_r2(Y[va], r.predict(Xc[va]))
        s = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000,
                                tol=1e-5, fit_intercept=False, random_state=SEED)
        s.fit(X2[tr], Y[tr])
        r2_scheffe = macro_r2(Y[va], s.predict(X2[va]))
        gap_rows.append({"fold": len(gap_rows), "r2_ilr": r2_ilr, "r2_scheffe": r2_scheffe,
                         "delta_r2_ilr_minus_scheffe": r2_ilr - r2_scheffe})
    gap = pd.DataFrame(gap_rows)
    gap.to_csv(OUT / "model_gap_uncertainty.csv", index=False, encoding="utf-8-sig")
    gap_summary = {
        "mean_r2_ilr": float(gap.r2_ilr.mean()),
        "mean_r2_scheffe": float(gap.r2_scheffe.mean()),
        "mean_delta_r2": float(gap.delta_r2_ilr_minus_scheffe.mean()),
        "delta_r2_ci95_low": float(np.quantile(gap.delta_r2_ilr_minus_scheffe, 0.025)),
        "delta_r2_ci95_high": float(np.quantile(gap.delta_r2_ilr_minus_scheffe, 0.975)),
    }
    (OUT / "model_gap_summary.json").write_text(json.dumps(gap_summary, indent=2) + "\n", encoding="utf-8")

    # 2. Interpretability eligibility gate.
    support = pd.read_csv(OUT / "mixture_domain_support.csv").set_index("domain")
    coverage = (support.nonzero_share >= 0.40) & (support.max_share >= 0.05)
    sub = pd.read_csv(OUT / "substitution_bootstrap.csv")
    comp = pd.read_csv(OUT / "complementarity_bootstrap.csv")
    joint = pd.read_csv(OUT / "pairwise_joint_effect_at_reference.csv", index_col=0)

    def eligible_pair(a, b):
        return bool(coverage.get(a, False) and coverage.get(b, False))

    sub["coverage_eligible"] = [eligible_pair(r["from"], r["to"]) for r in sub.to_dict("records")]
    sub["sign_stable"] = sub.sign_stable_share >= 0.95
    sub["magnitude_reasonable"] = sub.S_mean.abs() <= 5.0
    sub["interpretable"] = sub.coverage_eligible & sub.sign_stable & sub.magnitude_reasonable
    sub.to_csv(OUT / "substitution_eligibility.csv", index=False, encoding="utf-8-sig")

    comp["coverage_eligible"] = [eligible_pair(r["domain_i"], r["domain_j"]) for r in comp.to_dict("records")]
    comp["sign_stable"] = comp.sign_stable_share >= 0.95
    joint_vals = np.abs(joint.to_numpy(dtype=float))
    threshold = float(np.quantile(joint_vals[np.triu_indices_from(joint_vals, 1)], 0.95))
    comp["joint_effect_abs"] = [float(abs(joint.loc[r["domain_i"], r["domain_j"]])) for r in comp.to_dict("records")]
    comp["magnitude_reasonable"] = comp.joint_effect_abs <= threshold
    comp["interpretable"] = comp.coverage_eligible & comp.sign_stable & comp.magnitude_reasonable
    comp.to_csv(OUT / "complementarity_eligibility.csv", index=False, encoding="utf-8-sig")

    eligibility_summary = {
        "coverage_rule": "nonzero_share>=0.40 and max_share>=0.05",
        "sign_stability_rule": "bootstrap sign-stable share>=0.95",
        "substitution_magnitude_rule": "|S|<=5.0",
        "complementarity_magnitude_rule": "|joint effect|<=P95 of pairwise joint effects",
        "substitution_interpretable_share": float(sub.interpretable.mean()),
        "complementarity_interpretable_share": float(comp.interpretable.mean()),
        "low_coverage_domains": support.loc[~coverage].index.tolist(),
    }
    (OUT / "effect_eligibility_summary.json").write_text(json.dumps(eligibility_summary, indent=2) + "\n", encoding="utf-8")

    print("Model gap:", gap_summary)
    print("Eligibility:", eligibility_summary)


if __name__ == "__main__":
    main()
